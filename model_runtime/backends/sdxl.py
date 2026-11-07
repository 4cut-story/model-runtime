"""SDXL 만화풍 그림 (GPU PC 전용).

기본은 SDXL-Lightning 8 step LoRA로 그린다 (30 step 대비 약 7배 빠름). Lightning은 guidance 0으로
그리므로 negative prompt가 효과가 없다. 768px은 16GB에 다 올라가 CPU offload 없이 돈다.
VAE tiling·slicing은 1024px로 되돌려도 VRAM이 넘치지 않게 켜 둔다.

keep_in_ram이면 내릴 때 지우지 않고 시스템 RAM으로 옮겨 둔다. 다음 그림 작업은 디스크에서 다시
읽지 않고 GPU로 옮기기만 한다 (GPU PC 측정: 디스크에서 86~194초 → RAM에서 1.2~1.4초, RAM 약 5.5GB).

캐릭터 참조(conversation-service 캐릭터 DB)가 오면 IP-Adapter를 처음 한 번 올린다
(Plus, ViT-H 인코더, 약 4초, VRAM +2GB). 참조 embedding은 미리 계산해 둔 것을
ip_adapter_image_embeds로 넣고, 두 명이 나오는 컷은 왼쪽·오른쪽 마스크로 자리를 나눈다.
Plus는 얼굴뿐 아니라 머리 모양·옷까지 따르므로, 앞쪽 step은 약하게 걸어 구도는 프롬프트가 정하고
뒤쪽 step은 강하게 걸어 정체성을 넣는다(IpSchedule). Plus-Face는 얼굴만 잡아 두 명 컷에서 옷·머리가
컷마다 바뀌었다. embedding은 ViT-H 인코더 출력이라 Plus와 Plus-Face가 같은 것을 쓴다.
어댑터를 올린 뒤 참조가 없는 컷은 강도 0으로 그린다(텍스트만으로 그린 것과 픽셀이 같다).
GPU PC 벤치: 4컷 5.45초 → 약 5.6초, 예약 VRAM 약 12.5GiB.

두 명 이상 나오는 컷에 자리별 인물 프롬프트(images[].regions)가 오면 cross-attention에서 자리마다
그 인물 프롬프트 결과를 섞는다(regional.py). guidance > 1(CFG)이면 배치 앞 절반이 negative라
뒤 절반(프롬프트 쪽)에만 섞는다. 실패하면 자리 프롬프트 없이 다시 그린다.

컷에 인물 확인(images[].checks)이 오면 그린 뒤 CLIP(ViT-L/14, fp16 약 0.9GB)으로 자리마다
성별·나이·동물을 확인하고, 틀리면 시드를 바꿔 최대 두 번 더 그려 가장 잘 맞는 그림을
쓴다(check.py). 확인이 실패하면
처음 그림을 그대로 쓴다.
"""

from __future__ import annotations

import functools
import gc
import io
import logging
from typing import Any

from . import Reference, check, long_prompt, pose, regional
from .ip_schedule import IpSchedule

log = logging.getLogger(__name__)

# IP-Adapter 이미지 인코더 (ViT-H). 어댑터와 같은 레포 안 경로
IP_ADAPTER_ENCODER = "models/image_encoder"
# Plus 계열 embedding 모양: [이미지 수, 참조 수, 토큰, 차원]
EMBED_SHAPE = (1, 1, 257, 1280)
# 그린 뒤 인물 확인에 쓰는 CLIP
CHECK_MODEL = "openai/clip-vit-large-patch14"
# 뼈대 파이프라인이 SDXL 파이프라인에서 넘겨받는 부품
ADAPTER_PARTS = (
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "unet",
    "scheduler",
    "feature_extractor",
    "image_encoder",
)


class SdxlBackend:
    def __init__(
        self,
        model_name: str,
        cpu_offload: bool = False,
        lora: str = "",
        steps: int = 8,
        guidance: float = 0.0,
        keep_in_ram: bool = False,
        ip_adapter: str = "",
        ip_schedule: IpSchedule | None = None,
        variant: str = "fp16",
        image_check: bool = False,
        pose_adapter: str = "",
        pose_scale: float = 0.8,
        pose_factor: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.cpu_offload = cpu_offload
        self.lora = lora
        self.steps = steps
        self.guidance = guidance
        # CPU offload는 이미 컷마다 GPU↔CPU를 오가므로 RAM에 따로 둘 필요가 없다
        self.keep_in_ram = keep_in_ram and not cpu_offload
        self.ip_adapter = ip_adapter
        self.ip_schedule = ip_schedule or IpSchedule()
        # 체크포인트 레포에 fp16 variant 파일이 없으면(애니 계열 등) 비운다
        self.variant = variant or None
        self._pipe: Any = None
        self._on_gpu = False
        self._ip_loaded = False
        self.image_check = image_check
        self._clip: Any = None
        # 컷마다 인물 수·자리·키를 뼈대 그림으로 넣는 T2I-Adapter (HF 레포). 비우면 쓰지 않는다
        self.pose_adapter = pose_adapter
        self.pose_scale = pose_scale
        # 앞쪽 몇 할의 step에만 뼈대를 건다. 구도만 잡고 나머지는 프롬프트에 맡긴다
        self.pose_factor = pose_factor
        self._adapter_pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            if not self._on_gpu:
                self._pipe.to("cuda")
                self._on_gpu = True
            return
        import torch
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16, variant=self.variant, use_safetensors=True
        )
        if self.lora:
            from huggingface_hub import hf_hub_download

            repo, _, filename = self.lora.partition(":")
            pipe.load_lora_weights(hf_hub_download(repo, filename))
            pipe.fuse_lora()
        if self.lora or self.steps <= 10:
            # Lightning(LoRA든 Lightning을 합친 체크포인트든)은 trailing timestep으로 학습됐다.
            # 체크포인트 기본(leading)으로 10 step 아래를 돌리면 흐리고 덜 그려진다
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self._pipe = pipe
        self._on_gpu = True
        self._ip_loaded = False

    def _ensure_ip_adapter(self) -> bool:
        """IP-Adapter를 올린다. 설정에서 끄면 False."""
        if not self.ip_adapter:
            return False
        self.load()
        if self._ip_loaded:
            return True
        repo, _, path = self.ip_adapter.partition(":")
        subfolder, _, weight = path.rpartition("/")
        self._pipe.load_ip_adapter(
            repo,
            subfolder=subfolder,
            weight_name=weight,
            image_encoder_folder=IP_ADAPTER_ENCODER,
        )
        if self.cpu_offload:
            # offload hook이 새 image_encoder도 관리하도록 다시 건다
            self._pipe.enable_model_cpu_offload()
        self._ip_loaded = True
        return True

    def encode_reference(self, image: bytes) -> bytes | None:
        """참조 그림 → embedding safetensors (키 "embeds", [1, 1, 257, 1280] fp16)."""
        import torch
        from PIL import Image
        from safetensors.torch import save

        if not self._ensure_ip_adapter():
            return None
        picture = Image.open(io.BytesIO(image)).convert("RGB")
        with torch.no_grad():
            embeds = self._pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=[picture],
                ip_adapter_image_embeds=None,
                device=self._pipe._execution_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        tensor = embeds[0].detach().to("cpu", dtype=torch.float16).contiguous()
        if tuple(tensor.shape) != EMBED_SHAPE:
            raise ValueError(f"embedding 모양이 {tuple(tensor.shape)}이다")
        data: bytes = save({"embeds": tensor}, metadata={"adapter": self.ip_adapter})
        return data

    def _ip_kwargs(
        self, refs: list[Reference], width: int, height: int, schedule: IpSchedule
    ) -> dict[str, Any]:
        """참조 embedding·강도·마스크. 참조가 없으면 0 embedding과 강도 0."""
        import torch
        from safetensors.torch import load

        device = self._pipe._execution_device
        if not refs:
            self._pipe.set_ip_adapter_scale(0.0)
            zeros = torch.zeros(EMBED_SHAPE, dtype=torch.float16, device=device)
            return {"ip_adapter_image_embeds": [self._with_negative(zeros)]}
        tensors = []
        for ref in refs:
            t = load(ref.embed)["embeds"]
            if tuple(t.shape) != EMBED_SHAPE:
                raise ValueError(f"{ref.character} embedding 모양이 {tuple(t.shape)}이다")
            tensors.append(t.to(device=device, dtype=torch.float16))
        embeds = self._with_negative(torch.cat(tensors, dim=1))
        regions = [r.region for r in refs]
        self._set_scales(refs, 0, schedule)
        if len(refs) == 1 and regions[0] == "full":
            return {"ip_adapter_image_embeds": [embeds]}
        if len(refs) > 1 and "full" in regions:
            # 두 명인데 자리가 없으면 나온 순서대로 왼쪽·오른쪽
            regions = ["left", "right"][: len(refs)]
        masks = _region_masks(regions, width, height)
        return {
            "ip_adapter_image_embeds": [embeds],
            "cross_attention_kwargs": {"ip_adapter_masks": [masks]},
        }

    def _with_negative(self, embeds: Any) -> Any:
        """guidance > 1(CFG)이면 앞에 0 embedding을 붙여 [negative; positive]로 넘긴다.

        diffusers가 CFG에서 embedding을 chunk(2)로 나누므로 한 벌만 넘기면 ValueError가 난다
        (회차 21, DreamShaper guidance 2).
        """
        if self.guidance <= 1:
            return embeds
        import torch

        return torch.cat([torch.zeros_like(embeds), embeds], dim=0)

    def _set_scales(self, refs: list[Reference], step: int, schedule: IpSchedule) -> None:
        """step에서 쓸 강도. 전체 참조 한 명은 숫자 하나, 마스크를 쓰면 참조별 목록."""
        scales = schedule.scales([r.scale for r in refs], step)
        if len(refs) == 1 and refs[0].region == "full":
            self._pipe.set_ip_adapter_scale(scales[0])
        else:
            self._pipe.set_ip_adapter_scale([scales])

    def _schedule_callback(self, refs: list[Reference], schedule: IpSchedule) -> Any:
        """step이 끝날 때마다 다음 step의 강도를 건다 (callback_on_step_end)."""

        def callback(pipe: Any, step: int, timestep: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            self._set_scales(refs, step + 1, schedule)
            return kwargs

        return callback

    def unload(self) -> None:
        import torch

        if self.keep_in_ram and self._pipe is not None:
            self._pipe.to("cpu")
        else:
            self._pipe = None
        self._clip = None
        self._adapter_pipe = None
        self._on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()

    def draw(self, image: dict[str, Any], references: list[Reference] | None = None) -> bytes:
        checks = check.parse(image.get("checks")) if self.image_check else []
        first = self._draw_once(image, references)
        if not checks:
            return first
        qs = check.questions(checks)
        try:
            best = (first, self._judge(first, qs))
            for k in range(1, check.RETRIES + 1):
                if best[1][0] == 0:
                    break
                again = dict(image, seed=int(image.get("seed", 0)) + k * check.SEED_STEP)
                data = self._draw_once(again, references)
                result = self._judge(data, qs)
                log.info("인물 확인 다시 그림 %d: 틀림 %d → %d", k, best[1][0], result[0])
                if check.better(result, best[1]):
                    best = (data, result)
            return best[0]
        except Exception:
            log.exception("인물 확인을 하지 못해 처음 그림을 쓴다")
            return first

    def _judge(self, png: bytes, qs: list[check.Question]) -> tuple[int, float]:
        """그림의 자리마다 CLIP으로 물어 (틀린 수, 맞는 말 확률 평균)을 낸다."""
        import torch
        from PIL import Image

        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(CHECK_MODEL, torch_dtype=torch.float16).to("cuda")
            self._clip = (model.eval(), CLIPProcessor.from_pretrained(CHECK_MODEL))
        model, processor = self._clip
        picture = Image.open(io.BytesIO(png)).convert("RGB")
        probs = []
        for q in qs:
            crop = picture.crop(check.box(q.region, picture.width, picture.height))
            inputs = processor(text=list(q.labels), images=crop, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.no_grad():
                logits = model(**inputs).logits_per_image[0]
            probs.append(logits.float().softmax(-1).tolist())
        return check.score(probs, qs)

    def _draw_once(self, image: dict[str, Any], references: list[Reference] | None) -> bytes:
        if self.pose_adapter:
            try:
                return self._draw_plain(image, references, posed=True)
            except Exception:
                # 뼈대는 그림을 낫게 하려는 것이라, 어긋나면 뼈대 없이 그린다
                log.exception("뼈대 그림으로 그리지 못해 뼈대 없이 다시 그린다")
        return self._draw_plain(image, references, posed=False)

    def _draw_plain(
        self, image: dict[str, Any], references: list[Reference] | None, posed: bool
    ) -> bytes:
        regions = regional.parse(image.get("regions"))
        if regions:
            try:
                return self._draw(image, references, regions, posed)
            except Exception:
                # 자리 프롬프트는 그림을 낫게 하려는 것이라, 어긋나면 예전 방식으로 그린다
                log.exception("자리 프롬프트로 그리지 못해 전체 프롬프트로만 다시 그린다")
        return self._draw(image, references, [], posed)

    def _draw(
        self,
        image: dict[str, Any],
        references: list[Reference] | None,
        regions: list[regional.RegionPrompt],
        posed: bool = False,
    ) -> bytes:
        import torch

        self.load()
        width = int(image.get("width", 768))
        height = int(image.get("height", 768))
        refs = list(references or [])
        extra: dict[str, Any] = {}
        schedule = self.ip_schedule.override(image.get("ipSchedule"))
        if (refs and self._ensure_ip_adapter()) or self._ip_loaded:
            extra = self._ip_kwargs(refs, width, height, schedule)
            if refs and schedule.active:
                extra["callback_on_step_end"] = self._schedule_callback(refs, schedule)
        step = [0]
        if regions:
            extra["callback_on_step_end"] = _count_steps(step, extra.get("callback_on_step_end"))
        pipe = self._pipe
        skeleton = pose.render(check.parse(image.get("checks")), width, height) if posed else None
        if skeleton is not None:
            pipe = self._with_adapter()
            extra.update(
                image=skeleton,
                adapter_conditioning_scale=self.pose_scale,
                adapter_conditioning_factor=self.pose_factor,
            )
            # adapter 파이프라인은 callback_on_step_end가 없어
            # 예전 callback(step, t, latents)로 부른다
            inner = extra.pop("callback_on_step_end", None)
            if inner is not None:
                extra["callback"] = lambda i, t, latents: inner(self._pipe, i, t, {})
        generator = torch.Generator(device="cpu").manual_seed(int(image.get("seed", 0)))
        negative = image.get("negativePrompt") if self.guidance > 1 else None
        text: dict[str, Any] = {"prompt": image["prompt"], "negative_prompt": negative}
        if self.guidance <= 1:
            # guidance 0(Lightning)은 negative가 필요 없어 긴 프롬프트를 조각으로 인코딩해 넣는다
            long = long_prompt.encode(self._pipe, image["prompt"])
            if long is not None:
                text = {"prompt_embeds": long[0], "pooled_prompt_embeds": long[1]}
        else:
            # guidance > 1(LoRA 없는 모델)은 negative도 같은 조각 수로 인코딩한다
            pair = long_prompt.encode_pair(self._pipe, image["prompt"], negative or "")
            if pair is not None:
                text = {
                    "prompt_embeds": pair[0],
                    "negative_prompt_embeds": pair[1],
                    "pooled_prompt_embeds": pair[2],
                    "negative_pooled_prompt_embeds": pair[3],
                }
        saved = self._install_regions(regions, width, height, step) if regions else None
        try:
            result = pipe(
                **text,
                width=width,
                height=height,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=generator,
                **extra,
            ).images[0]
        finally:
            if saved is not None:
                self._pipe.unet.set_attn_processor(saved)
        out = io.BytesIO()
        result.save(out, format="PNG")
        return out.getvalue()

    def _with_adapter(self) -> Any:
        """뼈대 그림을 받는 파이프라인. SDXL 부품(UNet 처리기·IP-Adapter 포함)을 함께 쓴다."""
        if self._adapter_pipe is None:
            import torch
            from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter

            adapter = T2IAdapter.from_pretrained(self.pose_adapter, torch_dtype=torch.float16)
            parts = {
                k: v
                for k, v in self._pipe.components.items()
                if k in ADAPTER_PARTS and v is not None
            }
            self._adapter_pipe = StableDiffusionXLAdapterPipeline(
                **parts, adapter=adapter.to("cuda")
            )
        return self._adapter_pipe

    def _install_regions(
        self, regions: list[regional.RegionPrompt], width: int, height: int, step: list[int]
    ) -> dict[str, Any]:
        """cross-attention 처리기를 자리 프롬프트를 섞는 것으로 바꾸고, 원래 처리기를 돌려준다."""
        import torch

        device = self._pipe._execution_device
        prepared = []
        for r in regions:
            with torch.no_grad():
                embeds = self._pipe.encode_prompt(
                    prompt=r.prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )[0]
            prepared.append((r.region, embeds))
        saved = dict(self._pipe.unet.attn_processors)
        swapped = {
            name: _regional_processor(proc, prepared, width, height, step, self.guidance > 1)
            if name.endswith("attn2.processor")
            else proc
            for name, proc in saved.items()
        }
        self._pipe.unet.set_attn_processor(swapped)
        return saved

    def model_info(self) -> dict[str, Any]:
        return {
            "sdxl": self.model_name,
            "lora": self.lora or None,
            "poseAdapter": self.pose_adapter or None,
            "steps": self.steps,
            "guidance": self.guidance,
            "cpuOffload": self.cpu_offload,
            "ipAdapter": self.ip_adapter or None,
            "imageCheck": CHECK_MODEL if self.image_check else None,
            "ipSchedule": {
                "earlyScale": self.ip_schedule.early_scale,
                "earlySteps": self.ip_schedule.early_steps,
                "lateBoost": self.ip_schedule.late_boost,
            },
        }


def _region_masks(regions: list[str], width: int, height: int) -> Any:
    """자리별 마스크 [1, 참조 수, H, W]. left는 왼쪽 절반, right는 오른쪽 절반, center는 가운데 60%,
    full은 전체."""
    from diffusers.image_processor import IPAdapterMaskProcessor
    from PIL import Image

    half = width // 2
    edge = width // 5
    boxes = {
        "left": (0, 0, half, height),
        "right": (half, 0, width, height),
        "center": (edge, 0, width - edge, height),
    }
    pictures = []
    for region in regions:
        mask = Image.new("L", (width, height), 255 if region == "full" else 0)
        if region in boxes:
            mask.paste(255, boxes[region])
        pictures.append(mask)
    masks = IPAdapterMaskProcessor().preprocess(pictures, height=height, width=width)
    return masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])


def _count_steps(step: list[int], inner: Any) -> Any:
    """step이 끝날 때마다 다음 step 번호를 step[0]에 적고, 원래 callback이 있으면 부른다."""

    def callback(pipe: Any, i: int, timestep: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        step[0] = i + 1
        return inner(pipe, i, timestep, kwargs) if inner is not None else kwargs

    return callback


def _regional_processor(
    inner: Any,
    prepared: list[tuple[str, Any]],
    width: int,
    height: int,
    step: list[int],
    cfg: bool = False,
) -> Any:
    """자리 안에서 인물 프롬프트 결과를 섞는 처리기.

    cfg이면 배치 앞 절반(negative)은 그대로 두고 뒤 절반(프롬프트)에만 섞는다.

    inner와 같은 종류라 IP-Adapter 강도 설정이 그대로 닿고, inner의 상태(가중치 등)를 함께 쓴다.
    """
    base_cls = type(inner)
    base_call = base_cls.__call__

    @functools.wraps(base_call)
    def call(
        self: Any,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        out = base_call(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs)
        if encoder_hidden_states is None or hidden_states.ndim != 3 or not regional.active(step[0]):
            return out
        batch, seq, _ = hidden_states.shape
        size = regional.grid(seq, width, height)
        if size is None:
            return out
        h, w = size
        ip = None
        if isinstance(encoder_hidden_states, tuple):
            ip = encoder_hidden_states[1]
        for region, embeds in prepared:
            text = embeds.to(hidden_states.dtype).expand(batch, -1, -1)
            ehs = (text, ip) if ip is not None else text
            part = base_call(self, attn, hidden_states, ehs, *args, **kwargs)
            start, end = regional.columns(region, w)
            mask = hidden_states.new_zeros(h, w)
            mask[:, start:end] = regional.WEIGHT
            mask = mask.reshape(1, seq, 1)
            if cfg:
                mask = mask.expand(batch, -1, -1).clone()
                mask[: batch // 2] = 0
            out = out * (1 - mask) + part * mask
        return out

    cls = type("Regional" + base_cls.__name__, (base_cls,), {"__call__": call})
    proc: Any = object.__new__(cls)
    proc.__dict__.update(inner.__dict__)
    return proc
