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
"""

from __future__ import annotations

import gc
import io
from typing import Any

from . import Reference
from .ip_schedule import IpSchedule

# IP-Adapter 이미지 인코더 (ViT-H). 어댑터와 같은 레포 안 경로
IP_ADAPTER_ENCODER = "models/image_encoder"
# Plus 계열 embedding 모양: [이미지 수, 참조 수, 토큰, 차원]
EMBED_SHAPE = (1, 1, 257, 1280)


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
        self._pipe: Any = None
        self._on_gpu = False
        self._ip_loaded = False

    def load(self) -> None:
        if self._pipe is not None:
            if not self._on_gpu:
                self._pipe.to("cuda")
                self._on_gpu = True
            return
        import torch
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        if self.lora:
            from huggingface_hub import hf_hub_download

            repo, _, filename = self.lora.partition(":")
            pipe.load_lora_weights(hf_hub_download(repo, filename))
            pipe.fuse_lora()
            # Lightning은 trailing timestep으로 학습됐다
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
            return {"ip_adapter_image_embeds": [zeros]}
        tensors = []
        for ref in refs:
            t = load(ref.embed)["embeds"]
            if tuple(t.shape) != EMBED_SHAPE:
                raise ValueError(f"{ref.character} embedding 모양이 {tuple(t.shape)}이다")
            tensors.append(t.to(device=device, dtype=torch.float16))
        embeds = torch.cat(tensors, dim=1)
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
        self._on_gpu = False
        gc.collect()
        torch.cuda.empty_cache()

    def draw(self, image: dict[str, Any], references: list[Reference] | None = None) -> bytes:
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
        generator = torch.Generator(device="cpu").manual_seed(int(image.get("seed", 0)))
        negative = image.get("negativePrompt") if self.guidance > 1 else None
        result = self._pipe(
            prompt=image["prompt"],
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=generator,
            **extra,
        ).images[0]
        out = io.BytesIO()
        result.save(out, format="PNG")
        return out.getvalue()

    def model_info(self) -> dict[str, Any]:
        return {
            "sdxl": self.model_name,
            "lora": self.lora or None,
            "steps": self.steps,
            "guidance": self.guidance,
            "cpuOffload": self.cpu_offload,
            "ipAdapter": self.ip_adapter or None,
            "ipSchedule": {
                "earlyScale": self.ip_schedule.early_scale,
                "earlySteps": self.ip_schedule.early_steps,
                "lateBoost": self.ip_schedule.late_boost,
            },
        }


def _region_masks(regions: list[str], width: int, height: int) -> Any:
    """자리별 마스크 [1, 참조 수, H, W]. left는 왼쪽 절반, right는 오른쪽 절반, full은 전체."""
    from diffusers.image_processor import IPAdapterMaskProcessor
    from PIL import Image

    half = width // 2
    boxes = {"left": (0, 0, half, height), "right": (half, 0, width, height)}
    pictures = []
    for region in regions:
        mask = Image.new("L", (width, height), 255 if region == "full" else 0)
        if region in boxes:
            mask.paste(255, boxes[region])
        pictures.append(mask)
    masks = IPAdapterMaskProcessor().preprocess(pictures, height=height, width=width)
    return masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])
