"""FLUX.1-schnell 만화풍 그림 (GPU PC 전용, IMAGE_MODEL=flux).

SDXL은 회차 23~27에서 설정을 바꿔도 83점 근처에 머물렀다. 감점 대부분이 프롬프트에 적은 행동·물건이
빠지거나 두 인물이 섞이는 것이었는데, CLIP 텍스트 인코더가 긴 문장을 잘 읽지 못하는 탓으로 봤다.
FLUX는 T5로 문장을 읽어 "왼쪽은 누구, 오른쪽은 누구, 무엇을 든다"를 더 잘 따른다.

schnell은 4 step, guidance 0으로 그리므로 negative prompt를 쓰지 않는다. FLUX_MODEL로 FLUX.1-dev를
고르면 FLUX_STEPS(28 권장)와 FLUX_GUIDANCE(3.5 권장)를 함께 바꾼다.
양자화본 폴더는 모델마다 따로 둔다.

transformer(약 12B)와 T5는 16GB에 다 올라가지 않아 FLUX_QUANT(nf4·int8·gguf·fp8·nunchaku)로
줄이고 enable_model_cpu_offload()로 쓰는 부분만 GPU에 올린다. 나머지는 시스템 RAM에 둔다.
gguf는 Q4~Q8 어느 파일이든 읽는다. nunchaku는 SVDQuant 4bit 파일을 nunchaku 패키지로 돌린다.
FLUX_LORA로 적은 step용 LoRA를 붙일 수 있다(회차 29: dev 28 step은 그림 점수 90.7로 schnell 86.8보다
높았지만 컷당 42초로 약 6배 느려 보류했다).
NF4는 처음 한 번 양자화해 quantized_dir에 저장하고 다음부터 그것을 읽는다.

IP-Adapter·자리 프롬프트·뼈대·CLIP 확인은 SDXL 전용이라 쓰지 않는다. 캐릭터 참조는 무시하고
(encode_reference가 None) 외모는 프롬프트 글로만 맞춘다.
"""

from __future__ import annotations

import gc
import io
import logging
import re
from pathlib import Path
from typing import Any

from . import Reference

log = logging.getLogger(__name__)

DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
# nf4·int8: bitsandbytes로 읽으면서 줄인다. gguf·fp8: 미리 줄인 한 파일(FLUX_FILE)을 읽는다.
# nunchaku: SVDQuant 4bit 한 파일(50번대 GPU는 fp4, 그 전은 int4)을 nunchaku 커널로 돌린다
QUANTS = ("auto", "nf4", "int8", "gguf", "fp8", "nunchaku")
# 미리 줄인 한 파일을 읽는 방식
SINGLE_FILE_QUANTS = ("gguf", "fp8", "nunchaku")


def _has_bnb() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def _bnb_args(torch: Any, mode: str) -> dict[str, Any]:
    if mode == "int8":
        return {"load_in_8bit": True}
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }


def _upcast_uncast(torch: Any, model: Any) -> None:
    """layerwise casting이 건너뛴 모듈(norm, pos_embed 등)을 bf16으로 올린다.

    fp8 한 파일을 fp8 그대로 읽으면 건너뛴 모듈은 fp8로 남아
    "mat1 and mat2 must have the same dtype"
    오류가 났다(회차 29 schnell-fp8 47컷 모두 실패). 건너뛰는 규칙은 diffusers와 같게 맞춘다.
    """
    from diffusers.hooks._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    patterns = list(DEFAULT_SKIP_MODULES_PATTERN)
    patterns += list(getattr(model, "_keep_in_fp32_modules", None) or [])
    patterns += list(getattr(model, "_skip_layerwise_casting_patterns", None) or [])

    def walk(module: Any, prefix: str) -> None:
        if prefix and any(re.search(p, prefix) for p in patterns):
            module.to(torch.bfloat16)
            return
        if isinstance(module, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
            return
        for param in module.parameters(recurse=False):
            if param.dtype == torch.float8_e4m3fn:
                param.data = param.data.to(torch.bfloat16)
        for name, child in module.named_children():
            walk(child, f"{prefix}.{name}" if prefix else name)

    walk(model, "")


class FluxBackend:
    def __init__(
        self,
        model_name: str = DEFAULT_FLUX_MODEL,
        quantized_dir: str = "",
        steps: int = 4,
        keep_in_ram: bool = True,
        max_sequence_length: int = 512,
        guidance: float = 0.0,
        quant: str = "auto",
        single_file: str = "",
        lora: str = "",
        lora_weight: str = "",
        lora_scale: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.quantized_dir = quantized_dir
        self.steps = steps
        self.keep_in_ram = keep_in_ram
        self.max_sequence_length = max_sequence_length
        # schnell은 0. dev는 guidance를 모델 입력으로 받는다(CFG처럼 두 번 돌지 않는다). 3.5 권장
        self.guidance = guidance
        # auto: bitsandbytes가 있으면 nf4, 없으면 fp8. 양자화본 폴더는 모델·모드마다 따로 둔다
        if quant not in QUANTS:
            raise ValueError(f"FLUX_QUANT는 {', '.join(QUANTS)} 중 하나여야 한다: {quant}")
        self.quant = quant
        self.single_file = single_file
        # 적은 step으로 그리게 하는 LoRA (예: Hyper-SD dev 8 step, scale 0.125)
        self.lora = lora
        self.lora_weight = lora_weight
        self.lora_scale = lora_scale
        self.mode = ""
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline

        transformer, t5 = self._load_heavy(torch)
        pipe = FluxPipeline.from_pretrained(
            self.model_name, transformer=transformer, text_encoder_2=t5, torch_dtype=torch.bfloat16
        )
        if self.lora:
            self._load_lora(pipe, transformer)
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        self._pipe = pipe
        log.info("FLUX 준비 완료 (%s)", self.mode)

    def _load_lora(self, pipe: Any, transformer: Any) -> None:
        if self.mode == "nunchaku":
            # nunchaku transformer는 diffusers LoRA 대신 자기 함수로 LoRA를 합친다
            path = self.lora
            if self.lora_weight:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(self.lora, self.lora_weight)
            transformer.update_lora_params(path)
            transformer.set_lora_strength(self.lora_scale)
            return
        kwargs = {"weight_name": self.lora_weight} if self.lora_weight else {}
        pipe.load_lora_weights(self.lora, adapter_name="speed", **kwargs)
        pipe.set_adapters(["speed"], adapter_weights=[self.lora_scale])

    def _load_heavy(self, torch: Any) -> tuple[Any, Any]:
        mode = self.quant
        if mode == "auto":
            mode = "nf4" if _has_bnb() else "fp8"
        self.mode = mode
        saved = Path(self.quantized_dir) if self.quantized_dir else None
        return self._transformer(torch, mode, saved), self._t5(torch, saved)

    def _transformer(self, torch: Any, mode: str, saved: Path | None) -> Any:
        from diffusers import FluxTransformer2DModel

        if mode in SINGLE_FILE_QUANTS and not self.single_file:
            raise ValueError(f"FLUX_QUANT={mode}에는 FLUX_FILE(한 파일 경로)이 필요하다")
        if mode == "nunchaku":
            from nunchaku import NunchakuFluxTransformer2dModel

            return NunchakuFluxTransformer2dModel.from_pretrained(self.single_file)
        if mode in ("gguf", "fp8"):
            # 8bit 한 파일(GGUF Q8_0 또는 fp8 e4m3fn safetensors)을 8bit 그대로 읽는다.
            # bf16으로 읽으면 RAM에 24GB가 올라가 31GB PC에서 넘친다
            common = dict(config=self.model_name, subfolder="transformer")
            if mode == "gguf":
                from diffusers import GGUFQuantizationConfig

                return FluxTransformer2DModel.from_single_file(
                    self.single_file,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16,
                    **common,
                )
            transformer = FluxTransformer2DModel.from_single_file(
                self.single_file, torch_dtype=torch.float8_e4m3fn, **common
            )
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
            )
            _upcast_uncast(torch, transformer)
            return transformer
        if saved is not None and (saved / "transformer").exists():
            return FluxTransformer2DModel.from_pretrained(
                saved / "transformer", torch_dtype=torch.bfloat16
            )
        from diffusers import BitsAndBytesConfig

        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=BitsAndBytesConfig(**_bnb_args(torch, mode)),
            torch_dtype=torch.bfloat16,
        )
        if saved is not None:
            transformer.save_pretrained(saved / "transformer")
        return transformer

    def _t5(self, torch: Any, saved: Path | None) -> Any:
        """T5(bf16 약 9.5GB)는 어느 모드든 bitsandbytes가 있으면 NF4로 줄여 RAM을 아낀다."""
        from transformers import T5EncoderModel

        if saved is not None and (saved / "text_encoder_2").exists():
            return T5EncoderModel.from_pretrained(
                saved / "text_encoder_2", torch_dtype=torch.bfloat16
            )
        if not _has_bnb():
            return T5EncoderModel.from_pretrained(
                self.model_name, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
            )
        from transformers import BitsAndBytesConfig

        t5 = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_2",
            quantization_config=BitsAndBytesConfig(**_bnb_args(torch, "nf4")),
            torch_dtype=torch.bfloat16,
        )
        if saved is not None:
            t5.save_pretrained(saved / "text_encoder_2")
        return t5

    def encode_reference(self, image: bytes) -> bytes | None:
        return None

    def unload(self) -> None:
        import torch

        if self._pipe is not None:
            if self.keep_in_ram:
                # offload 훅이 쓴 부분을 CPU로 돌려놓는다. 다음 작업은 디스크를 다시 읽지 않는다
                self._pipe.maybe_free_model_hooks()
            else:
                self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def draw(self, image: dict[str, Any], references: list[Reference] | None = None) -> bytes:
        import torch

        self.load()
        generator = torch.Generator(device="cpu").manual_seed(int(image.get("seed", 0)))
        picture = self._pipe(
            prompt=image["prompt"],
            width=int(image.get("width", 768)),
            height=int(image.get("height", 768)),
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            max_sequence_length=self.max_sequence_length,
            generator=generator,
        ).images[0]
        buf = io.BytesIO()
        picture.save(buf, format="PNG")
        return buf.getvalue()

    def model_info(self) -> dict[str, Any]:
        return {
            "flux": self.model_name,
            "quantization": self.mode or None,
            "steps": self.steps,
            "guidance": self.guidance,
            "lora": self.lora or None,
            "cpuOffload": True,
            "ipAdapter": None,
        }
