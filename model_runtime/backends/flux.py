"""FLUX.1-schnell 만화풍 그림 (GPU PC 전용, IMAGE_MODEL=flux).

SDXL은 회차 23~27에서 설정을 바꿔도 83점 근처에 머물렀다. 감점 대부분이 프롬프트에 적은 행동·물건이
빠지거나 두 인물이 섞이는 것이었는데, CLIP 텍스트 인코더가 긴 문장을 잘 읽지 못하는 탓으로 봤다.
FLUX는 T5로 문장을 읽어 "왼쪽은 누구, 오른쪽은 누구, 무엇을 든다"를 더 잘 따른다.

schnell은 4 step, guidance 0으로 그리므로 negative prompt를 쓰지 않는다. FLUX_MODEL로 FLUX.1-dev를
고르면 FLUX_STEPS(28 권장)와 FLUX_GUIDANCE(3.5 권장)를 함께 바꾼다.
양자화본 폴더는 모델마다 따로 둔다.

transformer(약 12B)와 T5는 16GB에 다 올라가지 않아 NF4(bitsandbytes가 있을 때) 또는 fp8 저장으로
줄이고 enable_model_cpu_offload()로 쓰는 부분만 GPU에 올린다. 나머지는 시스템 RAM에 둔다.
NF4는 처음 한 번 양자화해 quantized_dir에 저장하고 다음부터 그것을 읽는다.

IP-Adapter·자리 프롬프트·뼈대·CLIP 확인은 SDXL 전용이라 쓰지 않는다. 캐릭터 참조는 무시하고
(encode_reference가 None) 외모는 프롬프트 글로만 맞춘다.
"""

from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import Any

from . import Reference

log = logging.getLogger(__name__)

DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"


class FluxBackend:
    def __init__(
        self,
        model_name: str = DEFAULT_FLUX_MODEL,
        quantized_dir: str = "",
        steps: int = 4,
        keep_in_ram: bool = True,
        max_sequence_length: int = 512,
        guidance: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.quantized_dir = quantized_dir
        self.steps = steps
        self.keep_in_ram = keep_in_ram
        self.max_sequence_length = max_sequence_length
        # schnell은 0. dev는 guidance를 모델 입력으로 받는다(CFG처럼 두 번 돌지 않는다). 3.5 권장
        self.guidance = guidance
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
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        self._pipe = pipe
        log.info("FLUX 준비 완료 (%s)", self.mode)

    def _load_heavy(self, torch: Any) -> tuple[Any, Any]:
        from diffusers import FluxTransformer2DModel
        from transformers import T5EncoderModel

        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            self.mode = "fp8"
            transformer = FluxTransformer2DModel.from_pretrained(
                self.model_name, subfolder="transformer", torch_dtype=torch.bfloat16
            )
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
            )
            t5 = T5EncoderModel.from_pretrained(
                self.model_name, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
            )
            return transformer, t5
        self.mode = "nf4"
        saved = Path(self.quantized_dir) if self.quantized_dir else None
        if saved is not None and (saved / "transformer").exists():
            transformer = FluxTransformer2DModel.from_pretrained(
                saved / "transformer", torch_dtype=torch.bfloat16
            )
            t5 = T5EncoderModel.from_pretrained(
                saved / "text_encoder_2", torch_dtype=torch.bfloat16
            )
            return transformer, t5
        from diffusers import BitsAndBytesConfig
        from transformers import BitsAndBytesConfig as TfBnb

        nf4 = dict(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=BitsAndBytesConfig(**nf4),
            torch_dtype=torch.bfloat16,
        )
        t5 = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_2",
            quantization_config=TfBnb(**nf4),
            torch_dtype=torch.bfloat16,
        )
        if saved is not None:
            transformer.save_pretrained(saved / "transformer")
            t5.save_pretrained(saved / "text_encoder_2")
        return transformer, t5

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
            "cpuOffload": True,
            "ipAdapter": None,
        }
