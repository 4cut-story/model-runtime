"""SD 3.5 Large·Qwen-Image 그림 (GPU PC 전용, IMAGE_MODEL=sd35 또는 qwen). 회차 29 모델 비교용.

FLUX처럼 큰 transformer와 글 인코더(SD 3.5는 T5, Qwen-Image는 Qwen2.5-VL 7B)를 4bit로 줄이고
enable_model_cpu_offload()로 쓰는 부분만 GPU에 올린다.
- DIFFUSION_QUANT=nf4: bitsandbytes NF4로 읽으면서 줄인다.
- DIFFUSION_QUANT=nunchaku: 미리 줄인 SVDQuant 4bit 한 파일(DIFFUSION_FILE)을 읽는다(Qwen-Image만).
  Lightning 8 step을 합친 파일이면 DIFFUSION_STEPS=8, DIFFUSION_GUIDANCE=1로 그린다.
- DIFFUSION_QUANT=none: bf16 그대로 읽는다.

SD 3.5는 guidance_scale(CFG)로, Qwen-Image는 true_cfg_scale로 negative prompt를 쓴다.
1 이하면 negative prompt를 쓰지 않는다.
캐릭터 참조(IP-Adapter)는 쓰지 않는다.
"""

from __future__ import annotations

import gc
import io
import logging
from typing import Any

from . import Reference

log = logging.getLogger(__name__)

KINDS: dict[str, dict[str, Any]] = {
    "sd35": {"model": "stabilityai/stable-diffusion-3.5-large", "steps": 28, "guidance": 3.5},
    "qwen": {"model": "Qwen/Qwen-Image", "steps": 50, "guidance": 4.0},
}
QUANTS = ("nf4", "nunchaku", "none")


class DiffusionBackend:
    def __init__(
        self,
        kind: str,
        model_name: str = "",
        steps: int = 0,
        guidance: float = -1.0,
        quant: str = "nf4",
        single_file: str = "",
        keep_in_ram: bool = True,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"IMAGE_MODEL은 {', '.join(KINDS)} 중 하나여야 한다: {kind}")
        if quant not in QUANTS:
            raise ValueError(f"DIFFUSION_QUANT는 {', '.join(QUANTS)} 중 하나여야 한다: {quant}")
        if quant == "nunchaku" and (kind != "qwen" or not single_file):
            raise ValueError("DIFFUSION_QUANT=nunchaku는 qwen에서 DIFFUSION_FILE과 함께 쓴다")
        defaults = KINDS[kind]
        self.kind = kind
        self.model_name = model_name or str(defaults["model"])
        self.steps = steps or int(defaults["steps"])
        self.guidance = guidance if guidance >= 0 else float(defaults["guidance"])
        self.quant = quant
        self.single_file = single_file
        self.keep_in_ram = keep_in_ram
        self._pipe: Any = None

    def load(self) -> None:  # pragma: no cover - GPU 전용
        if self._pipe is not None:
            return
        import torch

        if self.kind == "sd35":
            self._pipe = self._sd35(torch)
        else:
            self._pipe = self._qwen(torch)
        # SD3·Qwen 파이프라인에는 enable_vae_tiling()이 없어 VAE에 바로 켠다.
        self._pipe.vae.enable_tiling()
        self._pipe.enable_model_cpu_offload()
        log.info("%s 준비 완료 (%s)", self.kind, self.quant)

    def _nf4(self, torch: Any, transformers: bool) -> Any:  # pragma: no cover - GPU 전용
        if self.quant != "nf4":
            return None
        if transformers:
            from transformers import BitsAndBytesConfig as Config
        else:
            from diffusers import BitsAndBytesConfig as Config
        return Config(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    def _sd35(self, torch: Any) -> Any:  # pragma: no cover - GPU 전용
        from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
        from transformers import T5EncoderModel

        transformer = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=self._nf4(torch, False),
            torch_dtype=torch.bfloat16,
        )
        t5 = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_3",
            quantization_config=self._nf4(torch, True),
            torch_dtype=torch.bfloat16,
        )
        return StableDiffusion3Pipeline.from_pretrained(
            self.model_name, transformer=transformer, text_encoder_3=t5, torch_dtype=torch.bfloat16
        )

    def _qwen(self, torch: Any) -> Any:  # pragma: no cover - GPU 전용
        from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
        from transformers import Qwen2_5_VLForConditionalGeneration

        if self.quant == "nunchaku":
            from nunchaku import NunchakuQwenImageTransformer2DModel

            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(self.single_file)
        else:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                self.model_name,
                subfolder="transformer",
                quantization_config=self._nf4(torch, False),
                torch_dtype=torch.bfloat16,
            )
        # 글 인코더(7B)는 어느 방식이든 NF4로 줄여 RAM을 아낀다
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            subfolder="text_encoder",
            quantization_config=self._nf4_always(torch),
            torch_dtype=torch.bfloat16,
        )
        return QwenImagePipeline.from_pretrained(
            self.model_name,
            transformer=transformer,
            text_encoder=encoder,
            torch_dtype=torch.bfloat16,
        )

    def _nf4_always(self, torch: Any) -> Any:  # pragma: no cover - GPU 전용
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    def call_args(self, image: dict[str, Any]) -> dict[str, Any]:
        """파이프라인에 넘길 값. negative prompt는 guidance가 1보다 클 때만 쓴다."""
        args: dict[str, Any] = {
            "prompt": image["prompt"],
            "width": int(image.get("width", 768)),
            "height": int(image.get("height", 768)),
            "num_inference_steps": self.steps,
        }
        negative = image.get("negativePrompt") or None
        if self.kind == "sd35":
            args["guidance_scale"] = self.guidance
            args["max_sequence_length"] = 512
        else:
            args["true_cfg_scale"] = self.guidance
        if negative and self.guidance > 1:
            args["negative_prompt"] = negative
        return args

    def encode_reference(self, image: bytes) -> bytes | None:
        return None

    def unload(self) -> None:  # pragma: no cover - GPU 전용
        import torch

        if self._pipe is not None:
            if self.keep_in_ram:
                self._pipe.maybe_free_model_hooks()
            else:
                self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def draw(  # pragma: no cover - GPU 전용
        self, image: dict[str, Any], references: list[Reference] | None = None
    ) -> bytes:
        import torch

        self.load()
        generator = torch.Generator(device="cpu").manual_seed(int(image.get("seed", 0)))
        picture = self._pipe(generator=generator, **self.call_args(image)).images[0]
        buf = io.BytesIO()
        picture.save(buf, format="PNG")
        return buf.getvalue()

    def model_info(self) -> dict[str, Any]:
        return {
            self.kind: self.model_name,
            "quantization": self.quant,
            "steps": self.steps,
            "guidance": self.guidance,
            "cpuOffload": True,
            "ipAdapter": None,
        }
