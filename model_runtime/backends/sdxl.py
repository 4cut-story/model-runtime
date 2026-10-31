"""SDXL 만화풍 그림 (GPU PC 전용).

16GB 카드에서 1024px은 약 15.9GB까지 쓰므로 VAE tiling·slicing과 CPU offload를 기본으로 켠다.
"""

from __future__ import annotations

import gc
import io
from typing import Any


class SdxlBackend:
    def __init__(self, model_name: str, cpu_offload: bool = True) -> None:
        self.model_name = model_name
        self.cpu_offload = cpu_offload
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self._pipe = pipe

    def unload(self) -> None:
        import torch

        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def draw(self, image: dict[str, Any]) -> bytes:
        import torch

        self.load()
        generator = torch.Generator(device="cpu").manual_seed(int(image.get("seed", 0)))
        result = self._pipe(
            prompt=image["prompt"],
            negative_prompt=image.get("negativePrompt"),
            width=int(image.get("width", 1024)),
            height=int(image.get("height", 1024)),
            num_inference_steps=30,
            guidance_scale=6.0,
            generator=generator,
        ).images[0]
        out = io.BytesIO()
        result.save(out, format="PNG")
        return out.getvalue()

    def model_info(self) -> dict[str, Any]:
        return {"sdxl": self.model_name, "cpuOffload": self.cpu_offload}
