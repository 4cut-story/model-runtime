"""SDXL 만화풍 그림 (GPU PC 전용).

기본은 SDXL-Lightning 8 step LoRA로 그린다 (30 step 대비 약 7배 빠름). Lightning은 guidance 0으로
그리므로 negative prompt가 효과가 없다. 768px은 16GB에 다 올라가 CPU offload 없이 돈다.
VAE tiling·slicing은 1024px로 되돌려도 VRAM이 넘치지 않게 켜 둔다.
"""

from __future__ import annotations

import gc
import io
from typing import Any


class SdxlBackend:
    def __init__(
        self,
        model_name: str,
        cpu_offload: bool = False,
        lora: str = "",
        steps: int = 8,
        guidance: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.cpu_offload = cpu_offload
        self.lora = lora
        self.steps = steps
        self.guidance = guidance
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
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

    def unload(self) -> None:
        import torch

        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def draw(self, image: dict[str, Any]) -> bytes:
        import torch

        self.load()
        generator = torch.Generator(device="cpu").manual_seed(int(image.get("seed", 0)))
        negative = image.get("negativePrompt") if self.guidance > 1 else None
        result = self._pipe(
            prompt=image["prompt"],
            negative_prompt=negative,
            width=int(image.get("width", 768)),
            height=int(image.get("height", 768)),
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=generator,
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
        }
