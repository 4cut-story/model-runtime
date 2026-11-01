"""SDXL 만화풍 그림 (GPU PC 전용).

기본은 SDXL-Lightning 8 step LoRA로 그린다 (30 step 대비 약 7배 빠름). Lightning은 guidance 0으로
그리므로 negative prompt가 효과가 없다. 768px은 16GB에 다 올라가 CPU offload 없이 돈다.
VAE tiling·slicing은 1024px로 되돌려도 VRAM이 넘치지 않게 켜 둔다.

keep_in_ram이면 내릴 때 지우지 않고 시스템 RAM으로 옮겨 둔다. 다음 그림 작업은 디스크에서 다시
읽지 않고 GPU로 옮기기만 한다 (GPU PC 측정: 디스크에서 86~194초 → RAM에서 1.2~1.4초, RAM 약 5.5GB).
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
        keep_in_ram: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cpu_offload = cpu_offload
        self.lora = lora
        self.steps = steps
        self.guidance = guidance
        # CPU offload는 이미 컷마다 GPU↔CPU를 오가므로 RAM에 따로 둘 필요가 없다
        self.keep_in_ram = keep_in_ram and not cpu_offload
        self._pipe: Any = None
        self._on_gpu = False

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

    def unload(self) -> None:
        import torch

        if self.keep_in_ram and self._pipe is not None:
            self._pipe.to("cpu")
        else:
            self._pipe = None
        self._on_gpu = False
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
