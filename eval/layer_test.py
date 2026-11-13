"""라이브 모드 인물 층 시험 (docs specs/003-live-mode FR-052 다시 정하기).

게시물 하나의 넓은 그림 컷마다
1) 인물 자리를 BiRefNet(MIT)으로 찾아(못 올리면 깊이 지도 앞쪽) 투명 PNG로 오려 내고,
2) 그 자리를 SDXL 인페인팅으로 다시 그려 인물 없는 배경을 따로 만들고,
3) 인물 PNG의 입 자리만 다시 그려 벌린 입 PNG를 만든 뒤,
4) 배경은 고정하고 인물 PNG만 1~2px 오르내리며 입을 여닫는 미리보기(GIF)를 만든다.
단계마다 걸린 시간을 적는다. 운영 워커와 따로 도는 시험 스크립트라 결과는 out 폴더에만 둔다.

  python eval/layer_test.py --post <게시물 id> --panels 1,3,7 \
      --out C:\\claude-work\\4cut\\layer-test
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

SITE = "https://4cut.java21.net"
BG_PROMPT = (
    "empty background of the same place, no people, nobody, webtoon style, clean lines, "
    "same lighting and colors"
)
MOUTH_PROMPT = "same character, mouth open, talking, webtoon style, same face"


def fetch(url: str) -> bytes:
    if not url.startswith(SITE):
        raise ValueError(url)
    req = urllib.request.Request(url, headers={"User-Agent": "layer-test"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        return bytes(r.read())


class Timer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, float]] = []

    def run(self, name: str, fn: Any) -> Any:
        t = time.perf_counter()
        out = fn()
        self.rows.append((name, time.perf_counter() - t))
        print(f"{name}: {self.rows[-1][1]:.2f}s", flush=True)
        return out


def person_mask(image: Any, depth: Any) -> Any:
    """인물 자리(0~255). BiRefNet을 못 쓰면 깊이 지도 앞쪽."""
    from PIL import Image

    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        model = person_mask.__dict__.get("model")
        if model is None:
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            model.to("cuda").eval().half()
            person_mask.__dict__["model"] = model
        tf = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        x = tf(image).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            pred = model(x)[-1].sigmoid().cpu()[0].squeeze().float()
        mask = transforms.ToPILImage()(pred).resize(image.size)
        # 배경의 밝은 물건까지 잡지 않게 깊이 앞쪽과 겹치는 곳만 쓴다
        near = depth.point(lambda v: 255 if v > 110 else 0)
        from PIL import ImageChops

        return ImageChops.multiply(mask, near.filter(_max(15)))
    except Exception as e:  # 시험 스크립트: 실패하면 깊이 지도로
        print(f"BiRefNet 실패, 깊이 지도로: {e}", flush=True)
        return (
            depth.point(lambda v: 255 if v > 170 else 0)
            .convert("L")
            .resize(image.size, Image.BILINEAR)
        )


def _max(size: int) -> Any:
    from PIL import ImageFilter

    return ImageFilter.MaxFilter(size)


def load_pipes(lora: str, offload: bool) -> tuple[Any, Any]:
    import torch
    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )
    from huggingface_hub import hf_hub_download

    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    repo, _, filename = lora.partition(":")
    base.load_lora_weights(hf_hub_download(repo, filename))
    base.fuse_lora()
    base.scheduler = EulerDiscreteScheduler.from_config(
        base.scheduler.config, timestep_spacing="trailing"
    )
    base.enable_vae_tiling()
    if offload:
        # 운영 워커가 그림 모델을 올릴 자리(약 6GB)를 남긴다
        base.enable_model_cpu_offload()
    else:
        base.to("cuda")
    inpaint = StableDiffusionXLInpaintPipeline(**base.components)
    return base, inpaint


def inpaint(pipe: Any, image: Any, mask: Any, prompt: str, strength: float, seed: int) -> Any:
    import torch

    w, h = image.size
    # SDXL은 8의 배수
    W, H = (w // 8) * 8, (h // 8) * 8
    out = pipe(
        prompt=prompt,
        image=image.resize((W, H)),
        mask_image=mask.resize((W, H)),
        width=W,
        height=H,
        strength=strength,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    return out.resize((w, h))


def panel(
    pipes: tuple[Any, Any],
    post: str,
    index: int,
    person: dict[str, Any] | None,
    out: Path,
    t: Timer,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.open(io.BytesIO(fetch(f"{SITE}/api/images/posts/{post}?panel={index}&frame=0")))
    img = img.convert("RGB")
    depth = Image.open(
        io.BytesIO(fetch(f"{SITE}/api/images/posts/{post}?panel={index}&frame=depth"))
    ).convert("L")
    depth = depth.resize(img.size)
    W, H = img.size
    mask = t.run(f"{index}: 인물 자리 찾기", lambda: person_mask(img, depth))
    alpha = mask.filter(ImageFilter.GaussianBlur(1.2))

    # 배경: 인물 자리를 넓혀 지우고 다시 그린다
    hole = (
        mask.point(lambda v: 255 if v > 40 else 0)
        .filter(_max(31))
        .filter(ImageFilter.GaussianBlur(6))
    )
    bg = t.run(
        f"{index}: 배경 따로 그리기",
        lambda: inpaint(pipes[1], img, hole, BG_PROMPT, 0.99, 1000 + index),
    )

    char = img.copy()
    char.putalpha(alpha)

    # 입: 얼굴 자리 아래쪽 작은 상자만 다시 그린다
    opened = None
    if person:
        cx, cy, s = person["x"] * W, person["y"] * H, person["size"] * H
        box = (cx - 0.22 * s, cy + 0.12 * s, cx + 0.22 * s, cy + 0.42 * s)
        m = Image.new("L", img.size, 0)
        ImageDraw.Draw(m).ellipse(box, fill=255)
        m = m.filter(ImageFilter.GaussianBlur(3))
        crop = (
            max(0, int(cx - 1.2 * s)),
            max(0, int(cy - 1.2 * s)),
            min(W, int(cx + 1.2 * s)),
            min(H, int(cy + 1.2 * s)),
        )
        # 얼굴 둘레만 잘라 크게 그려야 입이 뭉개지지 않는다
        face = img.crop(crop)
        fm = m.crop(crop)
        scale = 768 / max(face.size)
        size = (int(face.width * scale), int(face.height * scale))
        drawn = t.run(
            f"{index}: 벌린 입 그리기",
            lambda: inpaint(
                pipes[1], face.resize(size), fm.resize(size), MOUTH_PROMPT, 0.75, 2000 + index
            ),
        ).resize(face.size)
        full = img.copy()
        full.paste(Image.composite(drawn, face, fm), crop[:2])
        opened = full.copy()
        opened.putalpha(alpha)

    bg.save(out / f"{index}-bg.png")
    char.save(out / f"{index}-char.png")
    if opened:
        opened.save(out / f"{index}-char-open.png")

    # 미리보기: 배경 고정, 인물만 오르내리고 입을 여닫는다(12fps 4초)
    frames = []
    talk = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    import math

    for f in range(48):
        dy = round(1.5 * math.sin(f / 48 * 2 * math.pi * 1.0))
        layer = opened if opened and talk[f % len(talk)] else char
        frame = bg.copy()
        frame.paste(layer, (0, dy), layer)
        frames.append(frame.resize((W // 2, H // 2)))
    frames[0].save(
        out / f"{index}-preview.gif", save_all=True, append_images=frames[1:], duration=83, loop=0
    )
    sheet = Image.new("RGB", (W, H * 3), "white")
    sheet.paste(img, (0, 0))
    sheet.paste(bg, (0, H))
    checker = Image.new("RGB", (W, H), (200, 200, 200))
    d = ImageDraw.Draw(checker)
    for y in range(0, H, 32):
        for x in range(0, W, 32):
            if (x // 32 + y // 32) % 2:
                d.rectangle((x, y, x + 31, y + 31), fill=(150, 150, 150))
    checker.paste(opened or char, (0, 0), opened or char)
    sheet.paste(checker, (0, 2 * H))
    sheet.save(out / f"{index}-sheet.png")
    return {"panel": index}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True)
    ap.add_argument("--panels", default="1,3,7")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--lora", default="ByteDance/SDXL-Lightning:sdxl_lightning_8step_lora.safetensors"
    )
    ap.add_argument("--offload", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    live = json.loads(fetch(f"{SITE}/api/v1/posts/{a.post}/live"))
    t = Timer()
    pipes = t.run("SDXL 올리기", lambda: load_pipes(a.lora, a.offload))
    for i in [int(x) for x in a.panels.split(",")]:
        people = live["panels"][i - 1].get("people") or []
        panel(pipes, a.post, i, people[0] if people else None, out, t)
    import torch

    print(f"최대 VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
    (out / "times.json").write_text(json.dumps(t.rows, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
