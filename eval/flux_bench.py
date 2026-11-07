"""FLUX.1-schnell 사전 측정 (GPU PC에서 한 번 손으로 돌린다).

SDXL(DreamShaper XL Lightning)이 회차 23~27에서 83점 안팎에 머물러, 글 이해가 나은 FLUX.1-schnell
(Apache 2.0)이 16GB에 들어가는지, 한 컷에 몇 초인지, 회차 27에서 틀린 컷을 맞게 그리는지 본다.
transformer와 T5 인코더를 bitsandbytes 4비트(NF4)로 읽고 model CPU offload로 돌린다.
처음 실행은 bf16 원본을 받아 양자화한 뒤 --out 아래 quantized/에 저장하고, 다음부터는 그것을 읽는다.

    pip install bitsandbytes
    python eval/flux_bench.py --cache D:\\hf-cache --out D:\\4cut\\flux-bench

결과: --out에 컷 PNG와 bench.json(로드 시간, 컷별 시간, 최대 VRAM).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = "black-forest-labs/FLUX.1-schnell"
STYLE = "simple cartoon webtoon panel, thick outlines, flat colors, simple background."

# 회차 27에서 두 채점이 심각 오류로 짚은 컷 (콘티 scene·setting·인물 외모를 문장으로 옮김)
PANELS = {
    "funny-15-p1": (
        "Living room with a sofa. A brown bear with small round ears, wearing an orange "
        "scarf, happily presses the button of a round robot vacuum on the floor. Beside "
        "him a gray rabbit with long ears, wearing a blue hoodie, looks worried and "
        "points at a striped sock on the floor."
    ),
    "funny-15-p3": (
        "Close-up in a living room. A round robot vacuum is stuck under a sofa with a "
        "striped sock in it. A small brown puppy sits next to it, guarding it. A gray "
        "rabbit with long ears in a blue hoodie gasps with both hands on her cheeks."
    ),
    "funny-12-p1": (
        "Supermarket aisle with freezer shelves and a dumpling sample stand with a "
        "steaming tray. A brown bear cub with small round ears, wearing a red cap, drools "
        "and points at the tray. Next to him a white rabbit mother with long ears, "
        "wearing a green apron, holds a shopping basket and raises an eyebrow."
    ),
    "mystery-08-p2": (
        "Daytime, pine trees, a wooden mountain cabin with a half-open door. Two small "
        "children of the same height creep toward the door: a little boy swordsman in a "
        "blue robe holding a wooden sword, brave face, and a little girl in a pink robe "
        "holding a paper lantern, peeking with wide eyes."
    ),
    "love-07-p1": (
        "Rainy street. A young man with short black hair in a navy coat holds a small "
        "umbrella tilted over a young woman with a brown ponytail in a yellow raincoat; "
        "his own shoulder is wet. She looks at him, he smiles."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Hugging Face 캐시 폴더")
    ap.add_argument("--out", required=True, help="결과 폴더")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.environ["HF_HOME"] = args.cache

    import torch
    from diffusers import BitsAndBytesConfig, FluxPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as TfBnb
    from transformers import T5EncoderModel

    out = Path(args.out)
    saved = out / "quantized"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if (saved / "transformer").exists():
        transformer = FluxTransformer2DModel.from_pretrained(
            saved / "transformer", torch_dtype=torch.bfloat16
        )
        t5 = T5EncoderModel.from_pretrained(saved / "text_encoder_2", torch_dtype=torch.bfloat16)
    else:
        nf4 = dict(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            REPO,
            subfolder="transformer",
            quantization_config=BitsAndBytesConfig(**nf4),
            torch_dtype=torch.bfloat16,
        )
        t5 = T5EncoderModel.from_pretrained(
            REPO,
            subfolder="text_encoder_2",
            quantization_config=TfBnb(**nf4),
            torch_dtype=torch.bfloat16,
        )
        transformer.save_pretrained(saved / "transformer")
        t5.save_pretrained(saved / "text_encoder_2")
    pipe = FluxPipeline.from_pretrained(
        REPO, transformer=transformer, text_encoder_2=t5, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    load_s = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    times = {}
    for name, scene in PANELS.items():
        g = torch.Generator("cpu").manual_seed(args.seed)
        t = time.perf_counter()
        image = pipe(
            prompt=f"{STYLE} {scene}",
            width=args.size,
            height=args.size,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            max_sequence_length=256,
            generator=g,
        ).images[0]
        times[name] = round(time.perf_counter() - t, 2)
        image.save(out / f"{name}.png")
        print(name, times[name], "s", flush=True)
    result = {
        "loadSeconds": round(load_s, 1),
        "panelSeconds": times,
        "peakVramGiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "size": args.size,
        "steps": args.steps,
        "torch": torch.__version__,
    }
    (out / "bench.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
