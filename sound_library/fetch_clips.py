"""후보 클립을 원본 zip에서 꺼내 미리 듣기용 MP3(최대 30초, 음량 맞춤)로 바꾼다.

python fetch_clips.py --candidates candidates.jsonl --out audio/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remote_zip import RemoteZip
from select_clips import FMA_FULL_ZIP, FMA_ZIP, FSD_PARTS


def key(c: dict) -> str:
    return f"{c['source'].lower()}-{c['source_id']}"


def convert(raw: bytes, suffix: str, out: Path) -> float:
    # Windows는 열려 있는 임시 파일을 다른 프로세스가 못 연다. 폴더째 만들고 닫은 뒤 ffmpeg에 넘긴다
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix}"
        src.write_bytes(raw)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(src),
                "-t",
                "30",
                "-af",
                "loudnorm=I=-18:TP=-1.5:LRA=11,afade=t=in:d=0.05",
                "-ar",
                "44100",
                "-b:a",
                "128k",
                str(out),
            ],
            check=True,
        )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(x) for x in a.candidates.read_text(encoding="utf-8").splitlines()]
    zipped = [c for c in cands if not c.get("preview_url")]
    zips = (
        {
            "eval": RemoteZip(FSD_PARTS["eval"]),
            "dev": RemoteZip(FSD_PARTS["dev"]),
            "fma": RemoteZip([FMA_ZIP]),
        }
        if zipped
        else {}
    )
    if any(c.get("fma_full") for c in zipped):
        zips["fma_full"] = RemoteZip([FMA_FULL_ZIP])

    def one(c: dict) -> str:
        out = a.out / f"{key(c)}.mp3"
        if out.exists() and out.stat().st_size > 0:
            return "skip"
        if c.get("preview_url"):  # Freesound 미리 듣기 MP3 (4차부터)
            try:
                req = urllib.request.Request(
                    c["preview_url"], headers={"User-Agent": "4cut-sound-library"}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    convert(r.read(), ".mp3", out)
            except Exception as e:
                out.unlink(missing_ok=True)
                return f"fail {key(c)} {e}"
            return "ok"
        if c["source"] == "FMA":
            z = zips["fma_full" if c.get("fma_full") else "fma"]
        else:
            z = zips[c["split"]]
        try:
            convert(z.read(c["zip_name"]), Path(c["zip_name"]).suffix, out)
        except Exception as e:
            out.unlink(missing_ok=True)
            return f"fail {key(c)} {e}"
        return "ok"

    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(one, cands))
    fails = [r for r in results if r.startswith("fail")]
    print(f"{len(results) - len(fails)}개 준비, 실패 {len(fails)}개")
    for r in fails:
        print(r)


if __name__ == "__main__":
    main()
