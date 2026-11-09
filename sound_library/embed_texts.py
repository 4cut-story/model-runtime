"""배경음 설명 임베딩(BGE-M3 1024차원)을 미리 만들어 파일로 둔다.

임베딩 서버에 닿지 않는 PC에서 register_sounds.py를 돌릴 때 쓴다. 서버에 닿는 곳에서 만들고,
결과를 register_sounds.py --text-embeddings로 넘긴다. 글은 register_sounds.py와 같은 함수로 만든다.

  python embed_texts.py --candidates data/candidates-1.jsonl --clap data/clap-1.jsonl.gz \
      --descriptions data/descriptions-1.jsonl --out data/text-1.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from categories import ALL
from fetch_clips import key
from register_sounds import TEXT_MODEL, embed_texts, jsonl, search_text, select_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--clap", type=Path, required=True)
    ap.add_argument("--descriptions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embedding-url", default="https://emb.java21.net")
    a = ap.parse_args()

    labels = {c.code: c.label for c in ALL}
    desc = jsonl(a.descriptions)
    _, rows = select_rows(a.candidates, jsonl(a.clap), desc)
    lines = []
    for i in range(0, len(rows), 32):
        batch = rows[i : i + 32]
        texts = [search_text(desc[key(c)], labels[c["category"]]) for c in batch]
        for c, v in zip(batch, embed_texts(a.embedding_url, texts), strict=True):
            emb = [round(x, 6) for x in v]
            lines.append(json.dumps({"key": key(c), "model": TEXT_MODEL, "embedding": emb}))
        print(f"{min(i + 32, len(rows))}/{len(rows)}", flush=True)
    a.out.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8")))
    print(f"{a.out} {len(lines)}개")


if __name__ == "__main__":
    main()
