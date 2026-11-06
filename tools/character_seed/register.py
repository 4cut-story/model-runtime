"""고른 참조 그림을 캐릭터 DB에 등록한다 (conversation-service characters 표 + MinIO).

conversation-service의 /internal/characters와 같은 일을 한다. 그 API는 게이트웨이 밖으로 열려 있지 않아서
여기서는 DB와 MinIO에 직접 쓴다. 키 규칙은 CharacterKeys.java, 임베딩은 GeminiEmbedder.java와 같다.

  python register.py --catalog catalog.yaml --out out --picks picks.yaml --env local.env [--dry-run]

- 이름이 같은 캐릭터가 있으면 public_id를 그대로 두고 외모·설명·참조 그림을 바꾼다 (다시 돌려도 된다).
- 참조 그림: characters/{publicId}/reference.png
- IP-Adapter embedding(draw.py가 계산): characters/{publicId}/ip-adapter/plus-face-sdxl-vit-h.safetensors
- 설명 임베딩: description_embedding 컬럼이 있을 때만 (pgvector). 없으면 이름으로만 찾는다.

필요한 패키지: psycopg[binary], minio, pyyaml
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import secrets
import urllib.request
from pathlib import Path

import yaml

ADAPTER = "plus-face-sdxl-vit-h"
EMBED_MODEL = "gemini-embedding-001"
ALPHABET = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def public_id() -> str:
    return "".join(ALPHABET[b & 63] for b in secrets.token_bytes(21))


def load_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def connect(env: dict[str, str]):
    import psycopg

    m = re.match(r"jdbc:postgresql://([^:/]+):(\d+)/([^?]+)(?:\?currentSchema=(\w+))?", env["DB_URL"])
    if not m:
        raise SystemExit("DB_URL 형식을 읽지 못했다")
    host, port, db, schema = m.groups()
    schema = env.get("DB_SCHEMA") or schema
    conn = psycopg.connect(host=host, port=int(port), dbname=db, user=env["DB_USERNAME"],
                           password=env["DB_PASSWORD"], options=f"-c search_path={schema},public")
    return conn, schema


def gemini(text: str, key: str) -> list[float]:
    body = {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": text.strip()}]},
            "taskType": "SEMANTIC_SIMILARITY", "outputDimensionality": 768}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=15) as r:
        v = json.load(r)["embedding"]["values"]
    if len(v) != 768:
        raise RuntimeError(f"임베딩 차원이 {len(v)}이다")
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalog.yaml")
    ap.add_argument("--out", default="out")
    ap.add_argument("--picks", default="picks.yaml")
    ap.add_argument("--env", default="local.env")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from minio import Minio

    env = load_env(Path(args.env))
    catalog = yaml.safe_load(Path(args.catalog).read_text(encoding="utf-8"))
    picks = yaml.safe_load(Path(args.picks).read_text(encoding="utf-8")) or {}
    manifest = json.loads((Path(args.out) / "manifest.json").read_text(encoding="utf-8"))
    endpoint = re.sub(r"^https?://", "", env["MINIO_ENDPOINT"])
    s3 = Minio(endpoint, access_key=env["MINIO_ACCESS_KEY"], secret_key=env["MINIO_SECRET_KEY"],
               secure=env["MINIO_ENDPOINT"].startswith("https"), region=env.get("MINIO_REGION", "us-east-1"))
    bucket = env["MINIO_BUCKET"]
    conn, schema = connect(env)
    with conn.cursor() as cur:
        cur.execute("select 1 from information_schema.columns where table_schema=%s and table_name='characters'"
                    " and column_name='description_embedding'", (schema,))
        has_vector = cur.fetchone() is not None
    print(f"스키마 {schema}, 설명 임베딩 컬럼 {'있음' if has_vector else '없음 (이름으로만 찾음)'}")

    done = 0
    for theme, spec in catalog["themes"].items():
        for ch in spec["characters"]:
            name = ch["name"]
            seed = picks.get(name)
            if seed is None:
                print(f"건너뜀 {name}: picks에 없음")
                continue
            d = Path(args.out) / manifest[name]["dir"]
            png = (d / f"seed{seed}.png").read_bytes()
            emb_path = d / f"seed{seed}.ipadapter.safetensors"
            emb = emb_path.read_bytes() if emb_path.exists() else None
            look = ch["look"].strip()
            desc = (ch.get("description") or look).strip()
            scale = float(ch.get("scale", 0.6))
            if args.dry_run:
                print(f"[dry-run] {theme} {name} seed{seed} embed={'Y' if emb else 'N'}")
                continue
            with conn.cursor() as cur:
                cur.execute("select public_id from characters where name=%s", (name,))
                row = cur.fetchone()
                pid = row[0] if row else public_id()
                ref_key = f"characters/{pid}/reference.png"
                emb_key = f"characters/{pid}/ip-adapter/{ADAPTER}.safetensors"
                s3.put_object(bucket, ref_key, io.BytesIO(png), len(png), content_type="image/png")
                if emb:
                    s3.put_object(bucket, emb_key, io.BytesIO(emb), len(emb),
                                  content_type="application/octet-stream")
                if row:
                    cur.execute(
                        "update characters set concept=%s, look=%s, description=%s, reference_image_key=%s,"
                        " ip_adapter_embed_key=%s, ip_adapter_model=%s, ip_adapter_scale=%s, active=true,"
                        " updated_at=current_timestamp(6) where public_id=%s",
                        (theme, look, desc, ref_key, emb_key if emb else None, ADAPTER if emb else None, scale, pid))
                else:
                    cur.execute(
                        "insert into characters (public_id, name, concept, look, description, reference_image_key,"
                        " ip_adapter_embed_key, ip_adapter_model, ip_adapter_scale) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (pid, name, theme, look, desc, ref_key, emb_key if emb else None, ADAPTER if emb else None,
                         scale))
                if has_vector and env.get("GEMINI_API_KEY"):
                    v = gemini(desc, env["GEMINI_API_KEY"])
                    cur.execute("update characters set description_embedding=%s::vector, embedding_model=%s"
                                " where public_id=%s", ("[" + ",".join(f"{x:.7f}" for x in v) + "]", EMBED_MODEL, pid))
            conn.commit()
            done += 1
            print(f"{'바꿈' if row else '등록'} {theme} {name} {pid} seed{seed}")
    print(f"끝: {done}명")


if __name__ == "__main__":
    main()
