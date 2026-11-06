"""cast.py로 그린 캐릭터(정체성 x 7 테마 x 6 표정)를 캐릭터 DB에 등록한다.

conversation-service V5 구조(characters + character_looks)를 쓴다. 내부 API가 게이트웨이 밖으로 열려 있지 않아서
DB와 MinIO에 직접 쓴다. 키 규칙은 CharacterKeys.java, 설명 임베딩은 GeminiEmbedder.java와 같다.

  python register_cast.py --cast cast.json --png png --env local.env [--dry-run] [--limit 10] [--no-embed]

- 이름이 같은 캐릭터가 있으면 public_id를 두고 정보와 모습을 바꾼다 (다시 돌려도 된다, 끊겨도 이어서 돈다).
- 참조 그림: characters/{캐릭터}/looks/{모습}/reference.png (PNG)
- 설명 임베딩: 기본 표정(NEUTRAL) 모습만, character_looks.description_embedding 컬럼(pgvector)이 있을 때만.
  Gemini 무료 등급은 하루 요청 수가 적어서, 막히면 --no-embed로 그림만 올리고 나중에 다시 돌린다.

필요한 패키지: psycopg[binary], minio
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import secrets
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EMBED_MODEL = "gemini-embedding-001"
ALPHABET = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXPR = {"neutral": "NEUTRAL", "happy": "HAPPY", "sad": "SAD", "angry": "ANGRY", "surprised": "SURPRISED",
        "embarrassed": "EMBARRASSED"}
GENDER = {"f": "FEMALE", "m": "MALE"}
JOB_EN = {"casual": "", "student": "student", "office": "office worker", "doctor": "doctor", "nurse": "nurse", "chef": "chef",
          "firefighter": "firefighter", "police": "police officer", "detective": "detective", "teacher": "teacher",
          "farmer": "farmer", "barista": "barista", "florist": "florist", "painter": "painter", "mail": "mail carrier",
          "mechanic": "mechanic", "reporter": "reporter", "waiter": "waiter", "astronaut": "astronaut", "musician": "musician",
          "athlete": "athlete", "programmer": "programmer", "shopkeeper": "shopkeeper", "scientist": "scientist",
          "fighter": "martial artist"}
JOB_KO = {"casual": None}


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


def gemini_batch(texts: list[str], key: str) -> list[list[float]]:
    body = {"requests": [{"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]},
                          "taskType": "SEMANTIC_SIMILARITY", "outputDimensionality": 768} for t in texts]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents"
    for attempt in range(6):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = []
                for e in json.load(r)["embeddings"]:
                    v = e["values"]
                    n = math.sqrt(sum(x * x for x in v))
                    out.append([x / n for x in v])
                return out
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                time.sleep(20 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cast", default="cast.json")
    ap.add_argument("--png", default="png")
    ap.add_argument("--env", default="local.env")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from minio import Minio

    env = load_env(Path(args.env))
    cast = json.loads(Path(args.cast).read_text(encoding="utf-8"))
    if args.limit:
        cast = cast[: args.limit]
    png_root = Path(args.png)
    endpoint = re.sub(r"^https?://", "", env["MINIO_ENDPOINT"])
    s3 = Minio(endpoint, access_key=env["MINIO_ACCESS_KEY"], secret_key=env["MINIO_SECRET_KEY"],
               secure=env["MINIO_ENDPOINT"].startswith("https"), region=env.get("MINIO_REGION", "us-east-1"))
    bucket = env["MINIO_BUCKET"]
    conn, schema = connect(env)
    with conn.cursor() as cur:
        cur.execute("select 1 from information_schema.tables where table_schema=%s and table_name='character_looks'", (schema,))
        if cur.fetchone() is None:
            raise SystemExit("character_looks 표가 없다. conversation-service V5(PR #17)가 배포된 뒤 돌린다")
        cur.execute("select 1 from information_schema.columns where table_schema=%s and table_name='character_looks'"
                    " and column_name='description_embedding'", (schema,))
        has_vector = cur.fetchone() is not None
    embed = has_vector and not args.no_embed and bool(env.get("GEMINI_API_KEY"))
    print(f"스키마 {schema}, 설명 임베딩 {'넣음' if embed else '건너뜀 (컬럼 없음 또는 --no-embed)'}, 캐릭터 {len(cast)}명")
    pool = ThreadPoolExecutor(args.workers)
    t0 = time.time()

    for n, c in enumerate(cast, 1):
        job_en = JOB_EN.get(c["job"], "")
        job_ko = c["themes"].get("DAILY", {}).get("job") or None
        age = c["age"].upper()
        desc = f'{c["name"]}: {age.lower()} {"woman" if c["gender"] == "f" else "man"}' + (f", {job_en}" if job_en else "")
        if args.dry_run:
            print(f"[dry-run] {c['id']} {c['name']} {c['gender']} {age} {job_ko} looks={len(c['themes']) * len(EXPR)}")
            continue
        with conn.cursor() as cur:
            cur.execute("select id, public_id from characters where name=%s", (c["name"],))
            row = cur.fetchone()
            if row:
                cid, cpid = row
                cur.execute("update characters set description=%s, job=%s, gender=%s, age_group=%s, active=true,"
                            " updated_at=current_timestamp(6) where id=%s", (desc, job_ko, GENDER[c["gender"]], age, cid))
            else:
                cpid = public_id()
                cur.execute("insert into characters (public_id, name, description, job, gender, age_group, ip_adapter_scale)"
                            " values (%s,%s,%s,%s,%s,%s,0.6) returning id",
                            (cpid, c["name"], desc, job_ko, GENDER[c["gender"]], age))
                cid = cur.fetchone()[0]
            cur.execute("select concept, expression, public_id from character_looks where character_id=%s", (cid,))
            existing = {(r[0], r[1]): r[2] for r in cur.fetchall()}
            uploads, neutral = [], []
            for theme, info in c["themes"].items():
                look = info["look"]
                ldesc = look + (f", {job_en}" if job_en else "")
                for e, E in EXPR.items():
                    f = png_root / c["id"] / f"{theme}_{e}.png"
                    if not f.exists():
                        continue
                    lpid = existing.get((theme, E)) or public_id()
                    key = f"characters/{cpid}/looks/{lpid}/reference.png"
                    if (theme, E) in existing:
                        cur.execute("update character_looks set look=%s, description=%s, reference_image_key=%s,"
                                    " ip_adapter_embed_key=null, ip_adapter_model=null, updated_at=current_timestamp(6)"
                                    " where public_id=%s", (look, ldesc, key, lpid))
                    else:
                        cur.execute("insert into character_looks (public_id, character_id, concept, expression, look,"
                                    " description, reference_image_key) values (%s,%s,%s,%s,%s,%s,%s)",
                                    (lpid, cid, theme, E, look, ldesc, key))
                    uploads.append((key, f))
                    if E == "NEUTRAL":
                        neutral.append((lpid, ldesc))
            futs = [pool.submit(lambda k, p: s3.put_object(bucket, k, io.BytesIO(p.read_bytes()), p.stat().st_size,
                                                           content_type="image/png"), k, p) for k, p in uploads]
            for fu in futs:
                fu.result()
            if embed and neutral:
                vecs = gemini_batch([d for _, d in neutral], env["GEMINI_API_KEY"])
                for (lpid, _), v in zip(neutral, vecs):
                    cur.execute("update character_looks set description_embedding=%s::vector, embedding_model=%s"
                                " where public_id=%s", ("[" + ",".join(f"{x:.7f}" for x in v) + "]", EMBED_MODEL, lpid))
        conn.commit()
        if n % 10 == 0 or n == len(cast):
            print(f"{n}/{len(cast)} {c['name']} 모습 {len(uploads)}장, {time.time() - t0:.0f}s", flush=True)
    print("끝")


if __name__ == "__main__":
    main()
