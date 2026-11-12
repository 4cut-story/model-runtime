"""고른 배경음을 conversation-service DB(background_sounds)와 MinIO에 등록한다.

  python register_sounds.py --candidates candidates.jsonl --audio audio/ --clap clap.jsonl \
      --descriptions descriptions.jsonl --env local.env [--dry-run]

- 미리 듣기 MP3: sounds/{공개 ID}/audio.mp3 (audio/mpeg)
- 소리 임베딩: CLAP 512차원 (clap.jsonl), 설명 임베딩: BGE-M3 1024차원 (임베딩 서버로 이 스크립트가 만든다)
- 임베딩 서버에 닿지 않는 곳에서는 embed_texts.py로 미리 만든 파일을 --text-embeddings로 넘긴다
- (source, source_id)가 같으면 공개 ID와 파일을 두고 내용만 바꾼다. 다시 돌려도 된다.
- 설명에서 exclude=true(카테고리와 다른 소리, 노래가 든 음악 등)인 클립은 넣지 않는다.

필요한 패키지: psycopg[binary], minio
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import secrets
import urllib.request
from pathlib import Path

from categories import MOODS
from fetch_clips import key

ALPHABET = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
AUDIO_MODEL = "clap-larger-general"
TEXT_MODEL = "bge-m3"


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


def read_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8").splitlines()


def jsonl(path: Path) -> dict[str, dict]:
    out = {}
    for line in read_lines(path):
        if line.strip():
            x = json.loads(line)
            out[x["key"]] = x
    return out


def embed_texts(url: str, texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": TEXT_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())["data"]
    out = []
    for d in sorted(data, key=lambda x: x["index"]):
        v = d["embedding"]
        n = sum(x * x for x in v) ** 0.5
        out.append([x / n for x in v])
    return out


def select_rows(candidates: Path, clap: dict, desc: dict) -> tuple[list[dict], list[dict]]:
    """설명과 CLAP이 다 있고 제외되지 않은 후보. 들어 보니 다른 카테고리면 설명 쪽을 따른다."""
    from categories import ALL

    kinds = {c.code: c.kind for c in ALL}
    cands = [json.loads(x) for x in read_lines(candidates)]
    rows = [
        c for c in cands if key(c) in clap and key(c) in desc and not desc[key(c)].get("exclude")
    ]
    for c in rows:
        moved = desc[key(c)].get("category")
        if moved:
            c["category"], c["kind"] = moved, kinds[moved]
    return cands, rows


def search_text(d: dict, label: str) -> str:
    moods = ", ".join(MOODS[m][0] for m in d["moods"] if m in MOODS)
    places = ", ".join(d["places"])
    return f"{d['title']}. {label}. {d['description']} 분위기: {moods}. 장소: {places}"


def vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--clap", type=Path, required=True)
    ap.add_argument("--descriptions", type=Path, required=True)
    ap.add_argument("--env", type=Path, required=True)
    ap.add_argument("--embedding-url", default="https://emb.java21.net")
    ap.add_argument("--text-embeddings", type=Path, help="embed_texts.py가 만든 설명 임베딩 파일")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reupload", action="store_true", help="이미 등록된 소리의 MP3도 다시 올린다")
    a = ap.parse_args()

    import psycopg
    from categories import ALL
    from minio import Minio

    labels = {c.code: c.label for c in ALL}
    clap, desc = jsonl(a.clap), jsonl(a.descriptions)
    cands, rows = select_rows(a.candidates, clap, desc)
    print(f"등록할 배경음 {len(rows)}개 (후보 {len(cands)}개)")
    given = jsonl(a.text_embeddings) if a.text_embeddings else {}
    if given:
        missing = [key(c) for c in rows if key(c) not in given]
        if missing:
            raise SystemExit(f"--text-embeddings에 없는 클립 {len(missing)}개: {missing[:5]}")
    if a.dry_run:
        return

    env = load_env(a.env)
    m = re.match(
        r"jdbc:postgresql://([^:/]+):(\d+)/([^?]+)(?:\?currentSchema=(\w+))?", env["DB_URL"]
    )
    if not m:
        raise SystemExit("DB_URL 형식을 읽지 못했다")
    host, port, db, schema = m.groups()
    schema = env.get("DB_SCHEMA") or schema
    conn = psycopg.connect(
        host=host,
        port=int(port),
        dbname=db,
        user=env["DB_USERNAME"],
        password=env["DB_PASSWORD"],
        options=f"-c search_path={schema},public",
    )
    endpoint = re.sub(r"^https?://", "", env["MINIO_ENDPOINT"])
    s3 = Minio(
        endpoint,
        access_key=env["MINIO_ACCESS_KEY"],
        secret_key=env["MINIO_SECRET_KEY"],
        secure=env["MINIO_ENDPOINT"].startswith("https"),
        region=env.get("MINIO_REGION", "us-east-1"),
    )
    bucket = env["MINIO_BUCKET"]

    with conn.cursor() as cur:
        cur.execute(
            "select udt_schema from information_schema.columns where table_schema = %s"
            " and table_name = 'background_sounds' and column_name = 'audio_embedding'",
            (schema,),
        )
        found = cur.fetchone()
        if found is None:
            raise SystemExit(
                "background_sounds 표나 pgvector 컬럼이 없다. conversation-service V11이 배포된 뒤 돌린다"
            )
        vs = found[0]
        cur.execute("select source || ':' || source_id, public_id from background_sounds")
        known = dict(cur.fetchall())

    done = 0
    for i in range(0, len(rows), 32):
        batch = rows[i : i + 32]
        texts = [search_text(desc[key(c)], labels[c["category"]]) for c in batch]
        if given:
            vectors = [given[key(c)]["embedding"] for c in batch]
        else:
            vectors = embed_texts(a.embedding_url, texts)
        with conn.cursor() as cur:
            for c, tv in zip(batch, vectors, strict=True):
                d, k = desc[key(c)], key(c)
                pid = known.get(f"{c['source']}:{c['source_id']}") or public_id()
                audio_key = f"sounds/{pid}/audio.mp3"
                path = a.audio / f"{k}.mp3"
                # 이미 등록된 소리는 MP3가 MinIO에 있으니 설명만 바꿀 때는 다시 올리지 않는다
                if a.reupload or f"{c['source']}:{c['source_id']}" not in known:
                    s3.fput_object(bucket, audio_key, str(path), content_type="audio/mpeg")
                # 미리 듣기 MP3는 30초로 자른다 (fetch_clips -t 30)
                duration_ms = round(min(float(d.get("duration") or c["duration"]), 30.0) * 1000)
                cur.execute(
                    "insert into background_sounds (public_id, kind, category, title, description,"
                    " moods, places, duration_ms, audio_key, source, source_id, source_url,"
                    " original_title, author, license, license_url, embedding_model, audio_model,"
                    f" audio_embedding, description_embedding) values (%s, %s, %s, %s, %s, %s, %s,"
                    f' %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::"{vs}".vector,'
                    f' %s::"{vs}".vector) on conflict (source, source_id) do update set'
                    " kind = excluded.kind, category = excluded.category, title = excluded.title,"
                    " description = excluded.description, moods = excluded.moods,"
                    " places = excluded.places, duration_ms = excluded.duration_ms,"
                    " audio_key = excluded.audio_key, source_url = excluded.source_url,"
                    " original_title = excluded.original_title, author = excluded.author,"
                    " license = excluded.license, license_url = excluded.license_url,"
                    " embedding_model = excluded.embedding_model,"
                    " audio_model = excluded.audio_model, audio_embedding = excluded.audio_embedding,"
                    " description_embedding = excluded.description_embedding, active = true,"
                    " updated_at = current_timestamp",
                    (
                        pid,
                        c["kind"],
                        c["category"],
                        d["title"][:60],
                        d["description"],
                        ",".join(d["moods"]),
                        ",".join(d["places"])[:200],
                        duration_ms,
                        audio_key,
                        c["source"],
                        c["source_id"],
                        c["source_url"][:500],
                        (c["title_original"] or "")[:300] or None,
                        c["author"][:200],
                        c["license"],
                        c["license_url"],
                        TEXT_MODEL,
                        AUDIO_MODEL,
                        vec(clap[k]["embedding"]),
                        vec(tv),
                    ),
                )
        conn.commit()
        done += len(batch)
        print(f"{done}/{len(rows)}")

    # 다시 쓴 설명에서 빠진 소리(배경음인데 한 번 나는 소리 등)는 검색에 안 나오게 끈다
    dropped = [
        (c["source"], c["source_id"])
        for c in cands
        if key(c) in desc
        and desc[key(c)].get("exclude")
        and f"{c['source']}:{c['source_id']}" in known
    ]
    with conn.cursor() as cur:
        for src, sid in dropped:
            cur.execute(
                "update background_sounds set active = false, updated_at = current_timestamp"
                " where source = %s and source_id = %s",
                (src, sid),
            )
    conn.commit()
    print(f"끝 (설명에서 빠져 끈 소리 {len(dropped)}개)")


if __name__ == "__main__":
    main()
