"""자주 쓰는 소리부터 설명·등록한다 (4차부터, 기존 소리 다시 쓰기에도).

  python priority.py --env local.env [--register]       # 한 PC에서 설명·등록
  python priority.py --env local.env --gate              # GPU PC: 설명만 (운영 작업 양보)
  python priority.py --env local.env --register --no-describe   # 맥: 받은 설명으로 등록만

카테고리를 아래 묶음 순서로 나눠, 묶음마다 4차 신규 → 1~3차 다시 쓰기 순으로 describe_local.py를 돌리고
그 묶음이 다 써지면 바로 register_sounds.py로 MinIO·DB에 넣는다. 다시 돌리면 끝난 묶음은 건너뛴다.

- 4차는 CLAP이 끝난 소리만 쓴다. CLAP이 아직이면 다음 바퀴에 다시 본다.
- 1~3차 MP3는 -23 LUFS로 다시 받아(work/audio-norm) --reupload로 덮어쓴다.
- --register가 없으면 설명만 쓰고 등록은 하지 않는다(운영 DB를 바꾸므로 승인을 받고 켠다).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fetch_clips import key
from select_clips import jsonline

# 만화 장면에 자주 나오는 장소·소리 순서. 여기 없는 카테고리는 마지막 묶음이다
TIERS: list[tuple[str, set[str]]] = [
    ("비·눈", {"RAIN", "THUNDERSTORM", "SNOWFALL", "SNOW"}),
    ("바다·파도", {"OCEAN_WAVES", "BEACH", "HARBOR"}),
    ("도심 거리·차도", {"CITY_TRAFFIC", "STREET", "CAR_PASS", "CAR_HORN", "HIGHWAY"}),
    ("카페·식당", {"CAFE", "RESTAURANT", "FASTFOOD", "BAR", "KITCHEN", "COFFEE"}),
    # 집·거실·식당 장면에 장소 소리가 없어 음악으로 대체되던 일이 많아 앞으로 당긴다
    (
        "실내 방·집",
        {
            "HOUSE_NOISES", "CLOCK", "FAN", "FRIDGE", "BLACKOUT", "OFFICE", "KETTLE",
            "MICROWAVE", "SINK", "VACUUM", "LAUNDRY", "BATHROOM", "TOILET", "RADIO",
        },
    ),
    ("공원·밤 풀벌레", {"PARK", "NIGHT_INSECTS", "NIGHT", "INSECTS", "FROGS"}),
    ("숲·새소리", {"FOREST_BIRDS", "JUNGLE", "OWL", "CROW"}),
    ("바람", {"WIND", "WIND_CHIME"}),
    ("학교·교실", {"SCHOOL_KIDS", "PLAYGROUND", "LIBRARY", "WRITING"}),
    ("지하철·버스", {"SUBWAY_TRAIN", "STATION", "BUS", "TRAM"}),
    (
        "생활 효과음",
        {
            "DOOR_KNOCK", "DOOR_CREAK", "DOORBELL", "FOOTSTEPS", "PHONE", "MOBILE",
            "NOTIFICATION", "ALARM", "DISHES", "GLASS_CLINK", "KIOSK",
        },
    ),
]  # fmt: skip
REST = "나머지"
HERE = Path(__file__).resolve().parent


def tier(category: str) -> int:
    for i, (_, codes) in enumerate(TIERS):
        if category in codes:
            return i
    return len(TIERS)


def rest_groups(size: int = 400) -> list[tuple[str, set[str]]]:
    """나머지 카테고리를 4차 후보 수 기준 약 size개씩 묶는다. 묶음마다 바로 등록하려고 나눈다."""
    from collections import Counter

    counts = Counter(
        c["category"]
        for c in read(HERE / "data/candidates-4.jsonl")
        if tier(c["category"]) == len(TIERS)
    )
    groups: list[tuple[str, set[str]]] = []
    cur: set[str] = set()
    total = 0
    for code, k in sorted(counts.items()):
        cur.add(code)
        total += k
        if total >= size:
            groups.append((f"{REST} {len(groups) + 1}", cur))
            cur, total = set(), 0
    if cur:
        groups.append((f"{REST} {len(groups) + 1}", cur))
    return groups


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(jsonline(r) + "\n" for r in rows), "utf-8")


def note(log: Path, text: str) -> None:
    with open(log, "a", encoding="utf-8") as f:
        print(text, file=f, flush=True)


def run(args: list[str], log: Path) -> None:
    with open(log, "a", encoding="utf-8") as f:
        subprocess.run([sys.executable, *args], cwd=HERE, stdout=f, stderr=f, check=True)


def batches() -> list[dict]:
    """(차수, 설명 파일, CLAP 파일, MP3 폴더) 목록. 4차가 앞이다."""
    out = [
        {
            "n": 4,
            "desc": HERE / "data/descriptions-4.jsonl",
            "clap": HERE / "data/clap-4.jsonl",
            "audio": HERE / "work/audio4",
        }
    ]
    for n in (1, 2, 3):
        out.append(
            {
                "n": n,
                "desc": HERE / f"data/descriptions-{n}.v2.jsonl",
                "clap": HERE / f"data/clap-{n}.jsonl.gz",
                "audio": HERE / "work/audio-norm",
                "only": HERE / f"data/descriptions-{n}.jsonl",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=Path, required=True)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--wait", type=int, default=600, help="4차 CLAP을 기다릴 때 쉬는 초")
    ap.add_argument(
        "--model", default="qwen3:8b", help="설명 모델 (낮에 서버 Ollama를 쓸 때 qwen3.5:9b)"
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="GPU PC 공용: 운영 워커가 GPU를 쓰는 동안 기다리고, 한 개 쓸 때마다 모델을 내린다",
    )
    ap.add_argument(
        "--no-describe",
        action="store_true",
        help="설명은 쓰지 않고(다른 PC가 쓴다) 설명이 다 된 묶음만 등록한다",
    )
    ap.add_argument(
        "--no-fetch",
        action="store_true",
        help="1~3차 MP3 다시 받기·재업로드를 미룬다(낮에 맥을 쓰는 동안). 설명은 쓴다",
    )
    a = ap.parse_args()
    work = HERE / "work/prio"
    work.mkdir(parents=True, exist_ok=True)
    log = HERE / "work-priority.log"

    while True:
        pending = False
        tiers = TIERS + rest_groups()
        where = {code: i for i, (_, codes) in enumerate(tiers) for code in codes}
        for t in range(len(tiers) + 1):
            name = tiers[t][0] if t < len(tiers) else REST
            # 앞 묶음의 4차 CLAP을 기다리는 동안 나머지로 넘어가지 않는다
            if t >= len(TIERS) and pending:
                break
            for b in batches():
                n = b["n"]
                mark = work / f"reg-{n}-{t}.done"
                if mark.exists():
                    continue
                cands = [
                    c
                    for c in read(HERE / f"data/candidates-{n}.jsonl")
                    if where.get(c["category"], len(tiers)) == t
                ]
                if "only" in b:
                    keep = {d["key"] for d in read(b["only"]) if not d.get("exclude")}
                    cands = [c for c in cands if key(c) in keep]
                else:  # 4차: 끝내 못 받은 MP3(CLAP도 없다)는 기다리지 않는다
                    cands = [c for c in cands if (b["audio"] / f"{key(c)}.mp3").exists()]
                if not b["clap"].exists() and b["clap"].with_suffix(".jsonl.gz").exists():
                    b["clap"] = b["clap"].with_suffix(".jsonl.gz")
                clap = {x["key"] for x in read(b["clap"])}
                ready = [c for c in cands if key(c) in clap]
                if not cands:
                    mark.touch()
                    continue
                part = work / f"cand-{n}-{t}.jsonl"
                write(part, ready)
                note(log, f"[{name}] {n}차 {len(ready)}/{len(cands)}개 설명")
                if ready and not a.no_describe:
                    args = ["describe_local.py", "--candidates", str(part), "--out", str(b["desc"])]
                    args += ["--clap", str(b["clap"])]
                    args += ["--gate", "--keep-alive", "0"] if a.gate else ["--keep-alive", "5m"]
                    args += ["--model", a.model]
                    run(args, log)
                if len(ready) < len(cands):  # CLAP이 아직 덜 끝났다
                    pending = True
                    continue
                if a.no_describe and not {key(c) for c in ready} <= {
                    d["key"] for d in read(b["desc"])
                }:  # 다른 PC가 아직 쓰는 중
                    pending = True
                    continue
                if not a.register:
                    continue
                if a.no_fetch and n != 4:  # 밤에 받는다
                    continue
                desc = {d["key"]: d for d in read(b["desc"])}
                sub = work / f"desc-{n}-{t}.jsonl"
                write(sub, [desc[key(c)] for c in cands if key(c) in desc])
                if n != 4:
                    for _ in range(2):  # 원격 zip 읽기가 가끔 끊긴다(IncompleteRead)
                        run(
                            [
                                "fetch_clips.py",
                                "--candidates",
                                str(part),
                                "--out",
                                str(b["audio"]),
                                "--workers",
                                "1",
                            ],  # 맥 사용 중 네트워크를 덜 잡게
                            log,
                        )
                # 끝내 못 받은 MP3는 이번 등록에서 뺀다(기존 소리는 MinIO의 예전 MP3가 그대로 남는다)
                have = [c for c in ready if (b["audio"] / f"{key(c)}.mp3").exists()]
                if len(have) < len(ready):
                    note(log, f"[{name}] {n}차 MP3 없는 {len(ready) - len(have)}개는 건너뜀")
                part = work / f"cand-{n}-{t}.reg.jsonl"
                write(part, have)
                args = ["register_sounds.py", "--candidates", str(part), "--audio", str(b["audio"])]
                args += ["--clap", str(b["clap"]), "--descriptions", str(sub), "--env", str(a.env)]
                if n != 4:
                    args.append("--reupload")
                run(args, log)
                mark.touch()
                note(log, f"[{name}] {n}차 등록 끝")
        if not pending or a.no_describe:  # 등록만 할 때는 한 바퀴 돌고 끝낸다(바깥에서 다시 부른다)
            break
        time.sleep(a.wait)
    tiers = TIERS + rest_groups()
    left = [
        f"{n}-{t}"
        for n in (4, 1, 2, 3)
        for t in range(len(tiers) + 1)
        if not (work / f"reg-{n}-{t}.done").exists()
    ]
    if left:
        note(log, f"이번 바퀴 끝, 남은 묶음 {len(left)}개")
    else:
        (work / "all.done").touch()
        note(log, "전부 끝")


if __name__ == "__main__":
    os.chdir(HERE)
    main()
