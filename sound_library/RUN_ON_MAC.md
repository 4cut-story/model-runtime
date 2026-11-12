# 맥에서 사운드 라이브러리 돌리기 (수집·벡터화·등록 전부)

GPU PC는 운영 그림 작업 전용으로 비워 두기로 해서, 사운드 수집·CLAP 임베딩·설명(Ollama)·BGE-M3
설명 임베딩·등록까지 **전부 맥에서** 한다. 브랜치 `claude/freesound-cc0`.

## 준비물 (맥)
- `brew install ffmpeg`
- Ollama + `ollama pull qwen3:8b` (설명 생성용)
- Python 패키지: `pip install transformers torch librosa soundfile numpy "psycopg[binary]" minio`
  (CLAP `laion/larger_clap_general`, BGE-M3는 임베딩 서버 `emb.java21.net` 사용)
- 환경변수/`local.env`: `FREESOUND_API_KEY`, `DB_URL`(jdbc:postgresql://host:port/db?currentSchema=schema),
  `DB_USERNAME`, `DB_PASSWORD`, `DB_SCHEMA`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`
- conversation-service V11(background_sounds + pgvector)이 배포돼 있어야 함

맥에는 GPU 워커 로그가 없으므로 `--gate`는 쓰지 않는다(쓰면 자동 무효라 그냥 진행). `describe_local`의
기본 `--keep-alive 0`은 생성마다 모델을 내려 느리니, 맥 전용이면 `--keep-alive 5m`으로 올려 쓰면 빠르다.

## 음량 기준 (중요)
`fetch_clips.py`가 EBU R128 **-23 LUFS / 최대 -2 dBTP**로 2패스 정규화한다(라이브 모드가 배경음 60%·효과음
100%로 섞어서 모든 파일 기준이 같아야 함). 기존 1~3차 MP3도 이 기준으로 **다시 받아 정규화해 재업로드**해야 한다.

## 1) 4차 신규 (약 1만 개)
```bash
cd sound_library
export FREESOUND_API_KEY=...   # 그 키
python freesound_select.py --total 13000 --kinds AMBIENCE,EFFECT \
  --exclude data/candidates-1.jsonl data/candidates-2.jsonl data/candidates-3.jsonl \
  --out data/candidates-4.jsonl
python fetch_clips.py  --candidates data/candidates-4.jsonl --out work/audio          # -23 LUFS 2패스
python embed_clips.py  --candidates data/candidates-4.jsonl --audio work/audio --out data/clap-4.jsonl.gz
python describe_local.py --candidates data/candidates-4.jsonl --clap data/clap-4.jsonl.gz \
  --out data/descriptions-4.jsonl --keep-alive 5m
python register_sounds.py --candidates data/candidates-4.jsonl --audio work/audio \
  --clap data/clap-4.jsonl.gz --descriptions data/descriptions-4.jsonl --env local.env   # --dry-run 먼저
```

## 2) 기존 1~3차 다시 쓰기 + 같은 음량으로 재업로드
설명은 "들리는 소리" 항목 틀로 다시 쓰고(describe_local), MP3는 -23 LUFS로 다시 받아 재업로드한다.
`register_sounds.py`는 다시 쓴 설명에서 빠진 소리를 자동으로 `active=false`로 끈다.
```bash
for N in 1 2 3; do
  python describe_local.py --candidates data/candidates-$N.jsonl --only data/descriptions-$N.jsonl \
    --clap data/clap-$N.jsonl.gz --out data/descriptions-$N.v2.jsonl --keep-alive 5m
  python fetch_clips.py --candidates data/candidates-$N.jsonl --out work/audio          # -23 LUFS로 다시
  python register_sounds.py --candidates data/candidates-$N.jsonl --audio work/audio \
    --clap data/clap-$N.jsonl.gz --descriptions data/descriptions-$N.v2.jsonl --env local.env --reupload
done
```
(`--reupload`를 줘야 음량이 바뀐 MP3가 MinIO에 덮어써진다. 설명만 바꿀 때는 생략.)

## 확인
- `/sounds`에 새 소리가 보이고, 설명이 "들리는 소리" 틀로 바뀌었는지
- `select count(*), count(*) filter (where active) from background_sounds;`
- MP3 음량: `ffmpeg -i <mp3> -af loudnorm=I=-23:TP=-2:LRA=11:print_format=json -f null -` 의 input_i가 -23 근처
