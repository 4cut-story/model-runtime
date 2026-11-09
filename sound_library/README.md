# 배경음 라이브러리 수집

만화에 붙일 배경음(환경음·효과음·배경음악)을 공개 데이터셋에서 모아 설명을 달고 벡터로 만들어
conversation-service의 `background_sounds` 표와 MinIO에 넣는 스크립트다. 서비스 코드가 아니라 손으로 한 번씩 돌린다.

| 순서 | 스크립트 | 하는 일 |
| --- | --- | --- |
| 1 | `select_clips.py` | FSD50K·FMA 메타데이터에서 CC0·CC BY 클립만 골라 카테고리별로 고르게 후보를 뽑는다 |
| 2 | `fetch_clips.py` | 큰 zip을 통째로 받지 않고 HTTP Range로 클립만 꺼내 30초 이하 128k MP3로 만든다 |
| 3 | `embed_clips.py` | CLAP(laion/larger_clap_general)으로 소리 임베딩 512차원과 분위기·악기·템포 점수를 만든다 |
| 4 | `evidence.py` | 설명을 쓸 근거(원본 제목·태그·라벨, CLAP 점수)를 클립마다 한 줄로 보여 준다 |
| 5 | `merge_descriptions.py` | 직접 쓴 한국어 설명을 `descriptions.jsonl`로 바꾼다 |
| 6 | `register_sounds.py` | 설명을 BGE-M3로 임베딩하고 MP3를 MinIO에, 행을 DB에 넣는다 (다시 돌려도 된다) |

`describe_clips.py`는 Gemini 무료 등급으로 설명을 쓰던 스크립트다. 하루 20번 제한이 있어 1차 43개만 이걸로 썼고,
나머지는 Claude(유료 API가 아닌 대화)가 근거를 보고 직접 썼다.

## 라이선스

- 받는 것: CC0 1.0, CC BY 2.0·2.5·3.0·4.0. NC(비상업), ND(변경 금지), SA(동일 조건), Sampling+는 받지 않는다.
- 클립마다 원본 주소, 원래 제목, 작가, 라이선스와 라이선스 주소를 DB에 넣고 web `/sounds`에 보여 준다.
- CLAP 모델은 Apache-2.0이다.

## 1차 결과 (1,000개)

- 후보 1,000개: 환경음 550, 효과음 150, 음악 300
- 설명을 보고 뺀 것 74개(같은 소리 중복, 말소리가 중심인 클립, 카테고리와 맞지 않는 소리)
- 등록 대상 926개, 57개 카테고리

`data/`에 1차 후보(`candidates-1.jsonl`), 직접 쓴 설명 원문(`desc-claude-1.raw`), 합친 설명(`descriptions-1.jsonl`)을 둔다.
2차는 `select_clips.py --exclude data/candidates-1.jsonl`로 1차와 겹치지 않게 뽑는다.
CLAP 임베딩 파일(약 6MB)과 MP3는 레포에 넣지 않는다.
