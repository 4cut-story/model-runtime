# model-runtime

모델 실행 전용 Python FastAPI 서버. `/transcribe`(WhisperX large-v3 + pyannote), `/images`(SDXL 만화풍 + IP-Adapter). 상태·DB·비즈니스 로직은 두지 않는다. `eval/`에 Phase 0 품질 검증(G1~G6) 스크립트와 점수를 둔다.

## 기준
- 상위 원칙: `docs`의 `.specify/memory/constitution.md`
- 아키텍처: `docs`의 `docs/architecture.md` (v3.0)
- **계약 기준 버전: common `v0.1.0` (태그 예정)** — 계약은 이 버전에 고정해 쓰고, 올릴 때 이 줄을 함께 고친다.
- 포트: 8001 (`common/service-catalog.yaml`)

## 기술
Python, FastAPI, pytest + TestClient, Schemathesis. 모델은 테스트에서 가짜 객체로 대체한다.

## 규칙
- 서비스 간 통신은 `common/contracts`의 OpenAPI대로만 한다. API를 바꾸려면 `common` PR → 태그가 먼저다.
- 환경별 차이는 프로필과 환경 변수로만 둔다. 비밀 값은 커밋하지 않는다.
- 녹음과 전사 텍스트는 커밋하지 않는다.
- 테스트 이름은 한국어 `@DisplayName`(또는 테스트 설명)으로 쓰고 FR 번호와 연결한다.
