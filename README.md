# model-runtime

모델 실행 전용 Python FastAPI 서버. `/transcribe`(WhisperX large-v3 + pyannote), `/images`(SDXL 만화풍 + IP-Adapter). 상태·DB·비즈니스 로직은 두지 않는다. `eval/`에 Phase 0 품질 검증(G1~G6) 스크립트와 점수를 둔다.

- 기술: Python, FastAPI, pytest + TestClient, Schemathesis. 모델은 테스트에서 가짜 객체로 대체한다.
- 설계와 스펙: [docs](https://github.com/4cut-story/docs)
