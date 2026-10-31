`docs`의 constitution을 상위 원칙으로 따른다: https://github.com/4cut-story/docs/blob/main/.specify/memory/constitution.md

# model-runtime Constitution

## 이 레포의 범위
모델 실행 전용 Python FastAPI 서버. `/transcribe`(WhisperX large-v3 + pyannote), `/images`(SDXL 만화풍 + IP-Adapter). 상태·DB·비즈니스 로직은 두지 않는다. `eval/`에 Phase 0 품질 검증(G1~G6) 스크립트와 점수를 둔다.

## 레포 원칙
- 상위 원칙과 충돌하면 상위 원칙을 따른다. 이 문서에는 이 레포에만 해당하는 규칙만 더한다.
- 다른 서비스와는 `common/contracts`의 태그된 계약대로만 통신한다.
- 테스트 없는 코드는 머지하지 않는다 (상위 원칙 V).
- 녹음과 전사 텍스트는 커밋하지 않는다.

**Version**: 0.1.0 | **Ratified**: 2026-10-31 | **Last Amended**: 2026-10-31
