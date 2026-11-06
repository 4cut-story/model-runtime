# 캐릭터 시드 (운영자 도구)

테마(컨셉)별 캐릭터 참조 그림을 GPU PC에서 그리고 conversation-service 캐릭터 DB(REQ-018)에 등록한다.
워커 패키지(`model_runtime`)와 따로 돈다. 워커는 여전히 DB에 직접 닿지 않는다.

1. `catalog.yaml`에 테마별 캐릭터(이름, look, 선택: description, scale)를 적는다. look은 conversation-service `Concept.lookRule` 형식을 따른다.
2. 그리기 (운영 워커가 VRAM을 쓰므로 쉬는 틈에 워커를 멈추고 돌린다. 32명 x 후보 4장 약 5분):
   `PYTHONPATH=<model-runtime> python draw.py --catalog catalog.yaml --out out`
   → `out/<THEME>/<slug>/seed*.png`, 후보별 IP-Adapter embedding, `out/sheet_<THEME>.png`, `out/manifest.json`
3. 시트를 보고 `picks.yaml`에 `이름: 시드`를 적는다.
4. 등록: `python register.py --catalog catalog.yaml --out out --picks picks.yaml --env <DB/MinIO/Gemini .env>` (`--dry-run` 먼저)
   - `/internal/characters`가 게이트웨이 밖으로 열려 있지 않아 DB와 MinIO에 직접 쓴다. 키 규칙은 `CharacterKeys`, 설명 임베딩은 `GeminiEmbedder`와 같다.
   - 같은 이름이 있으면 public_id를 두고 바꾼다. 다시 돌려도 된다.
   - 설명 임베딩은 `description_embedding` 컬럼(pgvector)이 있을 때만 넣는다. 없으면 콘티 인물 이름이 같을 때만 쓰인다.

필요한 패키지: 그리기는 워커와 같은 `[gpu]` 환경, 등록은 `psycopg[binary] minio pyyaml`.
