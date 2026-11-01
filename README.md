# model-runtime

GPU PC에서 도는 **작업 워커**. 들어오는 포트 없이 conversation-service(`/internal/gpu-jobs`, 게이트웨이 경유)에 작업을 가져가러 가서 실행하고 결과를 보고한다. 상태·DB·비즈니스 로직은 두지 않는다. `eval/`에 Phase 0 품질 검증(G1~G6) 스크립트와 점수를 둔다.

| 작업 | 모델 묶음 | 실행 |
|---|---|---|
| TRANSCRIBE | WHISPER | WhisperX large-v3 + pyannote 화자 분리 (`HF_TOKEN` 없으면 화자 분리 생략) |
| LLM | LLM | 같은 PC의 Ollama(`http://localhost:11434`)에 프롬프트·JSON 스키마를 그대로 넘김 |
| IMAGE | SDXL | SDXL + Lightning 8 step LoRA(guidance 0), 768px, CPU offload 끔. `SDXL_LORA`·`SDXL_STEPS`·`SDXL_GUIDANCE`·`SDXL_CPU_OFFLOAD`로 바꾼다. PNG를 사전 서명 PUT 주소로 올림 |

- 한 번에 작업 하나, 모델 묶음 하나만 VRAM에 둔다. 다른 종류 작업을 받으면 들고 있던 묶음을 먼저 내린다 (Ollama는 `keep_alive: 0`).
- 모델을 올리기 전에 남은 VRAM이 모자라면 `VRAM_BUSY`(재시도 가능)로 보고한다.
- 작업 중 30초마다 heartbeat. 임대를 잃으면(409) 그 작업 결과는 버린다.
- 작업이 없으면 2초부터 10초까지 늘려 가며 다시 묻는다.

## GPU PC(Windows)에서 실행

```powershell
D:\4cut\gpu-runtime\.venv\Scripts\Activate.ps1   # CUDA 12.8 PyTorch가 설치된 venv
pip install -e "<이 레포 경로>[gpu]"
copy .env.example .env                            # GPU_WORKER_TOKEN, HF_TOKEN 채우기
python -m model_runtime
```

`.env`는 실행 위치(현재 폴더)에서 읽는다. 연결만 확인하려면 `WORKER_BACKEND=fake`로 돌린다 (모델 없이 고정 결과를 보고).

## 개발

```bash
pip install -e '.[dev]'
ruff check . && ruff format --check . && mypy model_runtime && pytest --cov=model_runtime --cov-fail-under=80
```

GPU가 있어야 도는 `backends/whisper.py`, `backends/sdxl.py`는 CI 커버리지에서 빼고 GPU PC에서 `eval/`로 검증한다.

- 설계와 스펙: [docs](https://github.com/4cut-story/docs) (`specs/001-conversation-4cut-comic/contracts/internal-api.md`)
