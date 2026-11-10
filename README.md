# model-runtime

GPU PC에서 도는 **작업 워커**. 들어오는 포트 없이 conversation-service(`/internal/gpu-jobs`, 게이트웨이 경유)에 작업을 가져가러 가서 실행하고 결과를 보고한다. 상태·DB·비즈니스 로직은 두지 않는다. `eval/`에 Phase 0 품질 검증(G1~G6) 스크립트와 점수를 둔다.

| 작업 | 모델 묶음 | 실행 |
|---|---|---|
| TRANSCRIBE | WHISPER | WhisperX large-v3 + pyannote 화자 분리 (`HF_TOKEN` 없으면 화자 분리 생략, 최소 화자 수 `DIARIZE_MIN_SPEAKERS` 기본 2) |
| LLM | LLM | 같은 PC의 Ollama(`http://localhost:11434`)에 프롬프트·JSON 스키마를 그대로 넘김 |
| IMAGE | SDXL | SDXL + Lightning 8 step LoRA(guidance 0), 768px, CPU offload 끔. `SDXL_LORA`·`SDXL_STEPS`·`SDXL_GUIDANCE`·`SDXL_CPU_OFFLOAD`로 바꾼다. `MODEL_KEEP_IN_RAM=true`(기본)면 내릴 때 RAM에 옮겨 두고 다음 작업에서 GPU로 옮기기만 한다(1~2초). PNG를 사전 서명 PUT 주소로 올림. 컷에 `references`(캐릭터 참조)가 있으면 IP-Adapter Plus(`SDXL_IP_ADAPTER`)를 처음 한 번 올리고(약 4초, VRAM +2GB) 얼굴·머리 모양·옷을 유지해 그린다. 앞쪽 4 step은 강도 0.2 이하로 걸어 구도는 프롬프트를 따르고, 뒤쪽 step은 캐릭터 강도 + 0.2로 건다(`SDXL_IP_EARLY_SCALE`·`SDXL_IP_EARLY_STEPS`·`SDXL_IP_LATE_BOOST`, early steps 0이면 스케줄 없음). 그림 작업에 `ipSchedule`(`earlyScale`·`earlySteps`·`lateBoost`)이 오면 그 컷은 그 값을 쓴다(conversation-service 설정으로 GPU PC를 다시 띄우지 않고 고친다). 75토큰(CLIP 한도)을 넘는 프롬프트는 쉼표 뒤에서 75토큰 조각으로 나눠 인코딩해 이어 붙인다(`backends/long_prompt.py`). guidance > 1이면 negative도 같은 조각 수로 인코딩한다. 다른 SDXL 체크포인트(Animagine XL 4 등)는 `SDXL_MODEL`로 바꾸고, fp16 variant 파일이 없으면 `SDXL_VARIANT=`(빈 값), LoRA 없이 `SDXL_LORA=`·`SDXL_STEPS=28`·`SDXL_GUIDANCE=5`로 쓴다. embedding은 `embedUrl`로 받고, 없으면 `imageUrl`로 계산해 `embedUploadUrl`에 올린다. `region` left/right는 마스크로 자리를 나누고, center는 가운데 60%에만 건다(캐릭터 한 명과 참조 없는 사람이 함께 나오는 컷) |
| SPEECH | SDXL (내리지 않음) | 라이브 모드 대사 음성. `SPEECH_ENABLED=true`일 때만 가져간다(서버가 SPEECH를 모르면 가져가기가 거절된다). 같은 PC의 GPT-SoVITS v4 api_v2 서버(`TTS_URL`, 기본 `http://127.0.0.1:9880`, CUDAGraph 끔, VRAM 약 2.2GB)에 대사마다 목소리의 감정 참조 음성(`VOICES_DIR/<목소리 ID>/<감정>.wav`와 그 녹음의 글 `.txt`, 없으면 `neutral`)을 넘겨 읽고, ffmpeg로 MP3(64kbps)로 바꿔 줄마다 `uploadUrl`에 올린다. 결과는 `{lines: [{key, durationMs}]}`이고 못 읽은 줄은 뺀다. 서버에 닿지 않으면 `TTS_UNAVAILABLE`(재시도 가능) |

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
