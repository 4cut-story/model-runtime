# model-runtime

**GPU PC에서 도는 작업 워커. 받아 적기, 로컬 LLM, 그림 그리기, 대사 읽기를 맡아요.**

[← 4cut story 소개](https://github.com/4cut-story) · [AI 모델](https://github.com/4cut-story/.github/blob/main/docs/models.md) · [하드웨어](https://github.com/4cut-story/.github/blob/main/docs/hardware.md)

## 하는 일

| 작업 | 하는 일 | 모델 |
| --- | --- | --- |
| TRANSCRIBE | 녹음을 글로 받아 적고 화자를 나눠요 | WhisperX large-v3, pyannote |
| LLM | 클라우드 LLM이 안 될 때 콘티를 써요 | Ollama의 Qwen3 8B |
| IMAGE | 컷 그림을 그려요 (캐릭터 참조 그림 최대 4장) | FLUX.2 klein 4B, 실패하면 DreamShaper XL Lightning |
| SPEECH | 라이브 모드 대사를 가상 인물 목소리로 읽어요 | GPT-SoVITS v4 |

- 들어오는 포트를 열지 않아요. 워커가 HTTPS로 conversation-service에 일을 가져가러 가고(pull), 결과는 미리 서명된 주소로 올린 뒤 보고해요.
- 한 번에 작업 하나씩 처리하고, 남은 VRAM이 모자라면 `VRAM_BUSY`로 돌려보내 나중에 다시 받아요.
- 작업 중에는 30초마다 살아 있다고 알리고, 일이 없으면 2초부터 10초까지 늘려 가며 다시 물어요.
- 상태, DB, 업무 규칙은 두지 않아요. 모두 conversation-service에 있어요.

## 시스템 안에서의 위치

```
GPU PC (RTX 5060 Ti 16GB)                          클러스터
model-runtime ──HTTPS로 일 가져가기(pull)──> api-gateway ──> conversation-service
   ├── Ollama (Qwen3 8B)
   └── GPT-SoVITS (대사 읽기)
```

## 기술

Python 3.11+ · PyTorch (CUDA 12.8) · diffusers · WhisperX · Ollama · GPT-SoVITS

## 실행

GPU PC(Windows)에서:

```powershell
pip install -e ".[gpu]"
copy .env.example .env      # GPU_WORKER_API_URL, GPU_WORKER_TOKEN, HF_TOKEN 채우기
python -m model_runtime
```

연결만 확인하려면 `WORKER_BACKEND=fake`로 띄워요. 모델 없이 고정 결과를 보고해요. compose로 띄우는 방법은 [deploy](https://github.com/4cut-story/deploy)의 `external/gpu-stack/`에 있어요.

## 검사

```bash
pip install -e '.[dev]'
ruff check . && ruff format --check . && mypy model_runtime && pytest --cov=model_runtime --cov-fail-under=80
```

GPU가 있어야 도는 모델 코드는 CI 커버리지에서 빼고 GPU PC에서 확인해요.

## 자세한 문서

| 문서 | 내용 |
| --- | --- |
| [작업과 실행 설정](docs/jobs.md) | 작업 종류별 실행 방식과 환경 변수 |
| [AI 모델](https://github.com/4cut-story/.github/blob/main/docs/models.md) | 단계별 모델과 설정값, 그림 모델 비교 |
| [하드웨어](https://github.com/4cut-story/.github/blob/main/docs/hardware.md) | GPU PC 구성, 모델 세 묶음을 GPU 한 장에 나눠 올리는 방법 |
| [품질 검증](eval/README.md) | 품질 기준과 확인 스크립트 |
| [GPU 작업 계약](https://github.com/4cut-story/docs/blob/main/specs/001-conversation-4cut-comic/contracts/internal-api.md) | 일 가져가기·보고 API |
