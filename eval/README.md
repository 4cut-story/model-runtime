# eval — Phase 0 품질 검증 (G1~G6)

| 파일 | 게이트 | 작업 |
|---|---|---|
| `stt_eval.py` | G3 전사·화자 구분, 시간 | 001 T003 |
| `scene_eval.py` | G1 장면 선택 (8B/14B 비교) | 001 T004 |
| `script_eval.py` | G2 콘티·검수 | 001 T005 |
| `image_eval.py` | G4 4컷 그림 일관성 | 001 T006 |
| `comic_eval.py` | 4컷 전체 반복 실행(전사→추출→콘티→그림→합성), 단계별 시간 | 프롬프트 개선 |
| `gallery.py` | `comic_eval.py` 결과를 HTML 갤러리로 묶기 | 프롬프트 개선 |
| `report.md` | G5 1시간 전체 시간, G6 모델 라이선스 | 001 T007 |

여기에는 스크립트와 점수만 둔다. 샘플 녹음과 전사는 MinIO에만 둔다.

## 4컷 반복 실행 (GPU PC)

`prompt_bundle.json`은 conversation-service에서 내보낸다(서버와 같은 프롬프트):
`PROMPT_BUNDLE_OUT=../model-runtime/eval/prompt_bundle.json ./mvnw test -Dtest=PromptBundleExport`

```
# 워커를 잠시 멈추고(VRAM), model-runtime venv에서
python eval/comic_eval.py --audio D:/4cut/samples/01.m4a --runs 3 --out C:/claude-work/4cut/eval-out
# 같은 전사로 다시: --transcript <out>/<시각>/transcript.json
python eval/gallery.py C:/claude-work/4cut/eval-out/<시각> --out C:/claude-work/4cut/eval-out/gallery.html
```
SDXL 설정은 워커와 같은 환경 변수(`SDXL_LORA`, `SDXL_STEPS`, `SDXL_GUIDANCE`, `SDXL_CPU_OFFLOAD`)를 읽는다.
