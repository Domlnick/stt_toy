# 로컬 STT — 아이폰 녹음 → 한국어 텍스트

아이폰으로 녹음한 음성(`.m4a`)을 macOS(Apple Silicon)에서 **로컬·오프라인**으로 받아쓰는 도구.
외부 서버로 업로드하지 않고, 모든 처리를 내 컴퓨터에서만 수행한다.

- 🎙 **입력**: 아이폰 음성 메모 `.m4a`
- 📝 **출력**: 텍스트(`.txt`) · 자막(`.srt`) · `.json` · 회의록 Markdown
- 🌐 **두 가지 사용법**: 브라우저 업로드(웹 UI) 또는 CLI 스크립트
- 🗣 **화자 구분(선택)**: 화자가 바뀔 때 줄바꿈 + 시각 표시 + 동시 발화 강조

> 설계 기준 문서: [`CLAUDES.md`](./CLAUDES.md)

---

## 목차

- [엔진](#엔진)
- [처리 흐름](#처리-흐름)
- [요구 사항](#요구-사항)
- [설치](#설치)
- [사용법](#사용법)
- [화자 구분 (선택)](#화자-구분-선택)
- [성능](#성능)
- [출력 위치](#출력-위치)
- [프로젝트 구조](#프로젝트-구조)
- [설정](#설정)
- [제약·주의](#제약주의)

---

## 엔진

| 엔진 | 모델 | 용도 | 특징 |
|---|---|---|---|
| **MLX Whisper** | `large-v3-turbo-q4` | 웹 UI 기본 | Apple GPU, 빠르고 정확 (1시간 ≈ 6분, M1) |
| **whisper.cpp** | `small` | CLI 스크립트 기본 | 외부 의존성 최소, 오프라인 폴백 |

---

## 처리 흐름

```text
웹 UI:  input/*.m4a  →  MLX Whisper(turbo, GPU)          →  output/{txt,srt,json}  →  postprocess(Markdown)
CLI :   input/*.m4a  →  ffmpeg(WAV)  →  whisper.cpp(small) →  output/{txt,srt,json}  →  postprocess(Markdown)
```

---

## 요구 사항

- macOS, **Apple Silicon**(M1 이상) — MLX 엔진 전용
- [Homebrew](https://brew.sh)
- `git`, `cmake`, `ffmpeg`
- Python 3.9+
- (선택) Hugging Face 계정 — 화자 구분 사용 시

---

## 설치

### 1. 사전 패키지

```bash
brew install git cmake ffmpeg
```

### 2. MLX Whisper (웹 UI 엔진)

```bash
python3 -m pip install --user mlx-whisper
```

첫 실행 시 모델(`large-v3-turbo-q4`, ~1.5GB)을 Hugging Face 에서 자동 다운로드한다(1회, 이후 오프라인).

### 3. whisper.cpp (CLI 엔진) — 선택

> 이 저장소는 whisper.cpp 소스/빌드 결과를 포함하지 않는다.

```bash
git clone https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp
cd vendor/whisper.cpp
cmake -B build
cmake --build build -j --config Release
sh ./models/download-ggml-model.sh small      # 한국어는 다국어 모델(.en 아님)
cd ../..
```

빌드 결과: `vendor/whisper.cpp/build/bin/whisper-cli`
모델을 바꾸면 [`config.sh`](./config.sh) 의 `MODEL_NAME` 도 함께 변경한다.

| 용도 | 모델 |
|---|---|
| 빠른 테스트 | `base` |
| 일반 사용 | `small` (기본) |
| 업무/회의록 | `medium` |
| 품질 우선 | `large-v3-turbo` |

---

## 사용법

녹음 파일을 `input/` 에 넣는다 (한글·공백 파일명 가능).

### 웹 UI (권장 — 브라우저 업로드, MLX 엔진)

```bash
python3 src/server.py            # http://127.0.0.1:8000
python3 src/server.py 9000       # 포트 지정
```

- 브라우저로 `http://127.0.0.1:8000` 접속 → `.m4a` 끌어다 놓기(또는 클릭 선택).
- 처리 중 **진행률 바**가 실제 % 로 표시된다.
- 결과는 화면 + `output/{txt,srt,json}` 에 저장.
- 로컬 전용(127.0.0.1) — 외부로 업로드하지 않는다.
- ⚠️ 서버를 띄우는 터미널 `PATH` 에 `ffmpeg`(보통 `/opt/homebrew/bin`)가 있어야 한다.

### CLI — 단일 파일

```bash
# MLX 엔진
python3 -u src/mlx_transcribe.py "input/회의 녹음.m4a"

# whisper.cpp 엔진
bash scripts/transcribe_one.sh "input/회의 녹음.m4a"
```

### CLI — 폴더 일괄 처리 (whisper.cpp)

```bash
bash scripts/transcribe_batch.sh
```

- `.m4a` 만 처리, 그 외 확장자는 스킵.
- 이미 결과(`output/txt/<이름>.txt`)가 있으면 스킵.
- 실패 파일은 `logs/failed.log` 에 기록.

### Markdown 회의록 변환 (후처리)

```bash
python3 src/postprocess.py                          # output/txt 전체
python3 src/postprocess.py "output/txt/회의 녹음.txt"   # 특정 파일
```

→ `output/md/<이름>.md` 에 원문 + 회의록 템플릿 생성. (표준 라이브러리만 사용)

---

## 화자 구분 (선택)

여러 사람이 말하는 녹음에서 **화자가 바뀔 때 줄바꿈**하고, 각 줄 앞에 `(MM:SS)` 시각을,
2명 이상 **동시 발화** 구간엔 `⚠️【동시발화】` 표시를 붙인다.

```text
(00:00) [화자1] 안녕하세요 회의 시작하겠습니다
(00:03) [화자2] 네 자료 공유합니다
(00:06) ⚠️【동시발화】 [화자1] 그건 아닌데
(00:07) [화자1] 다음 안건으로
```

### 설치 (무거움 — torch 동반)

```bash
python3 -m pip install --user "pyannote.audio>=3.1"
```

### Hugging Face 토큰 (모델이 gated)

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 약관 동의
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 약관 동의
3. [settings/tokens](https://huggingface.co/settings/tokens) 에서 토큰 발급

```bash
export HF_TOKEN=hf_xxxxx
python3 src/server.py
```

- 웹 UI 의 **"화자 구분"** 체크박스를 켜고 업로드하면 적용된다.
- CLI 직접:
  ```bash
  STT_DIARIZE=1 HF_TOKEN=hf_xxx python3 -u src/mlx_transcribe.py "input/회의.m4a"
  # 화자 수 고정: STT_NUM_SPEAKERS=2
  ```

### 한계

- 처리 시간 2~4배 (1시간 ≈ 15~25분, M1).
- 화자 라벨은 `화자1/화자2`(등장 순서) — 실제 이름 매핑은 수동.
- 동시 발화는 "겹쳤다" 표시만 가능 — 묻힌 말의 복원은 안 된다(음원 분리 아님).
- 잡음·원거리·말 겹침이 많으면 화자 경계가 부정확해진다.
- 미설치/토큰 없으면 화자 구분만 건너뛰고 일반 텍스트로 저장.

---

## 성능

M1, 119초 한국어 샘플 실측:

| 엔진/모델 | 속도 | 1시간 추정 | 정확도 |
|---|---|---|---|
| whisper.cpp small | ~5배속 | ~12분 | 낮음 |
| MLX turbo (fp16) | 7배속 | ~8분 | 우수 |
| **MLX turbo-q4 (기본)** | **10배속** | **~6분** | **우수** |
| MLX turbo-q4 + 화자 구분 | — | ~15~25분 | 녹음 품질 의존 |

> 실제 녹음(잡음·다화자)은 다소 느려질 수 있다.

---

## 출력 위치

| 경로 | 내용 |
|---|---|
| `output/wav/` | 변환된 WAV |
| `output/txt/` | 텍스트 (화자 구분 시 화자·시각·동시발화 표시) |
| `output/srt/` | 자막 |
| `output/json/` | JSON 결과 |
| `output/md/` | Markdown 회의록 |
| `logs/` | 실패/처리 로그 |

---

## 프로젝트 구조

```text
01.stt/
├── README.md            # 이 문서
├── CLAUDES.md           # 설계 기준 스펙
├── config.sh            # 공통 경로·모델·옵션 (모든 스크립트가 source)
├── requirements.txt
├── scripts/
│   ├── convert_audio.sh     # .m4a → 16kHz mono WAV (ffmpeg)
│   ├── transcribe_one.sh    # 단일 파일: 변환 → whisper.cpp → txt/srt/json
│   └── transcribe_batch.sh  # input/ 일괄 + 스킵 + 실패 로그
├── src/
│   ├── server.py            # 웹 UI (표준 라이브러리, 127.0.0.1)
│   ├── mlx_transcribe.py    # MLX Whisper 엔진 (+ 화자 구분 연동)
│   ├── diarize.py           # 화자 구분(pyannote) 래퍼 + 병합/포맷
│   └── postprocess.py       # txt → Markdown 회의록
├── input/               # 원본 녹음 (Git 제외)
├── output/{wav,txt,srt,json,md}/
├── models/              # 모델 (Git 제외)
└── vendor/              # whisper.cpp clone 위치
```

---

## 설정

공통 경로·모델·옵션은 [`config.sh`](./config.sh) 한 곳에서 관리한다. 모든 셸 스크립트가 이 파일을 `source` 한다.
MLX 모델 변경은 `src/mlx_transcribe.py` 의 `DEFAULT_MODEL`(정확도 우선이면 fp16 `mlx-community/whisper-large-v3-turbo`).

---

## 제약·주의

- 원본 음성 파일은 삭제·변경하지 않는다.
- 음성·WAV·결과·모델 파일은 Git에 커밋하지 않는다 ([`.gitignore`](./.gitignore)).
- 모든 처리는 로컬에서만 수행한다 — 외부 업로드 없음.
- 한국어는 `.en` 모델이 아닌 **다국어 모델**을 사용한다.
