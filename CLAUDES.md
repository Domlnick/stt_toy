# 아이폰 녹음 파일 기반 로컬 STT 프로젝트 구축 가이드

## 1. 프로젝트 개요

이 문서는 **아이폰으로 녹음한 음성 파일을 macOS 환경에서 로컬로 텍스트 추출**하기 위한 프로젝트 구축 기준을 정리한 문서이다.

기본 음성 인식 엔진은 [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp)를 사용한다.

이 프로젝트의 목표는 다음과 같다.

- 아이폰 음성 메모 파일(`.m4a`)을 로컬 Mac 환경에서 처리한다.
- 인터넷 연결 없이 음성 파일을 텍스트로 변환할 수 있도록 한다.
- 한국어 음성 인식을 지원한다.
- 결과물을 `.txt`, `.srt`, `.json`, `.md` 등으로 저장할 수 있도록 확장 가능하게 구성한다.
- 추후 화자분리, 요약, 회의록 자동 정리 기능을 확장할 수 있도록 유지보수 가능한 구조로 설계한다.

---

## 2. 기본 기술 스택

| 구분 | 기술 | 용도 |
|---|---|---|
| STT 엔진 | whisper.cpp | 로컬 음성 인식 |
| 오디오 변환 | FFmpeg | `.m4a` → `.wav` 변환 |
| 스크립트 | Shell Script | 단일 파일/폴더 일괄 처리 |
| 후처리 | Python | 텍스트 정리, Markdown 변환, 결과 가공 |
| 실행 환경 | macOS | Mac 로컬 실행 |
| 모델 | Whisper 다국어 모델 | 한국어 음성 인식 |
| 버전 관리 | Git | 소스 관리 및 변경 이력 추적 |

---

## 3. whisper.cpp 선택 기준

`whisper.cpp`는 OpenAI Whisper 모델을 C/C++ 기반으로 실행할 수 있게 만든 로컬 STT 프로젝트이다.

macOS, 특히 Apple Silicon 환경에서 사용하기 적합하며 다음 장점이 있다.

- 로컬 실행 가능
- 인터넷 없이 음성 인식 가능
- Apple Silicon 최적화 지원
- 비교적 가벼운 의존성
- CLI 기반 자동화가 쉬움
- 프로젝트 내부에 쉽게 포함 가능

다만 다음 한계가 있다.

- 기본적으로 화자분리 기능은 약함
- 아이폰 녹음 파일 `.m4a`는 바로 처리하기보다 FFmpeg 변환이 필요함
- 한국어 처리를 위해 `.en` 모델이 아닌 다국어 모델을 사용해야 함
- 장시간 녹음 파일은 처리 시간과 메모리 사용량을 고려해야 함

---

## 4. 1차 MVP 범위

초기 버전에서는 기능을 과하게 확장하지 않고 다음 범위만 구현한다.

### 4.1 필수 기능

- `input/` 폴더에 있는 아이폰 녹음 파일을 읽는다.
- `.m4a` 파일을 `.wav`로 변환한다.
- 변환된 `.wav` 파일을 `whisper.cpp`로 텍스트 추출한다.
- 결과를 `output/` 폴더에 저장한다.
- 최소 출력 형식은 `.txt`로 한다.
- 필요 시 `.srt`, `.json` 출력도 지원한다.

### 4.2 제외 기능

초기 버전에서는 다음 기능을 구현하지 않는다.

- GUI 앱
- 실시간 받아쓰기
- 화자분리
- 자동 요약
- 웹 서버
- 클라우드 업로드
- 데이터베이스 저장
- 사용자 계정/로그인

위 기능들은 기본 텍스트 추출 흐름이 안정화된 후 별도 단계에서 검토한다.

---

## 5. 추천 프로젝트 구조

```text
local-stt/
├── README.md
├── .gitignore
├── scripts/
│   ├── transcribe_one.sh
│   ├── transcribe_batch.sh
│   └── convert_audio.sh
├── src/
│   └── postprocess.py
├── input/
│   └── .gitkeep
├── output/
│   ├── wav/
│   │   └── .gitkeep
│   ├── txt/
│   │   └── .gitkeep
│   ├── srt/
│   │   └── .gitkeep
│   ├── json/
│   │   └── .gitkeep
│   └── md/
│       └── .gitkeep
├── models/
│   └── .gitkeep
└── vendor/
    └── whisper.cpp/
```

### 5.1 폴더 역할

| 폴더 | 역할 |
|---|---|
| `scripts/` | 실행용 Shell Script 보관 |
| `src/` | Python 후처리 코드 보관 |
| `input/` | 원본 아이폰 녹음 파일 위치 |
| `output/wav/` | 변환된 WAV 파일 저장 |
| `output/txt/` | 텍스트 추출 결과 저장 |
| `output/srt/` | 자막 파일 저장 |
| `output/json/` | JSON 결과 저장 |
| `output/md/` | 회의록/정리본 Markdown 저장 |
| `models/` | Whisper 모델 파일 저장 |
| `vendor/whisper.cpp/` | whisper.cpp 소스 또는 빌드 파일 위치 |

---

## 6. 설치 구성

### 6.1 필수 설치 항목

```bash
brew install git cmake ffmpeg
```

### 6.2 whisper.cpp 설치

```bash
git clone https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp
cd vendor/whisper.cpp
cmake -B build
cmake --build build -j --config Release
```

### 6.3 모델 다운로드

한국어 인식을 위해 `.en` 모델은 사용하지 않는다.

추천 모델은 다음과 같다.

| 용도 | 모델 | 설명 |
|---|---|---|
| 빠른 테스트 | `base` | 설치 확인용 |
| 일반 사용 | `small` | 속도와 정확도 균형 |
| 업무/회의록 | `medium` | 한국어 정확도 개선 |
| 품질 우선 | `large-v3-turbo` | 품질과 속도 균형 |

초기에는 `small` 모델로 시작한다.

```bash
cd vendor/whisper.cpp
sh ./models/download-ggml-model.sh small
```

---

## 7. 기본 처리 흐름

```text
아이폰 녹음 파일(.m4a)
        ↓
FFmpeg로 WAV 변환
        ↓
whisper.cpp 실행
        ↓
TXT/SRT/JSON 결과 생성
        ↓
필요 시 Markdown 회의록으로 후처리
```

---

## 8. 예시 명령어

### 8.1 오디오 변환

```bash
ffmpeg -y \
  -i input/recording.m4a \
  -ar 16000 \
  -ac 1 \
  -c:a pcm_s16le \
  output/wav/recording.wav
```

### 8.2 텍스트 추출

```bash
./vendor/whisper.cpp/build/bin/whisper-cli \
  -m ./vendor/whisper.cpp/models/ggml-small.bin \
  -f ./output/wav/recording.wav \
  -l ko \
  -otxt \
  -osrt \
  -oj
```

---

## 9. 추가 확장 검토 사항

### 9.1 화자분리

`whisper.cpp` 단독으로는 일반적인 회의 녹음의 화자분리를 안정적으로 처리하기 어렵다.

화자분리가 필요할 경우 다음 대안을 별도 검토한다.

| 대안 | 설명 |
|---|---|
| WhisperX | 단어 단위 타임스탬프, 화자분리 지원 |
| pyannote.audio | 화자분리 전문 라이브러리 |
| 수동 후처리 | 초기 MVP에서 가장 단순한 방식 |
| LLM 후처리 | 발화 내용을 기반으로 화자 추정 가능하지만 정확성 한계 있음 |

초기 버전에서는 화자분리를 제외하고, 텍스트 추출 안정화 후 별도 기능으로 검토한다.

### 9.2 회의록 후처리

텍스트 추출 이후 다음 형태로 가공할 수 있다.

- 전체 대화 원문
- 핵심 요약
- 안건
- 이슈
- 결정사항
- TODO
- 담당자
- 일정

단, 후처리는 STT 품질이 안정화된 후 진행한다.

---

## 10. 개발 제약 조건

### 10.1 코드 작성 원칙

- 코드는 유지보수하기 쉽게 작성한다.
- 함수와 스크립트는 하나의 책임만 가지도록 구성한다.
- 중복 코드는 최소화한다.
- 명령어, 경로, 모델명 등은 하드코딩을 최소화한다.
- 반복적으로 사용되는 값은 변수 또는 설정 파일로 분리한다.
- Shell Script는 실행 흐름을 명확히 알 수 있도록 단계별로 작성한다.
- Python 코드는 함수 단위로 분리하고, 예외 처리를 포함한다.
- 파일명, 경로, 확장자 처리는 공백과 한글 파일명도 고려한다.
- 장시간 음성 파일 처리 시 실패한 파일과 성공한 파일을 구분할 수 있게 로그를 남긴다.

### 10.2 파일 수정 제한

- 파일 수정 시 필요한 부분만 탐색하고 수정한다.
- 요청받은 범위 밖의 파일은 사용자 승인 없이 수정하지 않는다.
- 기존 파일 전체를 재작성하지 않는다.
- 기존 구조를 변경해야 할 경우 먼저 변경 이유와 영향 범위를 정리한다.
- 자동 포맷팅 도구를 사용할 경우, 의도하지 않은 대량 변경이 발생하지 않도록 범위를 제한한다.
- 수정 대상 파일과 수정 이유를 작업 기록에 남긴다.

### 10.3 의존성 추가 제한

- 불필요한 라이브러리 설치를 지양한다.
- 표준 기능으로 처리 가능한 작업은 외부 라이브러리를 추가하지 않는다.
- 라이브러리를 추가해야 할 경우 다음 항목을 검토한다.
  - 실제 필요한 기능인지
  - 유지보수가 활발한지
  - 라이선스 문제가 없는지
  - macOS에서 안정적으로 동작하는지
  - Apple Silicon에서 문제가 없는지
  - 기존 도구로 대체 가능한지
- 한 번만 사용할 기능을 위해 무거운 프레임워크를 추가하지 않는다.
- Python 패키지 추가 시 `requirements.txt` 또는 `pyproject.toml`에 명확히 기록한다.
- Node.js, 웹 서버, Electron 등은 MVP 단계에서 도입하지 않는다.

### 10.4 실행 안정성

- 스크립트 실행 전 필수 경로가 존재하는지 확인한다.
- 입력 파일이 없을 경우 명확한 에러 메시지를 출력한다.
- 지원하지 않는 확장자는 건너뛰고 로그를 남긴다.
- 변환 실패, STT 실패, 출력 저장 실패를 구분한다.
- 동일 파일을 재처리할 때 기존 결과를 덮어쓸지 여부를 명확히 처리한다.
- 원본 음성 파일은 절대 자동 삭제하지 않는다.
- 처리 중 생성되는 임시 파일은 별도 폴더에 저장하고 정리 기준을 둔다.

### 10.5 보안 및 개인정보 보호

- 음성 파일은 민감정보를 포함할 수 있으므로 외부 서버로 업로드하지 않는다.
- 기본 설계는 로컬 처리만 허용한다.
- 클라우드 API 사용은 기본 범위에서 제외한다.
- 로그에 원문 전체를 불필요하게 남기지 않는다.
- 개인 음성 파일, 변환 결과, 모델 파일은 Git에 커밋하지 않는다.
- 공유용 결과물 생성 시 개인정보 마스킹 가능성을 고려한다.

### 10.6 성능 및 리소스 관리

- 초기 모델은 `small`을 기준으로 한다.
- 정확도가 부족할 때만 `medium` 또는 `large-v3-turbo`로 변경한다.
- 장시간 파일은 구간 분할 처리 가능성을 고려한다.
- 처리 시간, 파일 크기, 모델명을 로그에 기록한다.
- 동일 입력에 대해 중복 변환이 발생하지 않도록 한다.
- 대용량 파일 처리 시 진행 상태를 확인할 수 있게 한다.

### 10.7 출력 파일 관리

- 출력 파일명은 원본 파일명을 기준으로 생성한다.
- 원본 파일명이 한글이거나 공백을 포함해도 처리 가능해야 한다.
- 출력 파일은 형식별 폴더에 저장한다.
- 결과물 덮어쓰기 정책을 명확히 한다.
- 실패한 파일 목록을 별도 로그로 남긴다.
- 출력 결과에 생성 시간, 사용 모델, 언어 옵션 등을 기록하는 방식을 검토한다.

---

## 11. Git 관리 규칙

### 11.1 기본 원칙

- Git에는 소스 코드, 설정 파일, 문서만 커밋한다.
- 원본 음성 파일, 변환된 WAV 파일, STT 결과물, 모델 파일은 기본적으로 커밋하지 않는다.
- 대용량 파일은 Git에 포함하지 않는다.
- 실험용 임시 파일은 커밋하지 않는다.
- 변경 전후 차이를 확인한 뒤 커밋한다.
- 커밋은 기능 단위로 작게 나눈다.
- 여러 목적의 변경을 하나의 커밋에 섞지 않는다.

### 11.2 .gitignore 권장 설정

```gitignore
# macOS
.DS_Store
.AppleDouble
.LSOverride

# Editor / IDE
.vscode/
.idea/
*.swp
*.swo

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.env
.env.*

# Logs
logs/
*.log

# Input audio files
input/*
!input/.gitkeep

# Output files
output/wav/*
output/txt/*
output/srt/*
output/json/*
output/md/*
!output/wav/.gitkeep
!output/txt/.gitkeep
!output/srt/.gitkeep
!output/json/.gitkeep
!output/md/.gitkeep

# Whisper models
models/*
!models/.gitkeep
vendor/whisper.cpp/models/*.bin
vendor/whisper.cpp/models/*.mlmodelc

# Build artifacts
vendor/whisper.cpp/build/
build/
dist/

# Temporary files
tmp/
temp/
*.tmp

# Audio conversion artifacts
*.wav
*.mp3
*.m4a
*.aac
*.flac

# STT result artifacts
*.srt
*.vtt
*.json
*.txt
```

주의: 위 설정은 프로젝트 루트 기준이다. 샘플 음성 파일이나 샘플 결과물을 문서화 목적으로 커밋해야 하는 경우에는 별도 `samples/` 폴더를 만들고 명시적으로 예외 처리한다.

### 11.3 커밋 메시지 규칙

커밋 메시지는 다음 형식을 권장한다.

```text
<type>: <summary>
```

예시:

```text
init: create local stt project structure
feat: add single audio transcription script
feat: add batch transcription script
fix: handle korean filename in audio conversion
chore: update gitignore for whisper outputs
docs: add project setup guide
refactor: split audio conversion logic
```

권장 타입:

| 타입 | 용도 |
|---|---|
| `init` | 초기 구성 |
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `chore` | 설정, 빌드, 기타 작업 |
| `refactor` | 기능 변화 없는 구조 개선 |
| `test` | 테스트 추가/수정 |

### 11.4 브랜치 전략

초기 개인 프로젝트 기준으로는 단순한 브랜치 전략을 사용한다.

```text
main
└── feature/audio-convert
└── feature/transcribe-one
└── feature/batch-transcribe
└── feature/postprocess-md
```

규칙:

- `main` 브랜치는 항상 실행 가능한 상태를 유지한다.
- 새 기능은 `feature/*` 브랜치에서 작업한다.
- 실험성 작업은 `experiment/*` 브랜치에서 작업한다.
- 작업 완료 후 diff를 확인하고 병합한다.
- 불필요한 브랜치는 병합 후 삭제한다.

### 11.5 Git 작업 전 확인 사항

작업 전:

```bash
git status
```

작업 후:

```bash
git diff
```

커밋 전:

```bash
git status
git diff --staged
```

실수 방지 원칙:

- `git add .`는 지양한다.
- 필요한 파일만 명시적으로 추가한다.
- 대용량 파일이 스테이징되지 않았는지 확인한다.
- 모델 파일, 음성 파일, 결과 파일이 포함되지 않았는지 확인한다.

예시:

```bash
git add README.md scripts/transcribe_one.sh src/postprocess.py .gitignore
git commit -m "feat: add single file transcription flow"
```

---

## 12. 작업 단계 계획

### 12.1 1단계: 프로젝트 초기 구성

- 프로젝트 폴더 생성
- 기본 디렉터리 구성
- `.gitignore` 작성
- README 초안 작성
- `whisper.cpp` clone 및 빌드
- `small` 모델 다운로드

### 12.2 2단계: 단일 파일 처리

- `.m4a` 입력 경로 받기
- FFmpeg로 WAV 변환
- `whisper-cli` 실행
- `.txt` 결과 저장
- 에러 메시지 정리

### 12.3 3단계: 일괄 처리

- `input/` 폴더 전체 탐색
- `.m4a` 파일만 처리
- 이미 처리된 파일 스킵
- 실패 목록 기록
- 처리 결과 요약 출력

### 12.4 4단계: 출력 형식 확장

- `.srt` 출력
- `.json` 출력
- Markdown 회의록 형태 변환
- 파일별 메타데이터 기록

### 12.5 5단계: 품질 개선

- 모델별 결과 비교
- `small`, `medium`, `large-v3-turbo` 비교
- 긴 음성 파일 처리 방식 검토
- 무음 구간 제거 또는 분할 처리 검토

### 12.6 6단계: 확장 기능 검토

- 화자분리
- 회의록 요약
- 드래그앤드롭 UI
- macOS 앱 패키징
- 로컬 LLM 연동

---

## 13. 완료 기준

1차 MVP 완료 기준은 다음과 같다.

- `input/` 폴더에 `.m4a` 파일을 넣으면 텍스트 추출이 가능하다.
- 원본 파일은 삭제되거나 변경되지 않는다.
- 결과 파일은 `output/txt/`에 저장된다.
- 한글 파일명과 공백 포함 파일명을 처리할 수 있다.
- 실패 시 원인을 확인할 수 있는 메시지가 출력된다.
- 모델 파일과 음성 파일은 Git에 포함되지 않는다.
- 새로 설치한 의존성은 문서에 기록되어 있다.
- `main` 브랜치는 항상 실행 가능한 상태를 유지한다.

---

## 14. 핵심 원칙 요약

- 작게 만들고 점진적으로 확장한다.
- 로컬 처리 원칙을 유지한다.
- 원본 음성 파일은 절대 훼손하지 않는다.
- 불필요한 라이브러리는 추가하지 않는다.
- 필요한 파일만 수정한다.
- Git에는 코드와 문서만 남긴다.
- 모델, 음성, 결과물은 Git에서 제외한다.
- 유지보수 가능한 구조를 우선한다.
- 기능 추가보다 안정적인 처리 흐름을 먼저 완성한다.
