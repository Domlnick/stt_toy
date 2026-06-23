#!/usr/bin/env bash
#
# 공통 설정 파일.
# 모든 스크립트는 이 파일을 source 하여 경로/모델/옵션을 공유한다.
# 하드코딩을 한 곳으로 모으기 위한 목적 (CLAUDES.md 10.1).
#
# 사용 예:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/../config.sh"

# 이 파일(config.sh) 기준으로 프로젝트 루트를 절대경로로 잡는다.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 입력/출력 경로
INPUT_DIR="$PROJECT_ROOT/input"
OUTPUT_DIR="$PROJECT_ROOT/output"
WAV_DIR="$OUTPUT_DIR/wav"
TXT_DIR="$OUTPUT_DIR/txt"
SRT_DIR="$OUTPUT_DIR/srt"
JSON_DIR="$OUTPUT_DIR/json"
MD_DIR="$OUTPUT_DIR/md"
LOG_DIR="$PROJECT_ROOT/logs"

# whisper.cpp 바이너리 및 모델 경로
WHISPER_DIR="$PROJECT_ROOT/vendor/whisper.cpp"
WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"

# 사용할 모델. 초기 기준은 small (CLAUDES.md 10.6).
# 정확도가 부족하면 medium / large-v3-turbo 로 변경.
MODEL_NAME="small"
MODEL="$WHISPER_DIR/models/ggml-${MODEL_NAME}.bin"

# 인식 언어. 한국어는 .en 모델이 아닌 다국어 모델 + ko 옵션 (CLAUDES.md 3장).
LANG_OPT="ko"

# 오디오 변환 파라미터 (whisper.cpp 권장: 16kHz mono 16bit PCM)
AUDIO_RATE="16000"
AUDIO_CHANNELS="1"
AUDIO_CODEC="pcm_s16le"
