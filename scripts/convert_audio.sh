#!/usr/bin/env bash
#
# 단일 책임: 오디오 파일 1개를 whisper 입력용 WAV로 변환한다.
# (.m4a 등) -> 16kHz mono 16bit PCM WAV
#
# 사용법:
#   scripts/convert_audio.sh <입력파일> <출력WAV경로>
#
# 종료 코드:
#   0  변환 성공
#   1  사용법/입력 오류
#   2  ffmpeg 변환 실패
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../config.sh
source "$SCRIPT_DIR/../config.sh"

# --- 인자 확인 ---
if [ "$#" -ne 2 ]; then
  echo "[convert_audio] 사용법: $0 <입력파일> <출력WAV경로>" >&2
  exit 1
fi

IN_FILE="$1"
OUT_FILE="$2"

# --- 사전 점검 ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[convert_audio] ffmpeg 가 설치되어 있지 않다. 'brew install ffmpeg' 먼저 실행." >&2
  exit 1
fi

if [ ! -f "$IN_FILE" ]; then
  echo "[convert_audio] 입력 파일이 없다: $IN_FILE" >&2
  exit 1
fi

# 출력 디렉터리 보장 (한글/공백 경로 대응 위해 인용)
OUT_DIR="$(dirname "$OUT_FILE")"
mkdir -p "$OUT_DIR"

# --- 변환 실행 ---
# -y : 기존 출력 덮어쓰기 (재처리 시 일관성)
if ffmpeg -y -i "$IN_FILE" \
    -ar "$AUDIO_RATE" \
    -ac "$AUDIO_CHANNELS" \
    -c:a "$AUDIO_CODEC" \
    "$OUT_FILE" </dev/null; then
  echo "[convert_audio] 변환 완료: $OUT_FILE"
  exit 0
else
  echo "[convert_audio] 변환 실패: $IN_FILE" >&2
  exit 2
fi
