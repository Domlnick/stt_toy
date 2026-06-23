#!/usr/bin/env bash
#
# 단일 파일 end-to-end 처리:
#   입력 오디오(.m4a 등) -> WAV 변환 -> whisper.cpp STT -> txt/srt/json 저장
#
# 사용법:
#   scripts/transcribe_one.sh <입력오디오파일>
#
# 결과 위치:
#   output/wav/<이름>.wav
#   output/txt/<이름>.txt
#   output/srt/<이름>.srt
#   output/json/<이름>.json
#
# 종료 코드:
#   0  성공
#   1  사용법/사전조건 오류
#   2  변환 실패
#   3  STT 실패
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../config.sh
source "$SCRIPT_DIR/../config.sh"

# --- 인자 확인 ---
if [ "$#" -ne 1 ]; then
  echo "[transcribe_one] 사용법: $0 <입력오디오파일>" >&2
  exit 1
fi

IN_FILE="$1"

# --- 사전 점검 (CLAUDES.md 10.4) ---
if [ ! -f "$IN_FILE" ]; then
  echo "[transcribe_one] 입력 파일이 없다: $IN_FILE" >&2
  exit 1
fi

if [ ! -x "$WHISPER_BIN" ]; then
  echo "[transcribe_one] whisper-cli 를 찾을 수 없다: $WHISPER_BIN" >&2
  echo "  -> README 의 'whisper.cpp 빌드' 단계를 먼저 실행하라." >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "[transcribe_one] 모델 파일이 없다: $MODEL" >&2
  echo "  -> README 의 '모델 다운로드' 단계를 먼저 실행하라 (모델: $MODEL_NAME)." >&2
  exit 1
fi

# --- 출력 경로 준비 (원본 파일명 기준, 한글/공백 대응) ---
BASE="$(basename "$IN_FILE")"   # 예: "회의 녹음.m4a"
NAME="${BASE%.*}"               # 확장자 제거 -> "회의 녹음"

mkdir -p "$WAV_DIR" "$TXT_DIR" "$SRT_DIR" "$JSON_DIR"

WAV_FILE="$WAV_DIR/$NAME.wav"

# --- 1) WAV 변환 (convert_audio.sh 재사용, 중복 로직 없음) ---
if ! "$SCRIPT_DIR/convert_audio.sh" "$IN_FILE" "$WAV_FILE"; then
  echo "[transcribe_one] WAV 변환 단계 실패: $IN_FILE" >&2
  exit 2
fi

# --- 2) whisper STT 실행 ---
# -of 로 출력 접두사를 지정하면 whisper 가 <접두사>.txt/.srt/.json 을 만든다.
# 일단 txt 폴더 접두사로 생성한 뒤, srt/json 은 형식별 폴더로 이동한다.
OUT_PREFIX="$TXT_DIR/$NAME"

START_TS="$(date +%s)"
if ! "$WHISPER_BIN" \
    -m "$MODEL" \
    -f "$WAV_FILE" \
    -l "$LANG_OPT" \
    -pp \
    -otxt -osrt -oj \
    -of "$OUT_PREFIX"; then
  echo "[transcribe_one] STT 실패: $WAV_FILE" >&2
  exit 3
fi
END_TS="$(date +%s)"
ELAPSED=$(( END_TS - START_TS ))

# --- 3) 형식별 폴더로 정리 (txt 는 그대로 두고 srt/json 만 이동) ---
[ -f "$OUT_PREFIX.srt" ]  && mv -f "$OUT_PREFIX.srt"  "$SRT_DIR/$NAME.srt"
[ -f "$OUT_PREFIX.json" ] && mv -f "$OUT_PREFIX.json" "$JSON_DIR/$NAME.json"

# --- 처리 메타 로그 (CLAUDES.md 10.6) ---
FSIZE="$(wc -c < "$IN_FILE" | tr -d ' ')"
echo "[transcribe_one] 완료: $NAME | 모델=$MODEL_NAME | 크기=${FSIZE}B | ${ELAPSED}s"
echo "  txt : $TXT_DIR/$NAME.txt"
echo "  srt : $SRT_DIR/$NAME.srt"
echo "  json: $JSON_DIR/$NAME.json"
exit 0
