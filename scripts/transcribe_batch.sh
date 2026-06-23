#!/usr/bin/env bash
#
# 일괄 처리:
#   input/ 폴더의 모든 .m4a 파일을 순회하며 transcribe_one.sh 로 처리한다.
#   - .m4a 외 확장자는 건너뛰고 로그를 남긴다.
#   - 이미 결과(txt)가 있는 파일은 건너뛴다 (중복 처리 방지).
#   - 실패한 파일은 logs/failed.log 에 기록한다.
#
# 사용법:
#   scripts/transcribe_batch.sh
#
# 종료 코드:
#   0  전부 처리(또는 스킵) 성공
#   1  사전조건 오류
#   4  하나 이상 실패
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../config.sh
source "$SCRIPT_DIR/../config.sh"

# --- 사전 점검 ---
if [ ! -d "$INPUT_DIR" ]; then
  echo "[batch] 입력 폴더가 없다: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$TXT_DIR"
FAILED_LOG="$LOG_DIR/failed.log"
: > "$FAILED_LOG"   # 실행마다 실패 로그 초기화

# --- 집계 변수 ---
total=0
done_cnt=0
skip_done=0
skip_ext=0
fail_cnt=0

# input/ 바로 아래 파일만 순회. 한글/공백 파일명 안전하게 처리.
# (.gitkeep 같은 숨김파일 제외 위해 glob 사용)
shopt -s nullglob
for f in "$INPUT_DIR"/*; do
  [ -f "$f" ] || continue
  base="$(basename "$f")"

  # 확장자 검사: .m4a 만 처리, 그 외는 스킵 + 로그
  ext="${base##*.}"
  ext_lower="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"
  if [ "$ext_lower" != "m4a" ]; then
    echo "[batch] 스킵(미지원 확장자): $base"
    skip_ext=$(( skip_ext + 1 ))
    continue
  fi

  total=$(( total + 1 ))
  name="${base%.*}"

  # 이미 처리됨? -> 스킵
  if [ -f "$TXT_DIR/$name.txt" ]; then
    echo "[batch] 스킵(이미 처리됨): $base"
    skip_done=$(( skip_done + 1 ))
    continue
  fi

  echo "[batch] 처리 시작: $base"
  # transcribe_one.sh 에 위임 (set -e 영향 피하려 if 로 감쌈)
  if "$SCRIPT_DIR/transcribe_one.sh" "$f"; then
    done_cnt=$(( done_cnt + 1 ))
  else
    echo "[batch] 실패: $base" >&2
    echo "$f" >> "$FAILED_LOG"
    fail_cnt=$(( fail_cnt + 1 ))
  fi
done
shopt -u nullglob

# --- 요약 출력 (CLAUDES.md 3단계) ---
echo "------------------------------------------"
echo "[batch] 요약"
echo "  대상(.m4a)   : $total"
echo "  처리 성공    : $done_cnt"
echo "  스킵(완료)   : $skip_done"
echo "  스킵(확장자) : $skip_ext"
echo "  실패         : $fail_cnt"
if [ "$fail_cnt" -gt 0 ]; then
  echo "  실패 목록    : $FAILED_LOG"
  exit 4
fi
exit 0
