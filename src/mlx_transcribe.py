#!/usr/bin/env python3
"""MLX Whisper 기반 단일 파일 트랜스크라이버 (빠르고 정확한 엔진).

Apple Silicon GPU 에서 large-v3-turbo 모델을 돌려 한국어를 인식한다.
- 입력 오디오(.m4a 등)를 직접 받는다 (mlx_whisper 내부에서 ffmpeg 디코드).
- 결과를 output/{txt,srt,json} 에 저장한다.
- 진행 상황을 stdout 으로 흘려보낸다 (웹 서버가 파싱해 progress bar 로 표시).

진행 출력 규약(서버가 파싱):
    DURATION <초>                      # 시작 시 1회
    PHASE stt                          # 음성 인식 시작
    [mm:ss.sss --> mm:ss.sss]  text    # mlx verbose 세그먼트 (실시간)
    PHASE diarize                      # 화자 구분 시작 (옵션 켰을 때만)
    DONE <txt경로>                     # 완료 시 1회

화자 구분(옵션): 환경변수로 켠다.
    STT_DIARIZE=1            # 화자 구분 활성화
    STT_NUM_SPEAKERS=2      # (선택) 화자 수 고정
    HF_TOKEN=hf_xxx         # pyannote 모델 다운로드용 토큰
켜면 출력 txt 가 화자/시각/동시발화 표시 형식으로 바뀐다(src/diarize.py).

주의: 첫 실행 시 모델(~1.5GB)을 Hugging Face 에서 자동 다운로드한다.
      이후에는 캐시를 사용한다.

사용법:
    python3 -u src/mlx_transcribe.py <오디오파일> [모델repo] [언어]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = PROJECT_ROOT / "output" / "txt"
SRT_DIR = PROJECT_ROOT / "output" / "srt"
JSON_DIR = PROJECT_ROOT / "output" / "json"
WAV_DIR = PROJECT_ROOT / "output" / "wav"

# 기본값. 정확도/속도 균형: large-v3-turbo q4 (M1 기준 ~10배속).
# 정확도를 더 원하면 mlx-community/whisper-large-v3-turbo (fp16).
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo-q4"
DEFAULT_LANG = "ko"

# 환청(노이즈·무음을 텍스트로 지어내기, 같은 말 반복) 억제 옵션.
#   condition_on_previous_text=False : 앞 (환청)텍스트에 안 휘둘려 반복 폭주 차단(가장 큼).
#   temperature 폴백               : 저신뢰 구간 재디코딩.
#   compression_ratio/logprob/no_speech : 반복·저신뢰·무음 구간 버림.
# 이 세트는 안정적(word_timestamps 불필요).
ANTI_HALLUC_BASE = {
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}

# word_timestamps 기본 OFF (속도 2배). 예전엔 무음 환청 억제
# (hallucination_silence_threshold)용으로 켰지만 DTW 정렬이 디코드를 2배로
# 느리게 만들고, 그 방어 역할은 이제 silero-VAD(src/vad.py) 가 대신한다:
#   - clip_timestamps 로 비발화 구간을 whisper 가 아예 건너뜀
#   - 발화구간 밖 세그먼트를 사후필터로 제거(diarize.filter_noise_segments)
# 정밀 단어 타임스탬프가 꼭 필요하면 STT_WORD_TS=1 로 되살린다(느려짐).
def word_ts_enabled() -> bool:
    return os.environ.get("STT_WORD_TS", "") in ("1", "true", "yes")


def vad_enabled() -> bool:
    return os.environ.get("STT_VAD", "1") not in ("0", "false", "no")


# STT_WORD_TS=1 일 때만 쓰는 무음 환청 억제(word_timestamps 동반, 느림).
ANTI_HALLUC_EXTRA = {
    "hallucination_silence_threshold": 2.0,
    "word_timestamps": True,
}


# ── Whisper 메타데이터 기반 노이즈/환청 사후필터 (VAD 독립, 타임스탬프 안전) ──
# 의미 토큰만 추출(CJK/라틴/숫자). NFC 정규화로 한글 자모 분리 방지.
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣぀-ヿ一-鿿]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text))


def repetition_score(text: str):
    """(최대 연속반복, 중복비율). '이거 이거 이거'->(3, ...)."""
    toks = _tokenize(text)
    n = len(toks)
    if n == 0:
        return 0, 0.0
    max_run = run = 1
    for i in range(1, n):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        if run > max_run:
            max_run = run
    return max_run, 1.0 - (len(set(toks)) / n)


def is_repetitive(text: str, max_run_thr: int = 5,
                  dup_ratio_thr: float = 0.6, min_tokens: int = 6) -> bool:
    """반복 환청 여부.

    - 연속반복(max_run)은 토큰수 무관 적용: 5연속이면 환청('고개를'×N).
      정상 강조('네 네 네 네'=4)는 보존되게 임계 5.
    - 중복비율(dup_ratio)은 짧은 정상 발화 오판 방지로 min_tokens 이상에서만.
    """
    toks = _tokenize(text)
    n = len(toks)
    if n == 0:
        return False
    max_run, dup_ratio = repetition_score(text)
    if max_run >= max_run_thr:
        return True
    return n >= min_tokens and dup_ratio >= dup_ratio_thr


def filter_low_confidence_segments(
    segments: list,
    *,
    no_speech_thr: float = 0.6,
    logprob_thr: float = -1.0,
    compression_ratio_thr: float = 2.4,
    min_text_len: int = 1,
    return_dropped: bool = False,
):
    """노이즈/환청 세그먼트 제거(입력 불변). 규칙 OR:

      R1 노이즈: no_speech_prob > thr AND avg_logprob < thr (둘 다 만족해야 — 보수적).
                 발화가 있는 동시발화는 no_speech 가 낮아 안 걸림.
      R2 압축률: compression_ratio > thr (반복 환청 시그니처).
      R3 반복:   is_repetitive(text) ('이거 이거…').
      R4 빈것:   토큰 0 또는 글자수 < min_text_len.
    메타 키 없는 세그먼트는 보수적으로 보존.
    """
    kept, dropped = [], []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        nsp, alp, cr = (seg.get("no_speech_prob"), seg.get("avg_logprob"),
                        seg.get("compression_ratio"))
        reason = None
        if len(_tokenize(text)) == 0 or len(text) < min_text_len:
            reason = "empty"
        elif (nsp is not None and alp is not None
              and nsp > no_speech_thr and alp < logprob_thr):
            reason = "noise"
        elif cr is not None and cr > compression_ratio_thr:
            reason = "compression"
        elif is_repetitive(text):
            reason = "repetition"
        (dropped if reason else kept).append(seg)
    return (kept, dropped) if return_dropped else kept


def collapse_repeated_segments(segments: list, max_tokens: int = 4) -> list:
    """연속된 동일-짧은-텍스트 세그먼트를 1개로 합친다(세그먼트 경계 넘는 반복 환청).

    per-세그먼트 필터는 '불편함' 단일 토막이 여러 세그먼트로 흩어지면 못 잡는다.
    바로 앞 통과 세그먼트와 토큰열이 같고 짧으면(<=max_tokens) 중복으로 보고 버린다.
    정상 발화가 동일 짧은 문장을 연속 반복하는 일은 드물어 안전(첫 1개는 보존).
    """
    kept = []
    prev_key = None
    for s in segments:
        toks = _tokenize(s.get("text") or "")
        key = tuple(toks)
        if toks and len(toks) <= max_tokens and key == prev_key:
            continue  # 직전과 동일 짧은 텍스트 → 중복 제거
        kept.append(s)
        prev_key = key
    return kept


def filter_disabled() -> bool:
    return os.environ.get("STT_FILTER", "1") in ("0", "false", "no")


def apply_meta_filter(result: dict) -> None:
    """Whisper 메타 기반 노이즈필터(B)를 result 에 적용(in place). 전멸 시 원본 유지."""
    if filter_disabled():
        return
    segs = result.get("segments") or []
    if not segs:
        return
    kept, dropped = filter_low_confidence_segments(segs, return_dropped=True)
    kept = collapse_repeated_segments(kept)   # 세그먼트 경계 넘는 반복 정리
    if not kept:
        print(f"FILTER(meta) 전부 노이즈 판정 → 원본 유지({len(segs)})",
              file=sys.stderr, flush=True)
        return
    if len(kept) != len(segs):
        print(f"FILTER(meta) {len(segs)} → {len(kept)} segments",
              file=sys.stderr, flush=True)
        result["segments"] = kept
        result["text"] = "".join(s.get("text", "") for s in kept)


def apply_vad_filter(result: dict, regions: list) -> None:
    """silero-VAD 발화구간(regions)과 안 겹치는 세그먼트를 노이즈로 제거(in place).

    word_timestamps 를 껐을 때 잃는 무음 환청 억제를 이 필터가 대신한다.
    diarize.filter_noise_segments 재사용(타임스탬프 불변). 전멸 시 원본 유지.
    diarize 경로는 pyannote turns 로 자체 필터하므로 여기선 건너뛴다.
    """
    if filter_disabled() or not regions:
        return
    import diarize
    segs = result.get("segments") or []
    if not segs:
        return
    filtered = diarize.filter_noise_segments(segs, regions)
    if not filtered:
        print(f"FILTER(vad) 전부 노이즈 판정 → 원본 유지({len(segs)})",
              file=sys.stderr, flush=True)
        return
    if len(filtered) != len(segs):
        print(f"FILTER(vad) {len(segs)} → {len(filtered)} segments",
              file=sys.stderr, flush=True)
        result["segments"] = filtered
        result["text"] = "".join(s.get("text", "") for s in filtered)


def transcribe_robust(mlx_whisper, audio: str, model: str, lang: str) -> dict:
    """환청 억제 옵션으로 트랜스크라이브.

    기본은 word_timestamps OFF(빠름, ~2배). STT_WORD_TS=1 이면 무음 환청 억제용
    (hallucination_silence_threshold)으로 켜되, numba 크래시 시 BASE 로 폴백한다.
    """
    if not word_ts_enabled():
        return mlx_whisper.transcribe(
            audio, path_or_hf_repo=model, language=lang, verbose=True,
            **ANTI_HALLUC_BASE,
        )
    try:
        return mlx_whisper.transcribe(
            audio, path_or_hf_repo=model, language=lang, verbose=True,
            **ANTI_HALLUC_BASE, **ANTI_HALLUC_EXTRA,
        )
    except Exception as exc:
        # word_timestamps(numba) 등 실패 → 안정 옵션만으로 재시도.
        print(f"(word_timestamps 비활성 폴백: {exc})", file=sys.stderr, flush=True)
        return mlx_whisper.transcribe(
            audio, path_or_hf_repo=model, language=lang, verbose=True,
            **ANTI_HALLUC_BASE,
        )


def probe_duration(audio_path: Path) -> float:
    """ffprobe 로 오디오 길이(초)를 구한다. 실패 시 0.0."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def to_wav(audio_path: Path) -> Path:
    """16kHz mono WAV 로 변환(화자 구분 시 whisper/pyannote 공용 입력)."""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = WAV_DIR / f"{audio_path.stem}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
        capture_output=True, check=True,
    )
    return wav_path


def srt_timestamp(seconds: float) -> str:
    """초 -> SRT 타임스탬프 HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(name: str, result: dict, txt_body: str | None = None) -> Path:
    """txt/srt/json 결과를 형식별 폴더에 저장하고 txt 경로를 반환한다.

    txt_body 가 주어지면(화자 구분 결과) txt 는 그 본문으로 대체한다.
    """
    for d in (TXT_DIR, SRT_DIR, JSON_DIR):
        d.mkdir(parents=True, exist_ok=True)

    txt_path = TXT_DIR / f"{name}.txt"
    if txt_body is not None:
        txt_path.write_text(txt_body, encoding="utf-8")
    else:
        txt_path.write_text(result.get("text", "").strip() + "\n", encoding="utf-8")

    segments = result.get("segments", []) or []
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        srt_lines.append(str(i))
        srt_lines.append(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}")
        srt_lines.append(seg.get("text", "").strip())
        srt_lines.append("")
    (SRT_DIR / f"{name}.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    (JSON_DIR / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return txt_path


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: python3 -u src/mlx_transcribe.py <오디오파일> [모델repo] [언어]",
              file=sys.stderr)
        return 1

    audio_path = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    lang = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LANG

    if not audio_path.is_file():
        print(f"입력 파일이 없다: {audio_path}", file=sys.stderr)
        return 1

    diarize_on = os.environ.get("STT_DIARIZE", "") in ("1", "true", "yes")

    # 길이 먼저 알려 진행률 계산이 가능하게 한다.
    dur = probe_duration(audio_path)
    print(f"DURATION {dur}", flush=True)

    try:
        import mlx_whisper
    except ImportError:
        print("mlx_whisper 가 설치되어 있지 않다. 'pip3 install --user mlx-whisper' 실행.",
              file=sys.stderr)
        return 1

    # 16kHz WAV 로 미리 변환한다. VAD(silero)와, 화자 구분 시 pyannote 가
    # 같은 입력을 쓴다. 변환 실패해도 원본으로 STT 는 진행(VAD 만 건너뜀).
    stt_input = audio_path
    try:
        stt_input = to_wav(audio_path)
    except Exception as exc:
        if diarize_on:  # 화자 구분은 WAV 필수 → 치명적
            print(f"WAV 변환 실패(ffmpeg 확인): {exc}", file=sys.stderr)
            return 3
        print(f"WAV 변환 실패, 원본으로 STT 진행(VAD 생략): {exc}",
              file=sys.stderr, flush=True)

    # silero-VAD 발화구간 검출. 노이즈/환청 사후필터 전용으로 쓴다.
    #
    # 주의: 예전엔 이 구간을 clip_timestamps 로 whisper 에 넘겨 비발화 구간을
    # 건너뛰게 했으나, 이 mlx_whisper build 는 clip 경계 seek 을 나쁘게 처리해
    # 환청 무한반복('나랑 나랑…')에 빠졌다(실측: 무클립 56s → clip 499s, 9배 악화).
    # → clip_timestamps 는 쓰지 않는다. 발화구간은 필터에만 재활용.
    regions: list = []
    if vad_enabled() and stt_input != audio_path:
        try:
            import vad as vad_mod
            regions, total_dur, speech_dur = vad_mod.speech_segments(stt_input)
            frac = speech_dur / total_dur if total_dur else 1.0
            print(f"VAD speech {speech_dur:.0f}/{total_dur:.0f}s "
                  f"({frac * 100:.0f}%), {len(regions)} regions",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"VAD 실패, 필터 생략: {exc}", file=sys.stderr, flush=True)
            regions = []

    print("PHASE stt", flush=True)
    try:
        # verbose=True -> 세그먼트를 stdout 으로 실시간 출력 (서버가 파싱).
        result = transcribe_robust(mlx_whisper, str(stt_input), model, lang)
    except Exception as exc:
        print(f"STT 실패: {exc}", file=sys.stderr)
        return 3

    # B: Whisper 메타 기반 노이즈/환청 필터(항상, ON/OFF 공통).
    apply_meta_filter(result)
    # A': silero-VAD 사후필터. diarize 경로는 pyannote turns 로 자체 필터하므로 제외.
    if not diarize_on:
        apply_vad_filter(result, regions)

    txt_body = None
    if diarize_on:
        try:
            txt_body = run_diarize_phase(stt_input, result)
        except Exception as exc:
            # 화자 구분만 실패하면 STT 본문은 살리되, 실패 사실을 명확히 알린다.
            # (조용히 일반 텍스트로 폴백하면 사용자가 화자 구분된 줄 착각함)
            print(f"DIARIZE_ERROR {exc}", flush=True)
            txt_body = None

    try:
        txt_path = write_outputs(audio_path.stem, result, txt_body=txt_body)
    except Exception as exc:
        print(f"결과 저장 실패: {exc}", file=sys.stderr)
        return 3

    print(f"DONE {txt_path}", flush=True)
    return 0


def run_diarize_phase(wav_path: Path, result: dict) -> str:
    """pyannote 로 화자 구분 후 (MM:SS)·화자·동시발화 표시 본문을 만든다."""
    print("PHASE diarize", flush=True)
    import diarize  # 같은 디렉터리(src/) 모듈

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    num = os.environ.get("STT_NUM_SPEAKERS", "")
    num_speakers = int(num) if num.isdigit() and int(num) > 0 else None

    turns, overlaps = diarize.run_diarization(
        str(wav_path), token=token, num_speakers=num_speakers,
    )
    segments = result.get("segments", []) or []

    # A: VAD 사후필터 — 음성구간(turns)과 거의 안 겹치는 환청 제거(turns 재활용, 비용 0).
    # B(메타) 다음 단계라 합집합 효과. 전멸 시 원본 유지. result 갱신으로 srt/json 도 일관.
    if not filter_disabled():
        speech_turns = [(t0, t1) for t0, t1, _ in turns]
        filtered = diarize.filter_noise_segments(segments, speech_turns)
        if filtered:
            if len(filtered) != len(segments):
                print(f"FILTER(vad) dropped {len(segments) - len(filtered)}/{len(segments)}",
                      file=sys.stderr, flush=True)
            segments = filtered
            result["segments"] = segments
            result["text"] = "".join(s.get("text", "") for s in segments)

    labeled = diarize.label_segments(segments, turns, overlaps)
    return diarize.build_diarized_text(labeled)


if __name__ == "__main__":
    raise SystemExit(main())
