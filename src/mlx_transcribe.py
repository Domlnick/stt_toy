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
import subprocess
import sys
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

    # 화자 구분 시 whisper/pyannote 가 같은 16kHz WAV 를 쓰도록 미리 변환.
    stt_input = audio_path
    if diarize_on:
        try:
            stt_input = to_wav(audio_path)
        except Exception as exc:
            print(f"WAV 변환 실패(ffmpeg 확인): {exc}", file=sys.stderr)
            return 3

    print("PHASE stt", flush=True)
    try:
        # verbose=True -> 세그먼트를 stdout 으로 실시간 출력 (서버가 파싱).
        result = mlx_whisper.transcribe(
            str(stt_input),
            path_or_hf_repo=model,
            language=lang,
            verbose=True,
        )
    except Exception as exc:
        print(f"STT 실패: {exc}", file=sys.stderr)
        return 3

    txt_body = None
    if diarize_on:
        try:
            txt_body = run_diarize_phase(stt_input, result)
        except Exception as exc:
            # 화자 구분만 실패하면 STT 본문은 살린다(완전 실패 아님).
            print(f"화자 구분 실패(텍스트만 저장): {exc}", file=sys.stderr)
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
    labeled = diarize.label_segments(segments, turns, overlaps)
    return diarize.build_diarized_text(labeled)


if __name__ == "__main__":
    raise SystemExit(main())
