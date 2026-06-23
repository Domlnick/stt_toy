#!/usr/bin/env python3
"""화자 분리(speaker diarization) — pyannote.audio 래퍼 + 병합/포맷 함수.

Whisper 가 만든 "무슨 말"(세그먼트)에 pyannote 가 만든 "누가 언제"(화자 구간)를
시간축으로 겹쳐 매칭한다. 결과:
  - 같은 화자가 이어 말하면 한 덩어리(엔터 X), 화자 바뀌면 엔터.
  - 각 줄 맨 앞에 음성 파일 기준 (MM:SS) 표시.
  - 2명 이상 동시 발화 구간은 별도 표시·강조.

주의:
  - pyannote 모델은 gated. Hugging Face 토큰 필요(`HF_TOKEN`), 약관 동의 2개:
      pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0
    첫 다운로드만 인터넷, 이후 로컬 처리.
  - 동시 발화는 "겹쳤다"는 표시만 가능. 묻힌 말의 복원은 안 된다(음원 분리 아님).

병합/포맷 함수(assign·label·build)는 pyannote 없이도 import 가능하게
모듈 최상단에서 pyannote 를 import 하지 않는다(테스트/문법검사 용이).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

OVERLAP_MARK = "⚠️【동시발화】"
UNKNOWN_SPEAKER = "화자?"

# 세그먼트가 "동시 발화"로 표시될 임계값.
#   겹친 시간이 이 초 이상이거나, 세그먼트 길이의 이 비율 이상이면 표시.
OVERLAP_MIN_SEC = 0.3
OVERLAP_MIN_FRAC = 0.2


def mmss(seconds: float) -> str:
    """초 -> (MM:SS). 60분 넘어가면 분이 60 이상으로 계속 커진다(73:20 식)."""
    total = int(round(seconds))
    if total < 0:
        total = 0
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def _overlap_dur(a0: float, a1: float, b0: float, b1: float) -> float:
    """두 구간 [a0,a1], [b0,b1] 의 겹친 길이(초). 안 겹치면 0."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(
    seg_start: float, seg_end: float,
    turns: List[Tuple[float, float, str]],
) -> Optional[str]:
    """세그먼트와 가장 많이 겹치는 화자 라벨을 고른다. 없으면 None."""
    best_label = None
    best_dur = 0.0
    for t0, t1, label in turns:
        d = _overlap_dur(seg_start, seg_end, t0, t1)
        if d > best_dur:
            best_dur = d
            best_label = label
    return best_label


def seg_is_overlapped(
    seg_start: float, seg_end: float,
    overlaps: List[Tuple[float, float]],
) -> bool:
    """세그먼트가 동시 발화 구간과 충분히 겹치면 True."""
    seg_len = max(seg_end - seg_start, 1e-6)
    total = 0.0
    for o0, o1 in overlaps:
        total += _overlap_dur(seg_start, seg_end, o0, o1)
    return total >= OVERLAP_MIN_SEC or (total / seg_len) >= OVERLAP_MIN_FRAC


def label_segments(
    segments: List[dict],
    turns: List[Tuple[float, float, str]],
    overlaps: List[Tuple[float, float]],
) -> List[dict]:
    """각 Whisper 세그먼트에 화자(화자1/화자2…)와 동시발화 여부를 단다.

    화자 라벨(SPEAKER_00 등)은 등장 순서대로 화자1, 화자2… 로 다시 매긴다.
    반환: [{start,end,text,speaker,overlap}, ...] (start 기준 정렬).
    """
    segs = sorted(segments, key=lambda s: s.get("start", 0.0))
    order: List[str] = []
    out: List[dict] = []
    for s in segs:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        raw = assign_speaker(start, end, turns)
        if raw is not None and raw not in order:
            order.append(raw)
        speaker = f"화자{order.index(raw) + 1}" if raw is not None else UNKNOWN_SPEAKER
        out.append({
            "start": start,
            "end": end,
            "text": (s.get("text") or "").strip(),
            "speaker": speaker,
            "overlap": seg_is_overlapped(start, end, overlaps),
        })
    return out


def build_diarized_text(labeled: List[dict]) -> str:
    """라벨된 세그먼트 -> 화자 전환/동시발화 변화마다 줄바꿈한 본문.

    한 줄 형식:  (MM:SS) [⚠️【동시발화】 ] [화자N] 텍스트…
      - 맨 앞 (MM:SS) = 그 줄 첫 세그먼트의 시작 시각.
      - 같은 (화자, 동시발화여부) 가 이어지면 같은 줄에 이어 붙임(엔터 X).
    """
    lines: List[str] = []
    cur_key = None
    cur_start = 0.0
    cur_speaker = ""
    cur_overlap = False
    parts: List[str] = []

    def flush():
        if not parts:
            return
        mark = (OVERLAP_MARK + " ") if cur_overlap else ""
        body = " ".join(p for p in parts if p).strip()
        lines.append(f"({mmss(cur_start)}) {mark}[{cur_speaker}] {body}")

    for seg in labeled:
        key = (seg["speaker"], seg["overlap"])
        if key != cur_key:
            flush()
            cur_key = key
            cur_start = seg["start"]
            cur_speaker = seg["speaker"]
            cur_overlap = seg["overlap"]
            parts = [seg["text"]]
        else:
            parts.append(seg["text"])
    flush()
    return "\n".join(lines) + ("\n" if lines else "")


def run_diarization(
    audio_path: str,
    token: str,
    num_speakers: Optional[int] = None,
    prefer_mps: bool = True,
) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float]]]:
    """pyannote 로 화자 구간과 동시 발화 구간을 구한다.

    반환: (turns, overlaps)
      turns    = [(start, end, "SPEAKER_00"), ...]
      overlaps = [(start, end), ...]  (2명 이상 동시)
    pyannote/torch 는 이 함수 안에서만 import 한다(없으면 명확한 에러).
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio/torch 미설치. 'pip3 install --user pyannote.audio' 실행."
        ) from exc

    if not token:
        raise RuntimeError(
            "Hugging Face 토큰 없음. 환경변수 HF_TOKEN 설정 필요"
            "(pyannote/speaker-diarization-3.1 약관 동의 후 발급)."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token,
    )
    if pipeline is None:
        raise RuntimeError(
            "pyannote 파이프라인 로드 실패. 토큰/약관 동의 확인."
        )

    if prefer_mps:
        try:
            if torch.backends.mps.is_available():
                pipeline.to(torch.device("mps"))
        except Exception:
            pass  # mps 불가 시 CPU 폴백

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(audio_path, **kwargs)

    turns: List[Tuple[float, float, str]] = [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    overlaps: List[Tuple[float, float]] = []
    try:
        overlap_tl = diarization.get_overlap()  # pyannote.core Timeline
        overlaps = [(float(seg.start), float(seg.end)) for seg in overlap_tl]
    except Exception:
        overlaps = _overlaps_from_turns(turns)  # 폴백: 직접 계산

    return turns, overlaps


def _overlaps_from_turns(
    turns: List[Tuple[float, float, str]],
) -> List[Tuple[float, float]]:
    """화자 구간들에서 2명 이상 겹친 시간대를 직접 계산(폴백)."""
    out: List[Tuple[float, float]] = []
    n = len(turns)
    for i in range(n):
        a0, a1, _ = turns[i]
        for j in range(i + 1, n):
            b0, b1, _ = turns[j]
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return out
