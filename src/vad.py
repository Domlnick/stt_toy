#!/usr/bin/env python3
"""silero-vad 기반 음성구간 검출 (경량, CPU, 초단위).

용도(mlx_transcribe.py): 노이즈/환청 사후필터.
    발화구간과 안 겹치는 세그먼트를 환청/노이즈로 보고 제거한다
    (diarize.filter_noise_segments 에 speech_turns 로 투입).

pyannote(무거움, 화자 구분용)와 달리 VAD 만 하므로 가볍고 빠르다.
diarize 를 안 켜도 환청 방어(예전 word_timestamps 의 hallucination_silence
역할)를 이 VAD 가 대체한다.

주의: 이 발화구간을 whisper 의 clip_timestamps 로 넘겨 비발화 구간을 건너뛰게
하는 실험은 폐기했다 — 이 mlx_whisper build 가 clip 경계에서 환청 무한반복에
빠져 오히려 9배 느려졌다(무클립 56s → clip 499s). 필터 용도로만 쓴다.

설치: python3 -m pip install --user silero-vad
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

_MODEL = None  # 모듈 전역 캐시(모델 1회 로드)


def _model():
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad
        _MODEL = load_silero_vad()
    return _MODEL


def speech_segments(
    wav_path: Path | str,
    *,
    sampling_rate: int = 16000,
    merge_gap: float = 0.6,
    pad: float = 0.2,
    min_speech_ms: int = 200,
    min_silence_ms: int = 300,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """16kHz mono WAV 에서 발화구간을 (start, end) 초 리스트로 반환.

    후처리:
      - pad     : 각 구간 앞뒤로 늘려 경계 단어 잘림 방지.
      - merge_gap: 간격이 이보다 짧은 이웃 구간을 하나로 병합
                   (clip_timestamps 조각수↓ → whisper 문맥/오버헤드 안정).

    반환: (regions, total_dur, speech_dur)
    """
    from silero_vad import get_speech_timestamps, read_audio

    wav = read_audio(str(wav_path), sampling_rate=sampling_rate)
    total_dur = len(wav) / sampling_rate
    ts = get_speech_timestamps(
        wav, _model(), sampling_rate=sampling_rate, return_seconds=True,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )

    regions: List[Tuple[float, float]] = []
    for t in ts:
        s = max(0.0, float(t["start"]) - pad)
        e = min(total_dur, float(t["end"]) + pad)
        if regions and s - regions[-1][1] < merge_gap:
            regions[-1] = (regions[-1][0], e)  # 이웃과 병합
        else:
            regions.append((s, e))

    speech_dur = sum(e - s for s, e in regions)
    return regions, total_dur, speech_dur
