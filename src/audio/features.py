"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Acoustic feature extraction and signal measurement module.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa


@dataclass
class PauseInterval:
    start_time: float
    end_time: float
    duration: float
    is_long_pause: bool = False


@dataclass
class AcousticFeatures:
    duration: float
    # Energy / Volume metrics
    mean_rms_db: float
    rms_std_db: float
    dynamic_range_db: float
    volume_energy_trend: List[Dict[str, float]] = field(default_factory=list)
    # Pausing & Activity
    total_pause_duration: float = 0.0
    pause_ratio: float = 0.0
    speech_activity_ratio: float = 0.0
    pause_count: int = 0
    pauses_per_minute: float = 0.0
    pauses: List[PauseInterval] = field(default_factory=list)
    # Pitch / Vocal Variation
    mean_f0_hz: float = 0.0
    std_f0_hz: float = 0.0
    f0_range_hz: float = 0.0
    voiced_ratio: float = 0.0
    f0_contour: List[Dict[str, float]] = field(default_factory=list)
    is_monotone: bool = False


def extract_energy_profile(
    y: np.ndarray,
    sr: int = 16000,
    frame_length_sec: float = 0.05,
    hop_length_sec: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Extract frame-by-frame RMS energy in dB.
    Returns: (times, rms_db_curve, mean_db, std_db, dynamic_range_db)
    """
    frame_len = int(frame_length_sec * sr)
    hop_len = int(hop_length_sec * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_len)

    # Convert to dB relative to full scale
    rms_safe = np.maximum(rms, 1e-6)
    rms_db = 20.0 * np.log10(rms_safe)

    # Calculate statistics on speech-active frames (within 35 dB of peak) to prevent silence frames skewing volume stats
    peak_db = float(np.max(rms_db))
    active_mask = rms_db > (peak_db - 35.0)
    if np.sum(active_mask) > 10:
        active_db = rms_db[active_mask]
        mean_db = float(np.mean(active_db))
        std_db = float(np.std(active_db))
        p95 = float(np.percentile(active_db, 95))
        p05 = float(np.percentile(active_db, 5))
        dynamic_range_db = float(max(0.0, p95 - p05))
    else:
        mean_db = float(np.mean(rms_db))
        std_db = float(np.std(rms_db))
        dynamic_range_db = 0.0

    return times, rms_db, mean_db, std_db, dynamic_range_db


def detect_pauses(
    y: np.ndarray,
    sr: int = 16000,
    min_pause_duration: float = 0.35,
    long_pause_threshold: float = 1.2,
    top_db: float = 28.0,
) -> Tuple[List[PauseInterval], float, float]:
    """
    Detect silent intervals and speech pauses using librosa.effects.split.
    Non-silent intervals are identified; the gaps between them represent pauses.
    """
    duration = float(len(y)) / float(sr)
    if duration <= 0:
        return [], 0.0, 0.0

    # Non-silent intervals
    non_silent = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)

    pauses: List[PauseInterval] = []
    last_end = 0.0

    for interval in non_silent:
        start_sec = float(interval[0]) / float(sr)
        end_sec = float(interval[1]) / float(sr)

        # Check gap between previous non-silent end and current start
        gap = start_sec - last_end
        if gap >= min_pause_duration:
            is_long = gap >= long_pause_threshold
            pauses.append(
                PauseInterval(
                    start_time=round(last_end, 3),
                    end_time=round(start_sec, 3),
                    duration=round(gap, 3),
                    is_long_pause=is_long,
                )
            )
        last_end = end_sec

    # Check trailing pause
    trailing_gap = duration - last_end
    if trailing_gap >= min_pause_duration:
        pauses.append(
            PauseInterval(
                start_time=round(last_end, 3),
                end_time=round(duration, 3),
                duration=round(trailing_gap, 3),
                is_long_pause=trailing_gap >= long_pause_threshold,
            )
        )

    total_pause = sum(p.duration for p in pauses)
    pause_ratio = total_pause / max(0.001, duration)

    return pauses, total_pause, pause_ratio


def extract_pitch_contour(
    y: np.ndarray,
    sr: int = 16000,
    fmin: float = 65.0,
    fmax: float = 350.0,
    hop_length_sec: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Extract fundamental frequency (F0) using librosa.yin.
    Returns: (times, f0_curve, mean_f0, std_f0, f0_range, voiced_ratio)
    """
    hop_length = int(hop_length_sec * sr)
    # Estimate F0 with YIN
    try:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    except Exception:
        # Fallback if YIN fails
        f0 = np.full(int(len(y) / hop_length) + 1, np.nan)

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Filter voiced frames: valid finite positive within pitch limits
    voiced = np.isfinite(f0) & (f0 >= fmin) & (f0 <= fmax)
    voiced_f0 = f0[voiced]

    voiced_ratio = float(np.sum(voiced)) / float(max(1, len(f0)))

    if len(voiced_f0) > 10:
        mean_f0 = float(np.mean(voiced_f0))
        std_f0 = float(np.std(voiced_f0))
        p90 = float(np.percentile(voiced_f0, 90))
        p10 = float(np.percentile(voiced_f0, 10))
        f0_range = float(max(0.0, p90 - p10))
    else:
        mean_f0 = 0.0
        std_f0 = 0.0
        f0_range = 0.0

    return times, f0, mean_f0, std_f0, f0_range, voiced_ratio


def extract_acoustic_features(
    y: np.ndarray,
    sr: int = 16000,
    min_pause_sec: float = 0.35,
    long_pause_threshold_sec: float = 1.2,
) -> AcousticFeatures:
    """
    Run full acoustic analysis on speech waveform.
    """
    duration = float(len(y)) / float(sr)
    if duration <= 0:
        return AcousticFeatures(
            duration=0.0,
            mean_rms_db=-60.0,
            rms_std_db=0.0,
            dynamic_range_db=0.0,
        )

    # 1. Energy Profile
    times_rms, rms_db, mean_rms_db, rms_std_db, dyn_range_db = extract_energy_profile(y, sr)
    # Downsample energy trend for lightweight plotting (approx 100 points)
    step = max(1, len(times_rms) // 100)
    energy_trend = [
        {"time": round(float(times_rms[i]), 2), "rms_db": round(float(rms_db[i]), 2)}
        for i in range(0, len(times_rms), step)
    ]

    # 2. Pause detection
    pauses, total_pause, pause_ratio = detect_pauses(
        y, sr, min_pause_duration=min_pause_sec, long_pause_threshold=long_pause_threshold_sec
    )
    speech_activity_ratio = max(0.0, 1.0 - pause_ratio)
    pauses_per_min = (len(pauses) / max(0.1, duration)) * 60.0

    # 3. Pitch Contour
    times_f0, f0, mean_f0, std_f0, f0_range, voiced_ratio = extract_pitch_contour(y, sr)
    step_f0 = max(1, len(times_f0) // 100)
    f0_contour = []
    for i in range(0, len(times_f0), step_f0):
        val = float(f0[i]) if (np.isfinite(f0[i]) and f0[i] > 0) else None
        f0_contour.append({"time": round(float(times_f0[i]), 2), "f0_hz": round(val, 1) if val else None})

    # Heuristic for monotone delivery: std_f0 < 12 Hz or f0_range < 25 Hz for voiced speech
    is_monotone = bool(voiced_ratio > 0.2 and (std_f0 < 12.0 or f0_range < 25.0))

    return AcousticFeatures(
        duration=round(duration, 3),
        mean_rms_db=round(mean_rms_db, 2),
        rms_std_db=round(rms_std_db, 2),
        dynamic_range_db=round(dyn_range_db, 2),
        volume_energy_trend=energy_trend,
        total_pause_duration=round(total_pause, 3),
        pause_ratio=round(pause_ratio, 3),
        speech_activity_ratio=round(speech_activity_ratio, 3),
        pause_count=len(pauses),
        pauses_per_minute=round(pauses_per_min, 2),
        pauses=pauses,
        mean_f0_hz=round(mean_f0, 2),
        std_f0_hz=round(std_f0, 2),
        f0_range_hz=round(f0_range, 2),
        voiced_ratio=round(voiced_ratio, 3),
        f0_contour=f0_contour,
        is_monotone=is_monotone,
    )
