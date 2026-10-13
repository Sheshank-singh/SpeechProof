"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Deterministic Rubric Scoring Engine.
Calculates transparent 0-100 scores across 5 speech delivery dimensions:
Pace, Fluency, Pausing, Vocal Variation, Volume Consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class CategoryScore:
    name: str
    score: float  # 0 to 100
    raw_metric_name: str
    raw_value: float
    unit: str
    target_range: str
    weight: float
    explanation: str
    warning: Optional[str] = None


@dataclass
class RubricConfig:
    # Target Speaking Rate (WPM)
    target_wpm_min: float = 125.0
    target_wpm_max: float = 165.0
    wpm_tolerance: float = 35.0  # beyond min-tol or max+tol drops score rapidly

    # Fluency (Fillers per minute)
    ideal_max_fillers_per_min: float = 1.0
    poor_fillers_per_min: float = 6.0

    # Pausing (Pause ratio & Long pauses)
    ideal_pause_ratio_min: float = 0.12
    ideal_pause_ratio_max: float = 0.28
    long_pause_threshold_sec: float = 1.4
    max_acceptable_long_pauses_per_min: float = 1.0

    # Vocal Variation (Pitch F0 Std in Hz)
    ideal_f0_std_min: float = 20.0
    ideal_f0_std_max: float = 55.0
    monotone_f0_std: float = 16.0

    # Volume Consistency (Active Speech RMS Std dB)
    ideal_rms_std_min: float = 2.0
    ideal_rms_std_max: float = 7.5
    excessive_rms_std: float = 9.5

    # Weights for overall score
    weight_pace: float = 0.20
    weight_fluency: float = 0.20
    weight_pausing: float = 0.20
    weight_vocal_variation: float = 0.20
    weight_volume_consistency: float = 0.20


@dataclass
class OverallScoreResult:
    overall_score: float  # 0 to 100
    categories: Dict[str, CategoryScore]
    config: RubricConfig
    data_quality_warnings: List[str] = field(default_factory=list)


def score_pace(wpm: float, config: RubricConfig, word_count: int) -> CategoryScore:
    """Calculate Pace score (0-100) based on Speaking Rate in WPM."""
    if word_count < 5:
        return CategoryScore(
            name="Pace",
            score=50.0,
            raw_metric_name="Speaking Rate",
            raw_value=round(wpm, 1),
            unit="WPM",
            target_range=f"{config.target_wpm_min:.0f}-{config.target_wpm_max:.0f} WPM",
            weight=config.weight_pace,
            explanation="Insufficient spoken words to evaluate speech tempo accurately.",
            warning="Low word count (<5 words). Pace score confidence is low.",
        )

    if config.target_wpm_min <= wpm <= config.target_wpm_max:
        # Ideal range
        score = 100.0
        exp = f"Speaking rate of {wpm:.1f} WPM is within the optimal delivery window ({config.target_wpm_min:.0f}–{config.target_wpm_max:.0f} WPM)."
    elif wpm < config.target_wpm_min:
        # Sluggish
        diff = config.target_wpm_min - wpm
        score = max(15.0, 100.0 - (diff / max(1.0, config.wpm_tolerance)) * 60.0)
        exp = f"Speaking rate of {wpm:.1f} WPM is slower than ideal. Audience engagement may drop."
    else:
        # Rushed
        diff = wpm - config.target_wpm_max
        score = max(15.0, 100.0 - (diff / max(1.0, config.wpm_tolerance)) * 60.0)
        exp = f"Speaking rate of {wpm:.1f} WPM is rushed. Audience comprehension may be impacted."

    return CategoryScore(
        name="Pace",
        score=round(score, 1),
        raw_metric_name="Speaking Rate",
        raw_value=round(wpm, 1),
        unit="WPM",
        target_range=f"{config.target_wpm_min:.0f}-{config.target_wpm_max:.0f} WPM",
        weight=config.weight_pace,
        explanation=exp,
    )


def score_fluency(fillers_per_min: float, filler_count: int, duration_sec: float, config: RubricConfig) -> CategoryScore:
    """Calculate Fluency score (0-100) based on filler word density."""
    if duration_sec < 2.0:
        return CategoryScore(
            name="Fluency",
            score=50.0,
            raw_metric_name="Filler Rate",
            raw_value=round(fillers_per_min, 1),
            unit="fillers/min",
            target_range=f"< {config.ideal_max_fillers_per_min:.1f}/min",
            weight=config.weight_fluency,
            explanation="Recording duration too short to reliably evaluate fluency.",
            warning="Clip duration is under 2 seconds.",
        )

    if fillers_per_min <= config.ideal_max_fillers_per_min:
        score = 100.0 - (fillers_per_min / max(0.1, config.ideal_max_fillers_per_min)) * 5.0
        exp = f"Excellent fluency: only {filler_count} filler words detected ({fillers_per_min:.1f}/min)."
    elif fillers_per_min >= config.poor_fillers_per_min:
        excess = fillers_per_min - config.poor_fillers_per_min
        score = max(10.0, 40.0 - (excess * 5.0))
        exp = f"High filler density: {filler_count} fillers detected ({fillers_per_min:.1f}/min), significantly impacting clarity."
    else:
        # Linear degradation between ideal and poor
        ratio = (fillers_per_min - config.ideal_max_fillers_per_min) / (
            config.poor_fillers_per_min - config.ideal_max_fillers_per_min
        )
        score = 95.0 - (ratio * 55.0)
        exp = f"Moderate filler presence: {filler_count} fillers detected ({fillers_per_min:.1f}/min)."

    return CategoryScore(
        name="Fluency",
        score=round(score, 1),
        raw_metric_name="Filler Rate",
        raw_value=round(fillers_per_min, 1),
        unit="fillers/min",
        target_range=f"≤ {config.ideal_max_fillers_per_min:.1f}/min",
        weight=config.weight_fluency,
        explanation=exp,
    )


def score_pausing(
    pause_ratio: float,
    long_pause_count: int,
    duration_sec: float,
    config: RubricConfig,
) -> CategoryScore:
    """Calculate Pausing score (0-100) from total pause ratio and frequency of awkward silences."""
    duration_min = max(0.1, duration_sec / 60.0)
    long_pauses_per_min = long_pause_count / duration_min

    # Pause ratio component (50 pts)
    if config.ideal_pause_ratio_min <= pause_ratio <= config.ideal_pause_ratio_max:
        ratio_score = 50.0
    elif pause_ratio < config.ideal_pause_ratio_min:
        diff = config.ideal_pause_ratio_min - pause_ratio
        ratio_score = max(10.0, 50.0 - (diff / 0.15) * 35.0)
    else:
        diff = pause_ratio - config.ideal_pause_ratio_max
        ratio_score = max(10.0, 50.0 - (diff / 0.25) * 35.0)

    # Long pause penalty component (50 pts)
    if long_pauses_per_min <= config.max_acceptable_long_pauses_per_min:
        penalty_score = 50.0 - (long_pauses_per_min * 10.0)
    else:
        excess = long_pauses_per_min - config.max_acceptable_long_pauses_per_min
        penalty_score = max(5.0, 40.0 - (excess * 15.0))

    total = ratio_score + penalty_score
    exp = (
        f"Pause ratio is {pause_ratio*100:.1f}%. "
        f"Detected {long_pause_count} prolonged pause(s) exceeding {config.long_pause_threshold_sec}s."
    )

    return CategoryScore(
        name="Pausing",
        score=round(total, 1),
        raw_metric_name="Pause Ratio",
        raw_value=round(pause_ratio * 100.0, 1),
        unit="%",
        target_range=f"{config.ideal_pause_ratio_min*100:.0f}%–{config.ideal_pause_ratio_max*100:.0f}%",
        weight=config.weight_pausing,
        explanation=exp,
    )


def score_vocal_variation(
    std_f0: float,
    f0_range: float,
    voiced_ratio: float,
    config: RubricConfig,
) -> CategoryScore:
    """Calculate Vocal Variation score (0-100) from pitch F0 dynamics."""
    if voiced_ratio < 0.15:
        return CategoryScore(
            name="Vocal Variation",
            score=50.0,
            raw_metric_name="Pitch F0 Std",
            raw_value=round(std_f0, 1),
            unit="Hz",
            target_range=f"{config.ideal_f0_std_min:.0f}–{config.ideal_f0_std_max:.0f} Hz",
            weight=config.weight_vocal_variation,
            explanation="Unvoiced or whispering audio makes pitch estimation uncertain.",
            warning="Voiced audio ratio below 15%. Pitch variation metric is ungrounded.",
        )

    if std_f0 < config.monotone_f0_std:
        # Monotone delivery
        score = max(15.0, 30.0 + (std_f0 / config.monotone_f0_std) * 20.0)
        exp = f"Pitch variation is flat (Std: {std_f0:.1f} Hz, Range: {f0_range:.1f} Hz). Monotone delivery detected."
    elif config.ideal_f0_std_min <= std_f0 <= config.ideal_f0_std_max:
        score = 90.0 + min(10.0, (f0_range / 50.0) * 10.0)
        exp = f"Healthy, expressive vocal modulation (Std: {std_f0:.1f} Hz, Range: {f0_range:.1f} Hz)."
    elif std_f0 < config.ideal_f0_std_min:
        ratio = (std_f0 - config.monotone_f0_std) / (config.ideal_f0_std_min - config.monotone_f0_std)
        score = 50.0 + (ratio * 40.0)
        exp = f"Mildly flat pitch modulation (Std: {std_f0:.1f} Hz). Practice emphasizing key phrases."
    else:
        # Excessively erratic intonation
        score = max(50.0, 95.0 - ((std_f0 - config.ideal_f0_std_max) / 30.0) * 35.0)
        exp = f"Very high pitch variability (Std: {std_f0:.1f} Hz), which can sound ungrounded or erratic."

    return CategoryScore(
        name="Vocal Variation",
        score=round(score, 1),
        raw_metric_name="Pitch F0 Std",
        raw_value=round(std_f0, 1),
        unit="Hz",
        target_range=f"{config.ideal_f0_std_min:.0f}–{config.ideal_f0_std_max:.0f} Hz",
        weight=config.weight_vocal_variation,
        explanation=exp,
    )


def score_volume_consistency(
    rms_std_db: float,
    dyn_range_db: float,
    duration_sec: float,
    config: RubricConfig,
) -> CategoryScore:
    """Calculate Volume Consistency score (0-100) from energy variability."""
    if duration_sec < 2.0:
        return CategoryScore(
            name="Volume Consistency",
            score=50.0,
            raw_metric_name="Volume Dynamics Std",
            raw_value=round(rms_std_db, 1),
            unit="dB",
            target_range=f"{config.ideal_rms_std_min:.1f}–{config.ideal_rms_std_max:.1f} dB",
            weight=config.weight_volume_consistency,
            explanation="Recording too short for reliable volume stability assessment.",
            warning="Duration under 2 seconds.",
        )

    if config.ideal_rms_std_min <= rms_std_db <= config.ideal_rms_std_max:
        score = 95.0
        exp = f"Balanced volume control (Std: {rms_std_db:.1f} dB, Dynamic Range: {dyn_range_db:.1f} dB)."
    elif rms_std_db < config.ideal_rms_std_min:
        # Robotic or over-compressed volume
        score = max(50.0, 95.0 - ((config.ideal_rms_std_min - rms_std_db) / 3.0) * 30.0)
        exp = f"Very rigid volume levels (Std: {rms_std_db:.1f} dB). Could use more dynamic vocal projection."
    elif rms_std_db <= config.excessive_rms_std:
        ratio = (rms_std_db - config.ideal_rms_std_max) / (config.excessive_rms_std - config.ideal_rms_std_max)
        score = max(30.0, 95.0 - (ratio * 50.0))
        exp = f"Inconsistent vocal volume or mic fading (Std: {rms_std_db:.1f} dB). Noticeable loudness drops."
    else:
        score = max(15.0, 45.0 - ((rms_std_db - config.excessive_rms_std) / 5.0) * 20.0)
        exp = f"High volume instability (Std: {rms_std_db:.1f} dB, Range: {dyn_range_db:.1f} dB). Trailing off or erratic projection."

    return CategoryScore(
        name="Volume Consistency",
        score=round(score, 1),
        raw_metric_name="Volume Dynamics Std",
        raw_value=round(rms_std_db, 1),
        unit="dB",
        target_range=f"{config.ideal_rms_std_min:.1f}–{config.ideal_rms_std_max:.1f} dB",
        weight=config.weight_volume_consistency,
        explanation=exp,
    )


def compute_rubric_scores(
    wpm: float,
    word_count: int,
    fillers_per_min: float,
    filler_count: int,
    pause_ratio: float,
    long_pause_count: int,
    std_f0: float,
    f0_range: float,
    voiced_ratio: float,
    rms_std_db: float,
    dyn_range_db: float,
    duration_sec: float,
    config: Optional[RubricConfig] = None,
) -> OverallScoreResult:
    """Deterministic, grounded calculation of all 5 categories and overall score."""
    if config is None:
        config = RubricConfig()

    warnings: List[str] = []

    cat_pace = score_pace(wpm, config, word_count)
    if cat_pace.warning:
        warnings.append(cat_pace.warning)

    cat_fluency = score_fluency(fillers_per_min, filler_count, duration_sec, config)
    if cat_fluency.warning:
        warnings.append(cat_fluency.warning)

    cat_pausing = score_pausing(pause_ratio, long_pause_count, duration_sec, config)
    if cat_pausing.warning:
        warnings.append(cat_pausing.warning)

    cat_vocal = score_vocal_variation(std_f0, f0_range, voiced_ratio, config)
    if cat_vocal.warning:
        warnings.append(cat_vocal.warning)

    cat_volume = score_volume_consistency(rms_std_db, dyn_range_db, duration_sec, config)
    if cat_volume.warning:
        warnings.append(cat_volume.warning)

    categories = {
        "Pace": cat_pace,
        "Fluency": cat_fluency,
        "Pausing": cat_pausing,
        "Vocal Variation": cat_vocal,
        "Volume Consistency": cat_volume,
    }

    # Normalized weighted sum
    total_weight = sum(c.weight for c in categories.values())
    weighted_sum = sum(c.score * c.weight for c in categories.values())
    overall = weighted_sum / max(0.001, total_weight)

    return OverallScoreResult(
        overall_score=round(overall, 1),
        categories=categories,
        config=config,
        data_quality_warnings=warnings,
    )
