"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Contrastive Speech Evaluation Module.
Compares matched pairs (ideal vs. flawed speech) and evaluates flaw grounding accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from .rubric import OverallScoreResult
from .flaw_detection import GroundedFlaw
from ..audio.features import AcousticFeatures
from ..transcription.transcriber import TranscriptResult


@dataclass
class MetricDelta:
    metric_name: str
    unit: str
    ideal_value: float
    flawed_value: float
    delta: float  # flawed - ideal
    percent_change: float


@dataclass
class CategoryDelta:
    category: str
    ideal_score: float
    flawed_score: float
    delta_score: float  # flawed - ideal (typically negative when flaw degrades performance)


@dataclass
class ContrastivePairResult:
    pair_id: str
    script_id: str
    script_title: str
    intended_flaw: str
    # Results
    ideal_overall_score: float
    flawed_overall_score: float
    overall_score_delta: float
    category_deltas: Dict[str, CategoryDelta]
    metric_deltas: Dict[str, MetricDelta]
    # Grounding & Verification
    intended_flaw_detected_in_flawed: bool
    intended_flaw_detected_in_ideal: bool
    flaw_detection_success: bool  # True if detected in flawed and absent in ideal
    flawed_evidence_count: int
    ideal_evidence_count: int
    target_evidence_flaws: List[GroundedFlaw] = field(default_factory=list)
    notes: str = ""


def evaluate_contrastive_pair(
    pair_id: str,
    script_id: str,
    script_title: str,
    intended_flaw: str,
    ideal_acoustic: AcousticFeatures,
    ideal_transcript: TranscriptResult,
    ideal_scores: OverallScoreResult,
    ideal_flaws: List[GroundedFlaw],
    flawed_acoustic: AcousticFeatures,
    flawed_transcript: TranscriptResult,
    flawed_scores: OverallScoreResult,
    flawed_flaws: List[GroundedFlaw],
    notes: str = "",
) -> ContrastivePairResult:
    """
    Compute rigorous contrastive delta metrics between an ideal recording and its flawed counterpart.
    Checks whether the system successfully ground-detected the intended delivery flaw.
    """
    # 1. Metric deltas
    metrics = {
        "speaking_rate": MetricDelta(
            metric_name="Speaking Rate",
            unit="WPM",
            ideal_value=round(ideal_transcript.wpm, 1),
            flawed_value=round(flawed_transcript.wpm, 1),
            delta=round(flawed_transcript.wpm - ideal_transcript.wpm, 1),
            percent_change=round(
                ((flawed_transcript.wpm - ideal_transcript.wpm) / max(1.0, ideal_transcript.wpm)) * 100.0, 1
            ),
        ),
        "filler_rate": MetricDelta(
            metric_name="Filler Rate",
            unit="fillers/min",
            ideal_value=round(ideal_transcript.fillers_per_minute, 1),
            flawed_value=round(flawed_transcript.fillers_per_minute, 1),
            delta=round(flawed_transcript.fillers_per_minute - ideal_transcript.fillers_per_minute, 1),
            percent_change=round(
                ((flawed_transcript.fillers_per_minute - ideal_transcript.fillers_per_minute) / max(0.1, ideal_transcript.fillers_per_minute + 0.1)) * 100.0, 1
            ),
        ),
        "pause_ratio": MetricDelta(
            metric_name="Pause Ratio",
            unit="%",
            ideal_value=round(ideal_acoustic.pause_ratio * 100.0, 1),
            flawed_value=round(flawed_acoustic.pause_ratio * 100.0, 1),
            delta=round((flawed_acoustic.pause_ratio - ideal_acoustic.pause_ratio) * 100.0, 1),
            percent_change=round(
                ((flawed_acoustic.pause_ratio - ideal_acoustic.pause_ratio) / max(0.01, ideal_acoustic.pause_ratio)) * 100.0, 1
            ),
        ),
        "pitch_std": MetricDelta(
            metric_name="Pitch F0 Std",
            unit="Hz",
            ideal_value=round(ideal_acoustic.std_f0_hz, 1),
            flawed_value=round(flawed_acoustic.std_f0_hz, 1),
            delta=round(flawed_acoustic.std_f0_hz - ideal_acoustic.std_f0_hz, 1),
            percent_change=round(
                ((flawed_acoustic.std_f0_hz - ideal_acoustic.std_f0_hz) / max(1.0, ideal_acoustic.std_f0_hz)) * 100.0, 1
            ),
        ),
        "volume_std": MetricDelta(
            metric_name="Volume RMS Std",
            unit="dB",
            ideal_value=round(ideal_acoustic.rms_std_db, 1),
            flawed_value=round(flawed_acoustic.rms_std_db, 1),
            delta=round(flawed_acoustic.rms_std_db - ideal_acoustic.rms_std_db, 1),
            percent_change=round(
                ((flawed_acoustic.rms_std_db - ideal_acoustic.rms_std_db) / max(1.0, ideal_acoustic.rms_std_db)) * 100.0, 1
            ),
        ),
    }

    # 2. Category score deltas
    cat_deltas: Dict[str, CategoryDelta] = {}
    for cat_name, ideal_cat in ideal_scores.categories.items():
        flawed_cat = flawed_scores.categories.get(cat_name)
        flawed_score = flawed_cat.score if flawed_cat else 0.0
        cat_deltas[cat_name] = CategoryDelta(
            category=cat_name,
            ideal_score=ideal_cat.score,
            flawed_score=flawed_score,
            delta_score=round(flawed_score - ideal_cat.score, 1),
        )

    # 3. Flaw grounding verification
    flaw_type_map = {
        "fillers": "filler_word",
        "pauses": "prolonged_pause",
        "rushed": "rushed_segment",
        "sluggish": "sluggish_segment",
        "monotone": "monotone_pitch",
        "volume": "volume_instability",
    }
    target_type = flaw_type_map.get(intended_flaw.lower(), intended_flaw.lower())

    matching_flawed_flaws = [f for f in flawed_flaws if f.flaw_type == target_type or target_type in f.flaw_type]
    matching_ideal_flaws = [f for f in ideal_flaws if f.flaw_type == target_type or target_type in f.flaw_type]

    detected_in_flawed = len(matching_flawed_flaws) > 0
    detected_in_ideal = len(matching_ideal_flaws) > 0

    # Success condition: flaw must be detected in flawed audio, and significantly more frequent than in ideal
    success = detected_in_flawed and (len(matching_flawed_flaws) > len(matching_ideal_flaws))

    return ContrastivePairResult(
        pair_id=pair_id,
        script_id=script_id,
        script_title=script_title,
        intended_flaw=intended_flaw,
        ideal_overall_score=ideal_scores.overall_score,
        flawed_overall_score=flawed_scores.overall_score,
        overall_score_delta=round(flawed_scores.overall_score - ideal_scores.overall_score, 1),
        category_deltas=cat_deltas,
        metric_deltas=metrics,
        intended_flaw_detected_in_flawed=detected_in_flawed,
        intended_flaw_detected_in_ideal=detected_in_ideal,
        flaw_detection_success=success,
        flawed_evidence_count=len(flawed_flaws),
        ideal_evidence_count=len(ideal_flaws),
        target_evidence_flaws=matching_flawed_flaws,
        notes=notes,
    )
