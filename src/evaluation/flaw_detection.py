"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Temporal Flaw Grounding Engine.
Emits structured, timestamped evidence objects for every detected delivery defect.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np

from ..audio.features import AcousticFeatures, PauseInterval
from ..transcription.transcriber import TranscriptResult, TranscriptSegment, FillerWordOccurrence
from .rubric import RubricConfig


@dataclass
class GroundedFlaw:
    flaw_type: str  # e.g., "prolonged_pause", "filler_word", "rushed_segment", "monotone_pitch", "volume_drop"
    start_time: float
    end_time: float
    duration: float
    severity: str  # "mild", "moderate", "severe"
    metric_name: str
    observed_value: float
    unit: str
    threshold: float
    rubric_category: str
    explanation: str
    recommendation: str
    evidence_source: str  # "audio_signal", "transcript_alignment", "acoustic_feature"
    confidence: float
    context_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ground_pause_flaws(
    pauses: List[PauseInterval],
    config: RubricConfig,
    segments: List[TranscriptSegment],
) -> List[GroundedFlaw]:
    """Ground prolonged awkward pauses into structured evidence objects."""
    flaws: List[GroundedFlaw] = []

    for p in pauses:
        if p.duration >= config.long_pause_threshold_sec:
            # Determine severity
            if p.duration >= 2.5:
                severity = "severe"
            elif p.duration >= 1.6:
                severity = "moderate"
            else:
                severity = "mild"

            # Find neighboring text context
            nearby_texts = []
            for seg in segments:
                if abs(seg.end - p.start_time) < 1.0 or abs(seg.start - p.end_time) < 1.0:
                    nearby_texts.append(seg.text)
            context = " ... ".join(nearby_texts) if nearby_texts else "Between speech phrases"

            flaw = GroundedFlaw(
                flaw_type="prolonged_pause",
                start_time=round(p.start_time, 2),
                end_time=round(p.end_time, 2),
                duration=round(p.duration, 2),
                severity=severity,
                metric_name="pause_duration",
                observed_value=round(p.duration, 2),
                unit="seconds",
                threshold=round(config.long_pause_threshold_sec, 2),
                rubric_category="Pausing",
                explanation=f"Awkward silence of {p.duration:.2f}s exceeds the natural delivery threshold of {config.long_pause_threshold_sec:.1f}s.",
                recommendation="Replace extended dead silences with deliberate transitions, bridging remarks, or forward-looking connective phrases.",
                evidence_source="audio_signal",
                confidence=0.98,
                context_text=context,
            )
            flaws.append(flaw)

    return flaws


def ground_filler_flaws(
    fillers: List[FillerWordOccurrence],
    config: RubricConfig,
) -> List[GroundedFlaw]:
    """Ground filler words into timestamped evidence objects."""
    flaws: List[GroundedFlaw] = []

    for f in fillers:
        flaw = GroundedFlaw(
            flaw_type="filler_word",
            start_time=round(f.start, 2),
            end_time=round(f.end, 2),
            duration=round(max(0.15, f.end - f.start), 2),
            severity="moderate" if f.filler in ["um", "uh", "like"] else "mild",
            metric_name="filler_occurrence",
            observed_value=1.0,
            unit="count",
            threshold=0.0,
            rubric_category="Fluency",
            explanation=f"Verbal filler '{f.filler}' introduces cognitive friction and weakens message authority.",
            recommendation=f"Pause in silence instead of uttering '{f.filler}'. Silence conveys poise; filler words convey hesitation.",
            evidence_source="transcript_alignment",
            confidence=round(f.confidence, 2),
            context_text=f.context,
        )
        flaws.append(flaw)

    return flaws


def ground_pacing_flaws(
    segments: List[TranscriptSegment],
    config: RubricConfig,
) -> List[GroundedFlaw]:
    """Ground segment-level tempo defects (rushed or sluggish bursts)."""
    flaws: List[GroundedFlaw] = []

    for seg in segments:
        seg_duration = seg.end - seg.start
        if seg_duration < 1.0 or len(seg.words) < 3:
            continue

        if seg.wpm > (config.target_wpm_max + 25.0):
            # Rushed segment
            flaw = GroundedFlaw(
                flaw_type="rushed_segment",
                start_time=round(seg.start, 2),
                end_time=round(seg.end, 2),
                duration=round(seg_duration, 2),
                severity="severe" if seg.wpm > 210.0 else "moderate",
                metric_name="segment_wpm",
                observed_value=round(seg.wpm, 1),
                unit="WPM",
                threshold=round(config.target_wpm_max, 1),
                rubric_category="Pace",
                explanation=f"Speech rate surged to {seg.wpm:.1f} WPM, which risks losing listener comprehension.",
                recommendation="Slow down on key points; enunciate critical nouns and verbs deliberately.",
                evidence_source="transcript_alignment",
                confidence=0.92,
                context_text=seg.text,
            )
            flaws.append(flaw)

        elif seg.wpm < (config.target_wpm_min - 35.0):
            # Sluggish segment
            flaw = GroundedFlaw(
                flaw_type="sluggish_segment",
                start_time=round(seg.start, 2),
                end_time=round(seg.end, 2),
                duration=round(seg_duration, 2),
                severity="mild",
                metric_name="segment_wpm",
                observed_value=round(seg.wpm, 1),
                unit="WPM",
                threshold=round(config.target_wpm_min, 1),
                rubric_category="Pace",
                explanation=f"Speech rate slowed to {seg.wpm:.1f} WPM, creating a drag in energy.",
                recommendation="Pick up the tempo slightly to keep the momentum engaging.",
                evidence_source="transcript_alignment",
                confidence=0.88,
                context_text=seg.text,
            )
            flaws.append(flaw)

    return flaws


def ground_pitch_flaws(
    acoustic: AcousticFeatures,
    config: RubricConfig,
) -> List[GroundedFlaw]:
    """Ground vocal variation and monotone delivery defects."""
    flaws: List[GroundedFlaw] = []

    if acoustic.voiced_ratio >= 0.20 and acoustic.std_f0_hz < config.monotone_f0_std:
        # Flag global monotone flaw across the presentation
        flaw = GroundedFlaw(
            flaw_type="monotone_pitch",
            start_time=0.0,
            end_time=acoustic.duration,
            duration=acoustic.duration,
            severity="moderate" if acoustic.std_f0_hz < 8.0 else "mild",
            metric_name="pitch_f0_std",
            observed_value=round(acoustic.std_f0_hz, 1),
            unit="Hz",
            threshold=round(config.monotone_f0_std, 1),
            rubric_category="Vocal Variation",
            explanation=f"Pitch modulation standard deviation of {acoustic.std_f0_hz:.1f} Hz is below dynamic speaking threshold ({config.ideal_f0_std_min:.0f} Hz).",
            recommendation="Practice vocal inflection drills: raise pitch at the end of provocative questions, lower it to emphasize conclusions.",
            evidence_source="acoustic_feature",
            confidence=0.90,
            context_text=f"Pitch range observed: {acoustic.f0_range_hz:.1f} Hz across {acoustic.duration:.1f}s",
        )
        flaws.append(flaw)

    return flaws


def ground_volume_flaws(
    acoustic: AcousticFeatures,
    config: RubricConfig,
) -> List[GroundedFlaw]:
    """Ground excessive volume fluctuation or projection drops."""
    flaws: List[GroundedFlaw] = []

    if acoustic.rms_std_db > config.excessive_rms_std:
        flaw = GroundedFlaw(
            flaw_type="volume_instability",
            start_time=0.0,
            end_time=acoustic.duration,
            duration=acoustic.duration,
            severity="moderate",
            metric_name="rms_std_db",
            observed_value=round(acoustic.rms_std_db, 1),
            unit="dB",
            threshold=round(config.excessive_rms_std, 1),
            rubric_category="Volume Consistency",
            explanation=f"Volume variability of {acoustic.rms_std_db:.1f} dB indicates erratic projection or microphone distance shifts.",
            recommendation="Maintain consistent diaphragm support and stable distance from the microphone.",
            evidence_source="acoustic_feature",
            confidence=0.89,
            context_text=f"Dynamic range: {acoustic.dynamic_range_db:.1f} dB",
        )
        flaws.append(flaw)

    return flaws


def detect_all_grounded_flaws(
    acoustic: AcousticFeatures,
    transcript: TranscriptResult,
    config: Optional[RubricConfig] = None,
) -> List[GroundedFlaw]:
    """
    Run complete Temporal Flaw Grounding suite.
    Returns sorted list of grounded flaw evidence objects ordered chronologically.
    """
    if config is None:
        config = RubricConfig()

    flaws: List[GroundedFlaw] = []

    # 1. Grounded pauses
    flaws.extend(ground_pause_flaws(acoustic.pauses, config, transcript.segments))

    # 2. Grounded fillers
    flaws.extend(ground_filler_flaws(transcript.fillers, config))

    # 3. Grounded pacing surges / drops
    flaws.extend(ground_pacing_flaws(transcript.segments, config))

    # 4. Vocal variation defects
    flaws.extend(ground_pitch_flaws(acoustic, config))

    # 5. Volume consistency defects
    flaws.extend(ground_volume_flaws(acoustic, config))

    # Sort chronologically by start_time
    flaws.sort(key=lambda x: (x.start_time, x.end_time))
    return flaws
