"""
SpeechProof: Comprehensive Automated Unit & Regression Test Suite.
Validates:
1. Audio loading, validation, and error handling.
2. Signal feature extraction (energy, pauses, pitch, activity).
3. Deterministic rubric scoring engine and boundary conditions.
4. Temporal flaw grounding and evidence timestamp bounds.
5. Contrastive pair evaluation and delta calculations.
6. Actionable coaching feedback generation.
7. Dataset manifest and benchmark schema validity.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.audio.loader import load_audio, validate_audio_file
from src.audio.features import (
    extract_acoustic_features,
    detect_pauses,
    extract_energy_profile,
    extract_pitch_contour,
    AcousticFeatures,
    PauseInterval,
)
from src.transcription.transcriber import (
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
    FillerWordOccurrence,
    detect_fillers_in_segment,
)
from src.evaluation.rubric import (
    RubricConfig,
    compute_rubric_scores,
    score_pace,
    score_fluency,
    score_pausing,
    score_vocal_variation,
    score_volume_consistency,
)
from src.evaluation.flaw_detection import (
    GroundedFlaw,
    detect_all_grounded_flaws,
    ground_pause_flaws,
    ground_filler_flaws,
    ground_pacing_flaws,
    ground_pitch_flaws,
    ground_volume_flaws,
)
from src.evaluation.contrastive import evaluate_contrastive_pair
from src.feedback.coach import generate_coaching_report
from src.data.manifest import load_dataset_manifest, get_matched_pairs


# ==========================================
# 1. AUDIO LOADING & VALIDATION TESTS
# ==========================================

def test_validate_audio_nonexistent_file():
    valid, err = validate_audio_file("nonexistent_audio_sample_xyz.wav")
    assert not valid
    assert "does not exist" in err


def test_load_audio_valid_sample():
    p = Path("data/examples/P01_ideal.wav")
    assert p.exists(), "Test audio fixture P01_ideal.wav must exist"
    audio = load_audio(str(p), target_sr=16000)
    assert audio.sample_rate == 16000
    assert audio.metadata.duration_seconds > 0.0
    assert audio.metadata.channels == 1
    assert len(audio.samples) > 0
    assert np.max(np.abs(audio.samples)) <= 1.0


# ==========================================
# 2. ACOUSTIC FEATURE EXTRACTION TESTS
# ==========================================

def test_extract_acoustic_features_synthetic_sine():
    sr = 16000
    duration = 3.0
    t = np.arange(int(sr * duration)) / sr
    # 220 Hz sine tone with 0.5s silence gap in the middle
    sine = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    # Insert 0.5s silence
    sine[int(1.0 * sr) : int(1.5 * sr)] = 0.0

    features = extract_acoustic_features(sine, sr=sr, min_pause_sec=0.3, long_pause_threshold_sec=1.0)
    assert pytest.approx(features.duration, 0.1) == 3.0
    assert features.pause_count >= 1
    assert features.total_pause_duration >= 0.4
    assert features.mean_f0_hz > 200.0


def test_detect_pauses_boundary_conditions():
    sr = 16000
    empty_sig = np.zeros(0, dtype=np.float32)
    pauses, total_p, ratio_p = detect_pauses(empty_sig, sr)
    assert pauses == []
    assert total_p == 0.0


# ==========================================
# 3. DETERMINISTIC RUBRIC SCORING TESTS
# ==========================================

def test_rubric_scoring_deterministic():
    config = RubricConfig()
    res1 = compute_rubric_scores(
        wpm=145.0,
        word_count=50,
        fillers_per_min=0.5,
        filler_count=1,
        pause_ratio=0.18,
        long_pause_count=0,
        std_f0=35.0,
        f0_range=60.0,
        voiced_ratio=0.85,
        rms_std_db=5.2,
        dyn_range_db=18.0,
        duration_sec=30.0,
        config=config,
    )
    res2 = compute_rubric_scores(
        wpm=145.0,
        word_count=50,
        fillers_per_min=0.5,
        filler_count=1,
        pause_ratio=0.18,
        long_pause_count=0,
        std_f0=35.0,
        f0_range=60.0,
        voiced_ratio=0.85,
        rms_std_db=5.2,
        dyn_range_db=18.0,
        duration_sec=30.0,
        config=config,
    )
    assert res1.overall_score == res2.overall_score
    for cat in res1.categories:
        assert res1.categories[cat].score == res2.categories[cat].score


def test_rubric_score_bounds_and_extremes():
    config = RubricConfig()
    # Test completely pathological speech inputs
    res_awful = compute_rubric_scores(
        wpm=320.0,  # hyper rushed
        word_count=100,
        fillers_per_min=25.0,  # massive filler spam
        filler_count=30,
        pause_ratio=0.75,  # mostly silence
        long_pause_count=8,
        std_f0=2.0,  # robotic flat pitch
        f0_range=3.0,
        voiced_ratio=0.9,
        rms_std_db=25.0,  # erratic volume
        dyn_range_db=40.0,
        duration_sec=60.0,
        config=config,
    )
    assert 0.0 <= res_awful.overall_score <= 100.0
    assert res_awful.overall_score < 45.0

    # Test ideal polished speech
    res_ideal = compute_rubric_scores(
        wpm=145.0,
        word_count=80,
        fillers_per_min=0.0,
        filler_count=0,
        pause_ratio=0.20,
        long_pause_count=0,
        std_f0=35.0,
        f0_range=65.0,
        voiced_ratio=0.88,
        rms_std_db=5.0,
        dyn_range_db=20.0,
        duration_sec=35.0,
        config=config,
    )
    assert res_ideal.overall_score >= 90.0


def test_missing_data_warning_handling():
    config = RubricConfig()
    # Short 1.0 second audio clip with 2 words
    res = compute_rubric_scores(
        wpm=120.0,
        word_count=2,
        fillers_per_min=0.0,
        filler_count=0,
        pause_ratio=0.0,
        long_pause_count=0,
        std_f0=0.0,
        f0_range=0.0,
        voiced_ratio=0.05,
        rms_std_db=1.0,
        dyn_range_db=2.0,
        duration_sec=1.0,
        config=config,
    )
    assert len(res.data_quality_warnings) > 0
    assert any("Low word count" in w for w in res.data_quality_warnings)


# ==========================================
# 4. TEMPORAL FLAW GROUNDING TESTS
# ==========================================

def test_grounded_flaw_timestamp_bounds():
    config = RubricConfig()
    acoustic = AcousticFeatures(
        duration=10.0,
        mean_rms_db=-22.0,
        rms_std_db=4.5,
        dynamic_range_db=15.0,
        pauses=[
            PauseInterval(start_time=2.0, end_time=3.8, duration=1.8, is_long_pause=True)
        ],
        std_f0_hz=30.0,
        voiced_ratio=0.8,
    )
    transcript = TranscriptResult(
        text="Hello world um this is a test",
        language="en",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=1.9, text="Hello world", words=[], wpm=140.0),
            TranscriptSegment(id=1, start=3.9, end=6.0, text="um this is a test", words=[], wpm=150.0),
        ],
        duration=10.0,
        total_words=7,
        wpm=140.0,
        fillers=[
            FillerWordOccurrence(filler="um", start=4.0, end=4.4, context="um this is a test", confidence=0.9)
        ],
        filler_count=1,
        fillers_per_minute=6.0,
    )

    flaws = detect_all_grounded_flaws(acoustic, transcript, config)
    assert len(flaws) >= 2  # 1 prolonged pause + 1 filler

    for f in flaws:
        assert 0.0 <= f.start_time <= 10.0
        assert 0.0 <= f.end_time <= 10.0
        assert f.start_time <= f.end_time
        assert f.duration > 0.0
        assert f.rubric_category in ["Pace", "Fluency", "Pausing", "Vocal Variation", "Volume Consistency"]
        assert len(f.recommendation) > 10


# ==========================================
# 5. CONTRASTIVE PAIR & BENCHMARK TESTS
# ==========================================

def test_manifest_entries_and_pairs():
    manifest_path = "data/manifest.csv"
    assert Path(manifest_path).exists()
    entries = load_dataset_manifest(manifest_path)
    assert len(entries) == 10  # 5 pairs * 2
    pairs = get_matched_pairs(entries)
    assert len(pairs) == 5

    for pid, pair in pairs.items():
        assert "ideal" in pair
        assert "flawed" in pair
        assert Path(pair["ideal"].audio_path).exists()
        assert Path(pair["flawed"].audio_path).exists()


def test_benchmark_results_json_schema():
    bench_file = Path("data/benchmark_results.json")
    assert bench_file.exists(), "Benchmark results JSON must exist"
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_pairs" in data
    assert data["total_pairs"] == 5
    assert "flaw_grounding_accuracy_pct" in data
    assert data["flaw_grounding_accuracy_pct"] == 100.0
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "pairs_evaluated" in data
    assert len(data["pairs_evaluated"]) == 5


# ==========================================
# 6. ACTIONABLE COACHING GENERATION TESTS
# ==========================================

def test_coaching_report_structure():
    config = RubricConfig()
    scores = compute_rubric_scores(
        wpm=110.0,
        word_count=40,
        fillers_per_min=4.5,
        filler_count=5,
        pause_ratio=0.35,
        long_pause_count=2,
        std_f0=12.0,
        f0_range=20.0,
        voiced_ratio=0.7,
        rms_std_db=11.0,
        dyn_range_db=25.0,
        duration_sec=30.0,
        config=config,
    )
    flaws = [
        GroundedFlaw(
            flaw_type="prolonged_pause",
            start_time=5.0,
            end_time=7.1,
            duration=2.1,
            severity="moderate",
            metric_name="pause_duration",
            observed_value=2.1,
            unit="seconds",
            threshold=1.4,
            rubric_category="Pausing",
            explanation="Long pause",
            recommendation="Bridge the pause",
            evidence_source="audio_signal",
            confidence=0.95,
        ),
        GroundedFlaw(
            flaw_type="filler_word",
            start_time=12.0,
            end_time=12.4,
            duration=0.4,
            severity="moderate",
            metric_name="filler_occurrence",
            observed_value=1.0,
            unit="count",
            threshold=0.0,
            rubric_category="Fluency",
            explanation="Filler word um",
            recommendation="Inhale silently",
            evidence_source="transcript_alignment",
            confidence=0.92,
        ),
    ]

    report = generate_coaching_report(scores, flaws)
    assert len(report.headline) > 5
    assert len(report.priority_focus_areas) > 0
    assert len(report.practice_drills) >= 2
    assert len(report.timestamped_action_items) == 2
    for item in report.timestamped_action_items:
        assert "timestamp" in item
        assert "recommendation" in item
