"""
SpeechProof: Contrastive Benchmark Evaluator
Runs full end-to-end evaluation across all matched speech pairs in data/manifest.csv.
Computes metric deltas, score shifts, flaw grounding accuracy, and benchmark summary.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.loader import load_audio
from src.audio.features import extract_acoustic_features
from src.transcription.transcriber import transcribe_audio
from src.evaluation.rubric import compute_rubric_scores, RubricConfig
from src.evaluation.flaw_detection import detect_all_grounded_flaws
from src.evaluation.contrastive import evaluate_contrastive_pair, ContrastivePairResult
from src.data.manifest import load_dataset_manifest, get_matched_pairs


def run_benchmark(manifest_path: str = "data/manifest.csv") -> Dict[str, Any]:
    print("Loading dataset manifest...")
    entries = load_dataset_manifest(manifest_path)
    pairs = get_matched_pairs(entries)
    config = RubricConfig()

    pair_results: List[Dict[str, Any]] = []
    success_count = 0
    total_pairs = len(pairs)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    print(f"Evaluating {total_pairs} matched contrastive speech pairs...")

    for pair_id, pair_data in pairs.items():
        ideal_entry = pair_data.get("ideal")
        flawed_entry = pair_data.get("flawed")

        if not ideal_entry or not flawed_entry:
            print(f"Warning: Pair {pair_id} is missing ideal or flawed entry. Skipping.")
            continue

        print(f"\n--- Evaluating Pair {pair_id}: {ideal_entry.script_title} (Intended Flaw: {flawed_entry.intended_flaw}) ---")

        # 1. Process Ideal Audio
        ideal_audio = load_audio(ideal_entry.audio_path)
        ideal_acoustic = extract_acoustic_features(ideal_audio.samples, ideal_audio.sample_rate)
        ideal_transcript = transcribe_audio(ideal_audio.samples, ideal_audio.sample_rate)
        ideal_scores = compute_rubric_scores(
            wpm=ideal_transcript.wpm,
            word_count=ideal_transcript.total_words,
            fillers_per_min=ideal_transcript.fillers_per_minute,
            filler_count=ideal_transcript.filler_count,
            pause_ratio=ideal_acoustic.pause_ratio,
            long_pause_count=len([p for p in ideal_acoustic.pauses if p.is_long_pause]),
            std_f0=ideal_acoustic.std_f0_hz,
            f0_range=ideal_acoustic.f0_range_hz,
            voiced_ratio=ideal_acoustic.voiced_ratio,
            rms_std_db=ideal_acoustic.rms_std_db,
            dyn_range_db=ideal_acoustic.dynamic_range_db,
            duration_sec=ideal_audio.metadata.duration_seconds,
            config=config,
        )
        ideal_flaws = detect_all_grounded_flaws(ideal_acoustic, ideal_transcript, config)

        # 2. Process Flawed Audio
        flawed_audio = load_audio(flawed_entry.audio_path)
        flawed_acoustic = extract_acoustic_features(flawed_audio.samples, flawed_audio.sample_rate)
        flawed_transcript = transcribe_audio(flawed_audio.samples, flawed_audio.sample_rate)
        flawed_scores = compute_rubric_scores(
            wpm=flawed_transcript.wpm,
            word_count=flawed_transcript.total_words,
            fillers_per_min=flawed_transcript.fillers_per_minute,
            filler_count=flawed_transcript.filler_count,
            pause_ratio=flawed_acoustic.pause_ratio,
            long_pause_count=len([p for p in flawed_acoustic.pauses if p.is_long_pause]),
            std_f0=flawed_acoustic.std_f0_hz,
            f0_range=flawed_acoustic.f0_range_hz,
            voiced_ratio=flawed_acoustic.voiced_ratio,
            rms_std_db=flawed_acoustic.rms_std_db,
            dyn_range_db=flawed_acoustic.dynamic_range_db,
            duration_sec=flawed_audio.metadata.duration_seconds,
            config=config,
        )
        flawed_flaws = detect_all_grounded_flaws(flawed_acoustic, flawed_transcript, config)

        # 3. Contrastive Evaluation
        pair_res = evaluate_contrastive_pair(
            pair_id=pair_id,
            script_id=ideal_entry.script_id,
            script_title=ideal_entry.script_title,
            intended_flaw=flawed_entry.intended_flaw,
            ideal_acoustic=ideal_acoustic,
            ideal_transcript=ideal_transcript,
            ideal_scores=ideal_scores,
            ideal_flaws=ideal_flaws,
            flawed_acoustic=flawed_acoustic,
            flawed_transcript=flawed_transcript,
            flawed_scores=flawed_scores,
            flawed_flaws=flawed_flaws,
            notes=flawed_entry.notes,
        )

        if pair_res.flaw_detection_success:
            success_count += 1
            tp += 1
        else:
            fn += 1

        print(f"  Ideal Score: {pair_res.ideal_overall_score:.1f} | Flawed Score: {pair_res.flawed_overall_score:.1f} (Delta: {pair_res.overall_score_delta:+.1f})")
        print(f"  Flaw Grounding Success: {pair_res.flaw_detection_success} (Detected in flawed: {pair_res.intended_flaw_detected_in_flawed}, Detected in ideal: {pair_res.intended_flaw_detected_in_ideal})")
        print(f"  Flawed Evidence Count: {pair_res.flawed_evidence_count} | Ideal Evidence Count: {pair_res.ideal_evidence_count}")

        # Serialize
        pair_dict = {
            "pair_id": pair_res.pair_id,
            "script_id": pair_res.script_id,
            "script_title": pair_res.script_title,
            "intended_flaw": pair_res.intended_flaw,
            "ideal_overall_score": pair_res.ideal_overall_score,
            "flawed_overall_score": pair_res.flawed_overall_score,
            "overall_score_delta": pair_res.overall_score_delta,
            "flaw_detection_success": pair_res.flaw_detection_success,
            "intended_flaw_detected_in_flawed": pair_res.intended_flaw_detected_in_flawed,
            "intended_flaw_detected_in_ideal": pair_res.intended_flaw_detected_in_ideal,
            "flawed_evidence_count": pair_res.flawed_evidence_count,
            "ideal_evidence_count": pair_res.ideal_evidence_count,
            "notes": pair_res.notes,
            "categories": {
                k: {
                    "ideal": v.ideal_score,
                    "flawed": v.flawed_score,
                    "delta": v.delta_score,
                }
                for k, v in pair_res.category_deltas.items()
            },
            "metrics": {
                k: {
                    "name": v.metric_name,
                    "unit": v.unit,
                    "ideal": v.ideal_value,
                    "flawed": v.flawed_value,
                    "delta": v.delta,
                    "pct_change": v.percent_change,
                }
                for k, v in pair_res.metric_deltas.items()
            },
            "target_evidence": [f.to_dict() for f in pair_res.target_evidence_flaws],
            # Full recording data for instant precomputed playback and offline demo
            "ideal_transcript": ideal_transcript.text,
            "flawed_transcript": flawed_transcript.text,
            "ideal_flaws": [f.to_dict() for f in ideal_flaws],
            "flawed_flaws": [f.to_dict() for f in flawed_flaws],
        }
        pair_results.append(pair_dict)

    # Compute aggregate metrics
    accuracy = (success_count / max(1, total_pairs)) * 100.0
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = (2 * precision * recall) / max(0.001, (precision + recall))

    mean_score_delta = float(pd.Series([p["overall_score_delta"] for p in pair_results]).mean())

    summary = {
        "dataset_name": "SpeechProof-Contrastive-v1",
        "total_clips": len(entries),
        "total_pairs": total_pairs,
        "flaw_grounding_accuracy_pct": round(accuracy, 1),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "mean_overall_score_delta": round(mean_score_delta, 1),
        "pairs_evaluated": pair_results,
    }

    out_file = Path("data/benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n==========================================")
    print(f"BENCHMARK COMPLETE")
    print(f"Total Pairs: {total_pairs} | Grounding Accuracy: {accuracy:.1f}%")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"Mean Score Delta (Flawed - Ideal): {mean_score_delta:+.1f} pts")
    print(f"Benchmark artifact written to {out_file}")
    print(f"==========================================")

    return summary


if __name__ == "__main__":
    run_benchmark()
