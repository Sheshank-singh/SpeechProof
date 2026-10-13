"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Dataset manifest loader and validator.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd


@dataclass
class ManifestEntry:
    clip_id: str
    pair_id: str
    script_id: str
    script_title: str
    language: str
    delivery_type: str  # "ideal" or "flawed"
    audio_path: str
    duration: float
    intended_flaw: str
    annotation_start: float
    annotation_end: float
    annotation_method: str
    recording_conditions: str
    notes: str


def load_dataset_manifest(manifest_path: str = "data/manifest.csv") -> List[ManifestEntry]:
    """Load and parse the contrastive dataset manifest CSV."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")

    df = pd.read_csv(p)
    entries: List[ManifestEntry] = []

    for _, row in df.iterrows():
        entry = ManifestEntry(
            clip_id=str(row["clip_id"]),
            pair_id=str(row["pair_id"]),
            script_id=str(row["script_id"]),
            script_title=str(row.get("script_title", row["script_id"])),
            language=str(row.get("language", "en")),
            delivery_type=str(row["delivery_type"]),
            audio_path=str(row["audio_path"]),
            duration=float(row["duration"]),
            intended_flaw=str(row.get("intended_flaw", "none")),
            annotation_start=float(row.get("annotation_start", 0.0)),
            annotation_end=float(row.get("annotation_end", 0.0)),
            annotation_method=str(row.get("annotation_method", "manual")),
            recording_conditions=str(row.get("recording_conditions", "studio_synthetic")),
            notes=str(row.get("notes", "")),
        )
        entries.append(entry)

    return entries


def get_matched_pairs(entries: List[ManifestEntry]) -> Dict[str, Dict[str, ManifestEntry]]:
    """Group manifest entries by pair_id with 'ideal' and 'flawed' slots."""
    pairs: Dict[str, Dict[str, ManifestEntry]] = {}
    for entry in entries:
        pid = entry.pair_id
        if pid not in pairs:
            pairs[pid] = {}
        pairs[pid][entry.delivery_type] = entry
    return pairs
