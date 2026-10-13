"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Audio ingestion, validation, and preprocessing module.
"""

from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa


@dataclass
class AudioMetadata:
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    rms_db: float
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class AudioData:
    samples: np.ndarray  # 1D float32 array, normalized [-1.0, 1.0]
    sample_rate: int
    metadata: AudioMetadata


def validate_audio_file(
    file_input: Union[str, Path, io.BytesIO],
    max_duration_seconds: float = 600.0,
    max_size_mb: float = 50.0,
) -> Tuple[bool, Optional[str]]:
    """Validate file size and structure before reading into memory."""
    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        if not p.exists():
            return False, f"File does not exist: {p}"
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File exceeds maximum allowed size ({size_mb:.1f}MB > {max_size_mb}MB)"
        if p.stat().st_size == 0:
            return False, "File is empty (0 bytes)"
    elif isinstance(file_input, io.BytesIO):
        size_mb = len(file_input.getvalue()) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File exceeds maximum allowed size ({size_mb:.1f}MB > {max_size_mb}MB)"
        if len(file_input.getvalue()) == 0:
            return False, "File buffer is empty (0 bytes)"
    return True, None


def load_audio(
    file_input: Union[str, Path, io.BytesIO],
    target_sr: int = 16000,
    max_duration_seconds: float = 600.0,
    filename: str = "audio_clip.wav",
) -> AudioData:
    """
    Load an audio file, convert to mono, resample to target_sr, and normalize.
    Supports WAV, MP3, OGG, FLAC via soundfile/librosa.
    """
    valid, err = validate_audio_file(file_input, max_duration_seconds=max_duration_seconds)
    if not valid:
        raise ValueError(err)

    if isinstance(file_input, (str, Path)):
        fname = Path(file_input).name
    else:
        fname = filename

    try:
        # Load audio using librosa or soundfile
        y, orig_sr = librosa.load(file_input, sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio file '{fname}': {str(e)}") from e

    if len(y) == 0:
        raise ValueError(f"Decoded audio array is empty for '{fname}'")

    duration = float(len(y)) / float(target_sr)
    if duration > max_duration_seconds:
        # Truncate safely or raise
        y = y[: int(max_duration_seconds * target_sr)]
        duration = max_duration_seconds

    # Ensure float32 normalized
    y = np.asarray(y, dtype=np.float32)
    max_val = np.max(np.abs(y))
    if max_val > 0.0:
        # Avoid hard clipping while preserving dynamics
        if max_val > 1.0:
            y = y / max_val

    # Calculate overall RMS energy (dB)
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    rms_db = 20.0 * math.log10(rms + 1e-12)

    meta = AudioMetadata(
        filename=fname,
        duration_seconds=round(duration, 3),
        sample_rate=target_sr,
        channels=1,
        rms_db=round(rms_db, 2),
        is_valid=True,
    )

    return AudioData(samples=y, sample_rate=target_sr, metadata=meta)
