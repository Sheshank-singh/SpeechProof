"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Audio slicing utility for grounded temporal interval playback.
"""

import io
import numpy as np
import soundfile as sf


def slice_audio_to_wav_bytes(
    y: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    padding_sec: float = 0.2,
) -> bytes:
    """Extract an audio slice around start_time and end_time and return WAV bytes."""
    start_sample = max(0, int((start_time - padding_sec) * sr))
    end_sample = min(len(y), int((end_time + padding_sec) * sr))

    if end_sample <= start_sample:
        end_sample = min(len(y), start_sample + int(0.5 * sr))

    sliced = y[start_sample:end_sample]
    buf = io.BytesIO()
    sf.write(buf, sliced, sr, format="WAV")
    buf.seek(0)
    return buf.getvalue()
