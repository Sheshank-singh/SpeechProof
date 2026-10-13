"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Speech transcription and filler word detection module.
"""

from dataclasses import dataclass, field
import re
from typing import List, Dict, Optional, Tuple, Set
import numpy as np


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float
    is_filler: bool = False


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    wpm: float = 0.0


@dataclass
class FillerWordOccurrence:
    filler: str
    start: float
    end: float
    context: str
    confidence: float


@dataclass
class TranscriptResult:
    text: str
    language: str
    segments: List[TranscriptSegment]
    duration: float
    total_words: int
    wpm: float
    fillers: List[FillerWordOccurrence] = field(default_factory=list)
    filler_count: int = 0
    fillers_per_minute: float = 0.0


DEFAULT_ENGLISH_FILLERS: Set[str] = {
    "um",
    "uh",
    "uhh",
    "umm",
    "er",
    "ah",
    "like",
    "you know",
    "actually",
    "basically",
    "i mean",
    "literally",
    "sort of",
    "kind of",
}

_WHISPER_MODEL = None


def get_whisper_model(model_size: str = "tiny.en", device: str = "cpu", compute_type: str = "int8"):
    """Singleton getter for WhisperModel to prevent re-instantiation across runs."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _WHISPER_MODEL


def detect_fillers_in_segment(
    segment_text: str,
    seg_start: float,
    seg_end: float,
    words: List[WordTimestamp],
    filler_set: Set[str] = DEFAULT_ENGLISH_FILLERS,
) -> List[FillerWordOccurrence]:
    """Detect filler words either with word-level timestamps or segment fallback."""
    occurrences: List[FillerWordOccurrence] = []

    # If word-level timestamps are present, check each word
    if words:
        for w in words:
            clean_word = re.sub(r"[^\w\s]", "", w.word.lower().strip())
            if clean_word in filler_set:
                w.is_filler = True
                occurrences.append(
                    FillerWordOccurrence(
                        filler=clean_word,
                        start=round(w.start, 2),
                        end=round(w.end, 2),
                        context=segment_text.strip(),
                        confidence=round(w.probability, 3),
                    )
                )
    else:
        # Regex search inside segment text
        for filler in filler_set:
            pattern = rf"\b{re.escape(filler)}\b"
            for m in re.finditer(pattern, segment_text, re.IGNORECASE):
                occurrences.append(
                    FillerWordOccurrence(
                        filler=filler.lower(),
                        start=round(seg_start, 2),
                        end=round(seg_end, 2),
                        context=segment_text.strip(),
                        confidence=0.85,
                    )
                )

    return occurrences


def transcribe_audio(
    y: np.ndarray,
    sr: int = 16000,
    model_size: str = "tiny.en",
    filler_set: Optional[Set[str]] = None,
) -> TranscriptResult:
    """
    Transcribe speech using faster-whisper with word timestamps.
    Calculates pacing (WPM), detects filler words, and associates timestamps.
    """
    if filler_set is None:
        filler_set = DEFAULT_ENGLISH_FILLERS

    duration = float(len(y)) / float(sr)
    if duration <= 0:
        return TranscriptResult(
            text="",
            language="en",
            segments=[],
            duration=0.0,
            total_words=0,
            wpm=0.0,
            fillers=[],
            filler_count=0,
            fillers_per_minute=0.0,
        )

    model = get_whisper_model(model_size=model_size)

    # Run faster-whisper on raw float32 array
    segments_gen, info = model.transcribe(
        y,
        beam_size=3,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
    )

    detected_lang = info.language if hasattr(info, "language") else "en"

    segments_list: List[TranscriptSegment] = []
    all_fillers: List[FillerWordOccurrence] = []
    all_text_parts: List[str] = []
    total_words = 0

    for idx, seg in enumerate(segments_gen):
        seg_text = seg.text.strip()
        all_text_parts.append(seg_text)

        word_objs: List[WordTimestamp] = []
        if seg.words:
            for w in seg.words:
                w_obj = WordTimestamp(
                    word=w.word,
                    start=round(float(w.start), 2),
                    end=round(float(w.end), 2),
                    probability=round(float(w.probability), 3),
                )
                word_objs.append(w_obj)
            seg_word_count = len(word_objs)
        else:
            seg_word_count = len(seg_text.split())

        total_words += seg_word_count
        seg_duration = max(0.2, seg.end - seg.start)
        seg_wpm = (seg_word_count / seg_duration) * 60.0

        # Filler detection
        fillers = detect_fillers_in_segment(seg_text, seg.start, seg.end, word_objs, filler_set)
        all_fillers.extend(fillers)

        segments_list.append(
            TranscriptSegment(
                id=idx,
                start=round(float(seg.start), 2),
                end=round(float(seg.end), 2),
                text=seg_text,
                words=word_objs,
                avg_logprob=round(float(seg.avg_logprob), 3) if hasattr(seg, "avg_logprob") else 0.0,
                no_speech_prob=round(float(seg.no_speech_prob), 3) if hasattr(seg, "no_speech_prob") else 0.0,
                wpm=round(seg_wpm, 1),
            )
        )

    full_transcript = " ".join(all_text_parts)
    overall_wpm = (total_words / max(0.1, duration)) * 60.0
    fillers_per_min = (len(all_fillers) / max(0.1, duration)) * 60.0

    return TranscriptResult(
        text=full_transcript,
        language=detected_lang,
        segments=segments_list,
        duration=round(duration, 2),
        total_words=total_words,
        wpm=round(overall_wpm, 1),
        fillers=all_fillers,
        filler_count=len(all_fillers),
        fillers_per_minute=round(fillers_per_min, 1),
    )
