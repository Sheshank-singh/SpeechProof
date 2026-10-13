# Devpost Project Submission: SpeechProof

## Project Title
**SpeechProof — Evidence-Grounded Contrastive Speech Analytics**

## Tagline
*From subjective speech feedback to reproducible, timestamped evidence.*

---

## 1. The Problem
Public speaking and verbal communication remain vital professional skills, yet existing speech coaching tools suffer from two major flaws:
1. **Subjective, Black-Box Scoring**: Users receive a single ambiguous number (e.g., "78% confidence") without knowing what acoustic behavior triggered the score or where in the speech it occurred.
2. **LLM Timestamp Hallucination**: Generic AI speech wrappers ask large language models to "find awkward pauses and fillers with timestamps," resulting in made-up time intervals that bear no resemblance to physical audio waveforms.

Speakers are left wondering: *What actually went wrong? Where did it happen? What physical evidence proves it? And how do I practice fixing it?*

---

## 2. Our Solution
**SpeechProof** is an evidence-grounded speech evaluation platform built for Track C of the Multimodal AI Hackathon 2026. 

Its core promise:
> **“Don’t just receive a speech score. Discover what happened, where it happened, what evidence supports it, and how to improve.”**

SpeechProof combines **local speech-to-text recognition**, **physical acoustic signal processing**, and a **transparent deterministic rubric engine** to:
1. Ground every delivery defect with millisecond timestamps and context.
2. Let users click any defect to listen to the exact audio slice.
3. Compute reproducible 0–100 scores across 5 core delivery dimensions (Pace, Fluency, Pausing, Vocal Variation, Volume Consistency).
4. Contrast matched pairs of ideal vs. flawed speeches side-by-side.
5. Provide grounded rehearsal drills directly linked to observed intervals.

---

## 3. How It Works
1. **Audio Ingestion**: Audio (WAV, MP3) is ingested, validated, downmixed to mono, and resampled to 16 kHz.
2. **Dual-Stream Analysis**:
   - *Acoustic Stream*: Librosa extracts frame-by-frame active RMS energy dynamics, YIN fundamental frequency ($F_0$) pitch contours, and silent pause intervals.
   - *Speech Recognition Stream*: A local `faster-whisper` model (`tiny.en`, int8 quantized) generates segment-level and word-level timestamps, cadence measurements, and lexical filler word matches.
3. **Temporal Flaw Grounding**: The engine identifies prolonged silences ($>1.4$s), verbal fillers ("um", "like", "you know"), tempo surges ($>165$ WPM), monotone pitch flattening ($F_0\text{ Std} < 16$ Hz), and volume drop-offs ($>9.5$ dB active RMS std).
4. **Deterministic Rubric Scoring**: Five documented 0–100 categories compute transparent scores based on exact equations, outputting data quality warnings if audio is clipped or unvoiced.
5. **Interactive Dashboard & Rehearsal**: The Streamlit interface displays score gauges, dynamic Plotly pitch/volume contours, an interactive Gantt timeline of defects, interval audio players, and step-by-step practice drills.

---

## 4. What Makes It Different
| Feature | Traditional Speech Coaches | Generic LLM Wrappers | **SpeechProof** |
| :--- | :--- | :--- | :--- |
| **Flaw Timestamps** | None or vague | Hallucinated / Approximate | **Waveform-Grounded (Millisecond precision)** |
| **Scoring Engine** | Proprietary black box | Uncontrolled LLM prompt output | **Deterministic 0–100 equations** |
| **Evidence Playback** | Full audio only | None | **Instant in-memory audio interval slicing** |
| **API Key Dependency**| Cloud-hosted subscription | Requires OpenAI/Anthropic API | **100% On-Device & Zero API Keys** |
| **Contrastive Verification**| Not supported | Not supported | **Side-by-side matched pair delta analytics** |

---

## 5. Technical Architecture
Built as a high-performance Python monolith:
- **Frontend / Interface**: Streamlit with custom CSS research theme & interactive Plotly data visualizations.
- **Speech Recognition**: Local `faster-whisper` (`tiny.en`) with int8 CPU execution and word-level timestamping.
- **Signal Processing**: `librosa`, `soundfile`, `scipy`, and `numpy` for active RMS energy curves, YIN pitch tracking, and silence segmentation.
- **Evaluation Engine**: Modular dataclass architecture (`rubric.py`, `flaw_detection.py`, `contrastive.py`).
- **Testing**: Comprehensive 11-test automated suite using `pytest`.

---

## 6. Contrastive Dataset Methodology
We constructed **`SpeechProof-Contrastive-v1`**, a controlled benchmark containing 5 matched pairs (10 audio recordings) comparing ideal vs. deliberately flawed delivery of identical scripts:
- **Pair P01**: Investor Pitch — Ideal vs. Verbal Fillers ("um", "like", "you know", "basically").
- **Pair P02**: Cloud Architecture Keynote — Ideal vs. Prolonged Awkward Silences ($>3.2$s dead air).
- **Pair P03**: Biometric Sensor Launch — Ideal vs. Hyper-Rushed Tempo ($>265$ WPM).
- **Pair P04**: Executive Strategy Briefing — Ideal vs. Monotone Robotic Delivery ($F_0\text{ Std} = 0$ Hz).
- **Pair P05**: Customer Advisory Postmortem — Ideal vs. Diaphragmatic Volume Drops ($-22$ dB fade).

---

## 7. Evaluation & Results
Running the automated benchmark evaluator (`scripts/evaluate_benchmark.py`) yielded:
- **Contrastive Grounding Accuracy**: **100.0%** (5 / 5 target flaws detected with 0 false positives).
- **Precision / Recall / F1**: **1.000 / 1.000 / 1.000**.
- **Mean Score Delta ($\Delta$)**: **-15.5 points** separation between ideal and flawed deliveries.
- **Execution Speed**: Full 10-clip benchmark evaluates in under 30 seconds on standard CPU.

---

## 8. Responsible AI & Limitations
- **Zero Demographic Bias**: SpeechProof does not attempt to predict speaker gender, age, ethnicity, dialect, or emotional state.
- **Privacy by Design**: All inference runs locally in-process; no audio or transcript ever leaves the user's computer.
- **Transparent Heuristics**: Scoring targets represent configurable coaching guidelines, not universal truths.

---

## 9. Instructions for Judges to Run the Project

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Git

### Quickstart Commands
```bash
# 1. Clone repository
git clone https://github.com/[YOUR_USERNAME]/SpeechProof.git
cd SpeechProof

# 2. Install dependencies (Streamlit, librosa, faster-whisper, plotly, pytest)
pip install -r requirements.txt

# 3. Run automated test suite
python -m pytest -v

# 4. Run contrastive benchmark evaluation
python scripts/evaluate_benchmark.py

# 5. Launch interactive web dashboard
streamlit run app.py
```
*Open `http://localhost:8501` in your browser.*

---

## 10. Submission Links
- **GitHub Repository**: `[INSERT GITHUB REPOSITORY URL]`
- **Demo Video (YouTube)**: `[INSERT UNLISTED YOUTUBE DEMO URL]`
- **Team**: Sheshank (Solo Developer)
