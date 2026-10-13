# Dataset Card: SpeechProof-Contrastive-v1

## 1. Dataset Overview
- **Dataset Name**: SpeechProof Contrastive Speech Evaluation Dataset (`SpeechProof-Contrastive-v1`)
- **Version**: 1.0.0
- **Curated By**: Sheshank (Solo Developer, Multimodal AI Hackathon 2026)
- **License**: MIT
- **Format**: WAV audio (16,000 Hz, 16-bit PCM mono) + CSV metadata manifest (`data/manifest.csv`)
- **Number of Audio Clips**: 10
- **Number of Matched Pairs**: 5
- **Total Duration**: 134.6 seconds
- **Languages Represented**: English (`en-US`)

---

## 2. Dataset Purpose & Rationale
Speech evaluation models often struggle with confounders: two completely different speakers delivering two completely different speeches cannot be fairly compared to prove whether an algorithm accurately detected an isolated flaw.

`SpeechProof-Contrastive-v1` solves this by introducing **controlled matched pairs**:
- Identical speech scripts are compared side-by-side.
- One version represents an ideal delivery benchmark.
- The paired version incorporates exactly one isolated, controlled delivery flaw.
- This setup enables rigorous verification of temporal flaw grounding, false positive rates, and rubric sensitivity.

---

## 3. Matched Pairs Manifest

| Pair ID | Script ID | Script Genre | Target Flaw | Duration (Ideal) | Duration (Flawed) | Ground Truth Annotation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P01** | `SCR_ELEVATOR_PITCH` | Investor Pitch | Fillers ("um", "like", "you know") | 13.9s | 17.7s | Lexical token insertions with millisecond bounds |
| **P02** | `SCR_TECH_KEYNOTE` | Cloud Architecture | Prolonged Awkward Pauses | 20.2s | 24.3s | 2 awkward silences ($>3.2$s) at 5.9s and 14.7s |
| **P03** | `SCR_PRODUCT_LAUNCH` | Health Tech Pitch | Rushed Cadence ($>210$ WPM) | 17.0s | 7.6s | Accelerated rate factor +7 across all clauses |
| **P04** | `SCR_EXEC_BRIEFING` | Executive Strategy | Monotone Flat Pitch ($F_0$) | 18.2s | 18.2s | Pitch flattened to carrier sine with $F_0\text{ Std} < 5$ Hz |
| **P05** | `SCR_INCIDENT_POSTMORTEM`| Customer Advisory | Volume Instability / Trailing | 18.7s | 18.7s | $-22$ dB attenuation on clause 2 & decaying ramp |

---

## 4. Collection & Annotation Methodology
- **Acoustic Generation**: Synthesized using native 16 kHz studio-quality speech synthesis and NumPy signal modulation.
- **Annotation Method**:
  - Script-level ground truth definitions.
  - Verification via physical acoustic waveform inspection using Librosa and faster-whisper word-level alignment.
- **Reproducibility**: The complete dataset can be regenerated in seconds by executing `python scripts/generate_dataset.py`.

---

## 5. Ethical Considerations & Limitations
- **Consent & Privacy**: All samples were generated procedurally; no unauthorized personal recordings or private conversational audio was harvested.
- **Non-Demographic**: The dataset does not label speaker gender, age, race, or dialect.
- **Limitations**: Sample size is intentionally bounded to 5 controlled pairs (10 clips) as a feasibility benchmark. It demonstrates proof of methodology rather than exhaustive population coverage across global accents.
