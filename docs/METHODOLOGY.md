# SpeechProof: Scientific Methodology & Evaluation Framework

## 1. Research Motivation

Public speaking and communication coaching tools frequently suffer from two critical shortcomings:
1. **Unsubstantiated Black-Box Scores**: Generic speech tools return an ungrounded overall percentage (e.g., "78% confidence") without detailing which acoustic measurements caused the penalty.
2. **LLM Hallucination of Timestamps**: Wrapping speech transcripts in a large language model prompt and asking it to "find awkward pauses and fillers with timestamps" frequently hallucinates timestamps that do not align with physical acoustic waveforms.

SpeechProof establishes a **reproducible, signal-grounded methodology**:
- Rubric scores are calculated through deterministic mathematical equations operating on verified signal measurements and aligned transcript tokens.
- Delivery flaws are ground-truth verified against physical waveform timestamps.
- Contrastive matched pairs are evaluated to demonstrate that isolated delivery flaws cause targeted, predictable metric and score shifts.

---

## 2. Metric Measurement Methodology

| Metric Name | Measurement Domain | Calculation Method | Target Coaching Band | Primary Failure Mode Addressed |
| :--- | :--- | :--- | :--- | :--- |
| **Speaking Rate (WPM)** | Lexical / Temporal | Total transcribed words divided by total duration (minutes). | 125 – 165 WPM | Rushed cadence ($>175$ WPM) or sluggish dragging ($<110$ WPM). |
| **Filler Word Density** | Lexical Alignment | Word-level Whisper timestamps matched against linguistic filler dictionary. | $\le 1.0$ fillers/min | Hesitations, vocal ticks ("um", "like", "basically"). |
| **Pause Ratio** | Audio Signal | `librosa.effects.split` at 28 dB threshold; non-vocal duration divided by total duration. | 12% – 28% of clip | Dead air, lack of breathing room, or excessive silence. |
| **Awkward Silence Duration** | Audio Signal | Continuous silent interval $> 1.4$s between voiced speech bursts. | 0 occurrences | Audience disengagement during train-of-thought lapses. |
| **Pitch Standard Deviation** | Acoustic F0 | YIN pitch estimation on voiced frames ($65$ to $350$ Hz). | 20 – 55 Hz | Monotone, robotic, unexpressive delivery ($<16$ Hz). |
| **Active Energy Std Dev** | Acoustic RMS | Frame-by-frame RMS (dB) on speech frames within 35 dB of peak. | 2.0 – 7.5 dB | Unstable mic distance, sentence drop-offs, trailing off ($>9.5$ dB). |

---

## 3. Contrastive Evaluation Methodology

To validate whether SpeechProof accurately isolates delivery defects without confounding variables, we designed a **matched-pair contrastive experimental setup**:

### 3.1 Experimental Design
For each presentation script $S_i$, two recordings are created under identical audio capture conditions ($16\text{ kHz mono}$):
1. **Ideal Delivery ($S_i^{\text{ideal}}$)**: Clear articulation, fluent cadence (135–150 WPM), 0 filler words, natural breathing pauses ($0.5$–$0.8$s), dynamic vocal inflection ($F_0\text{ Std} > 30$ Hz), and steady volume.
2. **Flawed Delivery ($S_i^{\text{flawed}}$)**: Exactly one controlled delivery defect is deliberately introduced, holding all other variables constant.

### 3.2 Controlled Variation Classes
- **Pair P01 (Fluency)**: Lexical filler word injection (5 prominent fillers: "um", "like", "you know", "basically", "uh").
- **Pair P02 (Pausing)**: Awkward silence injection (2 distinct 2.5s dead silences between clauses).
- **Pair P03 (Pace)**: Deliberate tempo acceleration to $>215$ WPM without breathing breaks.
- **Pair P04 (Vocal Variation)**: Monotone pitch flattening ($F_0\text{ Std} < 5$ Hz) simulating robotic prosody.
- **Pair P05 (Volume Consistency)**: Diaphragmatic projection drop ($-22$ dB attenuation on core explanatory segment).

### 3.3 Verification Criteria
A contrastive evaluation is scored as a **True Positive (Grounding Success)** if and only if:
1. The intended flaw type is detected in the flawed recording $S_i^{\text{flawed}}$.
2. The intended flaw is absent (or at least strictly fewer in count) in the ideal recording $S_i^{\text{ideal}}$.
3. The corresponding rubric category score in $S_i^{\text{flawed}}$ reflects a significant penalty relative to $S_i^{\text{ideal}}$.
