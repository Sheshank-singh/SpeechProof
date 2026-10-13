# SpeechProof: Limitations & Failure Modes

## 1. Algorithmic & Acoustic Limitations

1. **Acoustic Noise Sensitivity**:
   - In extremely noisy recording environments (e.g., bustling coffee shops or wind noise), energy-based silence segmentation (`librosa.effects.split`) can misinterpret ambient background noise bursts as speech activity.
   - *Mitigation*: The current pipeline applies a top-dB threshold (28 dB) and energy gating; future versions will integrate neural voice activity detection (Silero VAD).

2. **Whisper Hallucination on Non-Speech Audio**:
   - Like all autoregressive speech models, Whisper can hallucinate repetitive text when fed pure music or prolonged silent noise.
   - *Mitigation*: Our loader flags clips with duration $<2.0$s and outputs data quality warnings for low word counts ($<5$ words).

3. **Pitch Estimation on Unvoiced & Whispering Speech**:
   - The YIN fundamental frequency algorithm requires periodic vocal fold vibration. For whispered speech or vocal fry, voiced frame ratios fall below 15%.
   - *Mitigation*: The scoring engine explicitly detects low voiced frame ratios and raises a Data Quality Warning rather than assigning a false monotone penalty.

4. **Genre & Cultural Specificity of Rubric Targets**:
   - Target pacing (125–165 WPM) and pausing targets represent executive presentation norms in American and British English.
   - Fast-paced auctioneer cadences, sports commentary, or deliberate storytelling styles may legitimately diverge from these targets.
   - *Mitigation*: All rubric thresholds in `RubricConfig` are fully configurable parameters.

---

## 2. Dataset Scope Limitations

- The initial contrastive dataset (`SpeechProof-Contrastive-v1`) contains 5 matched pairs (10 clips).
- While achieving 100% precision and recall on these controlled targets, it functions as a **controlled validation benchmark**, not an exhaustive cross-cultural study.
- Cross-accent generalization across non-native English speakers remains a priority for future exploration.

---

## 3. Responsible AI & Prohibited Interpretations

SpeechProof strictly prohibits:
- Using scores to assess speaker intelligence, psychological competence, honesty, or employability.
- Applying the tool in high-stakes automated hiring without human review.
- Interpreting rubric points as medical speech pathology diagnoses.
