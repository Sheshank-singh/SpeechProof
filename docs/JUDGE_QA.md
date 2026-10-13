# SpeechProof: Hackathon Judge Q&A & Defensible Answers

### Q1: Why not just pass the audio to GPT-4o or Gemini 1.5 Pro and ask it for speech feedback and timestamps?
**Answer**:
Multimodal LLMs are powerful for qualitative summarization, but they have well-documented failure modes in temporal precision:
1. **Timestamp Hallucination**: LLMs frequently invent approximate timestamps (e.g., claiming a pause occurred at 12.0s when the audio waveform shows speech continuously until 14.5s).
2. **Non-Deterministic Scores**: The same recording passed twice with different temperatures produces divergent scores, destroying user trust and reproducibility.
3. **Black-Box Metrics**: An LLM cannot provide the mathematical formula showing how 4 fillers and a 140 WPM pace resulted in a numerical grade.
SpeechProof uses faster-whisper and Librosa for physical, verifiable waveform alignment, reserving deterministic equations for scores.

---

### Q2: How do you differentiate a healthy, deliberate rhetorical pause from an awkward silence?
**Answer**:
SpeechProof implements a two-tier pause evaluation:
1. Natural breathing and rhetorical punctuation pauses ($0.35$s to $1.1$s) are considered normal and are included in the target pause ratio window of 12% to 28%.
2. Dead silences exceeding the configured threshold ($1.4$s) are flagged as prolonged pause defects.
Furthermore, the user can adjust the `long_pause_threshold_sec` in `RubricConfig` depending on whether they are evaluating a rapid pitch or a theatrical keynote.

---

### Q3: Why is active-speech RMS energy used instead of standard global RMS for volume consistency?
**Answer**:
If you calculate standard deviation across all frames of a recording including silences, the frames during pauses have energy around $-100$ dB, creating an artificial standard deviation of $35+$ dB purely due to conversational pauses.
By isolating **active speech frames** (frames within 35 dB of the signal peak), SpeechProof measures the consistency of the speaker's actual voice projection (e.g., diaphragm support and microphone distance) without being skewed by intentional pauses.

---

### Q4: How is monotone speech detected acoustically?
**Answer**:
We apply the YIN fundamental frequency algorithm (`librosa.yin`) to track $F_0$ across voiced frames ($65$ to $350$ Hz). Dynamic human speech exhibits natural pitch modulation with a pitch standard deviation between $20$ Hz and $55$ Hz and a range of $50+$ Hz.
When pitch standard deviation falls below $16.0$ Hz during voiced speech, the vocal folds are exhibiting flat, uninflected prosody, triggering the `monotone_pitch` grounded defect.

---

### Q5: How large is your contrastive dataset and how did you create it?
**Answer**:
The initial benchmark (`SpeechProof-Contrastive-v1`) contains 5 matched pairs (10 audio clips total, 134.6 seconds of speech).
Each pair uses an identical script delivered under two controlled conditions: an ideal baseline and a version with exactly one isolated flaw (fillers, prolonged silences, rushed pace, monotone pitch, volume drop).
This matched-pair design controls for vocabulary, length, and topic, allowing us to evaluate the sensitivity and precision of our flaw grounding algorithms without confounding variables.

---

### Q6: Can SpeechProof run offline without internet or GPU access?
**Answer**:
Yes, 100%. The entire pipeline runs locally in-process on CPU using int8 quantized `faster-whisper` (`tiny.en`), `librosa`, `soundfile`, and `scipy`. First inference takes $\approx 1.5$ seconds for a 15-second clip, and subsequent cached runs take under 500ms. No external cloud endpoints or API keys are required.

---

### Q7: Does your system claim to detect speaker emotion or personality traits?
**Answer**:
No. As documented in `docs/LIMITATIONS.md` and our dataset card, SpeechProof strictly analyzes delivery mechanics (pace, fluency, pausing, pitch modulation, volume). We deliberately avoid making unscientific claims regarding speaker emotion, honesty, psychological state, or demographic identity.

---

### Q8: What happens if someone uploads music, silence, or whispering?
**Answer**:
SpeechProof includes rigorous input validation and data quality handling:
1. Files $<2.0$s or empty buffers are rejected with clear error messages.
2. Recordings with $<5$ words receive an explicit Data Quality Warning.
3. Audio with voiced ratio $<15\%$ (such as whispering) sets the Vocal Variation score confidence to low and issues a warning rather than penalizing the speaker with a false monotone defect.

---

### Q9: How are the category weights in the rubric determined?
**Answer**:
By default, the 5 categories (Pace, Fluency, Pausing, Vocal Variation, Volume Consistency) have equal 20% weights, totaling 100%. However, `RubricConfig` exposes each weight as a configurable parameter, allowing institutions or coaches to prioritize fluency for interview preparation or vocal modulation for theatrical training.

---

### Q10: What are the next technical milestones on your roadmap?
**Answer**:
1. Integration of neural voice activity detection (Silero VAD) to improve silence segmentation in high-noise environments.
2. Expanding the contrastive dataset with diverse accents and multilingual filler dictionaries (e.g., Spanish, Hindi, French).
3. Word-level phoneme articulation scoring using Wav2Vec2 CTC forced alignment for pronunciation feedback.
