# SpeechProof: 3-Minute Video Pitch Script & Shot List

## Video Metadata
- **Target Duration**: 3:00 (180 seconds)
- **Presenter**: Sheshank
- **Track**: Track C — Contrastive Speech Analytics & Temporal Flaw Grounding
- **Visual Format**: Screen recording of live dashboard + presenter audio / webcam PIP

---

## Shot List & Script Progression

### [0:00 – 0:25] The Problem: Why Current Speech Coaching Fails
- **Visual On Screen**: Title slide showing SpeechProof tagline: *"From subjective speech feedback to reproducible, timestamped evidence."* Split-screen showing a mock generic app giving an unhelpful "78% speech confidence" score vs. an LLM hallucinating timestamps.
- **Presenter Spoken Voiceover**:
  > *"Every speech coach and AI presentation tool promises to make you a better speaker. But in practice, you either get a mysterious single-number score like '78% confidence' with zero explanation of what went wrong, or a generic LLM wrapper that hallucinates timestamps that don't match the audio waveform.*
  > *Speakers don't need vague compliments or invented timestamps. They need to know: What happened? Where did it happen? What physical evidence proves it? And how do I fix it? This is why I built SpeechProof."*

---

### [0:25 – 0:50] The Solution: Introducing SpeechProof
- **Visual On Screen**: Transition to SpeechProof Streamlit dashboard Overview screen. Highlight the 5 rubric dimensions and core promise.
- **Presenter Spoken Voiceover**:
  > *"SpeechProof is an evidence-grounded speech analytics platform built for Track C of the Multimodal AI Hackathon. It replaces subjective impressions with deterministic signal processing, local speech recognition, and millisecond-level temporal flaw grounding.*
  > *SpeechProof analyzes real speech audio, evaluates five transparent delivery dimensions—Pace, Fluency, Pausing, Vocal Variation, and Volume Consistency—and outputs structured evidence objects with clickable audio playback."*

---

### [0:50 – 1:30] Live Demonstration: Analysis & Interval Playback
- **Visual On Screen**: Switch to **Speech Analysis** screen. Select `P01: Investor Pitch (Fillers Flawed)`. Click **Run Live SpeechProof Analysis**. Progress bar completes in ~1.5 seconds.
- **Presenter Spoken Voiceover**:
  > *"Let's see it in action. Here is an investor elevator pitch loaded into our live engine. When I click Analyze, faster-whisper transcribes the words with sub-second alignment while Librosa extracts frame-by-frame RMS loudness and YIN pitch modulation.*
  > *Notice our score: 68.4 out of 100. Fluency dropped to 21.2 because the system detected six verbal fillers. Look at our Grounded Flaws tab: every single 'um', 'like', and 'you know' has its exact timestamp, observed threshold, and linguistic context.*
  > *Best of all, I don't have to guess where it occurred—I can click the interval audio player right here to hear the exact three-hundred millisecond slice where 'like' slipped out."*
- **Visual Interaction**: Click play on one of the grounded audio snippet slices. Audio plays cleanly.

---

### [1:30 – 2:05] Technical Architecture & Deterministic Scoring
- **Visual On Screen**: Switch to **About & Architecture** screen showing the Mermaid architecture diagram and mathematical scoring equations.
- **Presenter Spoken Voiceover**:
  > *"Under the hood, SpeechProof operates as a lightweight, modular Python monolith running 100% locally on CPU without remote API keys. Active-speech RMS standard deviation isolates voice projection from room silence, while YIN pitch tracking extracts fundamental frequency standard deviation.*
  > *Crucially, our scoring engine is completely deterministic. We don't ask an LLM to guess a score between 1 and 100. Every single rubric category is calculated via documented, piecewise mathematical transformations of physical measurements. If your recording has low word counts or unvoiced whispers, SpeechProof outputs explicit Data Quality Warnings rather than fabricating false perfection."*

---

### [2:05 – 2:35] Contrastive Dataset & Empirical Benchmark
- **Visual On Screen**: Switch to **Contrastive Comparison** screen. Show `P02: Cloud Architecture (Ideal vs. Pauses Flawed)`. Highlight the -80 pt drop in the Pausing category and the 2 grounded silences. Then switch to **Benchmark & Methodology** screen.
- **Presenter Spoken Voiceover**:
  > *"To validate our system scientifically, I curated SpeechProof-Contrastive-v1: a benchmark of five matched pairs comparing ideal deliveries against deliberate, controlled delivery flaws across identical scripts.*
  > *Here in our Contrastive Comparison tab, look at Cloud Architecture Keynote. In the ideal recording, breathing pauses are natural at seven hundred milliseconds, scoring 91.8. In the flawed recording, two awkward pauses exceeding three seconds were inserted. The system correctly isolated the flaw, grounded both intervals, and penalized Pausing by exactly 80 points.*
  > *Across our full 10-clip benchmark, SpeechProof achieved 100% flaw grounding accuracy with zero false positives, demonstrating an average 15.5 point separation between ideal and flawed deliveries."*

---

### [2:35 – 3:00] Actionable Coaching & Conclusion
- **Visual On Screen**: Show Actionable Practice Drills tab and Export JSON button. Return to presenter/hero view.
- **Presenter Spoken Voiceover**:
  > *"Finally, SpeechProof closes the loop with targeted rehearsal drills. For every grounded defect, it prescribes concrete exercises—like the 'Two-Second Silent Inhale Drill' for fillers or 'Bridge Transitions' for awkward silences.*
  > *SpeechProof proves that speech analytics can be reproducible, transparent, and grounded in physical evidence. The entire repository is open source, fully tested with pytest, and ready for judges to run locally in one command. Thank you."*
