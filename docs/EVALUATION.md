# SpeechProof: Empirical Evaluation & Benchmark Report

## 1. Executive Summary

The SpeechProof benchmark was executed across the full contrastive evaluation dataset (`data/manifest.csv`) containing 5 matched pairs (10 distinct speech audio recordings).

| Benchmark Metric | Measured Result | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Total Speech Clips Evaluated** | 10 | 10 | Complete |
| **Total Matched Contrastive Pairs** | 5 | 5 | Complete |
| **Flaw Grounding Accuracy** | **100.0%** (5 / 5) | $\ge 80.0\%$ | **PASS** |
| **Flaw Detection Precision** | **1.000** | $\ge 0.850$ | **PASS** |
| **Flaw Detection Recall** | **1.000** | $\ge 0.850$ | **PASS** |
| **Flaw Detection F1 Score** | **1.000** | $\ge 0.850$ | **PASS** |
| **Mean Overall Score Delta ($\Delta$)** | **-15.5 pts** | $\le -10.0$ pts | **PASS** |

---

## 2. Granular Matched-Pair Evaluation Results

### Pair P01: Investor Elevator Pitch (Fluency Defect: Verbal Fillers)
- **Script**: *"Welcome investors. Today we are launching SpeechProof..."*
- **Ideal Recording**: Score = **90.7** | Fillers: 0 | Flaws Grounded: 0
- **Flawed Recording**: Score = **68.4** | Fillers: 6 (20.3/min) | Flaws Grounded: 6
- **Category Impact**: Fluency dropped from **99.5** to **21.2** ($\Delta = -78.3$ pts).
- **Temporal Alignment**: Grounded fillers at `0.2s–0.4s` ("um"), `3.3s–3.6s` ("like"), `7.6s–7.8s` ("um"), `11.8s–12.1s` ("you know"), `14.5s–14.8s` ("basically"), `16.2s–16.4s` ("uh").
- **Grounding Outcome**: **SUCCESS (True Positive)**.

---

### Pair P02: Cloud Architecture Keynote (Pausing Defect: Awkward Silences)
- **Script**: *"Our infrastructure team recently migrated forty backend microservices..."*
- **Ideal Recording**: Score = **91.8** | Awkward Pauses ($>1.4$s): 0 | Pause Ratio: 13.9% | Flaws Grounded: 0
- **Flawed Recording**: Score = **74.5** | Awkward Pauses ($>1.4$s): 2 | Pause Ratio: 31.8% | Flaws Grounded: 2
- **Category Impact**: Pausing dropped from **90.0** to **10.0** ($\Delta = -80.0$ pts).
- **Temporal Alignment**: Grounded awkward silences at `5.9s–9.2s` (duration: 3.28s) and `14.7s–18.0s` (duration: 3.31s).
- **Grounding Outcome**: **SUCCESS (True Positive)**.

---

### Pair P03: Biometric Sensor Launch (Pacing Defect: Rushed Delivery)
- **Script**: *"Next month we are rolling out our biometric wearable sensor..."*
- **Ideal Recording**: Score = **98.2** | Rate: 137.9 WPM | Flaws Grounded: 0
- **Flawed Recording**: Score = **80.4** | Rate: 268.4 WPM | Flaws Grounded: 3
- **Category Impact**: Pace dropped from **100.0** to **15.0** ($\Delta = -85.0$ pts).
- **Temporal Alignment**: Grounded segment surges exceeding target 165 WPM across all three delivery clauses.
- **Grounding Outcome**: **SUCCESS (True Positive)**.

---

### Pair P04: Quarterly Executive Briefing (Vocal Variation Defect: Monotone Delivery)
- **Script**: *"To accelerate growth this quarter, we are prioritizing three key initiatives..."*
- **Ideal Recording**: Score = **89.9** | Pitch $F_0$ Std: 49.6 Hz | Range: 104.2 Hz | Flaws Grounded: 0
- **Flawed Recording**: Score = **78.2** | Pitch $F_0$ Std: 0.0 Hz | Range: 0.0 Hz | Flaws Grounded: 1
- **Category Impact**: Vocal Variation dropped from **90.0** to **30.0** ($\Delta = -60.0$ pts).
- **Temporal Alignment**: Global monotone pitch grounded with standard deviation $0.0$ Hz ($<16.0$ Hz threshold).
- **Grounding Outcome**: **SUCCESS (True Positive)**.

---

### Pair P05: Customer Advisory Postmortem (Volume Consistency Defect: Loudness Drops)
- **Script**: *"Thank you for joining our customer advisory briefing..."*
- **Ideal Recording**: Score = **94.9** | Active RMS Std: 6.5 dB | Range: 22.3 dB | Flaws Grounded: 0
- **Flawed Recording**: Score = **86.7** | Active RMS Std: 10.3 dB | Range: 31.8 dB | Flaws Grounded: 1
- **Category Impact**: Volume Consistency dropped from **95.0** to **54.0** ($\Delta = -41.0$ pts).
- **Temporal Alignment**: Grounded volume instability with active energy standard deviation $10.3$ dB ($>9.5$ dB threshold).
- **Grounding Outcome**: **SUCCESS (True Positive)**.

---

## 3. Confusion Matrix & Evaluation Analysis

```
                      GROUND TRUTH DEFECT PRESENT (Flawed)   GROUND TRUTH DEFECT ABSENT (Ideal)
DETECTION POSITIVE                    5 (TP)                                0 (FP)
DETECTION NEGATIVE                    0 (FN)                                5 (TN)
```

- **Accuracy**: $10 / 10 = 100.0\%$
- **Precision**: $5 / (5 + 0) = 1.000$
- **Recall**: $5 / (5 + 0) = 1.000$
- **Specificity**: $5 / (5 + 0) = 1.000$
- **F1 Score**: $2 \times \frac{1.0 \times 1.0}{1.0 + 1.0} = 1.000$

All measured evaluation metrics are derived directly from the execution log of `scripts/evaluate_benchmark.py` and saved in `data/benchmark_results.json`.
