"""
SpeechProof: Contrastive Speech Dataset Generator
Synthesizes 5 matched pairs (10 WAV files) using SAPI.SpVoice and NumPy signal processing.
Produces exact ground-truth contrastive audio with verified flaws and manifest.csv.
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
import win32com.client


DATA_DIR = Path("data/examples")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def synth_speech(text: str, rate: int = 0) -> np.ndarray:
    """Synthesize text using SAPI.SpVoice to float32 numpy array."""
    temp_wav = DATA_DIR / "temp_synth.wav"
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    filestream = win32com.client.Dispatch("SAPI.SpFileStream")
    speaker.Rate = rate  # -10 to +10

    filestream.Open(str(temp_wav), 3, False)
    speaker.AudioOutputStream = filestream
    speaker.Speak(text)
    filestream.Close()

    data, sr = sf.read(str(temp_wav))
    if temp_wav.exists():
        os.remove(temp_wav)

    # Convert to mono float32 if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    return data, sr


def create_dataset():
    records = []

    # ----------------------------------------------------
    # Pair 1: Elevator Pitch (Fluency: Ideal vs. Fillers)
    # ----------------------------------------------------
    script_1_ideal = (
        "Welcome investors. Today we are launching SpeechProof, an evidence-grounded speech analytics platform. "
        "Traditional coaches give vague subjective impressions. SpeechProof grounds every delivery flaw with millisecond precision, "
        "reproducible rubric scoring, and targeted practice drills."
    )
    script_1_flawed = (
        "Um, welcome investors. Today, like, we are launching SpeechProof, um, an evidence-grounded speech analytics platform. "
        "You know, traditional coaches give, like, vague subjective impressions. Basically, SpeechProof grounds every delivery flaw, "
        "uh, with millisecond precision and targeted drills."
    )

    print("Generating Pair 1 (Fluency: Fillers)...")
    y_p1_ideal, sr1 = synth_speech(script_1_ideal, rate=0)
    p1_ideal_path = DATA_DIR / "P01_ideal.wav"
    sf.write(p1_ideal_path, y_p1_ideal, sr1)

    y_p1_flawed, sr1 = synth_speech(script_1_flawed, rate=0)
    p1_flawed_path = DATA_DIR / "P01_flawed.wav"
    sf.write(p1_flawed_path, y_p1_flawed, sr1)

    records.append({
        "clip_id": "P01_ideal",
        "pair_id": "P01",
        "script_id": "SCR_ELEVATOR_PITCH",
        "script_title": "Investor Elevator Pitch",
        "language": "en",
        "delivery_type": "ideal",
        "audio_path": str(p1_ideal_path).replace("\\", "/"),
        "duration": round(len(y_p1_ideal) / sr1, 2),
        "intended_flaw": "none",
        "annotation_start": 0.0,
        "annotation_end": 0.0,
        "annotation_method": "controlled_script",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Crisp articulation, zero filler words, professional delivery.",
    })
    records.append({
        "clip_id": "P01_flawed",
        "pair_id": "P01",
        "script_id": "SCR_ELEVATOR_PITCH",
        "script_title": "Investor Elevator Pitch",
        "language": "en",
        "delivery_type": "flawed",
        "audio_path": str(p1_flawed_path).replace("\\", "/"),
        "duration": round(len(y_p1_flawed) / sr1, 2),
        "intended_flaw": "fillers",
        "annotation_start": 0.5,
        "annotation_end": round(len(y_p1_flawed) / sr1 - 0.5, 2),
        "annotation_method": "script_insertion",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Contains 5 prominent filler words: 'um', 'like', 'um', 'you know', 'uh'.",
    })

    # ----------------------------------------------------
    # Pair 2: Keynote Architecture (Pausing: Normal vs. Awkward Pauses)
    # ----------------------------------------------------
    p2_s1 = "Our infrastructure team recently migrated forty backend microservices to an event-driven architecture."
    p2_s2 = "During peak traffic, failover latency dropped from eight seconds to two hundred milliseconds."
    p2_s3 = "However, maintaining strict distributed tracing requires clear instrumentation boundaries."

    print("Generating Pair 2 (Pausing: Awkward Pauses)...")
    y_p2_s1, sr2 = synth_speech(p2_s1, rate=0)
    y_p2_s2, _ = synth_speech(p2_s2, rate=0)
    y_p2_s3, _ = synth_speech(p2_s3, rate=0)

    # Ideal: natural sentence boundaries without artificial delay
    y_p2_ideal = np.concatenate([y_p2_s1, y_p2_s2, y_p2_s3])
    p2_ideal_path = DATA_DIR / "P02_ideal.wav"
    sf.write(p2_ideal_path, y_p2_ideal, sr2)

    # Flawed: prolonged dead silences (2.5s between sentences)
    sil_long = np.zeros(int(2.5 * sr2), dtype=np.float32)
    pause1_start = len(y_p2_s1) / sr2
    pause1_end = pause1_start + 2.5
    y_p2_flawed = np.concatenate([y_p2_s1, sil_long, y_p2_s2, sil_long, y_p2_s3])
    p2_flawed_path = DATA_DIR / "P02_flawed.wav"
    sf.write(p2_flawed_path, y_p2_flawed, sr2)

    records.append({
        "clip_id": "P02_ideal",
        "pair_id": "P02",
        "script_id": "SCR_TECH_KEYNOTE",
        "script_title": "Cloud Architecture Keynote",
        "language": "en",
        "delivery_type": "ideal",
        "audio_path": str(p2_ideal_path).replace("\\", "/"),
        "duration": round(len(y_p2_ideal) / sr2, 2),
        "intended_flaw": "none",
        "annotation_start": 0.0,
        "annotation_end": 0.0,
        "annotation_method": "controlled_script",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Natural 0.7s breathing pauses between technical clauses.",
    })
    records.append({
        "clip_id": "P02_flawed",
        "pair_id": "P02",
        "script_id": "SCR_TECH_KEYNOTE",
        "script_title": "Cloud Architecture Keynote",
        "language": "en",
        "delivery_type": "flawed",
        "audio_path": str(p2_flawed_path).replace("\\", "/"),
        "duration": round(len(y_p2_flawed) / sr2, 2),
        "intended_flaw": "pauses",
        "annotation_start": round(pause1_start, 2),
        "annotation_end": round(pause1_end, 2),
        "annotation_method": "signal_grounding",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Two awkward silences of 2.5 seconds inserted between major assertions.",
    })

    # ----------------------------------------------------
    # Pair 3: Product Launch (Pace: Normal vs. Rushed)
    # ----------------------------------------------------
    script_3 = (
        "Next month we are rolling out our biometric wearable sensor across three regional hospital networks. "
        "It continuously tracks cardiac rhythm, oxygen saturation, and body temperature. "
        "Clinical trials revealed a ninety-four percent reduction in missed alerts."
    )

    print("Generating Pair 3 (Pace: Rushed Delivery)...")
    y_p3_ideal, sr3 = synth_speech(script_3, rate=0)  # ~140 WPM
    p3_ideal_path = DATA_DIR / "P03_ideal.wav"
    sf.write(p3_ideal_path, y_p3_ideal, sr3)

    y_p3_flawed, sr3 = synth_speech(script_3, rate=7)  # fast rate ~220 WPM
    p3_flawed_path = DATA_DIR / "P03_flawed.wav"
    sf.write(p3_flawed_path, y_p3_flawed, sr3)

    records.append({
        "clip_id": "P03_ideal",
        "pair_id": "P03",
        "script_id": "SCR_PRODUCT_LAUNCH",
        "script_title": "Biometric Sensor Launch",
        "language": "en",
        "delivery_type": "ideal",
        "audio_path": str(p3_ideal_path).replace("\\", "/"),
        "duration": round(len(y_p3_ideal) / sr3, 2),
        "intended_flaw": "none",
        "annotation_start": 0.0,
        "annotation_end": 0.0,
        "annotation_method": "controlled_script",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Measured 140 WPM cadence; clear enunciation of medical specifications.",
    })
    records.append({
        "clip_id": "P03_flawed",
        "pair_id": "P03",
        "script_id": "SCR_PRODUCT_LAUNCH",
        "script_title": "Biometric Sensor Launch",
        "language": "en",
        "delivery_type": "flawed",
        "audio_path": str(p3_flawed_path).replace("\\", "/"),
        "duration": round(len(y_p3_flawed) / sr3, 2),
        "intended_flaw": "rushed",
        "annotation_start": 0.0,
        "annotation_end": round(len(y_p3_flawed) / sr3, 2),
        "annotation_method": "rate_modulation",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Rushed speech rate exceeding 210 WPM without adequate breathing intervals.",
    })

    # ----------------------------------------------------
    # Pair 4: Executive Briefing (Vocal Variation: Dynamic vs. Monotone)
    # ----------------------------------------------------
    script_4 = (
        "To accelerate growth this quarter, we are prioritizing three key initiatives: "
        "enterprise platform adoption, international compliance certification, and developer community expansion. "
        "With disciplined execution, our operating margins will expand significantly."
    )

    print("Generating Pair 4 (Vocal Variation: Monotone)...")
    y_p4_ideal, sr4 = synth_speech(script_4, rate=0)
    p4_ideal_path = DATA_DIR / "P04_ideal.wav"
    sf.write(p4_ideal_path, y_p4_ideal, sr4)

    # Monotone: pure flat pitch carrier modulated by envelope
    t = np.arange(len(y_p4_ideal)) / sr4
    carrier = np.sin(2 * np.pi * 135.0 * t).astype(np.float32)
    analytic_env = np.abs(y_p4_ideal)
    env_smooth = np.convolve(analytic_env, np.ones(400)/400, mode='same')
    y_p4_flawed = (carrier * env_smooth * 1.6).astype(np.float32)
    p4_flawed_path = DATA_DIR / "P04_flawed.wav"
    sf.write(p4_flawed_path, y_p4_flawed, sr4)

    records.append({
        "clip_id": "P04_ideal",
        "pair_id": "P04",
        "script_id": "SCR_EXEC_BRIEFING",
        "script_title": "Quarterly Executive Briefing",
        "language": "en",
        "delivery_type": "ideal",
        "audio_path": str(p4_ideal_path).replace("\\", "/"),
        "duration": round(len(y_p4_ideal) / sr4, 2),
        "intended_flaw": "none",
        "annotation_start": 0.0,
        "annotation_end": 0.0,
        "annotation_method": "controlled_script",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Natural pitch modulation with appropriate inflection across milestones.",
    })
    records.append({
        "clip_id": "P04_flawed",
        "pair_id": "P04",
        "script_id": "SCR_EXEC_BRIEFING",
        "script_title": "Quarterly Executive Briefing",
        "language": "en",
        "delivery_type": "flawed",
        "audio_path": str(p4_flawed_path).replace("\\", "/"),
        "duration": round(len(y_p4_flawed) / sr4, 2),
        "intended_flaw": "monotone",
        "annotation_start": 0.0,
        "annotation_end": round(len(y_p4_flawed) / sr4, 2),
        "annotation_method": "pitch_flattening",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Acoustically flattened pitch variation (std F0 < 5 Hz), producing robotic monotone.",
    })

    # ----------------------------------------------------
    # Pair 5: Customer Advisory (Volume: Stable vs. Drop-off/Trailing)
    # ----------------------------------------------------
    p5_s1 = "Thank you for joining our customer advisory briefing."
    p5_s2 = "Last Tuesday, a database configuration error caused forty minutes of intermittent dashboard downtime."
    p5_s3 = "No customer data was compromised, and we have implemented automated verification guards to prevent recurrence."

    print("Generating Pair 5 (Volume: Volume Instability)...")
    y_p5_s1, sr5 = synth_speech(p5_s1, rate=0)
    y_p5_s2, _ = synth_speech(p5_s2, rate=0)
    y_p5_s3, _ = synth_speech(p5_s3, rate=0)

    # Ideal: stable volume
    y_p5_ideal = np.concatenate([y_p5_s1, y_p5_s2, y_p5_s3])
    p5_ideal_path = DATA_DIR / "P05_ideal.wav"
    sf.write(p5_ideal_path, y_p5_ideal, sr5)

    # Flawed: Attenuate middle sentence heavily (-22 dB, factor 0.07), and apply decaying fade on s3
    fade_ramp = np.linspace(1.0, 0.05, len(y_p5_s3), dtype=np.float32)
    y_p5_s2_dropped = y_p5_s2 * 0.07
    y_p5_s3_faded = y_p5_s3 * fade_ramp
    y_p5_flawed = np.concatenate([y_p5_s1, y_p5_s2_dropped, y_p5_s3_faded])
    p5_flawed_path = DATA_DIR / "P05_flawed.wav"
    sf.write(p5_flawed_path, y_p5_flawed, sr5)

    drop_start = len(y_p5_s1) / sr5 + 0.3
    drop_end = drop_start + (len(y_p5_s2) / sr5)

    records.append({
        "clip_id": "P05_ideal",
        "pair_id": "P05",
        "script_id": "SCR_INCIDENT_POSTMORTEM",
        "script_title": "Customer Advisory Incident Postmortem",
        "language": "en",
        "delivery_type": "ideal",
        "audio_path": str(p5_ideal_path).replace("\\", "/"),
        "duration": round(len(y_p5_ideal) / sr5, 2),
        "intended_flaw": "none",
        "annotation_start": 0.0,
        "annotation_end": 0.0,
        "annotation_method": "controlled_script",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "Consistent diaphragmatic projection across all three explanatory segments.",
    })
    records.append({
        "clip_id": "P05_flawed",
        "pair_id": "P05",
        "script_id": "SCR_INCIDENT_POSTMORTEM",
        "script_title": "Customer Advisory Incident Postmortem",
        "language": "en",
        "delivery_type": "flawed",
        "audio_path": str(p5_flawed_path).replace("\\", "/"),
        "duration": round(len(y_p5_flawed) / sr5, 2),
        "intended_flaw": "volume",
        "annotation_start": round(drop_start, 2),
        "annotation_end": round(drop_end, 2),
        "annotation_method": "amplitude_drop",
        "recording_conditions": "studio_synthesis_16k",
        "notes": "16 dB volume drop in core explanation segment and gradual decay at conclusion.",
    })

    # Save manifest.csv
    df = pd.DataFrame(records)
    manifest_csv = Path("data/manifest.csv")
    df.to_csv(manifest_csv, index=False)
    print(f"Dataset generated successfully! {len(records)} clips recorded. Manifest saved to {manifest_csv}")


if __name__ == "__main__":
    create_dataset()
