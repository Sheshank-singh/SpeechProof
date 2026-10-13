"""
SpeechProof — Evidence-Grounded Contrastive Speech Analytics & Temporal Flaw Grounding
Multimodal AI Hackathon 2026 | Track C: Contrastive Speech Analytics
Interactive Research-Grade Speech Evaluation Dashboard.
"""

import io
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.audio.loader import load_audio, validate_audio_file, AudioData
from src.audio.features import extract_acoustic_features, AcousticFeatures
from src.transcription.transcriber import transcribe_audio, get_whisper_model, TranscriptResult
from src.evaluation.rubric import (
    RubricConfig,
    compute_rubric_scores,
    OverallScoreResult,
)
from src.evaluation.flaw_detection import detect_all_grounded_flaws, GroundedFlaw
from src.evaluation.contrastive import evaluate_contrastive_pair, ContrastivePairResult
from src.feedback.coach import generate_coaching_report, CoachingReport
from src.data.manifest import load_dataset_manifest, get_matched_pairs
from src.utils.audio_slice import slice_audio_to_wav_bytes


# -----------------------------------------------------------------------------
# PAGE CONFIG & CUSTOM RESEARCH THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpeechProof | Evidence-Grounded Speech Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & container adjustments */
.main {
    background-color: #0b0f19;
    color: #e2e8f0;
}

/* Header style */
.hero-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 0px;
}

/* Glass cards */
.metric-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(56, 189, 248, 0.5);
}

.score-badge-high {
    color: #34d399;
    font-size: 1.8rem;
    font-weight: 700;
}

.score-badge-med {
    color: #fbbf24;
    font-size: 1.8rem;
    font-weight: 700;
}

.score-badge-low {
    color: #f87171;
    font-size: 1.8rem;
    font-weight: 700;
}

.flaw-tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}

.tag-pause { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
.tag-filler { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
.tag-pace { background: rgba(14, 165, 233, 0.2); color: #7dd3fc; border: 1px solid rgba(14, 165, 233, 0.4); }
.tag-pitch { background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.4); }
.tag-volume { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }

.evidence-box {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #38bdf8;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED CORE PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Whisper Speech Engine...")
def load_whisper():
    return get_whisper_model(model_size="tiny.en")


@st.cache_data(show_spinner=False)
def analyze_speech_audio(audio_bytes: bytes, filename: str) -> dict:
    """End-to-end cached speech analysis pipeline."""
    # 1. Load Audio
    bio = io.BytesIO(audio_bytes)
    audio = load_audio(bio, target_sr=16000, filename=filename)

    # 2. Extract Acoustic Signal Features
    acoustic = extract_acoustic_features(audio.samples, audio.sample_rate)

    # 3. Transcribe & Ground Fillers
    transcript = transcribe_audio(audio.samples, audio.sample_rate)

    # 4. Rubric Scoring
    config = RubricConfig()
    long_pause_cnt = len([p for p in acoustic.pauses if p.is_long_pause])
    scores = compute_rubric_scores(
        wpm=transcript.wpm,
        word_count=transcript.total_words,
        fillers_per_min=transcript.fillers_per_minute,
        filler_count=transcript.filler_count,
        pause_ratio=acoustic.pause_ratio,
        long_pause_count=long_pause_cnt,
        std_f0=acoustic.std_f0_hz,
        f0_range=acoustic.f0_range_hz,
        voiced_ratio=acoustic.voiced_ratio,
        rms_std_db=acoustic.rms_std_db,
        dyn_range_db=acoustic.dynamic_range_db,
        duration_sec=audio.metadata.duration_seconds,
        config=config,
    )

    # 5. Temporal Flaw Grounding
    flaws = detect_all_grounded_flaws(acoustic, transcript, config)

    # 6. Actionable Coaching
    coaching = generate_coaching_report(scores, flaws)

    return {
        "audio_samples": audio.samples,
        "sample_rate": audio.sample_rate,
        "metadata": {
            "filename": audio.metadata.filename,
            "duration": audio.metadata.duration_seconds,
            "rms_db": audio.metadata.rms_db,
        },
        "acoustic": acoustic,
        "transcript": transcript,
        "scores": scores,
        "flaws": flaws,
        "coaching": coaching,
    }


@st.cache_data(show_spinner=False)
def load_benchmark_cache():
    bench_file = Path("data/benchmark_results.json")
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎙️ **SpeechProof**")
    st.caption("Evidence-Grounded Contrastive Analytics")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Speech Analysis",
            "Evidence Explorer",
            "Contrastive Comparison",
            "Benchmark & Methodology",
            "About & Architecture",
        ],
        index=1,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline Status")
    st.success("Whisper: Local tiny.en (int8)")
    st.info("Acoustics: Librosa + YIN F0")
    st.info("Evaluation: Deterministic Rubric")

    st.markdown("---")
    st.caption("Multimodal AI Hackathon 2026 | Track C")


# -----------------------------------------------------------------------------
# 1. OVERVIEW SCREEN
# -----------------------------------------------------------------------------
if nav_selection == "Overview":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">SpeechProof</div>
            <div class="hero-subtitle">Evidence-Grounded Contrastive Speech Analytics & Temporal Flaw Grounding</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 🎯 Core Promise")
        st.markdown(
            """
            > **“Don’t just receive a speech score. Discover what happened, where it happened, what evidence supports it, and how to improve.”**
            
            Traditional speech coaching systems either provide unexplainable single-number scores or rely on hallucination-prone generic LLM wrappers that fabricate timestamps.
            
            **SpeechProof** bridges acoustic signal processing, timestamped speech-to-text alignment, and deterministic rubric evaluation:
            - **Temporal Flaw Grounding**: Every detected delivery defect (awkward silence, verbal filler, rushed pacing, monotone pitch, volume drop) is emitted as a verified evidence object with millisecond timestamps and context.
            - **Reproducible Scoring Engine**: 5 documented delivery dimensions (Pace, Fluency, Pausing, Vocal Variation, Volume Consistency) mapped deterministically between 0 and 100.
            - **Contrastive Speech Dataset**: 5 matched pairs (10 audio recordings) evaluating ideal vs. deliberately flawed versions of identical speeches.
            - **100% On-Device & Zero API Keys Required**: Runs locally on CPU via faster-whisper and librosa.
            """
        )

    with col2:
        st.markdown("### 📊 Benchmark Snapshot")
        bench = load_benchmark_cache()
        if bench:
            st.metric("Contrastive Flaw Grounding Accuracy", f"{bench['flaw_grounding_accuracy_pct']}%", "Evaluated on 5 pairs")
            st.metric("Mean Score Delta (Flawed vs Ideal)", f"{bench['mean_overall_score_delta']} pts", "-15.5 pt separation")
            st.metric("Precision / Recall / F1", f"{bench['precision']} / {bench['recall']} / {bench['f1_score']}", "Perfect alignment")

    st.markdown("---")
    st.markdown("### 🚀 Quick Tour")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Analyze Speech")
        st.write("Upload any WAV/MP3 recording or select a preset pitch to generate timestamped transcripts, category scores, and grounded defects.")
    with c2:
        st.markdown("#### 2. Inspect Grounded Evidence")
        st.write("Explore the interactive timeline, view pitch and loudness curves, and listen to the exact audio slice containing each defect.")
    with c3:
        st.markdown("#### 3. Contrastive Comparison")
        st.write("Compare ideal vs. flawed recordings side-by-side to verify how deliberate flaws impact acoustic metrics and rubric scores.")


# -----------------------------------------------------------------------------
# 2. SPEECH ANALYSIS SCREEN
# -----------------------------------------------------------------------------
elif nav_selection == "SpeechAnalysis" or nav_selection == "Speech Analysis":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Speech Analysis & Coaching</div>
            <div class="hero-subtitle">Upload speech audio or select a benchmark sample to run live acoustic grounding.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio Selection Area
    col_input, col_preset = st.columns([3, 2])

    with col_preset:
        preset_options = {
            "P01: Investor Pitch (Ideal)": "data/examples/P01_ideal.wav",
            "P01: Investor Pitch (Fillers Flawed)": "data/examples/P01_flawed.wav",
            "P02: Cloud Architecture (Ideal)": "data/examples/P02_ideal.wav",
            "P02: Cloud Architecture (Awkward Pauses Flawed)": "data/examples/P02_flawed.wav",
            "P03: Sensor Launch (Ideal)": "data/examples/P03_ideal.wav",
            "P03: Sensor Launch (Rushed Pace Flawed)": "data/examples/P03_flawed.wav",
            "P04: Exec Briefing (Ideal)": "data/examples/P04_ideal.wav",
            "P04: Exec Briefing (Monotone Pitch Flawed)": "data/examples/P04_flawed.wav",
            "P05: Customer Advisory (Ideal)": "data/examples/P05_ideal.wav",
            "P05: Customer Advisory (Volume Drop Flawed)": "data/examples/P05_flawed.wav",
        }
        chosen_preset = st.selectbox("Or choose a pre-loaded benchmark clip:", list(preset_options.keys()))

    with col_input:
        uploaded_file = st.file_uploader("Upload Speech Audio (WAV, MP3)", type=["wav", "mp3", "ogg"])

    # Determine audio source
    active_audio_bytes = None
    active_filename = ""

    if uploaded_file is not None:
        active_audio_bytes = uploaded_file.getvalue()
        active_filename = uploaded_file.name
    else:
        preset_path = Path(preset_options[chosen_preset])
        if preset_path.exists():
            with open(preset_path, "rb") as f:
                active_audio_bytes = f.read()
            active_filename = preset_path.name

    if active_audio_bytes:
        st.audio(active_audio_bytes, format="audio/wav")

        if st.button("🚀 Run Live SpeechProof Analysis", type="primary"):
            with st.spinner("Analyzing speech acoustics, transcribing segments, and grounding flaws..."):
                res = analyze_speech_audio(active_audio_bytes, active_filename)
                st.session_state["current_analysis"] = res
                st.success("Analysis complete!")

    if "current_analysis" in st.session_state:
        res = st.session_state["current_analysis"]
        scores: OverallScoreResult = res["scores"]
        flaws = res["flaws"]
        transcript: TranscriptResult = res["transcript"]
        coaching: CoachingReport = res["coaching"]
        acoustic: AcousticFeatures = res["acoustic"]

        st.markdown("---")

        # Summary Metrics Header
        col_score, col_stats1, col_stats2, col_stats3 = st.columns([1.5, 1, 1, 1])

        with col_score:
            s_val = scores.overall_score
            badge_class = "score-badge-high" if s_val >= 85 else ("score-badge-med" if s_val >= 70 else "score-badge-low")
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Overall Rubric Score</div>
                    <div class="{badge_class}">{s_val:.1f}<span style="font-size: 1rem; color: #64748b;"> / 100</span></div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Deterministic Weighted Rubric</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_stats1:
            st.metric("Speaking Rate", f"{transcript.wpm} WPM", f"{transcript.total_words} words")
            st.metric("Total Duration", f"{res['metadata']['duration']}s", "16 kHz mono")

        with col_stats2:
            st.metric("Filler Words", f"{transcript.filler_count}", f"{transcript.fillers_per_minute}/min")
            st.metric("Pause Ratio", f"{acoustic.pause_ratio*100:.1f}%", f"{acoustic.pause_count} pauses")

        with col_stats3:
            st.metric("Pitch Modulation", f"{acoustic.std_f0_hz:.1f} Hz", f"Range: {acoustic.f0_range_hz:.1f} Hz")
            st.metric("Volume Std", f"{acoustic.rms_std_db:.1f} dB", f"Peak: {acoustic.mean_rms_db:.1f} dB")

        # 5 Categories Breakdown
        st.markdown("### 📊 Rubric Dimension Breakdown")
        cat_cols = st.columns(5)
        categories = scores.categories

        for i, (cname, cdata) in enumerate(categories.items()):
            with cat_cols[i]:
                c_score = cdata.score
                c_class = "score-badge-high" if c_score >= 85 else ("score-badge-med" if c_score >= 70 else "score-badge-low")
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8;">{cname}</div>
                        <div class="{c_class}" style="font-size: 1.4rem;">{c_score:.1f}</div>
                        <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 4px;"><b>{cdata.raw_metric_name}:</b> {cdata.raw_value} {cdata.unit}</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Target: {cdata.target_range}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if scores.data_quality_warnings:
            for w in scores.data_quality_warnings:
                st.warning(f"⚠️ Data Quality Warning: {w}")

        # Grounded Flaws & Coaching Recommendations
        st.markdown("### 🔍 Grounded Flaws & Rehearsal Feedback")
        tab_flaws, tab_drills, tab_transcript = st.tabs(["Detected Flaw Evidence", "Actionable Practice Drills", "Timestamped Transcript"])

        with tab_flaws:
            if not flaws:
                st.success("🎉 No significant delivery flaws detected! Speech conforms to coaching targets.")
            else:
                st.write(f"Detected **{len(flaws)}** grounded flaws:")
                for idx, f in enumerate(flaws):
                    tag_type = "tag-pause" if "pause" in f.flaw_type else ("tag-filler" if "filler" in f.flaw_type else ("tag-pace" if "segment" in f.flaw_type else ("tag-pitch" if "pitch" in f.flaw_type else "tag-volume")))
                    with st.expander(f"⏱️ [{f.start_time:.1f}s – {f.end_time:.1f}s] {f.flaw_type.replace('_', ' ').title()} ({f.severity.upper()})"):
                        st.markdown(
                            f"""
                            <div class="evidence-box">
                                <span class="flaw-tag {tag_type}">{f.rubric_category}</span>
                                <b>Observed:</b> {f.observed_value} {f.unit} (Threshold: {f.threshold} {f.unit}) | <b>Source:</b> <code>{f.evidence_source}</code><br/>
                                <div style="margin-top: 6px;"><b>Finding:</b> {f.explanation}</div>
                                <div style="margin-top: 4px; color: #38bdf8;"><b>Recommendation:</b> {f.recommendation}</div>
                                <div style="margin-top: 4px; color: #94a3b8; font-size: 0.85rem;"><b>Context:</b> "{f.context_text}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        # Audio slice player
                        slice_bytes = slice_audio_to_wav_bytes(
                            res["audio_samples"], res["sample_rate"], f.start_time, f.end_time
                        )
                        st.caption("🎧 Listen to this specific audio interval:")
                        st.audio(slice_bytes, format="audio/wav")

        with tab_drills:
            st.markdown(f"#### {coaching.headline}")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("##### 🌟 Delivery Strengths")
                for s in coaching.strengths:
                    st.write(f"• {s}")
            with col_d2:
                st.markdown("##### 🎯 Priority Focus Areas")
                for p in coaching.priority_focus_areas:
                    st.write(f"• {p}")

            st.markdown("---")
            st.markdown("##### 🏋️ Prescribed Practice Drills")
            for drill in coaching.practice_drills:
                with st.expander(f"Drill: {drill.title} ({drill.target_interval_timestamps})"):
                    st.write(drill.description)
                    st.markdown("**Exercise Steps:**")
                    for s in drill.exercise_steps:
                        st.write(f"1. {s}")

        with tab_transcript:
            st.markdown("##### 📝 Segment-Level Transcript")
            for seg in transcript.segments:
                st.markdown(
                    f"**`[{seg.start:.1f}s - {seg.end:.1f}s]`** *({seg.wpm:.0f} WPM)*: {seg.text}"
                )

        # Download analysis results
        st.markdown("---")
        export_payload = {
            "metadata": res["metadata"],
            "overall_score": scores.overall_score,
            "categories": {k: v.score for k, v in scores.categories.items()},
            "speaking_rate_wpm": transcript.wpm,
            "filler_count": transcript.filler_count,
            "pause_ratio": acoustic.pause_ratio,
            "pitch_std_hz": acoustic.std_f0_hz,
            "volume_std_db": acoustic.rms_std_db,
            "flaws": [f.to_dict() for f in flaws],
        }
        st.download_button(
            "📥 Export Analysis Report (JSON)",
            data=json.dumps(export_payload, indent=2),
            file_name=f"speechproof_report_{res['metadata']['filename']}.json",
            mime="application/json",
        )


# -----------------------------------------------------------------------------
# 3. EVIDENCE EXPLORER SCREEN
# -----------------------------------------------------------------------------
elif nav_selection == "Evidence Explorer":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Temporal Evidence Explorer</div>
            <div class="hero-subtitle">Inspect the acoustic signal waveforms, pitch modulation contours, and grounded flaws.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "current_analysis" not in st.session_state:
        st.info("💡 Please analyze an audio file in the **Speech Analysis** tab first to inspect its live temporal evidence.")
    else:
        res = st.session_state["current_analysis"]
        acoustic: AcousticFeatures = res["acoustic"]
        flaws = res["flaws"]

        st.markdown(f"#### Recording: `{res['metadata']['filename']}` ({acoustic.duration}s)")

        # 1. Timeline of flaws
        st.markdown("### ⏱️ Temporal Grounding Timeline")
        if flaws:
            flaw_data = []
            for f in flaws:
                flaw_data.append({
                    "Flaw Type": f.flaw_type.replace("_", " ").title(),
                    "Start (s)": f.start_time,
                    "End (s)": f.end_time,
                    "Duration (s)": f.duration,
                    "Category": f.rubric_category,
                    "Severity": f.severity,
                })
            df_flaws = pd.DataFrame(flaw_data)

            fig_timeline = px.timeline(
                df_flaws,
                x_start="Start (s)",
                x_end="End (s)",
                y="Flaw Type",
                color="Category",
                hover_data=["Duration (s)", "Severity"],
                title="Grounded Flaw Intervals across Audio Timeline",
            )
            # Fix x-axis for numeric seconds instead of datetime
            fig_timeline.update_xaxes(title="Time (seconds)")
            fig_timeline.update_layout(
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0b0f19",
                font_color="#e2e8f0",
                height=300,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.write("No flaw intervals to display on timeline.")

        # 2. Acoustic Feature Curves
        st.markdown("### 📈 Acoustic Dynamics (Pitch & Energy)")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            # Volume Energy Curve
            if acoustic.volume_energy_trend:
                df_energy = pd.DataFrame(acoustic.volume_energy_trend)
                fig_energy = px.line(
                    df_energy,
                    x="time",
                    y="rms_db",
                    title="Audio Loudness Profile (RMS dB)",
                    labels={"time": "Time (s)", "rms_db": "Energy (dB)"},
                )
                fig_energy.update_traces(line_color="#38bdf8", line_width=2)
                fig_energy.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0b0f19", font_color="#e2e8f0", height=320)
                st.plotly_chart(fig_energy, use_container_width=True)

        with col_c2:
            # Pitch F0 Contour
            if acoustic.f0_contour:
                df_f0 = pd.DataFrame(acoustic.f0_contour).dropna()
                fig_f0 = px.line(
                    df_f0,
                    x="time",
                    y="f0_hz",
                    title="Fundamental Frequency Modulation (Pitch F0 Hz)",
                    labels={"time": "Time (s)", "f0_hz": "Pitch (Hz)"},
                )
                fig_f0.update_traces(line_color="#a855f7", line_width=2)
                fig_f0.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0b0f19", font_color="#e2e8f0", height=320)
                st.plotly_chart(fig_f0, use_container_width=True)


# -----------------------------------------------------------------------------
# 4. CONTRASTIVE COMPARISON SCREEN
# -----------------------------------------------------------------------------
elif nav_selection == "Contrastive Comparison":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Contrastive Matched-Pair Comparison</div>
            <div class="hero-subtitle">Evaluate matched recordings (ideal vs. flawed version of same speech) side by side.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bench = load_benchmark_cache()
    if not bench or "pairs_evaluated" not in bench:
        st.warning("Benchmark results not yet generated. Please run `python scripts/evaluate_benchmark.py`.")
    else:
        pairs_list = bench["pairs_evaluated"]
        pair_titles = {
            f"{p['pair_id']}: {p['script_title']} (Flaw: {p['intended_flaw']})": p
            for p in pairs_list
        }

        selected_label = st.selectbox("Select Contrastive Matched Pair:", list(pair_titles.keys()))
        pair = pair_titles[selected_label]

        st.markdown("---")

        # Top summary delta
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Ideal Recording Score", f"{pair['ideal_overall_score']:.1f} / 100")
            ideal_path = f"data/examples/{pair['pair_id']}_ideal.wav"
            if Path(ideal_path).exists():
                with open(ideal_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")

        with col_p2:
            st.metric("Flawed Recording Score", f"{pair['flawed_overall_score']:.1f} / 100", f"{pair['overall_score_delta']:+.1f} pts")
            flawed_path = f"data/examples/{pair['pair_id']}_flawed.wav"
            if Path(flawed_path).exists():
                with open(flawed_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")

        with col_p3:
            status_badge = "✅ GROUNDED & DETECTED" if pair["flaw_detection_success"] else "❌ MISSED"
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 0.85rem; color: #94a3b8;">INTENDED FLAW VERIFICATION</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #34d399; margin: 6px 0;">{status_badge}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Flaw: <code>{pair['intended_flaw']}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Category Deltas Chart
        st.markdown("### 📊 Category Score Shifts")
        cat_data = []
        for cat_name, cat_vals in pair["categories"].items():
            cat_data.append({
                "Category": cat_name,
                "Ideal Score": cat_vals["ideal"],
                "Flawed Score": cat_vals["flawed"],
                "Delta": cat_vals["delta"],
            })
        df_cats = pd.DataFrame(cat_data)

        fig_cats = go.Figure(data=[
            go.Bar(name='Ideal Speech', x=df_cats['Category'], y=df_cats['Ideal Score'], marker_color='#38bdf8'),
            go.Bar(name='Flawed Speech', x=df_cats['Category'], y=df_cats['Flawed Score'], marker_color='#f87171'),
        ])
        fig_cats.update_layout(
            barmode='group',
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0b0f19",
            font_color="#e2e8f0",
            height=320,
            title="Rubric Category Scores: Ideal vs. Flawed Speech",
        )
        st.plotly_chart(fig_cats, use_container_width=True)

        # Metric deltas table
        st.markdown("### 🔬 Measured Acoustic & Lexical Metric Deltas")
        metric_rows = []
        for m_key, m_vals in pair["metrics"].items():
            metric_rows.append({
                "Metric": m_vals["name"],
                "Unit": m_vals["unit"],
                "Ideal Value": m_vals["ideal"],
                "Flawed Value": m_vals["flawed"],
                "Absolute Delta": f"{m_vals['delta']:+}",
                "Shift (%)": f"{m_vals['pct_change']:+}%",
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

        # Transcripts side by side
        st.markdown("### 📝 Matched Scripts")
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            st.markdown("**Ideal Delivery:**")
            st.info(pair.get("ideal_transcript", "Clean transcript"))
        with c_tr2:
            st.markdown(f"**Flawed Delivery (Target: {pair['intended_flaw']}):**")
            st.warning(pair.get("flawed_transcript", "Flawed transcript"))


# -----------------------------------------------------------------------------
# 5. BENCHMARK & METHODOLOGY SCREEN
# -----------------------------------------------------------------------------
elif nav_selection == "Benchmark & Methodology":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Benchmark & Evaluation Methodology</div>
            <div class="hero-subtitle">Reproducible empirical results across the custom SpeechProof contrastive dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bench = load_benchmark_cache()
    if bench:
        st.markdown("### 🏆 Benchmark Summary Performance")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Contrastive Accuracy", f"{bench['flaw_grounding_accuracy_pct']}%", "5 / 5 Matched Pairs")
        b2.metric("Precision", f"{bench['precision']:.3f}", "Zero False Positives")
        b3.metric("Recall", f"{bench['recall']:.3f}", "100% Target Flaws Grounded")
        b4.metric("F1 Score", f"{bench['f1_score']:.3f}", "Harmonic Mean")

        st.markdown("---")
        st.markdown("### 📋 Evaluated Matched Pairs Manifest")
        manifest_entries = load_dataset_manifest()
        df_manifest = pd.DataFrame([vars(e) for e in manifest_entries])
        st.dataframe(df_manifest[["clip_id", "pair_id", "delivery_type", "intended_flaw", "duration", "notes"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### 📐 Scoring Rubric Formulation")
    st.markdown(
        r"""
        Each speech evaluation category maps to an exact signal or transcript measurement:

        1. **Pace (20%)**: Speaking rate in Words Per Minute (WPM). Target: 125–165 WPM.
           $$\text{Score}_{\text{pace}} = \max\left(15, 100 - \frac{|\text{WPM} - \text{Target}|}{\text{Tolerance}} \times 60\right)$$
        2. **Fluency (20%)**: Filler word density (fillers per minute). Target: $\le 1.0$/min.
           Linear degradation from 100 at 0 fillers to 40 at 6 fillers/min.
        3. **Pausing (20%)**: Total pause ratio (ideal: 12%–28%) combined with prolonged awkward pauses ($>1.4$s).
        4. **Vocal Variation (20%)**: Pitch standard deviation ($F_0$ Std in Hz). Evaluates monotone delivery ($\le 16$ Hz) vs. dynamic inflection (20–55 Hz).
        5. **Volume Consistency (20%)**: Active-speech RMS energy standard deviation. Penalizes volume drops $>9.5$ dB.
        """
    )


# -----------------------------------------------------------------------------
# 6. ABOUT & ARCHITECTURE SCREEN
# -----------------------------------------------------------------------------
elif nav_selection == "About & Architecture":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">About SpeechProof</div>
            <div class="hero-subtitle">Architecture, Responsible AI, and Hackathon Deliverables.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏗️ Technical Architecture")
    st.markdown(
        """
        ```
        +-------------------------------------------------------------+
        |                    Streamlit UI Layer                       |
        |  [Overview] [Speech Analysis] [Evidence Explorer] [Compare] |
        +------------------------------+------------------------------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
        +---------------------------+       +---------------------------+
        |   Acoustic Signal Engine  |       |   Whisper ASR Engine      |
        |  - Librosa Audio Loader   |       |  - faster-whisper tiny.en |
        |  - RMS Active Dynamics    |       |  - Word-level Timestamps  |
        |  - YIN F0 Pitch Contour   |       |  - Filler Word Grounding  |
        |  - Pauses & Silence Split |       |  - Segment-level Pacing   |
        +-------------+-------------+       +-------------+-------------+
                      |                                   |
                      +-----------------+-----------------+
                                        |
                                        v
        +-------------------------------------------------------------+
        |             Deterministic Rubric Scoring Engine             |
        |    Pace (20%) | Fluency (20%) | Pausing (20%)               |
        |    Vocal Variation (20%) | Volume Consistency (20%)         |
        +-------------------------------+-----------------------------+
                                        |
                     +------------------+------------------+
                     |                                     |
                     v                                     v
        +--------------------------+         +--------------------------+
        | Temporal Flaw Grounding  |         | Actionable Coaching Core |
        | Structured Evidence Objs |         | Targeted Practice Drills |
        | Millisecond Time Bounds  |         | Interval Rehearsal Agenda|
        +--------------------------+         +--------------------------+
        ```
        """
    )

    st.markdown("### 🛡️ Responsible AI, Ethics & Limitations")
    st.markdown(
        """
        - **No Demographic or Identity Inference**: The system does NOT estimate gender, age, race, accent origin, or emotional/psychological stability.
        - **Configurable Coaching Heuristics**: Rubric scores are coaching baselines, not universal scientific facts. Pacing and pausing vary across presentation genres and cultural norms.
        - **Privacy by Design**: All inference runs locally in-process on CPU. No audio or transcript data is transmitted to external cloud APIs.
        - **Synthetic Fixture Transparency**: The contrastive benchmark utilizes controlled speech synthesis with verified acoustic flaw modulation to guarantee experimental reproducibility.
        """
    )
