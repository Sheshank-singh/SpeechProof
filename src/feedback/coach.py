"""
SpeechProof: Evidence-Grounded Contrastive Speech Analytics
Deterministic Actionable Coaching & Rehearsal Feedback Engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ..evaluation.rubric import OverallScoreResult, CategoryScore
from ..evaluation.flaw_detection import GroundedFlaw


@dataclass
class PracticeDrill:
    title: str
    target_flaw: str
    description: str
    exercise_steps: List[str]
    target_interval_timestamps: Optional[str] = None


@dataclass
class CoachingReport:
    headline: str
    strengths: List[str]
    priority_focus_areas: List[str]
    practice_drills: List[PracticeDrill]
    timestamped_action_items: List[Dict[str, str]] = field(default_factory=list)


DRILL_LIBRARY = {
    "prolonged_pause": PracticeDrill(
        title="The Bridge Transition Technique",
        target_flaw="prolonged_pause",
        description="Eliminate dead silences between arguments by pre-loading transitional bridges.",
        exercise_steps=[
            "Identify the exact timestamp of the silence in your recording.",
            "Write down a 3-word vocal bridge (e.g., 'Consequently...', 'Notice this distinction...', 'Here is why...').",
            "Re-record the 10-second window until the bridge flows naturally within 0.5s.",
        ],
    ),
    "filler_word": PracticeDrill(
        title="The 2-Second Silent Inhale Drill",
        target_flaw="filler_word",
        description="Rewire the impulse to utter 'um', 'uh', or 'like' into silent breathing pauses.",
        exercise_steps=[
            "Play back the grounded filler timestamps.",
            "Every time you feel the urge to say 'um' or 'like', close your lips and take a silent breath through your nose.",
            "Count to one silently before continuing to your next sentence.",
            "Deliver 30 seconds of speech with 0 fillers, embracing clean silence.",
        ],
    ),
    "rushed_segment": PracticeDrill(
        title="Pacing Anchoring & Syllable Enunciation",
        target_flaw="rushed_segment",
        description="Slow down hyper-fast delivery sections that risk listener comprehension.",
        exercise_steps=[
            "Read the transcript excerpt aloud with a metronome or tap your finger at 140 BPM.",
            "Lengthen the vowels of critical nouns (e.g., 'a-r-c-h-i-t-e-c-t-u-r-e').",
            "Insert a deliberate 0.5s breathing pause between independent clauses.",
        ],
    ),
    "monotone_pitch": PracticeDrill(
        title="The Pitch Staircase Exercise",
        target_flaw="monotone_pitch",
        description="Expand fundamental vocal inflection across key statements and queries.",
        exercise_steps=[
            "Take your opening statement and speak it at 3 distinct pitch levels (low, neutral, high).",
            "Raise your pitch half a step on rhetorical questions.",
            "Drop your pitch to your resonant chest voice when declaring conclusions.",
        ],
    ),
    "volume_instability": PracticeDrill(
        title="Diaphragmatic Projection Stability",
        target_flaw="volume_instability",
        description="Prevent volume drop-offs and trailing off at the ends of sentences.",
        exercise_steps=[
            "Place your hand on your diaphragm to ensure full breath support through the final word of each phrase.",
            "Keep an unvarying 6-inch distance from your microphone.",
            "Avoid letting your vocal energy decay on the last 3 words of a sentence.",
        ],
    ),
}


def generate_coaching_report(
    score_result: OverallScoreResult,
    flaws: List[GroundedFlaw],
) -> CoachingReport:
    """
    Generate deterministic, grounded coaching recommendations directly linked
    to detected measurements and timestamped evidence intervals.
    """
    categories = score_result.categories
    strengths: List[str] = []
    priority_focus: List[str] = []

    # 1. Identify strengths (score >= 85)
    for name, cat in categories.items():
        if cat.score >= 85.0:
            strengths.append(f"{name} ({cat.score:.0f}/100): {cat.explanation}")

    # 2. Identify priority areas (score < 75)
    sorted_cats = sorted(categories.values(), key=lambda c: c.score)
    for cat in sorted_cats:
        if cat.score < 80.0:
            priority_focus.append(
                f"{cat.name} ({cat.score:.0f}/100): Measured {cat.raw_value} {cat.unit} (Target: {cat.target_range}). {cat.explanation}"
            )

    if not priority_focus:
        priority_focus.append("All categories performed within target coaching bands. Focus on continuous micro-refinement.")

    # 3. Compile grounded drills
    drills: List[PracticeDrill] = []
    seen_flaw_types = set()
    action_items: List[Dict[str, str]] = []

    for flaw in flaws:
        if flaw.flaw_type not in seen_flaw_types:
            seen_flaw_types.add(flaw.flaw_type)
            template = DRILL_LIBRARY.get(flaw.flaw_type)
            if template:
                drill = PracticeDrill(
                    title=template.title,
                    target_flaw=flaw.flaw_type,
                    description=template.description,
                    exercise_steps=template.exercise_steps,
                    target_interval_timestamps=f"{flaw.start_time:.1f}s – {flaw.end_time:.1f}s",
                )
                drills.append(drill)

        action_items.append({
            "timestamp": f"{flaw.start_time:.1f}s – {flaw.end_time:.1f}s",
            "category": flaw.rubric_category,
            "flaw": flaw.flaw_type.replace("_", " ").title(),
            "recommendation": flaw.recommendation,
            "context": flaw.context_text or "Audio interval",
        })

    # Overall headline
    if score_result.overall_score >= 88.0:
        headline = "Executive-Grade Delivery: High Polish with Targeted Micro-Refinements"
    elif score_result.overall_score >= 70.0:
        headline = "Strong Communication Foundation: Targeted Flaw Elimination Recommended"
    else:
        headline = "High Improvement Potential: Rehearsal Required on Core Delivery Fundamentals"

    return CoachingReport(
        headline=headline,
        strengths=strengths,
        priority_focus_areas=priority_focus,
        practice_drills=drills,
        timestamped_action_items=action_items[:10],  # Top 10 items
    )
