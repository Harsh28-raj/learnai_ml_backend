"""Weakness detector. Pure functions over MasteryState lists."""

from collections.abc import Mapping, Sequence

from app.engine.concepts import CONCEPTS, concept_name, prerequisite_layers
from app.engine.learner_model import RELATIVE_WEAKNESS_GAP, MasteryState

STRONG_THRESHOLD = 75
AVERAGE_THRESHOLD = 60
HIGH_PRIORITY = 55  # Section 20: lowered from 60 so a single SPEC §11 miss (56.9) alerts the tutor
MEDIUM_PRIORITY = 40
REVISION_THRESHOLD = 60


def classify(mastery: float, learner_average: float | None = None) -> str:
    """Strong >= 75, Average 60-74, Weak < 60.

    Relative rule (kept from Phase 1): below 75 and at least 15 points under the
    learner's own average practiced mastery also counts as Weak.
    """
    if mastery >= STRONG_THRESHOLD:
        return "Strong"
    if mastery < AVERAGE_THRESHOLD:
        return "Weak"
    if learner_average is not None and mastery <= learner_average - RELATIVE_WEAKNESS_GAP:
        return "Weak"
    return "Average"


def practiced_average(states: Sequence[MasteryState]) -> float | None:
    practiced = [s.mastery for s in states if s.attempts > 0]
    return sum(practiced) / len(practiced) if practiced else None


def _recent_accuracy_fraction(state: MasteryState) -> float:
    if state.recent_results:
        return sum(1 for r in state.recent_results if r.get("correct")) / len(state.recent_results)
    if state.attempts:
        return state.correct / state.attempts
    return 0.0


def priority_score(state: MasteryState) -> float:
    trend_term = 100 if state.trend == "declining" else 50 if state.trend == "stable" else 0
    score = (
        0.45 * (100 - state.mastery)
        + 0.20 * min(state.consecutive_wrong, 3) / 3 * 100
        + 0.15 * (100 - _recent_accuracy_fraction(state) * 100)
        + 0.10 * trend_term
        + 0.10 * min(state.attempts, 10) / 10 * 100
    )
    return round(max(0.0, min(100.0, score)), 1)


def priority_label(score: float) -> str:
    if score >= HIGH_PRIORITY:
        return "HIGH"
    if score >= MEDIUM_PRIORITY:
        return "MEDIUM"
    return "LOW"


def weakness_reason(state: MasteryState) -> str:
    parts = [
        f"mastery {round(state.mastery)}%",
        f"{state.attempts} attempt{'s' if state.attempts != 1 else ''}",
        f"{round(_recent_accuracy_fraction(state) * 100)}% recent accuracy",
    ]
    if state.consecutive_wrong:
        n = state.consecutive_wrong
        parts.append(f"{n} consecutive mistake{'s' if n != 1 else ''}")
    parts.append(f"trend {state.trend}")
    text = ", ".join(parts)
    return text[0].upper() + text[1:]


def tutor_alert(state: MasteryState, label: str) -> str | None:
    if label != "HIGH":
        return None
    return f"Your tutor noticed you're struggling with {concept_name(state.concept_id)}."


def recommended_revision(concept_id: str, states_by_id: Mapping[str, MasteryState]) -> dict:
    """Closest prerequisite layer that has a gap (< 60), weakest first; else the concept itself.

    Unpracticed prerequisites count too: a low prior means it was never learned.
    """
    for layer in prerequisite_layers(concept_id):
        gaps = [states_by_id[c] for c in layer if c in states_by_id and states_by_id[c].mastery < REVISION_THRESHOLD]
        if gaps:
            weakest = min(gaps, key=lambda s: s.mastery)
            return _revision(weakest.concept_id, weakest.mastery, is_prerequisite=True)
    own = states_by_id.get(concept_id)
    return _revision(concept_id, own.mastery if own else None, is_prerequisite=False)


def _revision(concept_id: str, mastery: float | None, is_prerequisite: bool) -> dict:
    return {
        "concept": concept_id,
        "name": concept_name(concept_id),
        "mastery": round(mastery, 1) if mastery is not None else None,
        "is_prerequisite": is_prerequisite,
    }


def assess_concept(state: MasteryState, states_by_id: Mapping[str, MasteryState],
                   learner_average: float | None) -> dict:
    priority = priority_score(state)
    label = priority_label(priority)
    return {
        "concept": state.concept_id,
        "name": concept_name(state.concept_id),
        "category": state.category,
        "mastery": round(state.mastery, 1),
        "attempts": state.attempts,
        "classification": classify(state.mastery, learner_average),
        "priority": priority,
        "label": label,
        "reason": weakness_reason(state),
        "tutor_alert": tutor_alert(state, label),
        "recommended_revision": recommended_revision(state.concept_id, states_by_id),
    }


def build_weakness_report(states: Sequence[MasteryState]) -> dict:
    """{strong, average, weak, untried}. Only practiced concepts are classified."""
    states_by_id = {s.concept_id: s for s in states}
    avg = practiced_average(states)
    report: dict[str, list] = {"strong": [], "average": [], "weak": [], "untried": []}
    for s in states:
        if s.attempts == 0:
            report["untried"].append(s.concept_id)
            continue
        entry = assess_concept(s, states_by_id, avg)
        report[entry["classification"].lower()].append(entry)
    report["strong"].sort(key=lambda e: -e["mastery"])
    report["average"].sort(key=lambda e: -e["priority"])
    report["weak"].sort(key=lambda e: -e["priority"])
    return report


def concept_assessment(concept_id: str, states: Sequence[MasteryState]) -> dict | None:
    """Assessment for a single concept within the learner's full state list."""
    states_by_id = {s.concept_id: s for s in states}
    state = states_by_id.get(concept_id)
    if state is None or concept_id not in CONCEPTS:
        return None
    return assess_concept(state, states_by_id, practiced_average(states))
