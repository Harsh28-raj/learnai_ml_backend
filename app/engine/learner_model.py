"""Learner knowledge model. Pure functions only: no DB access, no I/O."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from app.engine.concepts import CATEGORIES, CONCEPTS, lesson_id_for

DIFFICULTIES = ("Easy", "Medium", "Hard")
CORRECT_GAIN = {"Easy": 3.0, "Medium": 5.0, "Hard": 8.0}
WRONG_LOSS = {"Easy": 8.0, "Medium": 6.0, "Hard": 3.0}
FAST_ANSWER_SECONDS = 20
RECENT_WINDOW = 10

MASTERED_THRESHOLD = 85
STRENGTH_THRESHOLD = 75
WEAKNESS_THRESHOLD = 65
# A concept also counts as a (relative) weakness when it sits this far below the
# learner's own average attempted mastery. Without this, advanced learners would
# never have weaknesses (e.g. Elena's RAG chunking at 68% in SPEC Section 6.3).
RELATIVE_WEAKNESS_GAP = 15


@dataclass(frozen=True)
class MasteryState:
    concept_id: str
    category: str
    mastery: float = 0.0
    attempts: int = 0
    correct: int = 0
    recent_results: list[dict] = field(default_factory=list)
    consecutive_wrong: int = 0
    confidence: float = 0.0
    trend: str = "stable"


def normalize_difficulty(difficulty: str) -> str:
    d = str(difficulty).strip().capitalize()
    if d not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty '{difficulty}', expected one of {DIFFICULTIES}")
    return d


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def compute_confidence(attempts: int, consecutive_wrong: int) -> float:
    return min(1.0, attempts / 10) * (1 - 0.15 * min(consecutive_wrong, 3))


def _accuracy(results: Sequence[dict]) -> float:
    return sum(1 for r in results if r.get("correct")) / len(results)


def compute_trend(recent_results: Sequence[dict]) -> str:
    if len(recent_results) < 4:
        return "stable"
    last3 = recent_results[-3:]
    prev3 = recent_results[-6:-3]
    diff = _accuracy(last3) - _accuracy(prev3)
    if diff > 0.15:
        return "improving"
    if diff < -0.15:
        return "declining"
    return "stable"


def recent_accuracy(recent_results: Sequence[dict]) -> int | None:
    if not recent_results:
        return None
    return round(_accuracy(recent_results) * 100)


def update_mastery(
    state: MasteryState,
    is_correct: bool,
    difficulty: str,
    time_taken: float | None,
    ts: str | None = None,
) -> MasteryState:
    """Return a new state after one answered question. Does not mutate `state`."""
    difficulty = normalize_difficulty(difficulty)
    m = state.mastery

    if is_correct:
        delta = CORRECT_GAIN[difficulty] * (1 - m / 100) * 1.5
        delta = max(delta, 1.0)
        if time_taken is not None and time_taken < FAST_ANSWER_SECONDS:
            delta += 1.0
        new_mastery = clamp(m + delta)
        consecutive_wrong = 0
    else:
        delta = WRONG_LOSS[difficulty] * (m / 100) * 1.5
        delta = max(delta, 1.0)
        new_mastery = clamp(m - delta)
        consecutive_wrong = state.consecutive_wrong + 1

    result = {
        "correct": bool(is_correct),
        "difficulty": difficulty,
        "time_taken": time_taken,
        "ts": ts or datetime.now(timezone.utc).isoformat(),
    }
    recent = (list(state.recent_results) + [result])[-RECENT_WINDOW:]
    attempts = state.attempts + 1

    return replace(
        state,
        mastery=round(new_mastery, 2),
        attempts=attempts,
        correct=state.correct + (1 if is_correct else 0),
        recent_results=recent,
        consecutive_wrong=consecutive_wrong,
        confidence=round(compute_confidence(attempts, consecutive_wrong), 3),
        trend=compute_trend(recent),
    )


def category_score(category: str, states: Iterable[MasteryState]) -> float:
    """Attempt-weighted mean mastery for a category (unattempted concepts weigh 1)."""
    total = weight_sum = 0.0
    for s in states:
        if s.category != category:
            continue
        w = s.attempts if s.attempts > 0 else 1
        total += s.mastery * w
        weight_sum += w
    return total / weight_sum if weight_sum else 0.0


def _concept_name(concept_id: str) -> str:
    c = CONCEPTS.get(concept_id)
    return c["name"] if c else concept_id


def _weakness_reason(s: MasteryState, relative_only: bool, avg: float) -> str:
    score = round(s.mastery)
    acc = recent_accuracy(s.recent_results)
    parts = []
    if relative_only:
        parts.append(f"Mastery of {score}% is well below your average of {round(avg)}% across practiced concepts.")
    else:
        parts.append(f"Mastery of {score}% is below the {WEAKNESS_THRESHOLD}% proficiency threshold.")
    if acc is not None:
        parts.append(f"Recent accuracy: {acc}% over the last {len(s.recent_results)} questions.")
    if s.consecutive_wrong >= 2:
        parts.append(f"{s.consecutive_wrong} wrong answers in a row.")
    if s.trend == "declining":
        parts.append("Performance is trending down.")
    elif s.trend == "improving":
        parts.append("Improving, but not yet solid.")
    return " ".join(parts)


def _strength_description(s: MasteryState) -> str:
    acc = recent_accuracy(s.recent_results)
    text = f"Strong command at {round(s.mastery)}% mastery across {s.attempts} practice questions."
    if acc is not None:
        text += f" Recent accuracy {acc}%."
    return text


def find_weaknesses(states: Sequence[MasteryState]) -> list[MasteryState]:
    practiced = [s for s in states if s.attempts > 0]
    if not practiced:
        return []
    avg = sum(s.mastery for s in practiced) / len(practiced)
    weak = [
        s
        for s in practiced
        if s.mastery < WEAKNESS_THRESHOLD
        or (s.mastery < STRENGTH_THRESHOLD and s.mastery <= avg - RELATIVE_WEAKNESS_GAP)
    ]
    return sorted(weak, key=lambda s: s.mastery)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_profile_view(learner: Any, states: Sequence[MasteryState]) -> dict:
    """Frontend `LearnerProfile` (SPEC Section 5) with camelCase keys.

    `learner` may be an ORM object or a dict with snake_case fields.
    """
    practiced = [s for s in states if s.attempts > 0]
    avg = sum(s.mastery for s in practiced) / len(practiced) if practiced else 0.0
    weak_states = find_weaknesses(states)
    weak_ids = {s.concept_id for s in weak_states}

    strengths = [
        {
            "name": _concept_name(s.concept_id),
            "score": round(s.mastery),
            "description": _strength_description(s),
        }
        for s in sorted(practiced, key=lambda s: -s.mastery)
        if s.mastery >= STRENGTH_THRESHOLD and s.concept_id not in weak_ids
    ]
    weaknesses = [
        {
            "name": _concept_name(s.concept_id),
            "score": round(s.mastery),
            "reason": _weakness_reason(s, relative_only=s.mastery >= WEAKNESS_THRESHOLD, avg=avg),
            "recommendedLessonId": lesson_id_for(s.concept_id),
        }
        for s in weak_states
    ]

    return {
        "id": _get(learner, "id"),
        "name": _get(learner, "name"),
        "level": _get(learner, "level"),
        "role": _get(learner, "role"),
        "goal": _get(learner, "goal"),
        "targetRole": _get(learner, "target_role"),
        "experience": _get(learner, "experience"),
        "knownLanguages": list(_get(learner, "known_languages") or []),
        "knownTopics": list(_get(learner, "known_topics") or []),
        "targetTopics": list(_get(learner, "target_topics") or []),
        "learningPace": _get(learner, "learning_pace"),
        "dailyCommitmentMinutes": _get(learner, "daily_commitment_minutes"),
        "streakDays": _get(learner, "streak_days"),
        "questionsSolved": _get(learner, "questions_solved"),
        "accuracyRate": round(_get(learner, "accuracy_rate") or 0),
        "conceptsMastered": sum(1 for s in states if s.mastery >= MASTERED_THRESHOLD),
        "skills": {cat: round(category_score(cat, states)) for cat in CATEGORIES},
        "strengths": strengths,
        "weaknesses": weaknesses,
        "conceptStates": [
            {
                "concept": s.concept_id,
                "mastery": round(s.mastery, 1),
                "attempts": s.attempts,
                "correct": s.correct,
                "recentAccuracy": recent_accuracy(s.recent_results),
                "confidence": round(s.confidence, 2),
                "trend": s.trend,
            }
            for s in states
        ],
    }
