"""Adaptive decision: turn a scored answer/quiz into an AdaptiveEvent (SPEC 8.2). Pure."""

from datetime import datetime, timedelta, timezone

from app.engine.concepts import concept_name, dependents_of
from app.engine.learner_model import DIFFICULTIES

REDUCE_BELOW = 60
INCREASE_AT = 85


def threshold_action(score: float) -> str:
    """SPEC 8.1: < 60 reduced, 60-84 maintained, >= 85 increased."""
    if score < REDUCE_BELOW:
        return "reduced"
    if score < INCREASE_AT:
        return "maintained"
    return "increased"


def difficulty_action(current: str, nxt: str) -> str:
    delta = DIFFICULTIES.index(nxt) - DIFFICULTIES.index(current)
    return "reduced" if delta < 0 else "increased" if delta > 0 else "maintained"


def decide_action(score: float, current: str, next_diff: dict) -> tuple[str, bool]:
    """Return (action, overridden). The difficulty engine wins when it disagrees,
    except when it simply cannot move further (already at Easy / Hard)."""
    by_score = threshold_action(score)
    nxt = next_diff["difficulty"]
    by_difficulty = difficulty_action(current, nxt)
    if by_score == by_difficulty:
        return by_score, False
    at_floor = by_score == "reduced" and current == nxt == DIFFICULTIES[0]
    at_ceiling = by_score == "increased" and current == nxt == DIFFICULTIES[-1]
    if at_floor or at_ceiling:
        return by_score, False
    return by_difficulty, True


def human_timestamp(dt: datetime, now: datetime | None = None, utc_offset_minutes: int = 0) -> str:
    """'Today at 2:15 PM', 'Yesterday at 9:05 AM' or 'Oct 1 at 4:40 PM' in the viewer's offset."""
    tz = timezone(timedelta(minutes=utc_offset_minutes))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(tz)
    today = (now or datetime.now(timezone.utc)).astimezone(tz).date()
    hour = local.hour % 12 or 12
    clock = f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"
    if local.date() == today:
        day = "Today"
    elif local.date() == today - timedelta(days=1):
        day = "Yesterday"
    else:
        day = f"{local.strftime('%b')} {local.day}"
    return f"{day} at {clock}"


def build_adaptive_decision(
    concept_id: str,
    score: float,
    current_difficulty: str,
    next_diff: dict,
    assessment: dict | None,
    mastered: frozenset[str] | set[str] = frozenset(),
) -> dict:
    """Return {action, reason, recommendation, path_adjustment, overridden, threshold_action}.

    `assessment` is the weakness-detector entry for the concept (classification,
    label, reason, recommended_revision) or None. `mastered` holds concept ids the
    learner already masters, so they are not announced as newly unlocked.
    """
    name = concept_name(concept_id)
    score_i = round(score)
    action, overridden = decide_action(score, current_difficulty, next_diff)
    by_score = threshold_action(score)
    nxt = next_diff["difficulty"]

    if by_score == "reduced":
        reason = f"You scored {score_i}% on {name}, below the {REDUCE_BELOW}% threshold."
    elif by_score == "maintained":
        reason = f"You scored {score_i}% on {name}, inside the {REDUCE_BELOW}-{INCREASE_AT - 1}% on-track band."
    else:
        reason = f"You scored {score_i}% on {name}, at or above the {INCREASE_AT}% mastery threshold."
    if assessment:
        reason += (f" {name} is now rated {assessment['classification']} "
                   f"({assessment['label']} priority: {assessment['reason'].lower()}).")
    if overridden:
        reason += f" The score alone suggests '{by_score}', but the difficulty engine decided '{action}': {next_diff['reason']}"

    revision = (assessment or {}).get("recommended_revision") or {}
    prereq_gap = bool(revision.get("is_prerequisite"))

    if action == "reduced":
        recommendation = f"Next {name} questions drop to {nxt} difficulty."
        if prereq_gap:
            recommendation += f" Revise '{revision['name']}' first; it is a prerequisite you are still shaky on."
            path_adjustment = f"Prerequisite revision '{revision['name']}' injected before '{name}'."
        else:
            recommendation += f" Review the core intuition behind {name} with the tutor before retrying."
            path_adjustment = f"'{name}' marked as adapted: a remediation drill was inserted before moving on."
    elif action == "maintained":
        recommendation = f"Keep practicing {name} at {nxt} difficulty to consolidate."
        if prereq_gap:
            recommendation += f" A quick refresher on '{revision['name']}' would help."
        path_adjustment = f"No roadmap change: '{name}' remains the current focus."
    else:
        recommendation = f"Next {name} questions step up to {nxt} difficulty."
        # Only a genuine high score on a non-weak concept fast-tracks the roadmap. A
        # difficulty step-up while recovering (e.g. Easy -> Medium) leaves the path alone.
        still_weak = bool(assessment) and assessment.get("classification") == "Weak"
        unlock = [d for d in dependents_of(concept_id) if d not in mastered]
        if by_score != "increased" or still_weak:
            path_adjustment = (f"No roadmap change yet: '{name}' stays the current focus at {nxt} difficulty "
                               f"until it is no longer rated Weak." if still_weak else
                               f"No roadmap change: '{name}' stays the current focus at {nxt} difficulty.")
        elif unlock:
            recommendation += f" You're ready to start {concept_name(unlock[0])}."
            path_adjustment = (f"Fast-tracked: redundant '{name}' drills skipped and "
                               f"'{concept_name(unlock[0])}' unlocked early.")
        else:
            path_adjustment = f"Fast-tracked: redundant '{name}' drills skipped."

    return {
        "action": action,
        "reason": reason,
        "recommendation": recommendation,
        "path_adjustment": path_adjustment,
        "overridden": overridden,
        "threshold_action": by_score,
    }
