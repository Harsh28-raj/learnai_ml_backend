"""Action "assessment": cold-start profiling (SPEC 7) -> learner + baseline masteries + first path.

No LLM call except the whyThisPath polish inside the path computation.
"""

import re
from typing import Any, Literal

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete

from app.actions.common import parse_payload
from app.actions.path import compute_path_response
from app.actions.profile import load_profile_data
from app.db import session_scope
from app.engine.cold_start import BASELINE_CONFIDENCE, baseline_masteries, baseline_states, level_from_assessment
from app.engine.concepts import concept_name
from app.engine.goals import GOAL_LABELS, resolve_goal
from app.models import (
    AdaptiveEvent,
    Attempt,
    ChatMessage,
    ConceptState,
    LearningPath,
    Learner,
    QuestionServe,
)
from app.schemas import ActionError
from app.seed import DEMO_LEARNERS, POOL_LEARNERS

LEARNER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
DIAGNOSTIC_COUNT = 3


class AssessmentPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(None, max_length=120)
    experience_level: Literal["Beginner", "Some Programming", "Intermediate", "Advanced"]
    languages: list[str] = Field(default_factory=list, max_length=10)
    topics_known: list[str] = Field(default_factory=list, max_length=30)
    goal: str = Field("ML Engineer", max_length=200)
    pace: Literal["Relaxed", "Balanced", "Intensive"] = "Balanced"
    daily_minutes: int = Field(30, ge=10, le=240)
    overwrite: bool = False


def _upsert_learner(learner_id: str, p: AssessmentPayload) -> dict:
    if learner_id in DEMO_LEARNERS or learner_id in POOL_LEARNERS:
        raise ActionError("DEMO_LEARNER_PROTECTED", f"'{learner_id}' is a built-in learner and cannot be assessed.",
                          {"learner_id": learner_id})
    if not LEARNER_ID_RE.match(learner_id):
        raise ActionError("INVALID_PAYLOAD", "learner_id must be 3-64 chars: lowercase letters, digits, '-' or '_'.")

    level = level_from_assessment(p.experience_level, p.topics_known)
    goal_key = resolve_goal(p.goal)
    masteries = baseline_masteries(p.experience_level, p.languages, p.topics_known)
    fields = {
        "name": (p.name or learner_id).strip(),
        "level": level,
        "role": p.experience_level,
        "goal": p.goal or goal_key,
        "target_role": goal_key,
        "experience": f"Self-assessed: {p.experience_level}",
        "known_languages": [l for l in p.languages if l.strip() and l.strip().lower() != "none"],
        "known_topics": [t for t in p.topics_known if t.strip()],
        "target_topics": [],
        "learning_pace": p.pace,
        "daily_commitment_minutes": p.daily_minutes,
    }
    with session_scope() as db:
        learner = db.get(Learner, learner_id)
        if learner is not None and not p.overwrite:
            raise ActionError("LEARNER_EXISTS", f"Learner '{learner_id}' already exists; send overwrite=true to redo "
                                                "the assessment.", {"learner_id": learner_id})
        if learner is None:
            learner = Learner(id=learner_id, streak_days=0, questions_solved=0, accuracy_rate=0.0, **fields)
            db.add(learner)
        else:
            for model in (Attempt, AdaptiveEvent, ConceptState, ChatMessage, QuestionServe, LearningPath):
                db.execute(delete(model).where(model.learner_id == learner_id))
            for k, v in fields.items():
                setattr(learner, k, v)
            learner.streak_days, learner.questions_solved, learner.accuracy_rate = 0, 0, 0.0
        db.flush()
        for s in baseline_states(masteries):
            db.add(ConceptState(learner_id=learner_id, concept_id=s.concept_id, category=s.category,
                                mastery=s.mastery, attempts=0, correct=0, recent_results=[], consecutive_wrong=0,
                                confidence=BASELINE_CONFIDENCE, trend="stable"))
    return {"level": level, "goal_key": goal_key}


def _profile(learner_id: str) -> dict:
    with session_scope() as db:
        return load_profile_data(db, learner_id)


async def assessment(learner_id: str, payload: dict[str, Any]) -> dict:
    p = parse_payload(AssessmentPayload, payload)
    info = await run_in_threadpool(_upsert_learner, learner_id, p)
    path = await compute_path_response(learner_id)  # saves the first snapshot (+ whyThisPath polish)
    profile = await run_in_threadpool(_profile, learner_id)

    # Most uncertain = unfinished path concepts whose baseline is closest to 50 (coin-flip territory).
    pending = [n for n in path["nodes"] if n["status"] != "completed"]
    uncertain = sorted(pending, key=lambda n: (abs(n["mastery"] - 50), n["order"]))[:DIAGNOSTIC_COUNT]
    ids = [n["conceptId"] for n in uncertain]
    return {
        **profile,
        "path": path,
        "level": info["level"],
        "goal": info["goal_key"],
        "firstStep": {
            "type": "diagnostic",
            "conceptIds": ids,
            "message": (f"Welcome! Your path toward {GOAL_LABELS[info['goal_key']]} is ready. Start with a short "
                        f"diagnostic on {', '.join(concept_name(c) for c in ids)} so we can fine-tune it."
                        if ids else "Welcome! You already know everything on this path; try a Hard practice set."),
        },
    }
