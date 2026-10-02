"""Profile actions: get_profile, reset_learner."""

from typing import Any

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.actions.common import (
    get_learner_or_raise,
    learner_not_found,
    load_state_rows,
    ordered_states,
    parse_payload,
    recent_events,
    to_mastery_state,
)
from app.db import session_scope
from app.engine.learner_model import build_profile_view
from app.engine.weakness import build_weakness_report
from app.models import Learner
from app.schemas import ActionError
from app.seed import DEMO_LEARNERS, reset_demo_learner


class ProfilePayload(BaseModel):
    utc_offset_minutes: int = Field(0, ge=-720, le=840)


def load_profile_data(db: Session, learner_id: str, utc_offset_minutes: int = 0) -> dict:
    learner = get_learner_or_raise(db, learner_id)
    states = ordered_states({cid: to_mastery_state(r) for cid, r in load_state_rows(db, learner_id).items()})
    profile = build_profile_view(learner, states)
    profile["recentAdaptiveEvents"] = recent_events(db, learner_id, 5, utc_offset_minutes)
    return {"profile": profile, "weakness_report": build_weakness_report(states)}


def _get_profile_sync(learner_id: str, payload: dict[str, Any]) -> dict:
    p = parse_payload(ProfilePayload, payload)
    with session_scope() as db:
        return load_profile_data(db, learner_id, p.utc_offset_minutes)


def _reset_learner_sync(learner_id: str, payload: dict[str, Any]) -> dict:
    p = parse_payload(ProfilePayload, payload)
    if learner_id not in DEMO_LEARNERS:
        if _learner_exists_sync(learner_id):
            raise ActionError(
                "NOT_A_DEMO_LEARNER",
                f"'{learner_id}' is not a demo learner; re-run the assessment with overwrite=true instead.",
                {"learner_id": learner_id, "demo_learners": list(DEMO_LEARNERS)},
            )
        raise ActionError("LEARNER_NOT_FOUND", f"No learner with id '{learner_id}'.", {"learner_id": learner_id})
    with session_scope() as db:
        reset_demo_learner(db, learner_id)
        db.flush()
        return {"reset": True, **load_profile_data(db, learner_id, p.utc_offset_minutes)}


async def get_profile(learner_id: str, payload: dict[str, Any]) -> dict:
    return await run_in_threadpool(_get_profile_sync, learner_id, payload)


async def reset_learner(learner_id: str, payload: dict[str, Any]) -> dict:
    return await run_in_threadpool(_reset_learner_sync, learner_id, payload)


def _learner_exists_sync(learner_id: str) -> bool:
    with session_scope() as db:
        return db.get(Learner, learner_id) is not None


async def ensure_learner_exists(learner_id: str) -> None:
    if not await run_in_threadpool(_learner_exists_sync, learner_id):
        raise learner_not_found(learner_id)
