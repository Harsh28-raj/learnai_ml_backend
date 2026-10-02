"""Helpers shared by action handlers (DB <-> engine conversions, payload parsing)."""

from datetime import datetime, timezone
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.engine.adaptation import human_timestamp
from app.engine.concepts import CONCEPTS
from app.engine.learner_model import MasteryState
from app.models import AdaptiveEvent, ConceptState, Learner
from app.schemas import ActionError

T = TypeVar("T", bound=BaseModel)

CATALOG_ORDER = {cid: i for i, cid in enumerate(CONCEPTS)}


def learner_not_found(learner_id: str) -> ActionError:
    return ActionError("LEARNER_NOT_FOUND", f"No learner with id '{learner_id}'.", {"learner_id": learner_id})


def get_learner_or_raise(db: Session, learner_id: str) -> Learner:
    learner = db.get(Learner, learner_id)
    if learner is None:
        raise learner_not_found(learner_id)
    return learner


def parse_payload(model: type[T], payload: dict[str, Any]) -> T:
    try:
        return model.model_validate(payload)
    except ValidationError as e:
        raise ActionError(
            "INVALID_PAYLOAD",
            "Payload does not match the expected shape for this action.",
            [{"loc": list(err.get("loc", [])), "msg": err.get("msg")} for err in e.errors()],
        )


def to_mastery_state(row: ConceptState) -> MasteryState:
    return MasteryState(
        concept_id=row.concept_id,
        category=row.category,
        mastery=row.mastery,
        attempts=row.attempts,
        correct=row.correct,
        recent_results=list(row.recent_results or []),
        consecutive_wrong=row.consecutive_wrong,
        confidence=row.confidence,
        trend=row.trend,
    )


def apply_state(row: ConceptState, s: MasteryState) -> None:
    row.mastery = s.mastery
    row.attempts = s.attempts
    row.correct = s.correct
    row.recent_results = list(s.recent_results)
    row.consecutive_wrong = s.consecutive_wrong
    row.confidence = s.confidence
    row.trend = s.trend


def load_state_rows(db: Session, learner_id: str) -> dict[str, ConceptState]:
    rows = db.scalars(select(ConceptState).where(ConceptState.learner_id == learner_id)).all()
    return {r.concept_id: r for r in rows}


def ordered_states(states: dict[str, MasteryState]) -> list[MasteryState]:
    return sorted(states.values(), key=lambda s: CATALOG_ORDER.get(s.concept_id, len(CATALOG_ORDER)))


def as_utc(dt: datetime) -> datetime:
    # SQLite hands back naive datetimes; everything we store is UTC.
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def event_view(row: AdaptiveEvent, utc_offset_minutes: int = 0) -> dict:
    """SPEC 8.2 AdaptiveEvent (camelCase) plus ISO createdAt."""
    created = as_utc(row.created_at)
    return {
        "id": f"adapt-{row.id}",
        "timestamp": human_timestamp(created, utc_offset_minutes=utc_offset_minutes),
        "createdAt": created.isoformat(),
        "topic": row.topic,
        "score": round(row.score),
        "action": row.action,
        "reason": row.reason,
        "recommendation": row.recommendation,
        "pathAdjustment": row.path_adjustment,
    }


def recent_events(db: Session, learner_id: str, limit: int = 5, utc_offset_minutes: int = 0) -> list[dict]:
    rows = db.scalars(
        select(AdaptiveEvent)
        .where(AdaptiveEvent.learner_id == learner_id)
        .order_by(AdaptiveEvent.created_at.desc(), AdaptiveEvent.id.desc())
        .limit(limit)
    ).all()
    return [event_view(r, utc_offset_minutes) for r in rows]
