"""Action "evaluate": answer(s) -> learner model -> weakness -> difficulty -> AdaptiveEvent.

Two payload shapes:
  single: {question_id, concept_tested, category, difficulty, selected_option_index,
           time_taken_seconds, correct_index?, is_correct?}
  quiz:   {quiz: true, topic, category, answers: [single, ...]}
All DB writes for one call happen in one transaction (session_scope).
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from app.actions.path import record_path_change
from app.actions.common import (
    apply_state,
    event_view,
    get_learner_or_raise,
    load_state_rows,
    ordered_states,
    parse_payload,
    to_mastery_state,
)
from app.db import session_scope
from app.engine.adaptation import build_adaptive_decision
from app.engine.concepts import CONCEPTS, concept_name, resolve_concept, suggest_concepts
from app.engine.difficulty import next_difficulty
from app.engine.learner_model import MASTERED_THRESHOLD, MasteryState, category_score, normalize_difficulty, update_mastery
from app.engine.weakness import build_weakness_report, concept_assessment
from app.models import AdaptiveEvent, Attempt, ConceptState, Question
from app.schemas import ActionError

MAX_QUIZ_ANSWERS = 50


class AnswerIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str | None = Field(None, max_length=80)
    concept_tested: str | None = Field(None, max_length=200)
    category: str | None = Field(None, max_length=40)
    difficulty: str | None = None
    selected_option_index: int | None = Field(None, ge=0)
    time_taken_seconds: float | None = Field(None, ge=0)
    correct_index: int | None = Field(None, ge=0)
    is_correct: bool | None = None


class SinglePayload(AnswerIn):
    utc_offset_minutes: int = Field(0, ge=-720, le=840)


class QuizPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    quiz: Literal[True]
    topic: str | None = Field(None, max_length=200)
    category: str | None = Field(None, max_length=40)
    answers: list[AnswerIn] = Field(..., min_length=1, max_length=MAX_QUIZ_ANSWERS)
    utc_offset_minutes: int = Field(0, ge=-720, le=840)


@dataclass
class GradedAnswer:
    question_id: str | None
    concept_id: str
    difficulty: str
    is_correct: bool
    correct_index: int | None
    time_taken: float | None


def _where(index: int | None) -> dict:
    return {} if index is None else {"answer_index": index}


def _resolve(text: str | None, category: str | None, index: int | None) -> str:
    if not text:
        raise ActionError("INVALID_PAYLOAD", "concept_tested is required (or a quiz topic).", _where(index))
    cid = resolve_concept(text, category)
    if cid is None:
        raise ActionError(
            "UNKNOWN_CONCEPT",
            f"Could not match '{text}' to a known concept.",
            {"concept_tested": text, "suggestions": suggest_concepts(text), **_where(index)},
        )
    return cid


def _grade(db, a: AnswerIn, default_topic: str | None, default_category: str | None,
           index: int | None) -> GradedAnswer:
    stored = db.get(Question, a.question_id) if a.question_id else None

    if stored is not None:
        concept_id = stored.concept_id if stored.concept_id in CONCEPTS else _resolve(
            a.concept_tested or default_topic, a.category or default_category, index)
        raw_difficulty = stored.difficulty
    else:
        concept_id = _resolve(a.concept_tested or default_topic, a.category or default_category, index)
        raw_difficulty = a.difficulty
    if not raw_difficulty:
        raise ActionError("INVALID_PAYLOAD", "difficulty is required (Easy, Medium or Hard).", _where(index))
    try:
        difficulty = normalize_difficulty(raw_difficulty)
    except ValueError as e:
        raise ActionError("INVALID_PAYLOAD", str(e), _where(index))

    # Grading order: stored question -> payload correct_index -> payload is_correct.
    stored_payload = (stored.payload or {}) if stored is not None else {}
    stored_correct = stored_payload.get("correctIndex", stored_payload.get("correct_index"))
    if stored_correct is not None:
        if a.selected_option_index is None:
            raise ActionError("CANNOT_GRADE", "selected_option_index is required for a server-stored question.",
                              {"question_id": a.question_id, **_where(index)})
        correct_index, is_correct = int(stored_correct), a.selected_option_index == int(stored_correct)
    elif a.correct_index is not None and a.selected_option_index is not None:
        correct_index, is_correct = a.correct_index, a.selected_option_index == a.correct_index
    elif a.is_correct is not None:
        correct_index, is_correct = a.correct_index, a.is_correct
    else:
        raise ActionError(
            "CANNOT_GRADE",
            "Cannot grade this answer: question is not stored server-side and the payload has neither "
            "correct_index (with selected_option_index) nor is_correct.",
            {"question_id": a.question_id, **_where(index)},
        )
    return GradedAnswer(a.question_id, concept_id, difficulty, bool(is_correct), correct_index,
                        a.time_taken_seconds)


def _primary_concept(graded: list[GradedAnswer], topic: str | None, category: str | None) -> str:
    touched = [g.concept_id for g in graded]
    topic_id = resolve_concept(topic, category) if topic else None
    if topic_id in touched:
        return topic_id
    counts = Counter(touched)
    best = max(counts.values())
    return next(c for c in touched if counts[c] == best)


def _evaluate_sync(learner_id: str, payload: dict[str, Any], event_score: float | None = None) -> dict:
    """event_score: the real graded score (0-100) when the caller graded free text (tutor evaluate
    mode). Mastery still uses the binary is_correct; the AdaptiveEvent and decision use this score."""
    is_quiz = payload.get("quiz") is True
    if is_quiz:
        p = parse_payload(QuizPayload, payload)
        answers, topic, category, offset = p.answers, p.topic, p.category, p.utc_offset_minutes
    else:
        p = parse_payload(SinglePayload, payload)
        answers, topic, category, offset = [p], None, p.category, p.utc_offset_minutes

    with session_scope() as db:
        learner = get_learner_or_raise(db, learner_id)
        graded = [
            _grade(db, a, topic, category, i if is_quiz else None)
            for i, a in enumerate(answers)
        ]

        rows = load_state_rows(db, learner_id)
        for g in graded:
            if g.concept_id not in rows:  # concept added to the catalog after seeding
                row = ConceptState(learner_id=learner_id, concept_id=g.concept_id,
                                   category=CONCEPTS[g.concept_id]["category"], mastery=0.0,
                                   attempts=0, correct=0, recent_results=[], consecutive_wrong=0,
                                   confidence=0.0, trend="stable")
                db.add(row)
                rows[g.concept_id] = row
        before: dict[str, MasteryState] = {cid: to_mastery_state(r) for cid, r in rows.items()}
        states = dict(before)

        now = datetime.now(timezone.utc)
        last_difficulty: dict[str, str] = {}
        for g in graded:
            states[g.concept_id] = update_mastery(states[g.concept_id], g.is_correct, g.difficulty,
                                                  g.time_taken, ts=now.isoformat())
            last_difficulty[g.concept_id] = g.difficulty
            db.add(Attempt(learner_id=learner_id, concept_id=g.concept_id,
                           category=CONCEPTS[g.concept_id]["category"], difficulty=g.difficulty,
                           is_correct=g.is_correct, time_taken_seconds=g.time_taken,
                           question_id=g.question_id, created_at=now))

        n = len(graded)
        n_correct = sum(1 for g in graded if g.is_correct)
        answer_score = 100 * n_correct / n
        exact_score = float(event_score) if event_score is not None else answer_score
        score = round(exact_score)

        # SPEC 8.3 cumulative accuracy over answers (kept at 1 decimal to avoid rounding drift).
        solved = learner.questions_solved or 0
        learner.accuracy_rate = round(((learner.accuracy_rate or 0) * solved + answer_score * n) / (solved + n), 1)
        learner.questions_solved = solved + n

        touched = list(dict.fromkeys(g.concept_id for g in graded))
        for cid in touched:
            apply_state(rows[cid], states[cid])

        before_list, after_list = ordered_states(before), ordered_states(states)
        mastery_updates = []
        for cid in touched:
            s, cat = states[cid], states[cid].category
            mastery_updates.append({
                "concept": cid,
                "concept_name": concept_name(cid),
                "previous_score": round(before[cid].mastery, 1),
                "new_score": round(s.mastery, 1),
                "category": cat,
                "previous_category_score": round(category_score(cat, before_list), 1),
                "new_category_score": round(category_score(cat, after_list), 1),
                "trend": s.trend,
                "confidence": round(s.confidence, 2),
            })

        primary = _primary_concept(graded, topic, category)
        nd = next_difficulty(states[primary])
        assessment = concept_assessment(primary, after_list)
        mastered = {s.concept_id for s in after_list if s.mastery >= MASTERED_THRESHOLD}
        decision = build_adaptive_decision(primary, exact_score, last_difficulty[primary], nd, assessment, mastered)
        # Real roadmap effect of this answer (replaces the Phase 2 "intended change" text).
        path_summary = record_path_change(db, learner, before_list, after_list)

        event = AdaptiveEvent(
            learner_id=learner_id,
            topic=(topic or concept_name(primary))[:120],
            score=score,
            action=decision["action"],
            reason=decision["reason"],
            recommendation=decision["recommendation"],
            path_adjustment=path_summary,
            created_at=now,
        )
        db.add(event)
        db.flush()

        return {
            "results": [
                {"question_id": g.question_id, "is_correct": g.is_correct, "correct_index": g.correct_index,
                 "concept": g.concept_id}
                for g in graded
            ],
            "score": score,
            "mastery_updates": mastery_updates,
            "weakness_report": build_weakness_report(after_list),
            "next_difficulty": {"concept": primary, **nd},
            "adaptive_decision": {
                **event_view(event, offset),
                "concept": primary,
                "overridden": decision["overridden"],
                "thresholdAction": decision["threshold_action"],
            },
        }


async def evaluate(learner_id: str, payload: dict[str, Any]) -> dict:
    return await run_in_threadpool(_evaluate_sync, learner_id, payload)
