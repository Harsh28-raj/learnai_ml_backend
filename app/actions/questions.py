"""Action "generate_questions": adaptive MCQs with generation, independent verification and a
rate-limit-resilient serving chain.

Serving chain (first that yields questions wins, later ones only fill the gap):
  1. LLM generate (GROQ_MODEL_MAIN) -> validate -> verify (GROQ_MODEL_FAST); rejected slots are
     regenerated + re-verified ONCE.
  2. POOL: verified LLM questions from the DB for the same level, not yet served to this learner
     (same concept+difficulty -> same concept+adjacent difficulty -> same category+difficulty).
  3. Static curated bank (app/data/fallback_questions.json).
Python decides everything factual (concept, difficulty, domain/angle, ids, answer position,
whyThisQuestion); the LLM only writes question content. No DB session is held across LLM calls.
"""

import json
import logging
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, or_, select

from app.actions.common import get_learner_or_raise, load_state_rows, ordered_states, parse_payload, to_mastery_state
from app.config import get_settings
from app.db import session_scope
from app.engine.concepts import CONCEPTS, concept_name, normalize_category, resolve_concept, suggest_concepts
from app.engine.difficulty import next_difficulty
from app.engine.learner_model import DIFFICULTIES, MasteryState
from app.engine.question_policy import (
    GeneratedQuestion,
    assign_slots,
    select_concept,
    shuffle_options,
    validate_batch,
    why_this_question,
)
from app.engine.weakness import concept_assessment
from app.llm.groq_client import LLMError, chat_completion
from app.llm.prompts import build_question_messages
from app.llm.verifier import difficulty_mismatch, judge, verify_questions
from app.models import Question, QuestionServe
from app.schemas import ActionError, ActionResult, ErrorInfo

logger = logging.getLogger("learnai.questions")

FALLBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "fallback_questions.json"
RECENT_LIMIT = 20
TEMPERATURE = 0.5
MAX_REPORTED_ISSUES = 10


class GeneratePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str | None = Field(None, max_length=40)
    target_concept: str | None = Field(None, max_length=200)
    difficulty: str = "Adaptive"
    count: int = Field(3, ge=1, le=5)

    @field_validator("difficulty")
    @classmethod
    def _difficulty(cls, v: str) -> str:
        v = (v or "Adaptive").strip().capitalize()
        if v not in ("Adaptive", *DIFFICULTIES):
            raise ValueError("must be one of Adaptive, Easy, Medium, Hard")
        return v


# --------------------------------------------------------------------------- preparation (DB, sync)

def _recent_questions(db, learner_id: str) -> list[dict]:
    """Questions this learner generated or was served, newest first (anti-repetition window)."""
    served_at = (
        select(QuestionServe.question_id, QuestionServe.served_at)
        .where(QuestionServe.learner_id == learner_id).subquery()
    )
    seen_at = func.coalesce(served_at.c.served_at, Question.created_at)
    rows = db.execute(
        select(Question)
        .outerjoin(served_at, served_at.c.question_id == Question.id)
        .where(or_(Question.learner_id == learner_id, served_at.c.question_id.is_not(None)))
        .order_by(seen_at.desc())
        .limit(RECENT_LIMIT)
    ).scalars().all()
    return [
        {"concept_id": q.concept_id,
         "scenario_domain": (q.payload or {}).get("scenarioDomain"),
         "question_angle": (q.payload or {}).get("questionAngle"),
         "title": (q.payload or {}).get("title", ""),
         "question": (q.payload or {}).get("question", "")}
        for q in rows
    ]


def _prepare(learner_id: str, p: GeneratePayload) -> dict:
    category = None
    if p.category:
        category = normalize_category(p.category)
        if category is None:
            raise ActionError("INVALID_PAYLOAD", f"Unknown category '{p.category}'.",
                              {"supported_categories": sorted({c["category"] for c in CONCEPTS.values()})})
    target = None
    if p.target_concept:
        target = resolve_concept(p.target_concept, category)
        if target is None:
            raise ActionError("UNKNOWN_CONCEPT", f"Could not match '{p.target_concept}' to a known concept.",
                              {"target_concept": p.target_concept, "suggestions": suggest_concepts(p.target_concept)})

    with session_scope() as db:
        learner = get_learner_or_raise(db, learner_id)
        learner_ctx = {
            "name": learner.name, "level": learner.level, "goal": learner.goal,
            "known_topics": list(learner.known_topics or []), "known_languages": list(learner.known_languages or []),
        }
        states = ordered_states({cid: to_mastery_state(r) for cid, r in load_state_rows(db, learner_id).items()})
        recent = _recent_questions(db, learner_id)

    concept_id, selection_reason = select_concept(states, category, target)
    by_id = {s.concept_id: s for s in states}
    state = by_id.get(concept_id) or MasteryState(concept_id=concept_id, category=CONCEPTS[concept_id]["category"])
    nd = next_difficulty(state)
    requested = p.difficulty != "Adaptive"
    difficulty = p.difficulty if requested else nd["difficulty"]
    difficulty_reason = (f"{difficulty} requested explicitly." if requested else nd["reason"])
    assessment = concept_assessment(concept_id, states)

    return {
        "learner": learner_ctx,
        "level": learner_ctx["level"],
        "concept_id": concept_id,
        "selection_reason": selection_reason,
        "state": state,
        "next_difficulty": nd,
        "difficulty": difficulty,
        "difficulty_requested": requested,
        "difficulty_reason": difficulty_reason,
        "classification": assessment["classification"] if assessment and state.attempts else None,
        "slots": assign_slots(p.count, learner_ctx["level"], concept_id, recent),
        "recent_titles": [r["title"] for r in recent if r["title"]],
        "recent_texts": [r["question"] for r in recent if r["question"]],
    }


# --------------------------------------------------------------------------- generation + verification (async)

async def _call_llm(ctx: dict, slots: list[dict], feedback: list[str]) -> Any:
    concept = CONCEPTS[ctx["concept_id"]]
    messages = build_question_messages(
        learner=ctx["learner"],
        concept=concept,
        prerequisite_names=[concept_name(p) for p in concept["prerequisites"]],
        mastery=ctx["state"].mastery,
        classification=ctx["classification"],
        difficulty=ctx["difficulty"],
        slots=slots,
        recent_titles=ctx["recent_titles"],
        feedback=feedback,
    )
    try:
        return await chat_completion(messages, get_settings().groq_model_main, json_mode=True,
                                     temperature=TEMPERATURE, max_tokens=1200 + 900 * len(slots))
    except LLMError as e:
        if isinstance(e.details, dict) and e.details.get("reason") == "bad_json":
            return None  # treated as a validation failure, so it gets the regeneration round
        raise


def _label(slot: dict) -> str:
    return f"Question for {slot['scenario_domain']} / {slot['question_angle']}"


async def _verify(candidates: list[tuple[GeneratedQuestion, dict]], stats: dict, requested: str | None = None,
                  ) -> tuple[list[tuple[GeneratedQuestion, dict]], list[dict], list[str]]:
    """Return (passed, rejected slots, feedback lines). Raises LLMError if the verifier call fails."""
    if not candidates:
        return [], [], []
    stats["verify_calls"] += 1
    verdicts = await verify_questions([
        {"qid": f"q{slot['index']}", "question": q.question, "code_snippet": q.code_snippet, "options": q.options}
        for q, slot in candidates
    ])
    passed, rejected, feedback = [], [], []
    for q, slot in candidates:
        verdict = verdicts.get(f"q{slot['index']}")
        ok, issue = judge(verdict, q.correct_index)
        if ok and requested:
            issue = difficulty_mismatch(verdict.get("perceived_difficulty"), requested)
            ok = issue is None
        if ok:
            passed.append((q, slot))
        else:
            rejected.append(slot)
            stats["verifier_rejected"] += 1
            stats["issues"].append(f"{q.title}: {issue}")
            feedback.append(f"{_label(slot)} ('{q.title}') was rejected by an independent solver: {issue}.")
    stats["verified"] += len(passed)
    return passed, rejected, feedback


async def _generate_verified(ctx: dict, verify: bool) -> tuple[list[tuple[GeneratedQuestion, dict]], dict]:
    """Up to 2 generate calls + 2 verify calls. Never returns unverified questions when verify=True.

    stats["failure"] is set to an error code when the LLM path came up short.
    """
    stats: dict = {"model": get_settings().groq_model_main, "verifier_model": get_settings().groq_model_fast,
                   "llm_calls": 0, "verify_calls": 0, "verified": 0, "rejected": 0, "verifier_rejected": 0,
                   "issues": [], "failure": None}

    # Round 1 -------------------------------------------------------------
    try:
        stats["llm_calls"] += 1
        raw = await _call_llm(ctx, ctx["slots"], [])
    except LLMError as e:
        stats["failure"] = e.code
        return [], stats
    valid, errors, failed = validate_batch(raw, ctx["slots"], ctx["recent_texts"])
    if raw is None:
        errors = ["The response was not valid JSON."]
    stats["rejected"] += len(failed)
    stats["issues"].extend(errors)
    feedback = list(errors)
    retry_slots = list(failed)

    if verify:
        try:
            passed, rejected, vfeedback = await _verify(valid, stats, ctx["difficulty"])
        except LLMError as e:
            # Verifier unavailable: do not serve unverified questions; the pool fills the gap.
            stats["failure"] = e.code
            return [], stats
        stats["rejected"] += len(rejected)
        retry_slots += rejected
        feedback += vfeedback
    else:
        passed = valid

    # Round 2: regenerate ONLY the rejected slots, once ------------------
    if retry_slots:
        retry_slots.sort(key=lambda s: s["index"])
        try:
            stats["llm_calls"] += 1
            raw2 = await _call_llm(ctx, retry_slots, feedback)
            valid2, errors2, failed2 = validate_batch(raw2, retry_slots, ctx["recent_texts"],
                                                      [q.question for q, _ in passed])
            if raw2 is None:
                errors2 = ["The regenerated response was not valid JSON."]
            stats["rejected"] += len(failed2)
            stats["issues"].extend(errors2)
            if verify:
                passed2, rejected2, _ = await _verify(valid2, stats, ctx["difficulty"])
                stats["rejected"] += len(rejected2)
            else:
                passed2 = valid2
            passed += passed2
        except LLMError as e:
            stats["failure"] = e.code
            logger.warning("Question regeneration/verification failed (%s); keeping %d question(s)",
                           e.code, len(passed))

    passed.sort(key=lambda qs: qs[1]["index"])
    if len(passed) < len(ctx["slots"]) and stats["failure"] is None:
        stats["failure"] = "VERIFICATION_SHORTFALL" if stats["verifier_rejected"] else "GENERATION_FAILED"
    return passed, stats


def _finalize(ctx: dict, q: GeneratedQuestion, slot: dict, verified: bool) -> dict:
    qid = "gen-" + uuid.uuid4().hex[:10]
    options, correct = shuffle_options(q.options, q.correct_index, seed=qid)
    cid = ctx["concept_id"]
    return {
        "id": qid,
        "category": CONCEPTS[cid]["category"],
        "difficulty": ctx["difficulty"],
        "conceptId": cid,
        "conceptTested": concept_name(cid),
        "title": q.title,
        "question": q.question,
        "codeSnippet": q.code_snippet,
        "options": options,
        "correctIndex": correct,
        "explanation": q.explanation,
        "hint": q.hint,
        "whyThisQuestion": why_this_question(ctx["selection_reason"], ctx["state"], ctx["next_difficulty"],
                                             ctx["difficulty_requested"]),
        "recommendedNextDifficulty": ctx["next_difficulty"]["difficulty"],
        "scenarioDomain": slot["scenario_domain"],
        "questionAngle": slot["question_angle"],
        "source": "llm",
        "verified": verified,
    }


# --------------------------------------------------------------------------- pool (DB, sync)

def adjacent_difficulties(difficulty: str) -> list[str]:
    i = DIFFICULTIES.index(difficulty)
    return [DIFFICULTIES[j] for j in (i - 1, i + 1) if 0 <= j < len(DIFFICULTIES)]


def pool_tiers(concept_id: str, difficulty: str) -> list[tuple[str, Any]]:
    """Relaxation order: same concept+difficulty -> same concept+adjacent -> same category+difficulty."""
    category = CONCEPTS[concept_id]["category"]
    return [
        ("concept+difficulty", (Question.concept_id == concept_id) & (Question.difficulty == difficulty)),
        ("concept+adjacent_difficulty",
         (Question.concept_id == concept_id) & Question.difficulty.in_(adjacent_difficulties(difficulty))),
        ("category+difficulty", (Question.category == category) & (Question.difficulty == difficulty)),
    ]


def _pool_fill(learner_id: str, ctx: dict, need: int, exclude_ids: list[str]) -> list[dict]:
    """Verified LLM questions for this level that this learner has never been served."""
    out: list[dict] = []
    taken = set(exclude_ids)
    served = select(QuestionServe.question_id).where(QuestionServe.learner_id == learner_id)
    with session_scope() as db:
        for tier, condition in pool_tiers(ctx["concept_id"], ctx["difficulty"]):
            if len(out) >= need:
                break
            rows = db.scalars(
                select(Question)
                .where(Question.verified.is_(True), Question.source == "llm", Question.level == ctx["level"],
                       Question.id.not_in(served), condition)
                .order_by(Question.times_served.asc(), Question.created_at.desc())
                .limit(need + len(taken))
            ).all()
            for row in rows:
                if len(out) >= need:
                    break
                if row.id in taken:
                    continue
                taken.add(row.id)
                payload = dict(row.payload or {})
                if row.concept_id == ctx["concept_id"]:
                    # Rationale is rebuilt from THIS learner's facts, not the original requester's.
                    why = why_this_question(ctx["selection_reason"], ctx["state"], ctx["next_difficulty"],
                                            ctx["difficulty_requested"])
                else:
                    why = (f"From LearnAI's verified question bank: related {row.category} practice while "
                           f"new questions on {concept_name(ctx['concept_id'])} are unavailable.")
                payload.update({
                    "source": "pool",
                    "poolTier": tier,
                    "verified": True,
                    "whyThisQuestion": why,
                    "recommendedNextDifficulty": ctx["next_difficulty"]["difficulty"],
                })
                payload.pop("distractor_reasons", None)
                out.append(payload)
    return out


# --------------------------------------------------------------------------- static fallback

@lru_cache(maxsize=1)
def load_fallback_questions() -> list[dict]:
    with FALLBACK_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _fallback_questions(ctx: dict, count: int, code: str) -> list[dict]:
    cid = ctx["concept_id"]
    category = CONCEPTS[cid]["category"]
    bank = load_fallback_questions()
    ranked = ([q for q in bank if q["concept_id"] == cid]
              + [q for q in bank if q["concept_id"] != cid and q["category"] == category]
              + [q for q in bank if q["category"] != category])
    out = []
    for fb in ranked[:count]:
        qid = "fallback-" + uuid.uuid4().hex[:10]
        options, correct = shuffle_options(fb["options"], fb["correct_index"], seed=qid)
        out.append({
            "id": qid,
            "category": fb["category"],
            "difficulty": fb["difficulty"],
            "conceptId": fb["concept_id"],
            "conceptTested": concept_name(fb["concept_id"]),
            "title": fb["title"],
            "question": fb["question"],
            "codeSnippet": fb["code_snippet"],
            "options": options,
            "correctIndex": correct,
            "explanation": fb["explanation"],
            "hint": fb["hint"],
            "whyThisQuestion": f"Backup question from LearnAI's curated set: the AI generator is unavailable "
                               f"right now ({code}).",
            "recommendedNextDifficulty": ctx["next_difficulty"]["difficulty"],
            "scenarioDomain": fb["scenario_domain"],
            "questionAngle": fb["question_angle"],
            "source": "fallback",
            "verified": True,
            "fallback": True,
        })
    return out


# --------------------------------------------------------------------------- persistence (DB, sync)

def save_served_questions(learner_id: str, level: str, new_questions: list[dict], pool_ids: list[str]) -> None:
    """Insert new questions, bump pool counters, and record every serve -- one transaction."""
    with session_scope() as db:
        for q in new_questions:
            db.add(Question(id=q["id"], learner_id=learner_id, concept_id=q["conceptId"], category=q["category"],
                            difficulty=q["difficulty"], payload=q, verified=bool(q.get("verified")),
                            level=level, source="fallback" if q.get("source") == "fallback" else "llm",
                            times_served=1))
        db.flush()
        for qid in pool_ids:
            row = db.get(Question, qid)
            if row is not None:
                row.times_served = (row.times_served or 0) + 1
        for qid in [q["id"] for q in new_questions] + list(pool_ids):
            db.add(QuestionServe(learner_id=learner_id, question_id=qid))


# --------------------------------------------------------------------------- action

_FAILURE_MESSAGES = {
    "MODEL_RATE_LIMIT": "the AI service is rate limited right now",
    "MODEL_TIMEOUT": "the AI model timed out",
    "MODEL_ERROR": "the AI model returned an error",
    "LLM_NOT_CONFIGURED": "the AI generator is not configured",
    "VERIFICATION_SHORTFALL": "some generated questions failed independent answer verification",
    "GENERATION_FAILED": "the AI model did not produce enough valid questions",
}


async def generate_questions(learner_id: str, payload: dict[str, Any]) -> ActionResult:
    p = parse_payload(GeneratePayload, payload)
    ctx = await run_in_threadpool(_prepare, learner_id, p)
    verify = get_settings().verify_questions

    started = time.perf_counter()
    accepted, stats = await _generate_verified(ctx, verify)
    stats["latency_ms"] = round((time.perf_counter() - started) * 1000)

    questions = [_finalize(ctx, q, slot, verified=verify) for q, slot in accepted]
    need = p.count - len(questions)
    pool: list[dict] = []
    fallback: list[dict] = []
    if need > 0:
        pool = await run_in_threadpool(_pool_fill, learner_id, ctx, need, [])
        need -= len(pool)
    failure = stats["failure"] or "GENERATION_FAILED"
    if need > 0:
        fallback = _fallback_questions(ctx, need, failure)

    await run_in_threadpool(save_served_questions, learner_id, ctx["level"], questions + fallback,
                            [q["id"] for q in pool])
    served = questions + pool + fallback
    logger.info("generate_questions concept=%s difficulty=%s llm=%d pool=%d fallback=%d failure=%s",
                ctx["concept_id"], ctx["difficulty"], len(questions), len(pool), len(fallback), stats["failure"])

    data = {
        "count": len(served),
        "requested_count": p.count,
        "concept": ctx["concept_id"],
        "concept_name": concept_name(ctx["concept_id"]),
        "selection_reason": ctx["selection_reason"],
        "difficulty": ctx["difficulty"],
        "difficulty_reason": ctx["difficulty_reason"],
        "questions": served,
        "sources": {"llm": len(questions), "pool": len(pool), "fallback": len(fallback)},
        "generation": {
            "model": stats["model"],
            "verifier_model": stats["verifier_model"] if verify else None,
            "verification_enabled": verify,
            "llm_calls": stats["llm_calls"],
            "verify_calls": stats["verify_calls"],
            "verified": stats["verified"],
            "rejected": stats["rejected"],
            "rejection_issues": stats["issues"][:MAX_REPORTED_ISSUES],
            "latency_ms": stats["latency_ms"],
        },
    }
    if pool or fallback:
        reason = _FAILURE_MESSAGES.get(failure, "the AI generator was unavailable")
        parts = ([f"{len(pool)} from LearnAI's verified question pool"] if pool else []) + \
                ([f"{len(fallback)} from the curated question bank"] if fallback else [])
        warning = ErrorInfo(
            code=failure,
            message=f"{len(pool) + len(fallback)} of {p.count} question(s) were served without fresh generation "
                    f"({' and '.join(parts)}) because {reason}.",
        )
        if len(served) < p.count:
            raise ActionError(failure, warning.message, None, fallback_data=data)
        return ActionResult(data, warning)
    return ActionResult(data)
