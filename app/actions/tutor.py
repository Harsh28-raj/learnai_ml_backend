"""Action "tutor_chat": context-aware tutoring with layered guardrails.

Flow: clean input -> pre-classify (injection / distress / off-topic) -> load context ->
[off-topic: template reply, no LLM] -> LLM (mode-specific temperature/tokens) -> schema + output
checks -> at most ONE repair call -> safe fallback reply if still failing -> checkpoint
validation + independent verification (practice: one regeneration) -> optional evaluate-mode
mastery update -> persist the turn. Never logs message contents.
"""

import json
import logging
import time
import uuid
from typing import Any

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select

from app.actions.common import get_learner_or_raise, load_state_rows, ordered_states, parse_payload, to_mastery_state
from app.actions.evaluate import _evaluate_sync
from app.actions.path import path_context
from app.config import get_settings
from app.db import session_scope
from app.engine.concepts import CONCEPTS, concept_name, find_concept_in_text, resolve_concept
from app.engine.difficulty import next_difficulty
from app.engine.question_policy import GeneratedQuestion, shuffle_options
from app.engine.tutor_context import build_context, build_history, recommended_next_action
from app.engine.tutor_guardrails import (
    check_reply,
    classify_input,
    clean_message,
    distress_reply,
    off_topic_reply,
)
from app.engine.tutor_modes import CHECKPOINT_MODES, MODE_CONFIG, MODES, REPAIR_TEMPERATURE, detect_mode
from app.llm.groq_client import LLMError, chat_completion
from app.llm.tutor_prompts import build_system_prompt, build_tutor_messages, practice_regen_message, repair_message
from app.llm.verifier import judge, level_mismatch, verify_questions
from app.models import AdaptiveEvent, ChatMessage, Question, QuestionServe
from app.prompts import TUTOR_PROMPT_VERSION
from app.schemas import ActionError

logger = logging.getLogger("learnai.tutor")

# A distress turn must not push studying; these keep the door open gently.
DISTRESS_FOLLOW_UPS = ["Can we take things slowly next time?", "Can you remind me what I already do well?",
                       "Is it normal to find this hard?"]
GENERIC_FOLLOW_UPS = ["What should I study next?", "Can you quiz me on my weakest topic?",
                      "Can you explain that with an example?"]
PRACTICE_DROP_NOTE = ("\n\n_I couldn't produce a practice question I'm fully confident is correct this time. "
                      "Try the Practice tab for a verified set._")


class TutorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str | None = None
    conversation_id: str | None = Field(None, max_length=64)
    mode: str | None = None
    current_topic: str | None = Field(None, max_length=200)
    active_lesson_id: str | None = Field(None, max_length=200)
    question_id: str | None = Field(None, max_length=80)
    student_answer: str | None = Field(None, max_length=4000)
    record: bool = True
    utc_offset_minutes: int = Field(0, ge=-720, le=840)

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        v = v.strip().lower()
        if v not in MODES:
            raise ValueError(f"must be one of {', '.join(MODES)}")
        return v


class TutorReplyOut(BaseModel):
    message: str
    concept: str | None = None
    follow_up_suggestions: list[str] = []
    checkpoint_question: dict | None = None
    evaluation: dict | None = None

    @field_validator("message")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message must not be empty")
        return v.strip()

    @field_validator("follow_up_suggestions", mode="before")
    @classmethod
    def _follow_ups(cls, v: Any) -> list[str]:
        return [str(x).strip() for x in (v or []) if str(x).strip()] if isinstance(v, list) else []


class EvaluationOut(BaseModel):
    score: int
    correct_points: list[str] = []
    misconceptions: list[str] = []
    corrected_answer: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, v: Any) -> int:
        return max(0, min(100, int(round(float(v)))))


class CheckpointQuestion(GeneratedQuestion):
    title: str = "Checkpoint question"


# --------------------------------------------------------------------------- context loading (DB, sync)

def _resolve_chat_concept(p: TutorPayload, message: str, stored_question: dict | None,
                          last_concept: str | None, from_text: bool = True) -> str | None:
    for candidate in (p.current_topic, p.active_lesson_id):
        cid = resolve_concept(candidate) if candidate else None
        if cid:
            return cid
    if stored_question and stored_question.get("conceptId") in CONCEPTS:
        return stored_question["conceptId"]
    if from_text:
        found = find_concept_in_text(message) or resolve_concept(message)
        if found:
            return found
    return last_concept


def _load(learner_id: str, p: TutorPayload, message: str, injection: bool = False) -> dict:
    with session_scope() as db:
        learner = get_learner_or_raise(db, learner_id)
        learner_ctx = {
            "name": learner.name, "level": learner.level, "goal": learner.goal,
            "learning_pace": learner.learning_pace,
            "known_languages": list(learner.known_languages or []),
            "known_topics": list(learner.known_topics or []),
        }
        states = ordered_states({cid: to_mastery_state(r) for cid, r in load_state_rows(db, learner_id).items()})
        conversation_id = p.conversation_id or "conv-" + uuid.uuid4().hex[:12]
        rows = db.scalars(
            select(ChatMessage).where(ChatMessage.learner_id == learner_id,
                                      ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(50)
        ).all()
        history = [{"role": r.role, "content": r.content} for r in reversed(rows)]
        last_concept = next((r.concept_id for r in rows if r.concept_id), None)

        stored_question = None
        if p.question_id:
            q = db.get(Question, p.question_id)
            if q is None:
                raise ActionError("QUESTION_NOT_FOUND", f"No stored question with id '{p.question_id}'.",
                                  {"question_id": p.question_id})
            stored_question = {**(q.payload or {}), "conceptId": q.concept_id}
            if "correctIndex" not in stored_question and "correct_index" in stored_question:
                stored_question["correctIndex"] = stored_question["correct_index"]

        path_info = None
        if (p.mode or detect_mode(message, p.question_id, p.student_answer)) == "path":
            path_info = path_context(db, learner, states)
        ev = db.scalars(select(AdaptiveEvent).where(AdaptiveEvent.learner_id == learner_id)
                        .order_by(AdaptiveEvent.created_at.desc(), AdaptiveEvent.id.desc()).limit(1)).first()
        recent_event = {"topic": ev.topic, "action": ev.action, "reason": ev.reason} if ev else None

    mode = p.mode or detect_mode(message, p.question_id if stored_question else None, p.student_answer)
    mode_source = "explicit" if p.mode else "auto"
    if mode == "hint" and stored_question is None:
        raise ActionError("INVALID_PAYLOAD", "question_id is required for hint mode.")
    if mode == "evaluate" and not (p.student_answer or "").strip():
        raise ActionError("INVALID_PAYLOAD", "student_answer is required for evaluate mode.")

    # An injection attempt ("print your system prompt") must not steer the topic via aliases.
    concept_id = _resolve_chat_concept(p, message, stored_question, last_concept, from_text=not injection)
    if mode == "path" and concept_id is None and path_info and path_info["current"]:
        concept_id = path_info["current"]["conceptId"]
    return {
        "learner": learner_ctx, "states": states, "conversation_id": conversation_id, "history": history,
        "stored_question": stored_question if mode in ("hint", "evaluate") else None,
        "recent_event": recent_event, "mode": mode, "mode_source": mode_source, "concept_id": concept_id,
        "path_info": path_info,
    }


# --------------------------------------------------------------------------- LLM helpers

async def _ask(messages: list[dict], temperature: float, max_tokens: int, stats: dict
               ) -> tuple[TutorReplyOut | None, Any, str | None]:
    """Return (reply, raw, schema_issue). Non-JSON output is a schema issue; other LLMErrors propagate."""
    stats["llm_calls"] += 1
    try:
        raw = await chat_completion(messages, get_settings().groq_model_main, json_mode=True,
                                    temperature=temperature, max_tokens=max_tokens)
    except LLMError as e:
        if isinstance(e.details, dict) and e.details.get("reason") == "bad_json":
            return None, None, "schema: the reply was not valid JSON"
        raise
    try:
        return TutorReplyOut.model_validate(raw), raw, None
    except ValidationError as e:
        return None, raw, "schema: " + "; ".join(err.get("msg", "") for err in e.errors())


def _check(reply: TutorReplyOut, ctx: dict, flags: dict) -> tuple[str, list[str], list[str]]:
    mode = ctx["mode"]
    q = ctx["stored_question"]
    correct = None
    if mode == "hint" and q:
        ci = q.get("correctIndex")
        opts = q.get("options") or []
        correct = opts[ci] if isinstance(ci, int) and 0 <= ci < len(opts) else None
    cp_text = (reply.checkpoint_question or {}).get("question") if isinstance(reply.checkpoint_question, dict) else None
    text, violations, fixes = check_reply(
        reply.message, level=ctx["learner"]["level"], mode=mode, pace=ctx["learner"]["learning_pace"],
        correct_option=correct, distress=flags["distress"], support_text=get_settings().tutor_support_text,
        checkpoint_question=cp_text,
    )
    if mode == "evaluate":
        try:
            EvaluationOut.model_validate(reply.evaluation or {})
        except (ValidationError, TypeError, ValueError):
            violations.append("schema: evaluate mode requires an 'evaluation' object with a 0-100 score")
    return text, violations, fixes


def _safe_reply(ctx: dict, flags: dict) -> str:
    if flags["distress"]:
        return distress_reply(get_settings().tutor_support_text)
    q = ctx["stored_question"]
    if ctx["mode"] == "hint" and q and q.get("hint"):
        return f"Here's a nudge: {q['hint']} Take another look with that in mind."
    cid = ctx["concept_id"]
    if cid:
        return (f"**{concept_name(cid)} in short:** {CONCEPTS[cid]['summary']}\n\n"
                "Ask me to go deeper on any part, or say \"quiz me\" to test yourself.")
    return "Let's keep learning! Ask me about any AI or ML concept and I'll explain it at your level."


def _follow_ups(items: list[str], concept_id: str | None) -> list[str]:
    out = [s for s in items if len(s) <= 200][:3]
    name = concept_name(concept_id) if concept_id else None
    pad = ([f"Can you give me a real-world example of {name}?", f"Can you quiz me on {name}?",
            f"What should I learn after {name}?"] if name else GENERIC_FOLLOW_UPS)
    for s in pad:
        if len(out) >= 3:
            break
        if s not in out:
            out.append(s)
    return out


async def _validate_checkpoint(cp: dict | None, stats: dict, learner_level: str | None = None,
                               ) -> tuple[CheckpointQuestion | None, str | None]:
    if not cp:
        return None, "no checkpoint_question was provided"
    try:
        q = CheckpointQuestion.model_validate(cp)
    except ValidationError as e:
        return None, "invalid checkpoint: " + "; ".join(err.get("msg", "") for err in e.errors())
    if not get_settings().verify_questions:
        return q, None
    try:
        stats["verify_calls"] += 1
        verdicts = await verify_questions([{"qid": "c1", "question": q.question,
                                            "code_snippet": q.code_snippet, "options": q.options}])
    except LLMError as e:
        return None, f"the answer checker was unavailable ({e.code})"
    verdict = verdicts.get("c1")
    ok, issue = judge(verdict, q.correct_index)
    if ok and learner_level:
        issue = level_mismatch(verdict.get("perceived_difficulty"), learner_level)
        ok = issue is None
    return (q, None) if ok else (None, issue)


# --------------------------------------------------------------------------- persistence (DB, sync)

def _save_turn(learner_id: str, ctx: dict, user_content: str, assistant_content: str | None,
               concept_id: str | None, checkpoint: dict | None) -> None:
    with session_scope() as db:
        common = {"learner_id": learner_id, "conversation_id": ctx["conversation_id"], "mode": ctx["mode"],
                  "concept_id": concept_id, "prompt_version": TUTOR_PROMPT_VERSION}
        db.add(ChatMessage(role="user", content=user_content, **common))
        db.flush()  # keep user before assistant in id order
        if assistant_content is not None:
            db.add(ChatMessage(role="assistant", content=assistant_content, **common))
        if checkpoint:
            db.add(Question(id=checkpoint["id"], learner_id=learner_id, concept_id=checkpoint["conceptId"],
                            category=checkpoint["category"], difficulty=checkpoint["difficulty"],
                            payload=checkpoint, verified=bool(checkpoint.get("verified")),
                            level=ctx["learner"]["level"], source="llm", times_served=1))
            db.flush()
            db.add(QuestionServe(learner_id=learner_id, question_id=checkpoint["id"]))


def _checkpoint_payload(q: CheckpointQuestion, concept_id: str, ctx: dict) -> dict:
    qid = "chk-" + uuid.uuid4().hex[:10]
    options, correct = shuffle_options(q.options, q.correct_index, seed=qid)
    state = next((s for s in ctx["states"] if s.concept_id == concept_id), None)
    difficulty = next_difficulty(state)["difficulty"] if state else "Medium"
    return {
        "id": qid, "category": CONCEPTS[concept_id]["category"], "difficulty": difficulty,
        "conceptId": concept_id, "conceptTested": concept_name(concept_id),
        "title": f"Checkpoint: {concept_name(concept_id)}", "question": q.question, "codeSnippet": q.code_snippet,
        "options": options, "correctIndex": correct, "explanation": q.explanation, "hint": q.hint,
        "source": "llm", "verified": get_settings().verify_questions,
    }


# --------------------------------------------------------------------------- action

async def tutor_chat(learner_id: str, payload: dict[str, Any]) -> dict:
    p = parse_payload(TutorPayload, payload)
    message, err = clean_message(p.message)
    if err == "EMPTY_MESSAGE":
        raise ActionError("EMPTY_MESSAGE", "Message must not be empty.")
    if err == "MESSAGE_TOO_LONG":
        raise ActionError("MESSAGE_TOO_LONG", "Message must be at most 2000 characters.", {"max_chars": 2000})

    flags = classify_input(message)
    ctx = await run_in_threadpool(_load, learner_id, p, message, flags["injection_attempt"])
    mode, learner = ctx["mode"], ctx["learner"]
    settings = get_settings()
    stats = {"llm_calls": 0, "verify_calls": 0}
    started = time.perf_counter()
    guard = {"injectionAttempt": flags["injection_attempt"], "distress": flags["distress"],
             "offTopic": flags["off_topic"], "hinglish": flags["hinglish"], "violations": [], "fixes": [],
             "repaired": False,
             "fallbackUsed": False, "checkpoint": None}
    user_content = message + (f"\n\nMy answer: {p.student_answer}" if mode == "evaluate" else "")

    # Off-topic with no ML terms: template reply, no LLM call (saves rate limit).
    if flags["off_topic"] and not flags["distress"]:
        topic = concept_name(ctx["concept_id"]) if ctx["concept_id"] else None
        reply_text = off_topic_reply(topic)
        await run_in_threadpool(_save_turn, learner_id, ctx, user_content, reply_text, ctx["concept_id"], None)
        return _response(ctx, reply_text, _follow_ups([], ctx["concept_id"]), None, None, None, guard, stats, started)

    system = build_system_prompt(
        name=learner["name"], level=learner["level"], pace=learner["learning_pace"],
        languages=learner["known_languages"], context=build_context(
            learner, ctx["states"], ctx["concept_id"], ctx["recent_event"], ctx["stored_question"],
            ctx["path_info"]),
        mode=mode, flags=flags, support_text=settings.tutor_support_text,
    )
    messages = build_tutor_messages(system=system, history=build_history(ctx["history"]), message=message,
                                    student_answer=p.student_answer if mode == "evaluate" else None)
    cfg = MODE_CONFIG[mode]

    try:
        reply, raw, schema_issue = await _ask(messages, cfg["temperature"], cfg["max_tokens"], stats)
        violations = [schema_issue] if schema_issue else []
        text, fixes = "", []
        if reply is not None:
            text, v, fixes = _check(reply, ctx, flags)
            violations += v

        if violations:  # ONE repair call with the specific violation
            guard["violations"] = violations
            repair = list(messages)
            if raw is not None:
                repair.append({"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)[:6000]})
            repair.append(repair_message("; ".join(violations)))
            try:
                reply2, raw2, schema2 = await _ask(repair, REPAIR_TEMPERATURE, cfg["max_tokens"], stats)
                ok = reply2 is not None and not schema2
                if ok:
                    text2, v2, fixes2 = _check(reply2, ctx, flags)
                    ok = not v2
                if ok:
                    reply, raw, text, fixes = reply2, raw2, text2, fixes2
                    guard["repaired"] = True
                else:
                    reply = None
            except LLMError:
                reply = None
            if reply is None:
                guard["fallbackUsed"] = True
                text, fixes = _safe_reply(ctx, flags), []
    except LLMError as e:
        return await _llm_failure(learner_id, ctx, user_content, flags, e)

    guard["fixes"] = fixes
    concept_id = ctx["concept_id"] or (resolve_concept(reply.concept) if reply and reply.concept else None)

    # Checkpoint question: allowed in some modes, required in practice, always independently verified.
    checkpoint = None
    # No checkpoint in a distress turn: lead with warmth, no quizzing.
    if (reply is not None and not flags["distress"] and mode in CHECKPOINT_MODES
            and (reply.checkpoint_question or mode == "practice")):
        q, issue = await _validate_checkpoint(reply.checkpoint_question, stats, learner["level"])
        if issue and mode == "practice":
            try:
                regen = list(messages) + [
                    {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)[:6000]},
                    practice_regen_message(issue)]
                reply3, _, _ = await _ask(regen, cfg["temperature"], cfg["max_tokens"], stats)
                if reply3 is not None:
                    q, issue = await _validate_checkpoint(reply3.checkpoint_question, stats, learner["level"])
                    if q is not None:
                        text3, v3, fixes3 = _check(reply3, ctx, flags)
                        if not v3:
                            reply, text, guard["fixes"] = reply3, text3, fixes3
            except LLMError as e:
                q, issue = None, f"regeneration failed ({e.code})"
        if q is not None and concept_id:
            checkpoint = _checkpoint_payload(q, concept_id, ctx)
            guard["checkpoint"] = "verified" if checkpoint["verified"] else "unverified"
        else:
            guard["checkpoint"] = f"dropped: {issue or 'no concept to attach it to'}"
            if mode == "practice":
                text += PRACTICE_DROP_NOTE

    evaluation = mastery_update = adaptive_decision = None
    if mode == "evaluate" and reply is not None and not guard["fallbackUsed"]:
        ev = EvaluationOut.model_validate(reply.evaluation or {})
        evaluation = {"score": ev.score, "correctPoints": ev.correct_points, "misconceptions": ev.misconceptions,
                      "correctedAnswer": ev.corrected_answer, "recorded": False}
        if p.record and concept_id:
            result = await run_in_threadpool(_evaluate_sync, learner_id, {
                "concept_tested": concept_id, "difficulty": "Medium", "is_correct": ev.score >= 60,
                "utc_offset_minutes": p.utc_offset_minutes,
            }, ev.score)
            evaluation["recorded"] = True
            mastery_update = result["mastery_updates"][0]
            adaptive_decision = result["adaptive_decision"]

    follow_ups = (DISTRESS_FOLLOW_UPS if flags["distress"]
                  else _follow_ups(reply.follow_up_suggestions if reply else [], concept_id))
    await run_in_threadpool(_save_turn, learner_id, ctx, user_content, text, concept_id, checkpoint)
    if evaluation is not None or adaptive_decision is not None:
        ctx = {**ctx, "states": await run_in_threadpool(_reload_states, learner_id)}
    return _response({**ctx, "concept_id": concept_id}, text, follow_ups,
                     _public_checkpoint(checkpoint), evaluation, (mastery_update, adaptive_decision),
                     guard, stats, started)


def _reload_states(learner_id: str) -> list:
    with session_scope() as db:
        return ordered_states({cid: to_mastery_state(r) for cid, r in load_state_rows(db, learner_id).items()})


def _public_checkpoint(cp: dict | None) -> dict | None:
    if cp is None:
        return None
    keys = ("id", "category", "difficulty", "conceptId", "conceptTested", "question", "codeSnippet", "options",
            "correctIndex", "explanation", "hint", "verified")
    return {k: cp[k] for k in keys}


def _response(ctx: dict, text: str, follow_ups: list[str], checkpoint: dict | None, evaluation: dict | None,
              loop: tuple | None, guard: dict, stats: dict, started: float) -> dict:
    cid = ctx["concept_id"]
    action = recommended_next_action(cid, ctx["states"])
    latency = round((time.perf_counter() - started) * 1000)
    logger.info("tutor_chat mode=%s concept=%s flags=%s violations=%d repaired=%s fallback=%s llm_calls=%d "
                "verify_calls=%d latency_ms=%d", ctx["mode"], cid,
                ",".join(k for k in ("injectionAttempt", "distress", "offTopic") if guard[k]) or "-",
                len(guard["violations"]), guard["repaired"], guard["fallbackUsed"], stats["llm_calls"],
                stats["verify_calls"], latency)
    mastery_update, adaptive_decision = loop or (None, None)
    return {
        "conversationId": ctx["conversation_id"],
        "message": text,
        "mode": ctx["mode"],
        "modeSource": ctx["mode_source"],
        "concept": cid,
        "conceptName": concept_name(cid) if cid else None,
        "difficultyLevel": ctx["learner"]["level"],
        "followUpSuggestions": follow_ups,
        "checkpointQuestion": checkpoint,
        "recommendedNextAction": {"type": action["type"], "targetTopic": action["target_topic"],
                                  "reason": action["reason"]},
        "evaluation": evaluation,
        "masteryUpdate": mastery_update,
        "adaptiveDecision": adaptive_decision,
        "promptVersion": TUTOR_PROMPT_VERSION,
        "guardrails": guard,
        "generation": {"model": get_settings().groq_model_main, "llmCalls": stats["llm_calls"],
                       "verifyCalls": stats["verify_calls"], "latencyMs": latency},
    }


async def _llm_failure(learner_id: str, ctx: dict, user_content: str, flags: dict, e: LLMError):
    """LLM unavailable: keep the user message, return success=false with a useful fallback reply."""
    await run_in_threadpool(_save_turn, learner_id, ctx, user_content, None, ctx["concept_id"], None)
    cid = ctx["concept_id"]
    if flags["distress"]:
        reply = distress_reply(get_settings().tutor_support_text)
    elif cid:
        reply = f"**{concept_name(cid)} in short:** {CONCEPTS[cid]['summary']}"
    else:
        reply = "You're making progress just by asking questions. Keep going!"
    reply += "\n\nI'm having trouble reaching my full tutor engine right now; please try again in a moment."
    logger.warning("tutor_chat LLM failure code=%s mode=%s concept=%s", e.code, ctx["mode"], cid)
    raise ActionError(e.code, "The AI tutor is temporarily unavailable.", None, fallback_data={
        "conversation_id": ctx["conversation_id"],
        "reply": reply,
        "follow_ups": _follow_ups([], cid),
        "mode": ctx["mode"],
        "concept": cid,
    })
