"""Independent answer verification for generated MCQs (one call per batch, fast model, temp 0).

The verifier never sees correct_index, explanation, hint or distractor_reasons: it must
solve each question itself. A question passes only if the verifier's answer matches ours,
it reports a single correct option, no flaw, and confidence >= 0.6.
"""

from collections.abc import Sequence

from app.config import get_settings
from app.llm.groq_client import LLMError, chat_completion
from app.llm.prompts import build_verifier_messages

MIN_CONFIDENCE = 0.6
DIFFICULTY_LEVELS = ("Easy", "Medium", "Hard")
VERIFIER_EFFORT = "medium"  # solving needs real reasoning; the fast model keeps this cheap


def _coerce(item: dict) -> dict:
    try:
        answer = int(item.get("answer_index"))
    except (TypeError, ValueError):
        answer = -1
    try:
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    issue = item.get("issue")
    perceived = str(item.get("perceived_difficulty") or "").strip().capitalize()
    return {
        "perceived_difficulty": perceived if perceived in DIFFICULTY_LEVELS else None,
        "answer_index": answer,
        "confidence": confidence,
        "multiple_correct": bool(item.get("multiple_correct")),
        "flawed": bool(item.get("flawed")),
        "issue": str(issue).strip() if issue else None,
    }


async def verify_questions(questions: Sequence[dict]) -> dict[str, dict]:
    """questions: [{qid, question, code_snippet, options}] -> {qid: verdict}.

    Raises LLMError if the call fails or returns an unusable shape.
    """
    if not questions:
        return {}
    payload = [{"qid": q["qid"], "question": q["question"], "code_snippet": q.get("code_snippet"),
                "options": list(q["options"])} for q in questions]
    messages = build_verifier_messages(payload)
    raw = None
    for attempt in range(2):  # one retry if the model's JSON is invalid (often truncated reasoning)
        try:
            raw = await chat_completion(
                messages, get_settings().groq_model_fast, json_mode=True, temperature=0.0,
                max_tokens=400 * len(payload) + 400, reasoning_effort=VERIFIER_EFFORT,
            )
            break
        except LLMError as e:
            if attempt == 0 and isinstance(e.details, dict) and e.details.get("reason") == "bad_json":
                continue
            raise
    results = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(results, list):
        raise LLMError("MODEL_ERROR", "Verifier returned an unexpected shape.", details={"reason": "bad_verifier"})
    return {str(r.get("qid")): _coerce(r) for r in results if isinstance(r, dict) and r.get("qid") is not None}


def judge(verdict: dict | None, correct_index: int) -> tuple[bool, str | None]:
    """Return (accepted, rejection_issue)."""
    if verdict is None:
        return False, "verifier returned no verdict for this question"
    if verdict["multiple_correct"]:
        return False, "more than one option is defensible" + (f": {verdict['issue']}" if verdict["issue"] else "")
    if verdict["flawed"]:
        return False, "question is flawed" + (f": {verdict['issue']}" if verdict["issue"] else "")
    if verdict["answer_index"] != correct_index:
        return False, "an independent solver chose a different answer" + (
            f" ({verdict['issue']})" if verdict["issue"] else "")
    if verdict["confidence"] < MIN_CONFIDENCE:
        return False, f"solver confidence too low ({verdict['confidence']:.2f})"
    return True, None


def difficulty_mismatch(perceived: str | None, requested: str) -> str | None:
    """Reject only a two-level gap (Easy vs Hard); an unknown perception never rejects."""
    if perceived not in DIFFICULTY_LEVELS or requested not in DIFFICULTY_LEVELS:
        return None
    if abs(DIFFICULTY_LEVELS.index(perceived) - DIFFICULTY_LEVELS.index(requested)) >= 2:
        return f"an independent solver rated it {perceived} but {requested} was requested"
    return None


def level_mismatch(perceived: str | None, learner_level: str) -> str | None:
    """Checkpoints: no Easy questions for Advanced learners, no Hard ones for Beginners."""
    if learner_level == "Advanced" and perceived == "Easy":
        return "too easy for an Advanced learner (rated Easy)"
    if learner_level == "Beginner" and perceived == "Hard":
        return "too hard for a Beginner (rated Hard)"
    return None
