"""Tutor context: a compact, factual learner snapshot for the system prompt (pure functions).

The context block targets <= 700 tokens (estimated as chars/4). Conversation history is passed
separately as chat turns: last 6 messages, each truncated to 600 chars.
"""

import math
from collections.abc import Sequence

from app.engine.concepts import CONCEPTS, concept_name, dependents_of
from app.engine.difficulty import next_difficulty
from app.engine.learner_model import MASTERED_THRESHOLD, MasteryState
from app.engine.weakness import build_weakness_report, concept_assessment

CONTEXT_TOKEN_BUDGET = 700
HISTORY_MESSAGES = 6
HISTORY_CHARS = 600
MAX_KNOWN_TOPICS = 10


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _clip(text: str | None, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def wrap_learner_message(text: str, student_answer: str | None = None) -> str:
    out = f"<learner_message>\n{text}\n</learner_message>"
    if student_answer:
        out += f"\n<student_answer>\n{student_answer}\n</student_answer>"
    return out


def build_history(rows: Sequence[dict]) -> list[dict[str, str]]:
    """rows: oldest-first [{role, content}] -> last 6 chat turns, truncated, user turns tagged."""
    turns = []
    for r in list(rows)[-HISTORY_MESSAGES:]:
        content = _clip(r["content"], HISTORY_CHARS)
        if r["role"] == "user":
            turns.append({"role": "user", "content": wrap_learner_message(content)})
        else:
            turns.append({"role": "assistant", "content": content})
    return turns


def _sections(learner: dict, states: Sequence[MasteryState], concept_id: str | None, recent_event: dict | None,
              stored_question: dict | None, path_info: dict | None = None, *, topics: int, prereqs: int,
              strengths: int, event_chars: int, expl_chars: int) -> list[str]:
    by_id = {s.concept_id: s for s in states}
    report = build_weakness_report(states)
    lines = [
        f"LEARNER: {learner.get('name')} | level {learner.get('level')} | goal: {_clip(learner.get('goal'), 80)} "
        f"| pace {learner.get('learning_pace') or 'Balanced'}",
        f"Known languages: {', '.join(learner.get('known_languages') or []) or 'none'}"
        f" | Known topics: {', '.join((learner.get('known_topics') or [])[:topics]) or 'none'}",
    ]
    if concept_id and concept_id in CONCEPTS:
        s = by_id.get(concept_id)
        a = concept_assessment(concept_id, states) if s and s.attempts else None
        status = (f"{a['classification']} ({a['label']} priority: {a['reason']})" if a
                  else "not practiced yet")
        mastery = f"{round(s.mastery)}%" if s else "unknown"
        lines.append(f"CURRENT CONCEPT: {concept_name(concept_id)} | mastery {mastery} | "
                     f"trend {s.trend if s else 'n/a'} | confidence {s.confidence if s else 0:.2f} | status {status}")
        pre = CONCEPTS[concept_id]["prerequisites"][:prereqs]
        if pre:
            lines.append("Prerequisites: " + ", ".join(
                f"{concept_name(p)} {round(by_id[p].mastery)}%" if p in by_id else concept_name(p) for p in pre))
        if a and a["recommended_revision"]["is_prerequisite"]:
            rev = a["recommended_revision"]
            lines.append(f"Weakest prerequisite: {rev['name']} {round(rev['mastery'] or 0)}%")
    weak = report["weak"][:3]
    lines.append("TOP WEAKNESSES: " + ("; ".join(
        f"{w['name']} {round(w['mastery'])}% ({w['label']})" for w in weak) if weak else "none detected"))
    strong = report["strong"][:strengths]
    if strong:
        lines.append("STRENGTHS (bridge from these): " + ", ".join(f"{w['name']} {round(w['mastery'])}%" for w in strong))
    if recent_event:
        lines.append(f"LAST ADAPTIVE EVENT: {recent_event.get('topic')} | {recent_event.get('action')} | "
                     f"{_clip(recent_event.get('reason'), event_chars)}")
    if path_info and path_info.get("current"):
        cur = path_info["current"]
        lines.append(f"LEARNING PATH (goal: {path_info.get('goal')}): current node {cur['title']} "
                     f"({cur['status']}; {_clip(cur['reason'], event_chars or 80)})")
        if path_info.get("next"):
            lines.append("Next nodes: " + "; ".join(f"{n['title']} ({n['status']})" for n in path_info["next"]))
        if path_info.get("latest_change"):
            lines.append("Latest path change: " + _clip(path_info["latest_change"], event_chars or 80))
    if stored_question:
        q = stored_question
        opts = q.get("options") or []
        ci = q.get("correctIndex", q.get("correct_index"))
        lines.append("QUESTION THE LEARNER IS WORKING ON: " + _clip(q.get("question"), 400))
        if q.get("codeSnippet"):
            lines.append("Code: " + _clip(q.get("codeSnippet"), 300))
        lines.append("Options: " + " | ".join(f"[{i}] {_clip(o, 90)}" for i, o in enumerate(opts)))
        if isinstance(ci, int) and 0 <= ci < len(opts):
            lines.append(f"Correct option (CONFIDENTIAL in hint mode): [{ci}] {_clip(opts[ci], 120)}")
        lines.append("Reference explanation: " + _clip(q.get("explanation"), expl_chars))
    return lines


def build_context(learner: dict, states: Sequence[MasteryState], concept_id: str | None,
                  recent_event: dict | None = None, stored_question: dict | None = None,
                  path_info: dict | None = None) -> str:
    """Render the context block, shrinking optional parts until it fits the token budget."""
    budgets = [
        dict(topics=10, prereqs=4, strengths=3, event_chars=200, expl_chars=500),
        dict(topics=6, prereqs=3, strengths=2, event_chars=120, expl_chars=300),
        dict(topics=3, prereqs=2, strengths=0, event_chars=80, expl_chars=180),
        dict(topics=0, prereqs=1, strengths=0, event_chars=0, expl_chars=100),
    ]
    text = ""
    for b in budgets:
        text = "\n".join(_sections(learner, states, concept_id, recent_event, stored_question, path_info, **b))
        if estimate_tokens(text) <= CONTEXT_TOKEN_BUDGET:
            return text
    return text[: CONTEXT_TOKEN_BUDGET * 4]


def recommended_next_action(concept_id: str | None, states: Sequence[MasteryState]) -> dict:
    """{type: practice|revise|advance|continue, target_topic, reason} from real learner facts."""
    by_id = {s.concept_id: s for s in states}
    if concept_id and concept_id in by_id:
        s = by_id[concept_id]
        name = concept_name(concept_id)
        nd = next_difficulty(s)
        if s.attempts == 0:
            return {"type": "practice", "target_topic": name,
                    "reason": f"You haven't practiced {name} yet; start with {nd['difficulty']} questions."}
        a = concept_assessment(concept_id, states)
        if a["classification"] == "Weak":
            rev = a["recommended_revision"]
            if rev["is_prerequisite"]:
                return {"type": "revise", "target_topic": rev["name"],
                        "reason": f"{name} is at {round(s.mastery)}% and its prerequisite {rev['name']} is at "
                                  f"{round(rev['mastery'] or 0)}%; shoring that up first will make {name} click."}
            return {"type": "practice", "target_topic": name,
                    "reason": f"{name} is at {round(s.mastery)}% ({a['label']} priority); "
                              f"{nd['difficulty']} practice will rebuild it."}
        if s.mastery >= MASTERED_THRESHOLD:
            nxt = [d for d in dependents_of(concept_id) if by_id.get(d) is None or by_id[d].mastery < MASTERED_THRESHOLD]
            if nxt:
                return {"type": "advance", "target_topic": concept_name(nxt[0]),
                        "reason": f"You've mastered {name} ({round(s.mastery)}%); {concept_name(nxt[0])} builds on it."}
        return {"type": "practice", "target_topic": name,
                "reason": f"{name} is at {round(s.mastery)}%; keep practicing at {nd['difficulty']} to consolidate."}
    weak = build_weakness_report(states)["weak"]
    if weak:
        w = weak[0]
        return {"type": "practice", "target_topic": w["name"],
                "reason": f"Your top priority is {w['name']} at {round(w['mastery'])}% ({w['label']} priority)."}
    return {"type": "continue", "target_topic": None, "reason": "No weak spots detected; continue your learning path."}
