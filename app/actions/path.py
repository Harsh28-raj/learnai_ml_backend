"""Action "get_path" + path snapshot helpers shared with evaluate, tutor_chat and assessment.

A snapshot row is written only when the path's structure (hash) changes. whyThisPath is
polished by GROQ_MODEL_FAST once per (learner, path_hash) and cached on the snapshot; on any
LLM problem the Python template is used and success stays true.
"""

from typing import Any

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.actions.common import get_learner_or_raise, load_state_rows, ordered_states, parse_payload, to_mastery_state
from app.db import session_scope
from app.engine.difficulty import next_difficulty
from app.engine.goals import resolve_goal
from app.engine.learner_model import MasteryState
from app.engine.recommender import (
    before_after,
    build_path,
    current_and_next,
    daily_plan,
    diff_paths,
    milestones,
    why_facts,
    why_template,
)
from app.llm.why_path import polish_why
from app.models import Learner, LearningPath


class PathPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    goal: str | None = Field(None, max_length=200)  # "what if" preview: computed, never saved


# --------------------------------------------------------------------------- sync helpers

def knows_a_language(learner: Learner) -> bool:
    return any(str(l).strip() and str(l).strip().lower() != "none" for l in (learner.known_languages or []))


def learner_path(learner: Learner, states: list[MasteryState], goal_override: str | None = None) -> dict:
    return build_path(states, resolve_goal(goal_override or learner.goal),
                      target_topics=[] if goal_override else list(learner.target_topics or []),
                      pace=learner.learning_pace, daily_minutes=learner.daily_commitment_minutes or 30,
                      knows_a_language=knows_a_language(learner))


def latest_snapshot(db: Session, learner_id: str) -> LearningPath | None:
    return db.scalars(select(LearningPath).where(LearningPath.learner_id == learner_id)
                      .order_by(LearningPath.created_at.desc(), LearningPath.id.desc()).limit(1)).first()


def save_snapshot(db: Session, learner_id: str, path: dict, changes: list[dict], summary: str, why: str,
                  why_source: str) -> LearningPath:
    snap = LearningPath(learner_id=learner_id, goal=path["goal"], nodes=path["nodes"], path_hash=path["pathHash"],
                        changes=changes, before_after=summary, why_this_path=why, why_source=why_source)
    db.add(snap)
    db.flush()
    return snap


def record_path_change(db: Session, learner: Learner, before_states: list[MasteryState],
                       after_states: list[MasteryState]) -> str:
    """Called inside evaluate's transaction: diff the path before vs after this answer and snapshot it.

    Returns the real beforeAfter text for the AdaptiveEvent's pathAdjustment.
    """
    old = learner_path(learner, before_states)
    new = learner_path(learner, after_states)
    changes = diff_paths(old["nodes"], new["nodes"])
    summary = before_after(old["nodes"], new["nodes"], changes)
    latest = latest_snapshot(db, learner.id)
    if latest is None or latest.path_hash != new["pathHash"]:
        pace, daily = learner.learning_pace, learner.daily_commitment_minutes or 30
        save_snapshot(db, learner.id, new, changes, summary,
                      why_template(new, pace=pace, daily_minutes=daily), "template")
    return summary


def path_context(db: Session, learner: Learner, states: list[MasteryState]) -> dict:
    """Current node, next nodes and latest change for the tutor's path mode."""
    path = learner_path(learner, states)
    cur, nxt = current_and_next(path["nodes"])
    latest = latest_snapshot(db, learner.id)
    return {"current": cur, "next": nxt, "latest_change": latest.before_after if latest else None,
            "goal": path["goalLabel"]}


def _load(learner_id: str, goal_override: str | None) -> dict:
    with session_scope() as db:
        learner = get_learner_or_raise(db, learner_id)
        states = ordered_states({cid: to_mastery_state(r) for cid, r in load_state_rows(db, learner_id).items()})
        latest = latest_snapshot(db, learner_id)
        snap = None
        if latest is not None:
            snap = {"id": latest.id, "hash": latest.path_hash, "nodes": latest.nodes, "changes": latest.changes,
                    "before_after": latest.before_after, "why": latest.why_this_path, "why_source": latest.why_source}
        return {"name": learner.name, "pace": learner.learning_pace,
                "daily": learner.daily_commitment_minutes or 30, "states": states,
                "path": learner_path(learner, states, goal_override), "snapshot": snap}


def _persist(learner_id: str, snapshot_id: int | None, path: dict, changes: list, summary: str, why: str,
             why_source: str) -> None:
    with session_scope() as db:
        if snapshot_id is not None:
            snap = db.get(LearningPath, snapshot_id)
            if snap is not None:
                snap.why_this_path, snap.why_source = why, why_source
                return
        save_snapshot(db, learner_id, path, changes, summary, why, why_source)


# --------------------------------------------------------------------------- action

async def compute_path_response(learner_id: str, goal_override: str | None = None) -> dict:
    ctx = await run_in_threadpool(_load, learner_id, goal_override)
    path, snap, pace, daily = ctx["path"], ctx["snapshot"], ctx["pace"], ctx["daily"]
    nodes = path["nodes"]
    facts = why_facts(path, pace=pace, daily_minutes=daily)
    template = why_template(path, pace=pace, daily_minutes=daily)
    preview = goal_override is not None

    if preview:
        changes = diff_paths(snap["nodes"] if snap else None, nodes)
        summary = before_after(snap["nodes"] if snap else None, nodes, changes)
        why, why_source, changed_now = template, "template", bool(changes)
    elif snap and snap["hash"] == path["pathHash"]:
        changes, changed_now = snap["changes"], False
        # Keep showing the last real adaptation; otherwise say plainly that nothing changed.
        summary = snap["before_after"] if snap["changes"] else before_after(nodes, nodes, [])
        if snap["why_source"] == "llm" and snap["why"]:
            why, why_source = snap["why"], "llm"  # cache hit: no LLM call
        else:
            why, why_source = await polish_why(ctx["name"], facts, template)
            if why_source == "llm":
                await run_in_threadpool(_persist, learner_id, snap["id"], path, changes, summary, why, why_source)
    else:
        changes = diff_paths(snap["nodes"] if snap else None, nodes)
        summary = before_after(snap["nodes"] if snap else None, nodes, changes)
        changed_now = True
        why, why_source = await polish_why(ctx["name"], facts, template)
        await run_in_threadpool(_persist, learner_id, None, path, changes, summary, why, why_source)

    cur, nxt = current_and_next(nodes)
    by_id = {s.concept_id: s for s in ctx["states"]}
    practice_difficulty = next_difficulty(by_id[cur["conceptId"]])["difficulty"] if cur and cur["conceptId"] in by_id \
        else "Easy"
    return {
        "goal": path["goal"],
        "goalLabel": path["goalLabel"],
        "targets": path["targets"],
        "nodes": nodes,
        "milestones": milestones(nodes),
        "currentNode": cur,
        "nextNodes": nxt,
        "changes": changes,
        "changedNow": changed_now,
        "beforeAfter": summary,
        "whyThisPath": why,
        "whySource": why_source,
        "estimatedWeeksRemaining": path["estimatedWeeksRemaining"],
        "totalEstimatedMinutes": path["totalEstimatedMinutes"],
        "dailyPlan": daily_plan(path, ctx["states"], daily, practice_difficulty),
        "pathHash": path["pathHash"],
        "preview": preview,
    }


async def get_path(learner_id: str, payload: dict[str, Any]) -> dict:
    p = parse_payload(PathPayload, payload)
    return await compute_path_response(learner_id, p.goal)
