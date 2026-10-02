"""Recommendation engine: personalized learning path over the concept DAG. Pure functions.

build_path(): prerequisite closure of the goal targets -> classify each concept (completed /
revision / to learn) -> Kahn topological sort with a heap priority:
  a) revision nodes (weakness priority desc)  b) foundations of weak concepts
  c) goal relevance (DAG distance to a target)  d) category order  e) concept id.
"""

import hashlib
import heapq
import json
import math
from collections import deque
from collections.abc import Sequence

from app.engine.concepts import CATEGORIES, CONCEPTS, concept_name, dependents_of, prerequisite_layers
from app.engine.goals import GOAL_LABELS, goal_targets
from app.engine.learner_model import MasteryState
from app.engine.weakness import build_weakness_report

COMPLETED_MASTERY = 75
CATEGORY_ORDER = {c: i for i, c in enumerate(CATEGORIES)}
BASE_MINUTES = {"Python": 30, "Statistics": 40, "Machine Learning": 50, "Deep Learning": 60, "NLP": 60,
                "Generative AI": 60}
PACE_FACTOR = {"Relaxed": 1.2, "Balanced": 1.0, "Intensive": 0.85}
MIN_NODE_MINUTES = 10
STUDY_DAYS_PER_WEEK = 6
RECOMMENDED_AHEAD = 2


def ancestors(concept_id: str) -> set[str]:
    return {c for layer in prerequisite_layers(concept_id) for c in layer}


def _distances_to_targets(needed: set[str], targets: set[str]) -> dict[str, int]:
    """Shortest number of dependent-edges from each concept to any target (multi-source BFS backwards)."""
    dist = {t: 0 for t in targets}
    queue = deque(targets)
    while queue:
        cid = queue.popleft()
        for pre in CONCEPTS[cid]["prerequisites"]:
            if pre in needed and pre not in dist:
                dist[pre] = dist[cid] + 1
                queue.append(pre)
    return dist


def _closest_target(cid: str, targets: set[str]) -> str | None:
    seen, queue = {cid}, deque([cid])
    while queue:
        cur = queue.popleft()
        for dep in dependents_of(cur):
            if dep in targets:
                return dep
            if dep not in seen:
                seen.add(dep)
                queue.append(dep)
    return None


def estimate_minutes(concept_id: str, mastery: float, pace: str | None) -> int:
    raw = BASE_MINUTES[CONCEPTS[concept_id]["category"]] * (1 - mastery / 100) * PACE_FACTOR.get(pace or "Balanced", 1.0)
    return max(MIN_NODE_MINUTES, round(raw))


def priority_key(cid: str, *, is_revision: bool, weakness_priority: float, is_foundation: bool,
                 distance: int, category_first: bool = False) -> tuple:
    """Heap key (smaller = earlier): a) revisions by priority desc, b) foundations of weak concepts,
    c) goal relevance (DAG distance to a target), d) category order, e) concept id.

    category_first swaps c and d: for learners who know no programming language, Python
    foundations come before anything else."""
    cat = CATEGORY_ORDER[CONCEPTS[cid]["category"]]
    if is_revision:
        return (0, -weakness_priority, 0, cat, cid)
    group = 1 if is_foundation else 2
    return (group, 0, cat, distance, cid) if category_first else (group, 0, distance, cat, cid)


def build_path(states: Sequence[MasteryState], goal_key: str, *, target_topics: Sequence[str] = (),
               pace: str | None = "Balanced", daily_minutes: int = 30, knows_a_language: bool = True) -> dict:
    """Return {goal, targets, nodes, estimatedWeeksRemaining, totalEstimatedMinutes, pathHash}."""
    by_id = {s.concept_id: s for s in states}
    targets = goal_targets(goal_key, target_topics)
    target_set = set(targets)
    needed = set(targets)
    for t in targets:
        needed |= ancestors(t)

    report = build_weakness_report(states)
    weak = {e["concept"]: e for e in report["weak"]}

    def mastery(cid: str) -> float:
        return by_id[cid].mastery if cid in by_id else 0.0

    completed, revision, to_learn = set(), set(), set()
    for cid in needed:
        s = by_id.get(cid)
        if cid in weak and s and s.attempts > 0:
            revision.add(cid)
        elif mastery(cid) >= COMPLETED_MASTERY:
            completed.add(cid)
        else:
            to_learn.add(cid)

    # Foundations: unfinished ancestors of a weak concept get pulled forward.
    foundation_of: dict[str, str] = {}
    for w in sorted(revision, key=lambda c: -weak[c]["priority"]):
        for anc in ancestors(w):
            if anc in to_learn and anc not in foundation_of:
                foundation_of[anc] = w

    dist = _distances_to_targets(needed, target_set)
    scheduled = revision | to_learn

    def key(cid: str) -> tuple:
        return priority_key(cid, is_revision=cid in revision,
                            weakness_priority=weak[cid]["priority"] if cid in weak else 0.0,
                            is_foundation=cid in foundation_of, distance=dist.get(cid, 99),
                            category_first=not knows_a_language)

    # Kahn: a node is available once all of its scheduled prerequisites are placed.
    remaining = {cid: {p for p in CONCEPTS[cid]["prerequisites"] if p in scheduled} for cid in scheduled}
    heap = [key(c) for c, pre in remaining.items() if not pre]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        cid = heapq.heappop(heap)[-1]
        order.append(cid)
        for dep in dependents_of(cid):
            if dep in remaining and cid in remaining[dep]:
                remaining[dep].discard(cid)
                if not remaining[dep]:
                    heapq.heappush(heap, key(dep))

    done_order = [c for c in _topo(completed)]
    nodes: list[dict] = []
    for cid in done_order:
        s = by_id.get(cid)
        skipped = not s or s.attempts == 0
        reason = (f"Skipped: you already know this (mastery {round(mastery(cid))})." if skipped
                  else f"Completed: mastery {round(mastery(cid))} over {s.attempts} practice questions.")
        nodes.append(_node(cid, "completed", mastery(cid), 0, reason, False, skipped))

    for i, cid in enumerate(order):
        # Prerequisites not completed yet (they appear earlier in the path but are still to do).
        pending = [p for p in CONCEPTS[cid]["prerequisites"] if p in scheduled]
        is_rev = cid in revision
        moved = cid in foundation_of
        # The next 2 are "recommended" even if they build on the current node; beyond that, a node
        # with unfinished prerequisites is "locked" and an already-unlocked one stays "recommended".
        if i == 0:
            status = "current"
        elif is_rev or moved:
            status = "adapted"
        elif i <= RECOMMENDED_AHEAD:
            status = "recommended"
        elif pending:
            status = "locked"
        else:
            status = "recommended"
        if is_rev:
            e = weak[cid]
            reason = f"Revision added: {e['label']} priority weakness ({e['reason'].lower()})."
        elif moved:
            w = foundation_of[cid]
            reason = (f"Moved earlier: foundation for {concept_name(w)}, which you're working on "
                      f"(mastery {round(mastery(w))}).")
        elif cid in target_set:
            reason = f"Core concept for your goal: {GOAL_LABELS.get(goal_key, goal_key)}."
        else:
            t = _closest_target(cid, target_set)
            reason = f"Prerequisite for {concept_name(t)}." if t else "Builds toward your goal."
        if status == "locked":
            reason += " Unlocks after: " + ", ".join(concept_name(p) for p in pending) + "."
        nodes.append(_node(cid, status, mastery(cid), estimate_minutes(cid, mastery(cid), pace), reason,
                           is_rev, False, practiced=bool(by_id.get(cid) and by_id[cid].attempts),
                           foundation_for=foundation_of.get(cid)))

    for i, n in enumerate(nodes):
        n["order"] = i + 1
    total = sum(n["estimatedMinutes"] for n in nodes)
    weeks = math.ceil(total / (max(daily_minutes, 1) * STUDY_DAYS_PER_WEEK)) if total else 0
    return {
        "goal": goal_key,
        "goalLabel": GOAL_LABELS.get(goal_key, goal_key),
        "targets": targets,
        "nodes": nodes,
        "estimatedWeeksRemaining": weeks,
        "totalEstimatedMinutes": total,
        "pathHash": path_hash(nodes),
    }


def _topo(cids: set[str]) -> list[str]:
    out, seen = [], set()

    def visit(c: str) -> None:
        if c in seen:
            return
        seen.add(c)
        for p in CONCEPTS[c]["prerequisites"]:
            if p in cids:
                visit(p)
        out.append(c)

    for c in sorted(cids, key=lambda c: (CATEGORY_ORDER[CONCEPTS[c]["category"]], c)):
        visit(c)
    return out


def _node(cid: str, status: str, mastery: float, minutes: int, reason: str, is_revision: bool, skipped: bool,
          practiced: bool = True, foundation_for: str | None = None) -> dict:
    progress = 100 if status == "completed" else (round(mastery) if practiced else 0)
    return {
        "id": f"node-{cid.replace('_', '-')}",
        "conceptId": cid,
        "title": concept_name(cid),
        "category": CONCEPTS[cid]["category"],
        "status": status,
        "progress": progress,
        "mastery": round(mastery),
        "estimatedMinutes": minutes,
        "reason": reason,
        "isRevision": is_revision,
        "skipped": skipped,
        "prerequisites": list(CONCEPTS[cid]["prerequisites"]),
        "foundationFor": foundation_for,
        "order": 0,
    }


def path_hash(nodes: Sequence[dict]) -> str:
    """Structure only (order, status, revision, skipped): small mastery moves don't create snapshots."""
    sig = [(n["conceptId"], n["status"], n["isRevision"], n["skipped"]) for n in nodes]
    return hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:16]


def current_and_next(nodes: Sequence[dict]) -> tuple[dict | None, list[dict]]:
    pending = [n for n in nodes if n["status"] != "completed"]
    return (pending[0] if pending else None), pending[1:1 + RECOMMENDED_AHEAD]


def milestones(nodes: Sequence[dict]) -> list[dict]:
    stages: dict[str, list[dict]] = {}
    for n in nodes:
        stages.setdefault(n["category"], []).append(n)
    out = []
    for cat, ns in stages.items():
        statuses = {n["status"] for n in ns}
        status = ("completed" if statuses == {"completed"} else "current" if "current" in statuses else "upcoming")
        out.append({
            "id": "stage-" + cat.lower().replace(" ", "-"),
            "title": cat,
            "category": cat,
            "status": status,
            "nodeIds": [n["id"] for n in ns],
            "progress": round(sum(n["progress"] for n in ns) / len(ns)),
        })
    return out


# --------------------------------------------------------------------------- diff

def diff_paths(old: Sequence[dict] | None, new: Sequence[dict]) -> list[dict]:
    if not old:
        return []
    o = {n["conceptId"]: n for n in old}
    changes = []
    for n in new:
        cid, prev = n["conceptId"], o.get(n["conceptId"])
        if n["isRevision"] and not (prev and prev["isRevision"]):
            changes.append(_change("inserted_revision", n, f"Revision of {n['title']} added: {n['reason']}"))
        elif n["skipped"] and not (prev and prev["skipped"]):
            changes.append(_change("skipped", n, n["reason"]))
        elif n["status"] == "completed" and prev and prev["status"] != "completed":
            changes.append(_change("completed", n, f"{n['title']} completed (mastery {n['mastery']})."))
        elif prev and prev["status"] == "locked" and n["status"] not in ("locked", "completed"):
            changes.append(_change("unlocked", n, f"{n['title']} is now unlocked."))
    old_cur, _ = current_and_next(old)
    new_cur, _ = current_and_next(new)
    if new_cur and old_cur and new_cur["conceptId"] != old_cur["conceptId"]:
        old_done = any(n["conceptId"] == old_cur["conceptId"] and n["status"] == "completed" for n in new)
        if not old_done:
            changes.append(_change("reordered", new_cur,
                                   f"{new_cur['title']} is now your current focus (was {old_cur['title']})."))
    return changes


def _change(kind: str, node: dict, detail: str) -> dict:
    return {"type": kind, "conceptId": node["conceptId"], "title": node["title"], "detail": detail}


def before_after(old: Sequence[dict] | None, new: Sequence[dict], changes: Sequence[dict]) -> str:
    new_cur, _ = current_and_next(new)
    cur_title = new_cur["title"] if new_cur else "your goal (all concepts completed)"
    if not old:
        return f"Initial path created: start with {cur_title}."
    if not changes:
        return f"Roadmap unchanged: current focus remains {cur_title}."
    old_cur, _ = current_and_next(old)
    before = f"Before: next up was {old_cur['title']}." if old_cur else "Before: path complete."
    parts = []
    for kind, verb in (("inserted_revision", "revision added"), ("completed", "completed"),
                       ("skipped", "skipped"), ("unlocked", "unlocked")):
        titles = [c["title"] for c in changes if c["type"] == kind]
        if titles:
            parts.append(f"{', '.join(titles)} {verb}")
    after = "After: " + ("; ".join(parts) + "; " if parts else "") + f"current focus is {cur_title}."
    return f"{before} {after}"


# --------------------------------------------------------------------------- whyThisPath facts + template

def why_facts(path: dict, *, pace: str | None, daily_minutes: int) -> list[str]:
    nodes = path["nodes"]
    facts = [f"Goal: {path['goalLabel']}."]
    skipped = [n for n in nodes if n["skipped"]][:4]
    if skipped:
        facts.append("Skipped because you already know them: " +
                     ", ".join(f"{n['title']} (mastery {n['mastery']})" for n in skipped) + ".")
    revisions = [n for n in nodes if n["isRevision"]][:3]
    for n in revisions:
        facts.append(f"Revision added for {n['title']} (mastery {n['mastery']}).")
    by_cid = {n["conceptId"]: n for n in nodes}
    for n in [n for n in nodes if n.get("foundationFor")][:2]:
        weak = by_cid.get(n["foundationFor"])
        if weak:
            facts.append(f"{n['title']} is not a weak area itself; it comes first because it is a prerequisite of "
                         f"{weak['title']}, which you are struggling with (mastery {weak['mastery']}).")
    cur, nxt = current_and_next(nodes)
    if cur:
        facts.append(f"Start with: {cur['title']}." + (f" Then: {', '.join(n['title'] for n in nxt)}." if nxt else ""))
    facts.append(f"Pace: {pace or 'Balanced'}, {daily_minutes} minutes a day, "
                 f"about {_weeks(path['estimatedWeeksRemaining'])} remaining.")
    return facts


def _weeks(n: int) -> str:
    return f"{n} week" if n == 1 else f"{n} weeks"


def why_template(path: dict, *, pace: str | None, daily_minutes: int) -> str:
    nodes = path["nodes"]
    parts = [f"Your path is built for your goal: {path['goalLabel']}."]
    skipped = [n for n in nodes if n["skipped"]][:3]
    if skipped:
        parts.append("We skipped " + ", ".join(f"{n['title']} ({n['mastery']})" for n in skipped) +
                     " because you already know them.")
    revisions = [n for n in nodes if n["isRevision"]][:2]
    if revisions:
        parts.append("We added a revision of " + " and ".join(f"{n['title']} ({n['mastery']})" for n in revisions) +
                     " so the gap doesn't hold you back.")
    cur, _ = current_and_next(nodes)
    if cur:
        parts.append(f"Start with {cur['title']}.")
    parts.append(f"At {daily_minutes} minutes a day ({pace or 'Balanced'} pace), you'll finish in about "
                 f"{_weeks(path['estimatedWeeksRemaining'])}.")
    return " ".join(parts)


# --------------------------------------------------------------------------- daily plan

def daily_plan(path: dict, states: Sequence[MasteryState], daily_minutes: int, practice_difficulty: str) -> list[dict]:
    """Up to 1 revision (top weakness), practice on the current node, then a lesson; fits the budget."""
    cur, _ = current_and_next(path["nodes"])
    weak = build_weakness_report(states)["weak"]
    budget = max(daily_minutes, 5)
    tasks: list[dict] = []

    def add(kind: str, cid: str, title: str, minutes: int, **extra) -> None:
        nonlocal budget
        minutes = min(minutes, budget)
        if minutes >= 5:
            tasks.append({"id": f"plan-{kind}-{cid.replace('_', '-')}", "title": title, "type": kind,
                          "durationMinutes": minutes, "completed": False, "conceptId": cid, **extra})
            budget -= minutes

    share = max(5, (budget // 3) // 5 * 5)
    if weak and (not cur or weak[0]["concept"] != cur["conceptId"]) and budget >= 20:
        add("revision", weak[0]["concept"], f"Revise {weak[0]['name']}", min(15, share))
    if cur:
        add("practice", cur["conceptId"], f"Practice {cur['title']} ({practice_difficulty} questions)",
            min(15, max(share, 5)) if budget > 15 else budget, difficulty=practice_difficulty)
        if budget >= 10:  # a lesson never runs longer than the node's own estimate
            add("lesson", cur["conceptId"], f"Lesson: {cur['title']}",
                min(budget, max(MIN_NODE_MINUTES, cur["estimatedMinutes"])))
    return tasks
