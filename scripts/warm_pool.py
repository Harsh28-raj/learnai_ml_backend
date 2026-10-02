"""Pre-generate verified questions into the pool (manual; NOT part of unittest).

Generates through the running server as the internal pool learners (pool-beginner,
pool-intermediate, pool-advanced), so every question goes through the same validation and
independent verification as live traffic. Resumable: cells that already hold --per-cell
verified LLM questions are skipped. Counts are read from the same DATABASE_URL as the server.

    python scripts/warm_pool.py --only-demo --dry-run
    python scripts/warm_pool.py --only-demo --per-cell 2 --gap 25 --base http://127.0.0.1:8000
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.actions.common import load_state_rows, ordered_states, to_mastery_state  # noqa: E402
from app.actions.path import learner_path  # noqa: E402
from app.db import create_tables, session_scope  # noqa: E402
from app.engine.pool_plan import DIFFICULTIES, LEVELS, all_cells, cells_for, estimate, plan  # noqa: E402
from app.models import Learner, Question  # noqa: E402
from app.seed import DEMO_LEARNERS, POOL_LEARNERS  # noqa: E402

POOL_LEARNER_BY_LEVEL = {level: lid for lid, level in POOL_LEARNERS.items()}
RATE_LIMIT_WAIT = 60


def demo_concepts_by_level() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with session_scope() as db:
        for lid in DEMO_LEARNERS:
            learner = db.get(Learner, lid)
            if learner is None:
                continue
            states = ordered_states({c: to_mastery_state(r) for c, r in load_state_rows(db, lid).items()})
            nodes = learner_path(learner, states)["nodes"]
            out.setdefault(learner.level, []).extend(n["conceptId"] for n in nodes)
    return out


def current_counts() -> Counter:
    with session_scope() as db:
        rows = db.execute(
            select(Question.concept_id, Question.level, Question.difficulty, func.count())
            .where(Question.verified.is_(True), Question.source == "llm")
            .group_by(Question.concept_id, Question.level, Question.difficulty)
        ).all()
    return Counter({(c, l, d): n for c, l, d, n in rows})


def coverage_table(cells, counts, per_cell) -> str:
    lines = ["| level | difficulty | cells | full | questions |", "|---|---|---|---|---|"]
    for level in LEVELS:
        for diff in DIFFICULTIES:
            cs = [c for c in cells if c[1] == level and c[2] == diff]
            if cs:
                full = sum(1 for c in cs if counts.get(c, 0) >= per_cell)
                lines.append(f"| {level} | {diff} | {len(cs)} | {full} | {sum(counts.get(c, 0) for c in cs)} |")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--gap", type=float, default=25.0)
    ap.add_argument("--only-demo", action="store_true", help="only concepts on the 3 demo learners' paths")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    create_tables()
    cells = cells_for(demo_concepts_by_level()) if args.only_demo else all_cells()
    counts = current_counts()
    todo = plan(cells, counts, args.per_cell)
    est = estimate(todo, args.gap)
    print(f"cells in scope: {len(cells)} | already full: {len(cells) - len(todo)} | to fill: {est['cells']} "
          f"({est['questions']} questions)")
    print(f"requests: {est['requests']} | LLM calls: {est['llm_calls_min']}-{est['llm_calls_max']} "
          f"(generate + verify, + regeneration round) | ~{est['minutes_at_gap']} min at a {args.gap:.0f}s gap")
    if args.dry_run:
        print(coverage_table(cells, counts, args.per_cell))
        return 0

    with httpx.Client(base_url=args.base, timeout=180) as client:
        for i, ((cid, level, diff), need) in enumerate(todo, 1):
            body = client.post("/api/v1/learnai", json={
                "action": "generate_questions", "learner_id": POOL_LEARNER_BY_LEVEL[level],
                "payload": {"target_concept": cid, "difficulty": diff, "count": min(need, 5)},
            }).json()
            data = body.get("data") or {}
            made = (data.get("sources") or {}).get("llm", 0)
            code = (body.get("error") or {}).get("code")
            print(f"[{i}/{len(todo)}] {cid:<24} {level:<12} {diff:<6} +{made} verified" + (f" ({code})" if code else ""))
            if code == "MODEL_RATE_LIMIT" and made == 0:
                print(f"   rate limited: waiting {RATE_LIMIT_WAIT}s (cell retried on the next run)")
                time.sleep(RATE_LIMIT_WAIT)
            elif i < len(todo):
                time.sleep(args.gap)
    print(coverage_table(cells, current_counts(), args.per_cell))
    return 0


if __name__ == "__main__":
    sys.exit(main())
