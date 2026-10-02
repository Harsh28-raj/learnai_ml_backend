"""Run every Swagger example against a live server and report success + latency (manual check).

    python scripts/check_examples.py --base https://<service>.onrender.com --gap 15

Run it on a freshly seeded database: the "assessment: new beginner" example returns
LEARNER_EXISTS if new-learner-01 already exists (the overwrite example always works).
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api_docs import EXAMPLES  # noqa: E402
from app.protection import LLM_ACTIONS  # noqa: E402


def detail(name: str, body: dict) -> str:
    d = body.get("data") or {}
    if "tutor" in name:
        return f"mode={d.get('mode')} concept={d.get('concept')} checkpoint={'yes' if d.get('checkpointQuestion') else 'no'}"
    if name.startswith("generate"):
        return f"count={d.get('count')} sources={d.get('sources')} difficulty={d.get('difficulty')}"
    if name.startswith("evaluate"):
        ad = d.get("adaptive_decision") or {}
        return f"score={d.get('score')} action={ad.get('action')}"
    if "path" in name:
        return f"current={(d.get('currentNode') or {}).get('conceptId')} why={d.get('whySource')} weeks={d.get('estimatedWeeksRemaining')}"
    if name.startswith("assessment"):
        return f"level={d.get('level')} firstStep={len((d.get('firstStep') or {}).get('conceptIds', []))} concepts"
    return f"learner={(d.get('profile') or {}).get('id')}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--gap", type=float, default=15.0, help="seconds before each AI-backed example")
    args = ap.parse_args()

    failures = 0
    with httpx.Client(base_url=args.base, timeout=120) as client:
        t = time.perf_counter()
        health = client.get("/health").json()
        print(f"health: {health} ({(time.perf_counter() - t) * 1000:.0f} ms)\n")
        print(f"| example | action | success | latency | notes |\n|---|---|---|---|---|")
        for name, ex in EXAMPLES.items():
            v = ex["value"]
            if v["action"] in LLM_ACTIONS:
                time.sleep(args.gap)
            t = time.perf_counter()
            body = client.post("/api/v1/learnai", json=v).json()
            ms = (time.perf_counter() - t) * 1000
            err = body.get("error") or {}
            note = detail(name, body) if body.get("success") else f"{err.get('code')}: {err.get('message')}"
            if body.get("success") and err:
                note += f" (notice {err.get('code')})"
            failures += 0 if body.get("success") else 1
            print(f"| {name} | {v['action']} | {body.get('success')} | {ms:.0f} ms | {note} |")
    print(f"\n{len(EXAMPLES) - failures}/{len(EXAMPLES)} examples succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
