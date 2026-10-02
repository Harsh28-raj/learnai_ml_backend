"""Live smoke test for generate_questions (manual; NOT part of the unittest suite).

Calls a running server, which uses the real GROQ_API_KEY from its .env.

    uvicorn app.main:app --port 8000
    python scripts/smoke_generate.py [--base http://127.0.0.1:8000] [--count 2] [--dump out.json]
"""

import argparse
import json
import sys
import time

import httpx

LEARNERS = ("alex-beginner", "akshat-intermediate", "elena-advanced")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # model output may contain non-cp1252 characters
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--dump", help="write full responses to this JSON file")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="seconds between learners (Groq free tier has a low tokens-per-minute limit)")
    args = ap.parse_args()

    dump: dict = {}
    latencies: list[float] = []
    failures = 0
    with httpx.Client(base_url=args.base, timeout=120) as client:
        for i, learner_id in enumerate(LEARNERS):
            if i and args.gap:
                time.sleep(args.gap)
            started = time.perf_counter()
            resp = client.post("/api/v1/learnai", json={
                "action": "generate_questions", "learner_id": learner_id, "payload": {"count": args.count},
            })
            latency = time.perf_counter() - started
            latencies.append(latency)
            body = resp.json()
            dump[learner_id] = body

            print(f"\n=== {learner_id}  ({latency:.1f}s, HTTP {resp.status_code})")
            if body.get("success"):
                d = body["data"]
                gen = d.get("generation", {})
                print(f"concept={d['concept']} ({d['selection_reason']})  difficulty={d['difficulty']}  "
                      f"count={d['count']}/{d['requested_count']}  sources={d.get('sources')}")
                print(f"llm_calls={gen.get('llm_calls')}  verify_calls={gen.get('verify_calls')}  "
                      f"verified={gen.get('verified')}  rejected={gen.get('rejected')}  "
                      f"server_latency={gen.get('latency_ms')}ms")
                if body.get("error"):
                    print(f"notice: {body['error']['code']}: {body['error']['message']}")
                for e in gen.get("rejection_issues") or []:
                    print(f"  rejected: {e}")
                questions = d["questions"]
            else:
                failures += 1
                err = body.get("error") or {}
                print(f"FAILED  code={err.get('code')}  message={err.get('message')}")
                questions = (body.get("fallback_data") or {}).get("questions", [])
                if questions:
                    print("fallback questions served:")
            for q in questions:
                print(f"  - [{q['difficulty']}] ({q.get('source')}) {q['title']}")
                print(f"    domain={q['scenarioDomain']}  angle={q['questionAngle']}")
                print(f"    why: {q['whyThisQuestion']}")

    print(f"\naverage end-to-end latency: {sum(latencies) / len(latencies):.1f}s over {len(latencies)} calls")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2, ensure_ascii=False)
        print(f"full responses written to {args.dump}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
