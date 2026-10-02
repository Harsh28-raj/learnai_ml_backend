"""Live SPEC Section 11 demo for akshat-intermediate (manual; NOT part of unittest).

get_path (before) -> tutor_chat -> generate_questions (Bias vs Variance) -> wrong answer ->
get_profile (weakness + tutor alert) -> get_path (after: changes + beforeAfter) ->
Easy question -> 3 fast correct answers -> final mastery trend + path -> reset_learner.

    python scripts/demo_flow.py --base http://127.0.0.1:8000 --gap 20
"""

import argparse
import sys
import time

import httpx

LEARNER = "akshat-intermediate"


class Demo:
    def __init__(self, base: str, gap: float):
        self.client = httpx.Client(base_url=base, timeout=180)
        self.gap = gap

    def call(self, action: str, payload: dict | None = None, llm: bool = False) -> dict:
        if llm:
            time.sleep(self.gap)
        body = self.client.post("/api/v1/learnai", json={"action": action, "learner_id": LEARNER,
                                                         "payload": payload or {}}).json()
        if not body.get("success"):
            print(f"   ! {action} -> {body.get('error')}")
        elif body.get("error"):
            print(f"   (notice: {body['error']['code']}: {body['error']['message']})")
        return body


def step(n, title: str) -> None:
    print(f"\n{'=' * 8} STEP {n}: {title} {'=' * 8}")


def show_path(d: dict) -> None:
    cur = d.get("currentNode") or {}
    print(f"current: {cur.get('title')} [{cur.get('status')}] - {cur.get('reason')}")
    print("next: " + "; ".join(f"{n['title']} [{n['status']}]" for n in d.get("nextNodes", [])))
    print("order: " + " > ".join(
        f"{n['title']}{' (REV)' if n['isRevision'] else ''}" for n in d["nodes"] if n["status"] != "completed"))
    print(f"weeks remaining: {d['estimatedWeeksRemaining']} | beforeAfter: {d['beforeAfter']}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--gap", type=float, default=20.0, help="seconds before each LLM-backed call")
    args = ap.parse_args()
    demo = Demo(args.base, args.gap)

    demo.call("reset_learner")
    step(1, "get_path (before)")
    before = demo.call("get_path", llm=True)["data"]
    show_path(before)
    print("whyThisPath:", before["whyThisPath"], f"({before['whySource']})")

    step(2, 'tutor_chat "Why does my model overfit?"')
    t = demo.call("tutor_chat", {"message": "Why does my model overfit?"}, llm=True)
    td = t.get("data") or {}
    print(f"mode={td.get('mode')} concept={td.get('concept')} checkpoint={'yes' if td.get('checkpointQuestion') else 'no'}")
    print((td.get("message") or (t.get("fallback_data") or {}).get("reply", ""))[:600] + " ...")

    step(3, "generate_questions on Bias vs Variance")
    g = demo.call("generate_questions", {"target_concept": "Bias vs Variance", "count": 1}, llm=True)["data"]
    q = g["questions"][0]
    print(f"[{q['difficulty']}] ({q['source']}) {q['title']}\n{q['question']}")
    for i, o in enumerate(q["options"]):
        print(f"   ({i}) {o}")
    print("why:", q["whyThisQuestion"])

    step(4, "evaluate a WRONG answer")
    wrong = (q["correctIndex"] + 1) % 4
    e = demo.call("evaluate", {"question_id": q["id"], "selected_option_index": wrong, "time_taken_seconds": 40})["data"]
    mu, ad = e["mastery_updates"][0], e["adaptive_decision"]
    print(f"mastery {mu['concept']}: {mu['previous_score']} -> {mu['new_score']} (trend {mu['trend']})")
    print(f"next difficulty: {e['next_difficulty']['difficulty']} ({e['next_difficulty']['reason']})")
    print(f"adaptive event: action={ad['action']} score={ad['score']}\n   reason: {ad['reason']}\n"
          f"   pathAdjustment: {ad['pathAdjustment']}")

    step("4b", "two more misses on Overfitting (the topic the learner asked the tutor about)")
    quiz = {"quiz": True, "topic": "Overfitting", "answers": [
        {"concept_tested": "Overfitting", "difficulty": "Medium", "is_correct": False, "time_taken_seconds": 50}] * 2}
    e = demo.call("evaluate", quiz)["data"]
    mu, ad = e["mastery_updates"][0], e["adaptive_decision"]
    print(f"mastery {mu['concept']}: {mu['previous_score']} -> {mu['new_score']} | action={ad['action']} "
          f"score={ad['score']}\n   pathAdjustment: {ad['pathAdjustment']}")

    step(5, "get_profile: weakness report + tutor alert")
    p = demo.call("get_profile")["data"]
    for w in p["weakness_report"]["weak"][:3]:
        print(f"weak: {w['name']} {w['mastery']} {w['label']} | alert: {w['tutor_alert']}")

    step(6, "get_path (after): changes + beforeAfter")
    after = demo.call("get_path", llm=True)["data"]
    show_path(after)
    for c in after["changes"]:
        print(f"change: {c['type']} - {c['detail']}")

    step(7, "generate an Easy question + 3 fast correct answers")
    for i in range(3):
        g = demo.call("generate_questions", {"target_concept": "Bias vs Variance", "difficulty": "Easy", "count": 1},
                      llm=True)["data"]
        q = g["questions"][0]
        e = demo.call("evaluate", {"question_id": q["id"], "selected_option_index": q["correctIndex"],
                                   "time_taken_seconds": 12})["data"]
        mu = e["mastery_updates"][0]
        print(f"#{i + 1} ({q['source']}) {q['title'][:60]} -> mastery {mu['previous_score']} -> {mu['new_score']}, "
              f"next {e['next_difficulty']['difficulty']} [{e['next_difficulty']['rule_id']}], "
              f"action {e['adaptive_decision']['action']}")

    step(8, "final mastery trend and path")
    p = demo.call("get_profile")["data"]
    cs = next(c for c in p["profile"]["conceptStates"] if c["concept"] == "bias_variance")
    print(f"bias_variance: mastery {cs['mastery']} trend {cs['trend']} confidence {cs['confidence']}")
    final = demo.call("get_path")["data"]
    show_path(final)

    demo.call("reset_learner")
    print("\nreset_learner done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
