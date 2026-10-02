"""Phase 5 tests: recommender, path diff/snapshots, whyThisPath, daily plan, pending Phase 4 fixes,
warm-pool planning and the full SPEC Section 11 loop over HTTP. LLM always mocked."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from sqlalchemy import func, select

from tests import _support
from tests._support import app_db, call
from tests.test_questions import STEMS, approve_all, item, llm_response
from tests.test_tutor import CORRECT, checkpoint, reply, verify_ok

from app.config import Settings
from app.engine.cold_start import baseline_masteries, baseline_states
from app.engine.concepts import CONCEPTS
from app.engine.goals import goal_targets, resolve_goal
from app.engine.learner_model import MasteryState
from app.engine.pool_plan import cell_allowed, cells_for, estimate, plan
from app.engine.recommender import (
    ancestors,
    before_after,
    build_path,
    daily_plan,
    diff_paths,
    estimate_minutes,
    priority_key,
)
from app.engine.tutor_guardrails import balance_math, check_reply
from app.llm import why_path
from app.llm.groq_client import LLMError
from app.llm.verifier import difficulty_mismatch, level_mismatch
from app.models import AdaptiveEvent, LearningPath
from app.seed import DEMO_LEARNERS, demo_states

AKSHAT, ALEX, ELENA = "akshat-intermediate", "alex-beginner", "elena-advanced"


def st(cid, mastery, attempts=5, results="+++++", cw=0, trend="stable"):
    recent = [{"correct": c == "+", "difficulty": "Medium", "time_taken": 40, "ts": "t"} for c in results]
    return MasteryState(cid, CONCEPTS[cid]["category"], mastery, attempts, sum(c == "+" for c in results), recent,
                        cw, 0.5, trend)


def all_states(default=90.0, **overrides):
    out = {cid: st(cid, default) for cid in CONCEPTS}
    out.update(overrides)
    return list(out.values())


def pending(path):
    return [n["conceptId"] for n in path["nodes"] if n["status"] != "completed"]


def demo_path(lid):
    pr = DEMO_LEARNERS[lid]["profile"]
    return build_path(demo_states(lid), resolve_goal(pr["goal"]), target_topics=pr["target_topics"],
                      pace=pr["learning_pace"], daily_minutes=pr["daily_commitment_minutes"])


# =========================================================================== recommender (pure)

class TestGoals(unittest.TestCase):
    def test_resolve_free_text(self):
        cases = {"Become an ML Engineer": "ML Engineer", "Build GenAI Apps & Fine-tune LLMs": "GenAI Apps",
                 "Learn ML from Scratch": "ML from Scratch", "crack FAANG interviews": "Interviews",
                 "ml engneer": "ML Engineer", "": "ML Engineer", "something unrelated": "ML Engineer",
                 "Interviews": "Interviews"}
        for text, key in cases.items():
            with self.subTest(text=text):
                self.assertEqual(resolve_goal(text), key)

    def test_target_topics_added(self):
        t = goal_targets("GenAI Apps", ["Bias vs Variance", "Deep Learning", "MLOps"])
        self.assertIn("bias_variance", t)
        self.assertTrue({"neural_networks", "backpropagation", "cnns"} <= set(t))


class TestBuildPath(unittest.TestCase):
    def test_closure(self):
        p = build_path(all_states(default=10, ), "GenAI Apps")
        nodes = {n["conceptId"] for n in p["nodes"]}
        expected = set(p["targets"])
        for t in p["targets"]:
            expected |= ancestors(t)
        self.assertEqual(nodes, expected)
        self.assertNotIn("cnns", nodes)

    def test_skipping_unpracticed_known_concepts(self):
        states = baseline_states(baseline_masteries("Intermediate", ["Python"], ["NumPy", "Pandas"]))
        p = build_path(states, "ML Engineer")
        py = next(n for n in p["nodes"] if n["conceptId"] == "python_basics")
        self.assertEqual((py["status"], py["skipped"], py["progress"]), ("completed", True, 100))
        self.assertEqual(py["reason"], "Skipped: you already know this (mastery 85).")

    def test_revision_injection_and_priority_order(self):
        states = all_states(std_variance=st("std_variance", 30, results="+----", cw=3, trend="declining"),
                            numpy_arrays=st("numpy_arrays", 55, results="-+-++", trend="improving"))
        p = build_path(states, "ML Engineer")
        self.assertEqual(pending(p)[:2], ["std_variance", "numpy_arrays"])  # rule a: priority desc
        first, second = [n for n in p["nodes"] if n["status"] != "completed"][:2]
        self.assertTrue(first["isRevision"] and second["isRevision"])
        self.assertEqual((first["status"], second["status"]), ("current", "adapted"))
        self.assertIn("HIGH priority weakness", first["reason"])

    def test_foundation_first(self):
        states = all_states(bias_variance=st("bias_variance", 40, results="--+--", cw=2),
                            std_variance=st("std_variance", 50, attempts=0, results=""),
                            pandas_dataframes=st("pandas_dataframes", 50, attempts=0, results=""))
        p = build_path(states, "ML Engineer")
        self.assertEqual(pending(p), ["std_variance", "bias_variance", "pandas_dataframes"])  # rule b
        self.assertTrue(p["nodes"][-3]["reason"].startswith("Moved earlier: foundation for Bias vs Variance"))

    def test_goal_relevance_beats_category(self):
        states = all_states(tokenization=st("tokenization", 50, attempts=0, results=""),
                            linear_regression=st("linear_regression", 50, attempts=0, results=""))
        self.assertEqual(pending(build_path(states, "GenAI Apps")), ["tokenization", "linear_regression"])  # rule c

    def test_category_and_id_tiebreaks(self):
        k = dict(is_revision=False, weakness_priority=0.0, is_foundation=False, distance=1)
        self.assertLess(priority_key("numpy_arrays", **k), priority_key("probability_basics", **k))  # rule d
        self.assertLess(priority_key("distributions", **k), priority_key("std_variance", **k))  # rule e
        self.assertLess(priority_key("cnns", is_revision=True, weakness_priority=10, is_foundation=False, distance=9),
                        priority_key("python_basics", **k))

    def test_statuses_and_locking(self):
        p = demo_path(AKSHAT)
        todo = [n for n in p["nodes"] if n["status"] != "completed"]
        self.assertEqual(todo[0]["status"], "current")
        self.assertEqual([n["conceptId"] for n in todo[:3]], ["probability_basics", "std_variance", "bias_variance"])
        self.assertEqual([n["status"] for n in todo[1:3]], ["adapted", "adapted"])
        locked = [n for n in todo if n["status"] == "locked"]
        self.assertTrue(locked)
        self.assertTrue(all("Unlocks after:" in n["reason"] for n in locked))
        self.assertTrue(all(n["progress"] == 100 for n in p["nodes"] if n["status"] == "completed"))
        self.assertEqual([n["order"] for n in p["nodes"]], list(range(1, len(p["nodes"]) + 1)))

    def test_estimates(self):
        self.assertEqual(estimate_minutes("python_basics", 0, "Balanced"), 30)
        self.assertEqual(estimate_minutes("rag_chunking", 68, "Intensive"), 16)
        self.assertEqual(estimate_minutes("python_basics", 95, "Relaxed"), 10)  # min 10
        self.assertEqual(estimate_minutes("neural_networks", 0, "Relaxed"), 72)
        p = demo_path(ALEX)
        self.assertEqual(p["totalEstimatedMinutes"], sum(n["estimatedMinutes"] for n in p["nodes"]))
        self.assertEqual(p["estimatedWeeksRemaining"], -(-p["totalEstimatedMinutes"] // (30 * 6)))

    def test_foundation_fact_names_the_weak_concept(self):
        from app.engine.recommender import why_facts
        p = demo_path(AKSHAT)
        prob = next(n for n in p["nodes"] if n["conceptId"] == "probability_basics")
        self.assertEqual(prob["foundationFor"], "bias_variance")
        facts = " ".join(why_facts(p, pace="Balanced", daily_minutes=45))
        self.assertIn("Probability Fundamentals is not a weak area itself; it comes first because it is a "
                      "prerequisite of Bias vs Variance Tradeoff", facts)

    def test_no_language_learner_starts_with_python(self):
        states = baseline_states(baseline_masteries("Beginner", [], []))
        no_lang = build_path(states, "ML Engineer", knows_a_language=False)
        default = build_path(states, "ML Engineer")
        self.assertEqual(pending(no_lang)[:3], ["python_basics", "python_data_structures", "numpy_arrays"])
        self.assertEqual(pending(default)[0], "probability_basics")  # goal relevance first for everyone else
        k = dict(is_revision=False, weakness_priority=0.0, is_foundation=False)
        self.assertLess(priority_key("python_basics", distance=3, category_first=True, **k),
                        priority_key("probability_basics", distance=1, category_first=True, **k))

    def test_deterministic(self):
        self.assertEqual(demo_path(AKSHAT), demo_path(AKSHAT))

    def test_demo_learners_get_different_paths(self):
        sigs = {lid: tuple((n["conceptId"], n["status"]) for n in demo_path(lid)["nodes"]) for lid in DEMO_LEARNERS}
        self.assertEqual(len(set(sigs.values())), 3)
        self.assertEqual(pending(demo_path(ELENA)), ["rag_chunking"])

    def test_same_goal_different_masteries(self):
        beginner = build_path(baseline_states(baseline_masteries("Beginner", [], [])), "ML Engineer")
        intermediate = build_path(baseline_states(baseline_masteries(
            "Intermediate", ["Python"], ["NumPy", "Pandas", "Statistics"])), "ML Engineer")
        b = {n["conceptId"]: n for n in beginner["nodes"]}
        i = {n["conceptId"]: n for n in intermediate["nodes"]}
        self.assertNotEqual(b["python_basics"]["status"], "completed")
        self.assertTrue(i["python_basics"]["skipped"])
        self.assertGreater(beginner["totalEstimatedMinutes"], intermediate["totalEstimatedMinutes"])


class TestDiffAndText(unittest.TestCase):
    @staticmethod
    def n(cid, status, rev=False, skipped=False):
        return {"conceptId": cid, "title": CONCEPTS[cid]["name"], "status": status, "isRevision": rev,
                "skipped": skipped, "reason": "r", "mastery": 80}

    def test_change_types(self):
        old = [self.n("python_basics", "current"), self.n("numpy_arrays", "recommended"),
               self.n("matrix_shapes", "locked"), self.n("overfitting", "locked"), self.n("cnns", "locked")]
        new = [self.n("python_basics", "completed"), self.n("numpy_arrays", "completed", skipped=True),
               self.n("overfitting", "current", rev=True), self.n("matrix_shapes", "recommended"),
               self.n("cnns", "locked")]
        kinds = {c["type"]: c["conceptId"] for c in diff_paths(old, new)}
        self.assertEqual(kinds, {"completed": "python_basics", "skipped": "numpy_arrays",
                                 "inserted_revision": "overfitting", "unlocked": "matrix_shapes"})
        reorder = diff_paths([self.n("python_basics", "current"), self.n("numpy_arrays", "recommended")],
                             [self.n("numpy_arrays", "current"), self.n("python_basics", "recommended")])
        self.assertEqual([c["type"] for c in reorder], ["reordered"])

    def test_before_after_text(self):
        a = [self.n("python_basics", "current")]
        self.assertEqual(before_after(None, a, []), "Initial path created: start with Python Basics (Variables, Loops, Functions).")
        self.assertEqual(before_after(a, a, []), "Roadmap unchanged: current focus remains Python Basics (Variables, Loops, Functions).")
        b = [self.n("overfitting", "current", rev=True), self.n("python_basics", "recommended")]
        text = before_after(a, b, diff_paths(a, b))
        self.assertTrue(text.startswith("Before: next up was Python Basics"))
        self.assertIn("Overfitting & Underfitting revision added", text)
        self.assertIn("current focus is Overfitting & Underfitting", text)

    def test_daily_plan_fits_budget(self):
        p = demo_path(AKSHAT)
        for budget in (10, 15, 30, 45, 90):
            with self.subTest(budget=budget):
                tasks = daily_plan(p, demo_states(AKSHAT), budget, "Medium")
                self.assertLessEqual(sum(t["durationMinutes"] for t in tasks), budget)
                self.assertTrue(all(t["durationMinutes"] >= 5 and t["completed"] is False for t in tasks))
                types = [t["type"] for t in tasks]
                self.assertEqual(types, sorted(types, key=["revision", "practice", "lesson"].index))
        elena = demo_path("elena-advanced")
        lesson = [t for t in daily_plan(elena, demo_states("elena-advanced"), 90, "Medium") if t["type"] == "lesson"]
        self.assertEqual(lesson[0]["durationMinutes"], 16)  # capped at the node estimate, not the 90-min budget
        tasks = daily_plan(p, demo_states(AKSHAT), 45, "Medium")
        self.assertEqual([t["type"] for t in tasks], ["revision", "practice", "lesson"])
        self.assertEqual(tasks[0]["conceptId"], "bias_variance")
        self.assertEqual(tasks[1]["conceptId"], "probability_basics")


class TestWhyThisPath(unittest.TestCase):
    FACTS = ["Revision added for Bias vs Variance Tradeoff (mastery 54).", "Pace: Balanced, 45 minutes a day."]

    def test_number_validation(self):
        self.assertIsNone(why_path.validate_why("Your 54 mastery at 45 minutes a day.", self.FACTS))
        self.assertIn("numbers not in facts", why_path.validate_why("You'll finish in 3 weeks.", self.FACTS))
        self.assertEqual(why_path.validate_why(" ", self.FACTS), "empty")
        self.assertIn("numbers not in facts", why_path.validate_why("Done in three weeks.", self.FACTS))
        self.assertEqual(why_path.validate_why("word " * 120, self.FACTS), "too long")

    def run_polish(self, side_effect):
        with patch("app.llm.why_path.get_settings", return_value=Settings(groq_api_key="k")), \
                patch("app.llm.why_path.chat_completion", AsyncMock(side_effect=side_effect)) as m:
            out = asyncio.run(why_path.polish_why("Akshat", self.FACTS, "TEMPLATE"))
        return out, m

    def test_polish_paths(self):
        out, m = self.run_polish([{"why_this_path": "Fixing Bias vs Variance (54) first will pay off at 45 minutes a day."}])
        self.assertEqual(out[1], "llm")
        self.assertEqual(m.await_args.kwargs["temperature"], 0.3)
        self.assertEqual(self.run_polish([{"why_this_path": "Done in 3 weeks!"}])[0], ("TEMPLATE", "template"))
        self.assertEqual(self.run_polish([LLMError("MODEL_RATE_LIMIT", "x")])[0], ("TEMPLATE", "template"))

    def test_no_key_means_template_without_call(self):
        out, m = asyncio.run(why_path.polish_why("A", self.FACTS, "T")), None
        self.assertEqual(out, ("T", "template"))


# =========================================================================== pending Phase 4 fixes (pure)

class TestPhase4Fixes(unittest.TestCase):
    def test_difficulty_mismatch_rules(self):
        self.assertIsNotNone(difficulty_mismatch("Easy", "Hard"))
        self.assertIsNotNone(difficulty_mismatch("Hard", "Easy"))
        self.assertIsNone(difficulty_mismatch("Medium", "Hard"))
        self.assertIsNone(difficulty_mismatch(None, "Hard"))
        self.assertIsNotNone(level_mismatch("Easy", "Advanced"))
        self.assertIsNotNone(level_mismatch("Hard", "Beginner"))
        self.assertIsNone(level_mismatch("Easy", "Intermediate"))

    def test_katex_balance(self):
        self.assertEqual(balance_math("ok $x$ and $$y$$ and \\(z\\)")[1:], (0, None))
        text, n, issue = balance_math("Update \\[\\theta - \\eta g\n\nNext para $a + b")
        self.assertEqual((n, issue), (2, None))
        self.assertTrue(text.split("\n\n")[0].endswith("\\]"))
        self.assertTrue(text.endswith("$"))
        self.assertIn("without a matching", balance_math("orphan \\) closer")[2])
        code = "```python\nprint('$ and \\\\[')\n```"
        self.assertEqual(balance_math(code), (code, 0, None))  # never counted inside code
        self.assertEqual(balance_math("use `$PATH` here")[1], 0)
        self.assertEqual(balance_math("The GPU costs $50 per day and $1,200.50 a month."), (
            "The GPU costs $50 per day and $1,200.50 a month.", 0, None))
        self.assertEqual(balance_math("Budget $50 and the loss $L$ matter")[1:], (0, None))
        self.assertEqual(balance_math("Inline $2x + 1$ is math")[1:], (0, None))
        _, violations, _ = check_reply("bad \\] math", level="Advanced", mode="explanation")
        self.assertTrue(violations[0].startswith("katex_unbalanced"))


class TestPoolPlan(unittest.TestCase):
    def test_cell_rules(self):
        self.assertFalse(cell_allowed("cnns", "Beginner", "Hard"))
        self.assertFalse(cell_allowed("cnns", "Advanced", "Easy"))
        self.assertFalse(cell_allowed("python_basics", "Intermediate", "Hard"))
        self.assertTrue(cell_allowed("cnns", "Intermediate", "Hard"))
        cells = cells_for({"Beginner": ["python_basics"], "Advanced": ["rag_chunking", "rag_chunking"]})
        self.assertEqual(cells, [("python_basics", "Beginner", "Easy"), ("python_basics", "Beginner", "Medium"),
                                 ("rag_chunking", "Advanced", "Medium"), ("rag_chunking", "Advanced", "Hard")])

    def test_resume_and_estimate(self):
        cells = [("a", "Beginner", "Easy"), ("b", "Beginner", "Easy"), ("c", "Beginner", "Medium")]
        todo = plan(cells, {cells[0]: 2, cells[1]: 1}, per_cell=2)
        self.assertEqual(todo, [(cells[1], 1), (cells[2], 2)])
        self.assertEqual(estimate(todo, 30), {"cells": 2, "questions": 3, "requests": 2, "llm_calls_min": 4,
                                              "llm_calls_max": 8, "minutes_at_gap": 1.0})


# =========================================================================== HTTP

class PathApiCase(unittest.TestCase):
    def setUp(self):
        _support.client()
        _support.clear_generated_questions()
        for lid in DEMO_LEARNERS:
            self.assertTrue(call("reset_learner", learner_id=lid)["success"])

    @staticmethod
    def snapshots(lid=AKSHAT):
        with app_db.session_scope() as s:
            return s.scalar(select(func.count()).select_from(LearningPath).where(LearningPath.learner_id == lid))


class TestGetPathApi(PathApiCase):
    def test_shape_snapshot_and_cache(self):
        with patch("app.actions.path.polish_why", AsyncMock(return_value=("Polished 54 text.", "llm"))) as m:
            first = call("get_path")
            second = call("get_path")
        self.assertTrue(first["success"], first)
        d = first["data"]
        for key in ("goal", "targets", "nodes", "milestones", "currentNode", "nextNodes", "changes", "beforeAfter",
                    "whyThisPath", "estimatedWeeksRemaining", "totalEstimatedMinutes", "dailyPlan"):
            self.assertIn(key, d)
        self.assertEqual((d["goal"], d["currentNode"]["conceptId"]), ("ML Engineer", "probability_basics"))
        self.assertEqual(d["beforeAfter"], "Initial path created: start with Probability Fundamentals.")
        self.assertEqual(m.await_count, 1)  # cache hit on the second call: no new LLM call
        self.assertEqual((second["data"]["whyThisPath"], second["data"]["changedNow"]), ("Polished 54 text.", False))
        self.assertEqual(second["data"]["beforeAfter"],
                         "Roadmap unchanged: current focus remains Probability Fundamentals.")
        self.assertEqual(self.snapshots(), 1)
        self.assertEqual({s["category"] for s in d["milestones"]}, {n["category"] for n in d["nodes"]})

    def test_template_when_llm_unavailable(self):
        d = call("get_path", learner_id=ELENA)["data"]
        self.assertEqual(d["whySource"], "template")
        self.assertIn("Vector DB Embedding Chunking", d["whyThisPath"])
        self.assertEqual([n["conceptId"] for n in d["nodes"] if n["status"] != "completed"], ["rag_chunking"])

    def test_preview_goal_not_saved(self):
        call("get_path")
        d = call("get_path", {"goal": "GenAI Apps"})["data"]
        self.assertTrue(d["preview"])
        self.assertEqual(d["goal"], "GenAI Apps")
        self.assertEqual(self.snapshots(), 1)

    def test_evaluate_writes_real_path_adjustment(self):
        call("get_path")
        e = call("evaluate", {"concept_tested": "Bias vs Variance", "difficulty": "Medium", "is_correct": False})
        self.assertEqual(e["data"]["adaptive_decision"]["pathAdjustment"],
                         "Roadmap unchanged: current focus remains Probability Fundamentals.")
        quiz = {"quiz": True, "topic": "Overfitting", "answers": [
            {"concept_tested": "Overfitting", "difficulty": "Medium", "is_correct": False}] * 2}
        e = call("evaluate", quiz)
        adj = e["data"]["adaptive_decision"]["pathAdjustment"]
        self.assertTrue(adj.startswith("Before: next up was Probability Fundamentals."), adj)
        self.assertIn("Overfitting & Underfitting revision added", adj)
        d = call("get_path")["data"]
        self.assertFalse(d["changedNow"])  # evaluate already snapshotted it
        self.assertIn("inserted_revision", [c["type"] for c in d["changes"]])
        node = next(n for n in d["nodes"] if n["conceptId"] == "overfitting")
        self.assertTrue(node["isRevision"])

    def test_tutor_path_mode_uses_real_path(self):
        with patch("app.actions.tutor.chat_completion", AsyncMock(return_value=reply("Because..."))) as m:
            out = call("tutor_chat", {"message": "why am I learning this next?"})
        self.assertEqual(out["data"]["mode"], "path")
        system = m.await_args.args[0][0]["content"]
        self.assertIn("LEARNING PATH (goal: become an ML Engineer): current node Probability Fundamentals", system)
        self.assertIn("Next nodes: Standard Deviation & Variance (adapted)", system)

    def test_pool_learners_hidden(self):
        self.assertEqual(call("get_profile", learner_id="pool-beginner")["error"]["code"], "LEARNER_NOT_FOUND")
        self.assertEqual(call("get_path", learner_id="pool-advanced")["error"]["code"], "LEARNER_NOT_FOUND")
        with patch("app.actions.questions.chat_completion", AsyncMock(return_value=llm_response(item(0)))), \
                patch("app.actions.questions.verify_questions", AsyncMock(side_effect=approve_all)):
            g = call("generate_questions", {"target_concept": "overfitting", "difficulty": "Hard", "count": 1},
                     learner_id="pool-advanced")
        self.assertTrue(g["success"], g)
        self.assertEqual(g["data"]["difficulty"], "Hard")


class TestPhase4FixesApi(PathApiCase):
    def test_evaluate_mode_event_uses_real_score(self):
        ev = {"score": 40, "correct_points": ["a"], "misconceptions": ["b"], "corrected_answer": "c"}
        with patch("app.actions.tutor.chat_completion", AsyncMock(return_value=reply("Partly.", evaluation=ev))):
            out = call("tutor_chat", {"message": "grade this", "student_answer": "bias is noise",
                                      "current_topic": "Bias vs Variance"})
        d = out["data"]
        self.assertEqual((d["adaptiveDecision"]["score"], d["adaptiveDecision"]["action"]), (40, "reduced"))
        self.assertIn("You scored 40%", d["adaptiveDecision"]["reason"])
        with app_db.session_scope() as s:
            ev_row = s.scalars(select(AdaptiveEvent).where(AdaptiveEvent.learner_id == AKSHAT)).first()
            self.assertEqual(ev_row.score, 40)

    def test_generation_rejects_two_level_difficulty_gap(self):
        async def perceived_easy(questions):
            out = await approve_all(questions)
            for v in out.values():
                v["perceived_difficulty"] = "Easy"
            return out

        with patch("app.actions.questions.chat_completion",
                   AsyncMock(side_effect=[llm_response(item(0)), llm_response(item(1))])), \
                patch("app.actions.questions.verify_questions", AsyncMock(side_effect=perceived_easy)):
            out = call("generate_questions", {"difficulty": "Hard", "count": 1})
        g = out["data"]["generation"]
        self.assertEqual(out["data"]["sources"]["llm"], 0)
        self.assertTrue(any("rated it Easy but Hard was requested" in i for i in g["rejection_issues"]))

    def test_checkpoint_level_rules(self):
        async def perceived(level):
            async def v(questions):
                out = await verify_ok(questions)
                for x in out.values():
                    x["perceived_difficulty"] = level
                return out
            return v

        cases = [(ELENA, "Easy", False), (ALEX, "Hard", False), (AKSHAT, "Easy", True), (ELENA, "Hard", True)]
        for lid, level, kept in cases:
            with self.subTest(learner=lid, perceived=level):
                with patch("app.actions.tutor.chat_completion", AsyncMock(return_value=reply("x", cp=checkpoint()))), \
                        patch("app.actions.tutor.verify_questions",
                              AsyncMock(side_effect=asyncio.run(perceived(level)))):
                    d = call("tutor_chat", {"message": "What is overfitting?"}, learner_id=lid)["data"]
                self.assertEqual(d["checkpointQuestion"] is not None, kept, d["guardrails"]["checkpoint"])

    def test_unbalanced_katex_triggers_repair(self):
        with patch("app.actions.tutor.chat_completion",
                   AsyncMock(side_effect=[reply("Step \\) broken"), reply("Fixed $x$ math.")])) as m:
            d = call("tutor_chat", {"message": "What is gradient descent?"})["data"]
        self.assertEqual(m.await_count, 2)
        self.assertTrue(d["guardrails"]["repaired"])
        self.assertIn("katex_unbalanced", d["guardrails"]["violations"][0])


class TestSpecSection11FullLoop(PathApiCase):
    """SPEC 11 end to end over HTTP with the LLM mocked."""

    def test_loop(self):
        before = call("get_path")["data"]
        self.assertEqual(before["currentNode"]["conceptId"], "probability_basics")

        with patch("app.actions.tutor.chat_completion", AsyncMock(return_value=reply("Overfitting means...",
                                                                                   cp=checkpoint()))), \
                patch("app.actions.tutor.verify_questions", AsyncMock(side_effect=verify_ok)):
            t = call("tutor_chat", {"message": "Why does my model overfit?"})["data"]
        self.assertEqual((t["mode"], t["concept"]), ("explanation", "overfitting"))

        def gen(i, payload):
            with patch("app.actions.questions.chat_completion", AsyncMock(return_value=llm_response(item(i)))), \
                    patch("app.actions.questions.verify_questions", AsyncMock(side_effect=approve_all)):
                return call("generate_questions", payload)["data"]["questions"][0]

        q = gen(0, {"target_concept": "Bias vs Variance", "count": 1})
        wrong = (q["correctIndex"] + 1) % 4
        e = call("evaluate", {"question_id": q["id"], "selected_option_index": wrong, "time_taken_seconds": 40})["data"]
        self.assertEqual(e["adaptive_decision"]["action"], "reduced")
        self.assertEqual(e["next_difficulty"]["difficulty"], "Easy")

        weak = call("get_profile")["data"]["weakness_report"]["weak"]
        bv = next(w for w in weak if w["concept"] == "bias_variance")
        self.assertEqual(bv["label"], "HIGH")
        self.assertIn("struggling with Bias vs Variance", bv["tutor_alert"])

        after = call("get_path")["data"]
        self.assertTrue(after["beforeAfter"])
        self.assertIn("bias_variance", [n["conceptId"] for n in after["nodes"] if n["isRevision"]])

        masteries = []
        for i in (1, 2, 3):
            q = gen(i, {"target_concept": "Bias vs Variance", "difficulty": "Easy", "count": 1})
            self.assertEqual(q["difficulty"], "Easy")
            e = call("evaluate", {"question_id": q["id"], "selected_option_index": q["correctIndex"],
                                  "time_taken_seconds": 12})["data"]
            masteries.append(e["mastery_updates"][0]["new_score"])
        self.assertEqual(masteries, sorted(masteries))
        self.assertEqual(e["next_difficulty"]["difficulty"], "Medium")
        cs = next(c for c in call("get_profile")["data"]["profile"]["conceptStates"] if c["concept"] == "bias_variance")
        self.assertEqual(cs["trend"], "improving")
        self.assertTrue(call("get_path")["success"])


if __name__ == "__main__":
    unittest.main()
