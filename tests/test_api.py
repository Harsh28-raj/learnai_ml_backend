"""Integration tests through the HTTP endpoint against a temporary SQLite database."""

import unittest

from sqlalchemy import func, select

from tests import _support
from tests._support import app_db, call

from app.models import AdaptiveEvent, Attempt, ConceptState, Learner, Question

LEARNER = "akshat-intermediate"


def setUpModule():
    _support.client()


def count(model, learner_id: str = LEARNER) -> int:
    with app_db.session_scope() as s:
        return s.scalar(select(func.count()).select_from(model).where(model.learner_id == learner_id))


def concept_row(concept_id: str, learner_id: str = LEARNER) -> ConceptState:
    with app_db.session_scope() as s:
        return s.scalar(select(ConceptState).where(ConceptState.learner_id == learner_id,
                                                   ConceptState.concept_id == concept_id))


def learner_row(learner_id: str = LEARNER) -> Learner:
    with app_db.session_scope() as s:
        return s.get(Learner, learner_id)


WRONG_MEDIUM_BV = {
    "question_id": "ml-q-bv-1", "concept_tested": "Bias vs Variance", "category": "Machine Learning",
    "difficulty": "Medium", "selected_option_index": 0, "correct_index": 1, "time_taken_seconds": 38,
}


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.assertTrue(call("reset_learner")["success"])


class TestEvaluateSingle(ApiTestCase):
    def test_wrong_medium_answer(self):
        before = learner_row()
        out = call("evaluate", WRONG_MEDIUM_BV)
        self.assertTrue(out["success"], out)
        d = out["data"]
        self.assertEqual(d["results"], [{"question_id": "ml-q-bv-1", "is_correct": False, "correct_index": 1,
                                         "concept": "bias_variance"}])
        self.assertEqual(d["score"], 0)
        mu = d["mastery_updates"][0]
        self.assertEqual(mu["previous_score"], 54.0)
        self.assertAlmostEqual(mu["new_score"], 54 - 6 * 0.54 * 1.5, places=1)
        self.assertLess(mu["new_category_score"], mu["previous_category_score"])
        self.assertEqual(d["next_difficulty"]["difficulty"], "Easy")
        ev = d["adaptive_decision"]
        self.assertEqual(ev["action"], "reduced")
        self.assertTrue(ev["id"].startswith("adapt-"))
        self.assertTrue(ev["timestamp"].startswith("Today at "))
        for key in ("topic", "score", "reason", "recommendation", "pathAdjustment", "createdAt"):
            self.assertIn(key, ev)
        self.assertIn("bias_variance", [e["concept"] for e in d["weakness_report"]["weak"]])

        # DB state changed.
        self.assertEqual(count(Attempt), 1)
        self.assertEqual(count(AdaptiveEvent), 1)
        row = concept_row("bias_variance")
        self.assertAlmostEqual(row.mastery, 49.14, places=2)
        self.assertEqual(row.consecutive_wrong, 1)
        after = learner_row()
        self.assertEqual(after.questions_solved, before.questions_solved + 1)
        expected_acc = round(before.accuracy_rate * before.questions_solved / (before.questions_solved + 1), 1)
        self.assertAlmostEqual(after.accuracy_rate, expected_acc, places=1)

        prof = call("get_profile")["data"]
        self.assertEqual(len(prof["profile"]["recentAdaptiveEvents"]), 1)
        self.assertIn("weakness_report", prof)

    def test_is_correct_fallback(self):
        out = call("evaluate", {"concept_tested": "overfitting", "difficulty": "Hard", "is_correct": True,
                                "time_taken_seconds": 90})
        self.assertTrue(out["success"], out)
        self.assertTrue(out["data"]["results"][0]["is_correct"])
        self.assertIsNone(out["data"]["results"][0]["correct_index"])

    def test_server_side_grading_overrides_payload(self):
        with app_db.session_scope() as s:
            if s.get(Question, "stored-q1") is None:
                s.add(Question(id="stored-q1", learner_id=None, concept_id="regularization",
                               category="Machine Learning", difficulty="Hard",
                               payload={"question": "...", "options": ["a", "b", "c", "d"], "correct_index": 2}))
        # Payload lies about the answer and difficulty; the stored question wins.
        out = call("evaluate", {"question_id": "stored-q1", "concept_tested": "overfitting", "difficulty": "Easy",
                                "selected_option_index": 0, "correct_index": 0, "is_correct": True})
        res = out["data"]["results"][0]
        self.assertEqual((res["is_correct"], res["correct_index"], res["concept"]), (False, 2, "regularization"))
        with app_db.session_scope() as s:
            att = s.scalars(select(Attempt).where(Attempt.question_id == "stored-q1")).first()
            self.assertEqual(att.difficulty, "Hard")
        out = call("evaluate", {"question_id": "stored-q1", "selected_option_index": 2})
        self.assertTrue(out["data"]["results"][0]["is_correct"])

    def test_errors(self):
        out = call("evaluate", {**WRONG_MEDIUM_BV, "concept_tested": "Quantum Knitting"})
        self.assertEqual(out["error"]["code"], "UNKNOWN_CONCEPT")
        self.assertEqual(len(out["error"]["details"]["suggestions"]), 3)

        out = call("evaluate", {"concept_tested": "overfitting", "difficulty": "Easy", "selected_option_index": 1})
        self.assertEqual(out["error"]["code"], "CANNOT_GRADE")

        out = call("evaluate", {**WRONG_MEDIUM_BV, "difficulty": "Extreme"})
        self.assertEqual(out["error"]["code"], "INVALID_PAYLOAD")

        out = call("evaluate", {"quiz": True, "topic": "Overfitting", "answers": []})
        self.assertEqual(out["error"]["code"], "INVALID_PAYLOAD")

        out = call("evaluate", WRONG_MEDIUM_BV, learner_id="ghost")
        self.assertEqual(out["error"]["code"], "LEARNER_NOT_FOUND")
        self.assertEqual(count(Attempt), 0)

    def test_quiz_is_atomic(self):
        answers = [dict(WRONG_MEDIUM_BV), {**WRONG_MEDIUM_BV, "concept_tested": "Quantum Knitting"}]
        out = call("evaluate", {"quiz": True, "topic": "Bias vs Variance", "answers": answers})
        self.assertEqual(out["error"]["code"], "UNKNOWN_CONCEPT")
        self.assertEqual(out["error"]["details"]["answer_index"], 1)
        self.assertEqual(count(Attempt), 0)
        self.assertEqual(concept_row("bias_variance").mastery, 54.0)


class TestEvaluateQuiz(ApiTestCase):
    def test_quiz_mode(self):
        before = learner_row()
        answers = [
            {"question_id": f"q{i}", "concept_tested": c, "difficulty": "Medium",
             "selected_option_index": sel, "correct_index": 1, "time_taken_seconds": 40}
            for i, (c, sel) in enumerate([("Bias vs Variance", 1), ("Bias vs Variance", 1), ("Overfitting", 1),
                                          ("Overfitting", 0), ("Bias vs Variance", 1)])
        ]
        out = call("evaluate", {"quiz": True, "topic": "Bias vs Variance", "category": "Machine Learning",
                                "answers": answers})
        self.assertTrue(out["success"], out)
        d = out["data"]
        self.assertEqual(d["score"], 80)
        self.assertEqual(len(d["results"]), 5)
        self.assertEqual({m["concept"] for m in d["mastery_updates"]}, {"bias_variance", "overfitting"})
        self.assertEqual(d["next_difficulty"]["concept"], "bias_variance")
        self.assertEqual(d["adaptive_decision"]["topic"], "Bias vs Variance")
        self.assertIn(d["adaptive_decision"]["action"], ("maintained", "increased", "reduced"))

        self.assertEqual(count(Attempt), 5)
        after = learner_row()
        self.assertEqual(after.questions_solved, before.questions_solved + 5)
        expected = round((before.accuracy_rate * before.questions_solved + 80 * 5) / (before.questions_solved + 5), 1)
        self.assertAlmostEqual(after.accuracy_rate, expected, places=1)
        self.assertEqual(concept_row("bias_variance").attempts, 8 + 3)


class TestReset(ApiTestCase):
    def test_reset_restores_state(self):
        before = learner_row()
        call("evaluate", WRONG_MEDIUM_BV)
        call("evaluate", {"quiz": True, "topic": "Overfitting",
                          "answers": [{"concept_tested": "Overfitting", "difficulty": "Easy", "is_correct": False}]})
        self.assertEqual(count(Attempt), 2)
        self.assertEqual(count(AdaptiveEvent), 2)

        out = call("reset_learner")
        self.assertTrue(out["success"])
        self.assertEqual(count(Attempt), 0)
        self.assertEqual(count(AdaptiveEvent), 0)
        self.assertEqual(concept_row("bias_variance").mastery, 54.0)
        after = learner_row()
        self.assertEqual((after.questions_solved, after.accuracy_rate), (before.questions_solved, before.accuracy_rate))
        self.assertEqual(out["data"]["profile"]["skills"]["Machine Learning"], 71)
        self.assertEqual(out["data"]["profile"]["recentAdaptiveEvents"], [])


class TestSpecSection11Scenario(ApiTestCase):
    """Akshat misses a Medium Bias vs Variance question, is dropped to Easy, then recovers."""

    def test_closed_loop(self):
        out = call("evaluate", WRONG_MEDIUM_BV)["data"]
        self.assertEqual(out["adaptive_decision"]["action"], "reduced")
        self.assertEqual(out["next_difficulty"]["difficulty"], "Easy")
        mastery = out["mastery_updates"][0]["new_score"]

        difficulties = []
        for i in range(3):
            d = call("evaluate", {"question_id": f"easy-{i}", "concept_tested": "Bias vs Variance",
                                  "difficulty": "Easy", "selected_option_index": 2, "correct_index": 2,
                                  "time_taken_seconds": 12})["data"]
            new = d["mastery_updates"][0]["new_score"]
            self.assertGreater(new, mastery)
            mastery = new
            difficulties.append(d["next_difficulty"]["difficulty"])

        self.assertEqual(difficulties[:2], ["Easy", "Easy"])
        self.assertEqual(difficulties[2], "Medium")
        self.assertEqual(d["next_difficulty"]["rule_id"], "R4_FAST_STREAK")
        self.assertEqual(d["adaptive_decision"]["action"], "increased")
        self.assertEqual(concept_row("bias_variance").trend, "improving")


class TestRoutes(unittest.TestCase):
    def test_health_and_docs_disabled(self):
        h = _support.client().get("/health").json()
        self.assertEqual((h["status"], h["db"], h["llm_configured"]), ("ok", "ok", False))
        self.assertIsInstance(h["pool_size"], int)
        self.assertEqual(_support.client().get("/docs").status_code, 200)  # docs on by default (Phase 6)
        self.assertEqual(_support.client().get("/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
