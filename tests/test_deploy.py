"""Phase 6 tests: Swagger docs + examples, rate limiting, body limit, CORS, idempotent startup."""

import unittest
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, func, select

from tests import _support
from tests._support import app_db, call
from tests.test_questions import approve_all, item, llm_response
from tests.test_tutor import checkpoint, reply, verify_ok

from app.api_docs import EXAMPLES
from app.config import Settings, get_settings
from app.models import Learner, Question
from app.protection import RateLimiter, limiter
from app.seed import SEED_QUESTION_ID, ensure_pool_learners, ensure_seed_questions, seed_if_empty


class TestDocs(unittest.TestCase):
    def test_docs_redoc_and_openapi(self):
        c = _support.client()
        self.assertEqual(c.get("/docs").status_code, 200)
        self.assertEqual(c.get("/redoc").status_code, 200)
        spec = c.get("/openapi.json").json()
        self.assertEqual(spec["info"]["title"], "LearnAI ML Backend")
        for needle in ("getTutorReply()", "recordQuizScore()", "RATE_LIMITED", "45 s client timeout",
                       "local tutor mode", "alex-beginner", "fetch("):
            self.assertIn(needle, spec["info"]["description"])
        op = spec["paths"]["/api/v1/learnai"]["post"]
        self.assertEqual(set(op["requestBody"]["content"]["application/json"]["examples"]), set(EXAMPLES))
        self.assertIn("LearnAIResponseDoc", spec["components"]["schemas"])
        self.assertEqual(set(spec["paths"]), {"/health", "/api/v1/learnai"})

    def test_every_action_and_mode_has_an_example(self):
        actions = {e["value"]["action"] for e in EXAMPLES.values()}
        self.assertEqual(actions, {"get_profile", "reset_learner", "evaluate", "generate_questions", "tutor_chat",
                                   "get_path", "assessment"})
        self.assertTrue({"tutor_explanation", "tutor_simplify", "tutor_code", "tutor_hint", "tutor_evaluate",
                         "tutor_path", "evaluate_single", "evaluate_quiz", "assessment_new",
                         "assessment_overwrite"} <= set(EXAMPLES))


class TestExamplesRun(unittest.TestCase):
    """Every Swagger example succeeds on a freshly seeded DB (LLM mocked)."""

    def setUp(self):
        _support.client()
        with app_db.session_scope() as s:
            s.execute(delete(Learner).where(Learner.id == "new-learner-01"))
        for lid in ("alex-beginner", "akshat-intermediate", "elena-advanced"):
            call("reset_learner", learner_id=lid)

    def tearDown(self):
        _support.clear_generated_questions()

    def test_all_examples_succeed(self):
        ev = {"score": 45, "correct_points": ["a"], "misconceptions": ["b"], "corrected_answer": "c"}

        async def tutor_llm(messages, model, **kw):
            system = messages[0]["content"]
            if "MODE RULES (evaluate)" in system:
                return reply("Partly right.", evaluation=ev)
            if "MODE RULES (hint)" in system:
                return reply("Compare the training and validation numbers first.")
            return reply("Here is the idea.", cp=checkpoint() if "MODE RULES (explanation)" in system else None)

        gen = AsyncMock(side_effect=[llm_response(item(i), item(i + 1), item(i + 2)) for i in (0, 3)]
                        + [llm_response(item(6), item(7))])
        with patch("app.actions.tutor.chat_completion", AsyncMock(side_effect=tutor_llm)), \
                patch("app.actions.tutor.verify_questions", AsyncMock(side_effect=verify_ok)), \
                patch("app.actions.questions.chat_completion", gen), \
                patch("app.actions.questions.verify_questions", AsyncMock(side_effect=approve_all)):
            for name, ex in EXAMPLES.items():
                with self.subTest(example=name):
                    v = ex["value"]
                    out = call(v["action"], v["payload"], learner_id=v["learner_id"])
                    self.assertTrue(out["success"], (name, out.get("error")))

    def test_seed_question_is_gradable(self):
        with app_db.session_scope() as s:
            q = s.get(Question, SEED_QUESTION_ID)
            self.assertEqual((q.concept_id, q.source, q.verified), ("bias_variance", "fallback", True))
            right = q.payload["correctIndex"]
        out = call("evaluate", {"question_id": SEED_QUESTION_ID, "selected_option_index": right})
        self.assertTrue(out["data"]["results"][0]["is_correct"])


class TestRateLimit(unittest.TestCase):
    def setUp(self):
        _support.client()
        limiter.reset()

    def tearDown(self):
        limiter.reset()

    def test_limits_per_ip_and_bucket(self):
        s = get_settings()
        with patch.object(s, "rate_limit_other_per_min", 3), patch.object(s, "rate_limit_llm_per_min", 2):
            outs = [call("get_profile") for _ in range(4)]
            self.assertEqual([o["success"] for o in outs], [True, True, True, False])
            err = outs[-1]["error"]
            self.assertEqual(err["code"], "RATE_LIMITED")
            self.assertGreaterEqual(err["details"]["retry_after_seconds"], 1)
            # A different client IP (X-Forwarded-For first hop) has its own budget.
            r = _support.client().post("/api/v1/learnai", headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"},
                                       json={"action": "get_profile", "learner_id": "akshat-intermediate"})
            self.assertTrue(r.json()["success"])
            self.assertEqual(r.status_code, 200)
            # The AI bucket is separate and stricter.
            llm = [call("tutor_chat", {"message": "who won the IPL final?"}) for _ in range(3)]
            self.assertEqual([o["success"] for o in llm], [True, True, False])
            self.assertEqual(llm[-1]["error"]["details"]["bucket"], "ai")

    def test_window_and_eviction(self):
        rl = RateLimiter(max_keys=3)
        self.assertIsNone(rl.check("a", "other", 1, now=0))
        self.assertEqual(rl.check("a", "other", 1, now=10), 50)
        self.assertIsNone(rl.check("a", "other", 1, now=61))  # window slid
        for ip in ("b", "c", "d", "e"):
            rl.check(ip, "other", 5, now=70)
        self.assertEqual(len(rl), 3)  # bounded memory, LRU evicted
        self.assertIsNone(rl.check("x", "llm", 0))  # 0 disables


class TestBodyLimitAndCors(unittest.TestCase):
    def test_body_over_32kb_rejected(self):
        big = {"action": "tutor_chat", "learner_id": "akshat-intermediate", "payload": {"message": "x" * 40_000}}
        r = _support.client().post("/api/v1/learnai", json=big)
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.json()["error"]["code"], "PAYLOAD_TOO_LARGE")

    def test_cors_star(self):
        self.assertEqual(Settings(allowed_origins="*").cors_origins, ["*"])
        self.assertEqual(Settings(allowed_origins="http://a.com, http://b.com").cors_origins,
                         ["http://a.com", "http://b.com"])
        r = _support.client().get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(r.headers.get("access-control-allow-origin"), "http://localhost:5173")


class TestIdempotentStartup(unittest.TestCase):
    def test_seed_twice_is_a_no_op(self):
        _support.client()
        with app_db.session_scope() as s:
            before = s.scalar(select(func.count()).select_from(Learner))
            self.assertFalse(seed_if_empty(s))
            ensure_pool_learners(s)
            ensure_seed_questions(s)
            app_db.create_tables()
            self.assertEqual(s.scalar(select(func.count()).select_from(Learner)), before)


if __name__ == "__main__":
    unittest.main()
