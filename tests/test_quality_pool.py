"""Phase 3.5 tests: independent verification, targeted regeneration, the verified pool and the
serving chain. All LLM calls are mocked; no network."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, select

from tests import _support
from tests._support import app_db, call
from tests.test_questions import AKSHAT, MOCK_TARGET, VERIFY_TARGET, approve_all, item, llm_response

from app.config import get_settings
from app.llm import verifier
from app.llm.groq_client import LLMError
from app.models import Question, QuestionServe


def verdict(answer, confidence=0.95, multiple=False, flawed=False, issue=None):
    return {"answer_index": answer, "confidence": confidence, "multiple_correct": multiple, "flawed": flawed,
            "issue": issue}


class TestJudge(unittest.TestCase):
    def test_accepts_matching_confident_answer(self):
        self.assertEqual(verifier.judge(verdict(2), 2), (True, None))

    def test_rejections(self):
        cases = [
            (verdict(1), "different answer"),
            (verdict(2, multiple=True, issue="B and C both reduce variance"), "more than one option"),
            (verdict(2, flawed=True, issue="units inconsistent"), "flawed"),
            (verdict(2, confidence=0.5), "confidence too low"),
            (None, "no verdict"),
        ]
        for v, fragment in cases:
            with self.subTest(fragment=fragment):
                ok, issue = verifier.judge(v, 2)
                self.assertFalse(ok)
                self.assertIn(fragment, issue)


class TestVerifierCall(unittest.TestCase):
    def test_one_call_fast_model_temp0_and_no_answer_leak(self):
        captured = {}

        async def fake(messages, model, **kw):
            captured.update(messages=messages, model=model, **kw)
            return {"results": [{"qid": "q0", "answer_index": 1, "confidence": 0.9, "multiple_correct": False,
                                 "flawed": False, "issue": None},
                                {"qid": "q1", "answer_index": "x", "confidence": "high"}]}

        qs = [{"qid": "q0", "question": "Q zero?", "code_snippet": None, "options": ["a", "b", "c", "d"],
               "correct_index": 1, "explanation": "SECRET-EXPLANATION", "hint": "SECRET-HINT",
               "distractor_reasons": ["SECRET-REASON"]},
              {"qid": "q1", "question": "Q one?", "options": ["e", "f", "g", "h"]}]
        with patch("app.llm.verifier.chat_completion", side_effect=fake) as m:
            out = asyncio.run(verifier.verify_questions(qs))
        self.assertEqual(m.call_count, 1)
        self.assertEqual(captured["model"], get_settings().groq_model_fast)
        self.assertEqual(captured["temperature"], 0.0)
        self.assertTrue(captured["json_mode"])
        sent = captured["messages"][1]["content"]
        for secret in ("SECRET-EXPLANATION", "SECRET-HINT", "SECRET-REASON", "correct_index"):
            self.assertNotIn(secret, sent)
        payload = json.loads(sent[sent.index("["):])
        self.assertEqual(set(payload[0]), {"qid", "question", "code_snippet", "options"})
        self.assertEqual(out["q0"]["answer_index"], 1)
        self.assertEqual((out["q1"]["answer_index"], out["q1"]["confidence"]), (-1, 0.0))  # coerced, will be rejected

    def test_bad_shape_raises(self):
        with patch("app.llm.verifier.chat_completion", AsyncMock(return_value={"oops": 1})):
            with self.assertRaises(LLMError):
                asyncio.run(verifier.verify_questions([{"qid": "q0", "question": "x", "options": list("abcd")}]))


def pool_question(qid, concept="bias_variance", difficulty="Medium", level="Intermediate", category="Machine Learning",
                  verified=True, source="llm", times_served=0):
    return Question(id=qid, learner_id=None, concept_id=concept, category=category, difficulty=difficulty,
                    verified=verified, level=level, source=source, times_served=times_served, payload={
                        "id": qid, "category": category, "difficulty": difficulty, "conceptId": concept,
                        "conceptTested": concept, "title": f"Pool {qid}", "question": f"Pool question {qid}?",
                        "codeSnippet": None, "options": ["w", "x", "y", "z"], "correctIndex": 2,
                        "explanation": "\"y\" is correct.", "hint": "think", "source": "llm",
                        "scenarioDomain": "education", "questionAngle": "concept_check"})


class PoolCase(unittest.TestCase):
    def setUp(self):
        _support.client()
        self.assertTrue(call("reset_learner", learner_id=AKSHAT)["success"])
        with app_db.session_scope() as s:
            s.execute(delete(QuestionServe))
            s.execute(delete(Question).where(Question.id.not_in(["stored-q1", "seed-q-bias-variance"])))

    def add(self, *questions):
        with app_db.session_scope() as s:
            s.add_all(questions)

    def generate(self, side_effect, payload=None, verify=approve_all):
        mock, vmock = AsyncMock(side_effect=side_effect), AsyncMock(side_effect=verify)
        with patch(MOCK_TARGET, mock), patch(VERIFY_TARGET, vmock):
            out = call("generate_questions", payload or {"count": 3}, learner_id=AKSHAT)
        return out, mock, vmock


class TestRegeneration(PoolCase):
    def test_only_rejected_questions_are_regenerated(self):
        rejected_once = {"done": False}

        async def verify(questions):
            out = await approve_all(questions)
            if not rejected_once["done"]:  # first batch: wrong answer on the second question
                out[questions[1]["qid"]]["answer_index"] = (out[questions[1]["qid"]]["answer_index"] + 1) % 4
                rejected_once["done"] = True
            return out

        out, mock, vmock = self.generate([llm_response(item(0), item(1), item(2)), llm_response(item(3))],
                                         verify=verify)
        self.assertTrue(out["success"], out)
        d = out["data"]
        self.assertEqual(d["sources"], {"llm": 3, "pool": 0, "fallback": 0})
        self.assertEqual((mock.await_count, vmock.await_count), (2, 2))
        self.assertEqual(len(vmock.await_args_list[1].args[0]), 1)  # only the regenerated one re-verified
        self.assertIn("Write 1 Medium", mock.await_args_list[1].args[0][1]["content"])
        self.assertIn("independent solver", mock.await_args_list[1].args[0][-1]["content"])
        g = d["generation"]
        self.assertEqual((g["verified"], g["rejected"], g["llm_calls"], g["verify_calls"]), (3, 1, 2, 2))
        self.assertTrue(any("different answer" in i for i in g["rejection_issues"]))
        self.assertEqual(d["questions"][1]["title"], "Diagnosis case 3")

    def test_each_rejection_reason_triggers_regeneration(self):
        for tweak, fragment in ((dict(multiple_correct=True), "more than one"), (dict(flawed=True), "flawed"),
                                (dict(confidence=0.3), "confidence")):
            with self.subTest(fragment=fragment):
                self.setUp()

                async def verify(questions, tweak=tweak):
                    out = await approve_all(questions)
                    out[questions[0]["qid"]].update(tweak)
                    return out

                out, mock, _ = self.generate([llm_response(item(0)), llm_response(item(1))], {"count": 1},
                                             verify=verify)
                # Rejected twice -> never served unverified; topped up from the bank instead.
                self.assertEqual(out["data"]["sources"], {"llm": 0, "pool": 0, "fallback": 1})
                self.assertEqual(out["error"]["code"], "VERIFICATION_SHORTFALL")
                self.assertTrue(any(fragment in i for i in out["data"]["generation"]["rejection_issues"]))

    def test_verifier_failure_never_serves_unverified(self):
        self.add(pool_question("pool-a"))

        async def broken(_):
            raise LLMError("MODEL_RATE_LIMIT", "busy")

        out, mock, _ = self.generate([llm_response(item(0), item(1), item(2))], verify=broken)
        self.assertTrue(out["success"], out)
        self.assertEqual(out["error"]["code"], "MODEL_RATE_LIMIT")
        self.assertEqual(mock.await_count, 1)  # no regeneration once the verifier is down
        sources = [q["source"] for q in out["data"]["questions"]]
        self.assertEqual(sources, ["pool", "fallback", "fallback"])
        self.assertNotIn("llm", sources)

    def test_verification_disabled(self):
        with patch.object(get_settings(), "verify_questions", False):
            out, _, vmock = self.generate([llm_response(item(0))], {"count": 1})
        self.assertEqual(vmock.await_count, 0)
        self.assertFalse(out["data"]["questions"][0]["verified"])
        with app_db.session_scope() as s:
            self.assertFalse(s.get(Question, out["data"]["questions"][0]["id"]).verified)


class TestPool(PoolCase):
    def test_relaxation_order_and_filters(self):
        self.add(
            pool_question("p-category", concept="overfitting"),                    # tier 3
            pool_question("p-adjacent", difficulty="Easy"),                         # tier 2
            pool_question("p-exact"),                                               # tier 1
            pool_question("p-other-level", level="Advanced"),                       # wrong level
            pool_question("p-unverified", verified=False),                          # unverified
            pool_question("p-fallback-src", source="fallback"),                     # not an LLM question
            pool_question("p-far-difficulty", concept="overfitting", difficulty="Hard"),  # outside every tier
        )
        out, _, _ = self.generate([LLMError("MODEL_RATE_LIMIT", "busy")], {"count": 4})
        d = out["data"]
        self.assertTrue(out["success"], out)
        self.assertEqual([q["id"] for q in d["questions"][:3]], ["p-exact", "p-adjacent", "p-category"])
        self.assertEqual([q["poolTier"] for q in d["questions"][:3]],
                         ["concept+difficulty", "concept+adjacent_difficulty", "category+difficulty"])
        self.assertEqual(d["sources"], {"llm": 0, "pool": 3, "fallback": 1})
        self.assertIn("3 from LearnAI's verified question pool and 1 from the curated question bank", out["error"]["message"])
        q = d["questions"][0]
        self.assertEqual(q["whyThisQuestion"], "Generated because you scored 54% on Bias vs Variance Tradeoff.")

    def test_success_true_with_pool_fill_and_counters(self):
        self.add(pool_question("p1"), pool_question("p2"), pool_question("p3"))
        out, _, _ = self.generate([LLMError("MODEL_TIMEOUT", "slow")], {"count": 3})
        self.assertTrue(out["success"])
        self.assertEqual(out["error"]["code"], "MODEL_TIMEOUT")
        self.assertEqual(out["data"]["sources"], {"llm": 0, "pool": 3, "fallback": 0})
        self.assertIn("verified question pool", out["error"]["message"])
        with app_db.session_scope() as s:
            self.assertEqual({r.times_served for r in s.scalars(select(Question).where(Question.id.like("p%")))}, {1})
            served = {r.question_id for r in s.scalars(select(QuestionServe).where(QuestionServe.learner_id == AKSHAT))}
        self.assertEqual(served, {"p1", "p2", "p3"})

    def test_never_reserves_to_same_learner(self):
        self.add(pool_question("p1"))
        first, _, _ = self.generate([LLMError("MODEL_RATE_LIMIT", "busy")], {"count": 1})
        self.assertEqual(first["data"]["questions"][0]["id"], "p1")
        second, _, _ = self.generate([LLMError("MODEL_RATE_LIMIT", "busy")], {"count": 1})
        self.assertEqual(second["data"]["questions"][0]["source"], "fallback")

    def test_new_llm_questions_join_the_pool(self):
        out, _, _ = self.generate([llm_response(item(0))], {"count": 1})
        qid = out["data"]["questions"][0]["id"]
        with app_db.session_scope() as s:
            row = s.get(Question, qid)
            self.assertEqual((row.verified, row.level, row.source, row.times_served), (True, "Intermediate", "llm", 1))
            self.assertIsNotNone(s.scalars(select(QuestionServe).where(QuestionServe.question_id == qid)).first())

    def test_pool_question_graded_server_side(self):
        self.add(pool_question("p1"))
        out, _, _ = self.generate([LLMError("MODEL_RATE_LIMIT", "busy")], {"count": 1})
        q = out["data"]["questions"][0]
        ev = call("evaluate", {"question_id": q["id"], "selected_option_index": 2}, learner_id=AKSHAT)
        self.assertTrue(ev["data"]["results"][0]["is_correct"])


if __name__ == "__main__":
    unittest.main()
