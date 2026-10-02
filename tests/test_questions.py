"""Phase 3 tests: question policy, generate_questions action, fallback bank, Groq client.

No network: chat_completion is mocked for action tests, and the Groq client is exercised
through httpx.MockTransport.
"""

import asyncio
import json
import random
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx
from sqlalchemy import delete, select

from tests import _support
from tests._support import app_db, call

from app.config import Settings
from app.engine.concepts import CATEGORIES, CONCEPTS
from app.engine.learner_model import MasteryState
from app.engine.question_policy import (
    GeneratedQuestion,
    allowed_angles,
    assign_slots,
    is_duplicate,
    select_concept,
    shuffle_options,
    validate_batch,
    why_this_question,
)
from app.llm import groq_client
from app.llm.groq_client import LLMError
from app.models import Question, QuestionServe
from app.actions.questions import load_fallback_questions
from app.seed import demo_states

AKSHAT, ALEX = "akshat-intermediate", "alex-beginner"
_real_sleep = asyncio.sleep
MOCK_TARGET = "app.actions.questions.chat_completion"
VERIFY_TARGET = "app.actions.questions.verify_questions"


async def approve_all(questions):
    """Fake verifier: solves each test question by picking the 'Right conclusion' option."""
    return {q["qid"]: {"answer_index": next(i for i, o in enumerate(q["options"]) if o.startswith("Right")),
                       "confidence": 0.95, "multiple_correct": False, "flawed": False, "issue": None}
            for q in questions}

STEMS = [
    "A hospital readmission model shows 99% training accuracy but 61% validation accuracy. What is happening?",
    "Your credit default model's validation loss starts rising after epoch 5 while training loss keeps falling. Diagnose it.",
    "A crop-yield regression has high error on both the training and the test set. Which change helps most?",
    "Learning curves for a product recommender converge quickly to the same high error. What does this indicate?",
    "Adding degree-12 polynomial features drove a sports model's training error to zero while test error doubled. Why?",
]


def item(i: int, **over) -> dict:
    q = {
        "title": f"Diagnosis case {i}",
        "question": STEMS[i % len(STEMS)],
        "code_snippet": None,
        "options": [f"Right conclusion {i}", f"Distractor alpha {i}", f"Distractor beta {i}", f"Distractor gamma {i}"],
        "correct_index": 0,
        "explanation": "Because the gap between seen and unseen data reveals how well the model generalizes.",
        "hint": "Compare performance on seen versus unseen data.",
        "distractor_reasons": ["Wrong because alpha.", "Wrong because beta.", "Wrong because gamma."],
        "scenario_domain": "healthcare",
        "question_angle": "scenario_diagnosis",
    }
    q.update(over)
    return q


def llm_response(*items) -> dict:
    return {"questions": list(items)}


def slots(n: int) -> list[dict]:
    return [{"index": i, "scenario_domain": "healthcare", "question_angle": "concept_check"} for i in range(n)]


def state_with(cid: str, mastery: float, results: str, attempts: int | None = None) -> MasteryState:
    """results like '+-+' (oldest first), all Medium."""
    recent = [{"correct": c == "+", "difficulty": "Medium", "time_taken": 40, "ts": "t"} for c in results]
    return MasteryState(concept_id=cid, category=CONCEPTS[cid]["category"], mastery=mastery,
                        attempts=len(results) if attempts is None else attempts, recent_results=recent)


# =========================================================================== pure policy

class TestConceptSelection(unittest.TestCase):
    def test_target_concept(self):
        self.assertEqual(select_concept(demo_states(AKSHAT), "NLP", "overfitting"), ("overfitting", "requested"))

    def test_category_with_weakness(self):
        self.assertEqual(select_concept(demo_states(AKSHAT), "Machine Learning"), ("bias_variance", "weak"))

    def test_category_lowest_practiced(self):
        self.assertEqual(select_concept(demo_states(AKSHAT), "Python"), ("matrix_shapes", "lowest_practiced"))

    def test_category_new_topic_with_mastered_prereqs(self):
        self.assertEqual(select_concept(demo_states(AKSHAT), "Deep Learning"), ("neural_networks", "new_topic"))

    def test_category_nothing_unlocked(self):
        self.assertEqual(select_concept(demo_states(ALEX), "Deep Learning"), ("neural_networks", "next_in_category"))

    def test_no_category_top_weakness(self):
        self.assertEqual(select_concept(demo_states(AKSHAT)), ("bias_variance", "weak"))
        cid, reason = select_concept(demo_states(ALEX))
        self.assertEqual(reason, "weak")
        self.assertIn(cid, ("matrix_shapes", "std_variance"))

    def test_no_category_no_weakness(self):
        states = [state_with("overfitting", 70, "+-++"), state_with("python_basics", 80, "++++"),
                  state_with("linear_regression", 78, "+++-")]
        self.assertEqual(select_concept(states), ("overfitting", "lowest_practiced"))


class TestSlotRotation(unittest.TestCase):
    def test_avoids_recent_for_same_concept(self):
        recent = [
            {"concept_id": "bias_variance", "scenario_domain": "healthcare", "question_angle": "concept_check"},
            {"concept_id": "bias_variance", "scenario_domain": "finance/credit", "question_angle": "code_debugging"},
        ]
        for seed in range(20):
            out = assign_slots(3, "Intermediate", "bias_variance", recent, random.Random(seed))
            domains = [s["scenario_domain"] for s in out]
            angles = [s["question_angle"] for s in out]
            self.assertTrue(set(domains).isdisjoint({"healthcare", "finance/credit"}))
            self.assertTrue(set(angles).isdisjoint({"concept_check", "code_debugging"}))
            self.assertEqual(len(set(domains)), 3)
            self.assertEqual(len(set(angles)), 3)

    def test_least_recently_used_when_all_used(self):
        from app.engine.question_policy import DOMAINS
        recent = [{"concept_id": "overfitting", "scenario_domain": d, "question_angle": "concept_check"}
                  for d in DOMAINS]  # newest first: DOMAINS[-1] is the oldest
        out = assign_slots(1, "Advanced", "overfitting", recent, random.Random(1))
        self.assertEqual(out[0]["scenario_domain"], DOMAINS[-1])

    def test_beginner_never_gets_code_debugging(self):
        self.assertNotIn("code_debugging", allowed_angles("Beginner"))
        for seed in range(50):
            out = assign_slots(5, "Beginner", "std_variance", [], random.Random(seed))
            self.assertNotIn("code_debugging", [s["question_angle"] for s in out])
        self.assertIn("code_debugging", allowed_angles("Intermediate"))


class TestValidation(unittest.TestCase):
    def check_invalid(self, msg_part: str, **over):
        valid, errors, failed = validate_batch(llm_response(item(0, **over)), slots(1), [])
        self.assertEqual(valid, [])
        self.assertEqual(len(failed), 1)
        self.assertIn(msg_part, errors[0])

    def test_valid_item(self):
        valid, errors, failed = validate_batch(llm_response(item(0), item(1)), slots(2), [])
        self.assertEqual((len(valid), errors, failed), (2, [], []))

    def test_option_rules(self):
        self.check_invalid("exactly 4 options", options=["a1", "b2", "c3", "d4", "e5"])
        self.check_invalid("distinct", options=["Same", "same!", "Other", "Third"])
        self.check_invalid("must not be empty", options=["x1", "", "y2", "z3"])
        self.check_invalid("between 0 and 3", correct_index=4)

    def test_text_rules(self):
        self.check_invalid("must not be empty", explanation="  ")
        self.check_invalid("must not be empty", hint="")
        self.check_invalid("hint must not contain", hint="The answer is right conclusion 0, obviously.")
        self.check_invalid("letter or position", explanation="Option B is correct because of variance.")
        self.check_invalid("letter or position", hint="Look closely at the second option.")
        self.check_invalid("letter or position", explanation="Answer (3) ignores the validation gap.")
        # Ordinary prose is not a position reference.
        prose = "A model that can answer a question about unseen data generalizes; the choice a team makes matters."
        valid, _, _ = validate_batch(llm_response(item(0, explanation=prose)), slots(1), [])
        self.assertEqual(len(valid), 1)
        self.check_invalid("at most 15 non-blank lines (got 16)", code_snippet="\n".join(f"x{i} = {i}" for i in range(16)))
        # Blank lines do not count toward the limit.
        spaced = "\n\n".join(f"x{i} = {i}" for i in range(15))
        self.assertEqual(GeneratedQuestion.model_validate(item(0, code_snippet=spaced)).code_snippet, spaced)

    def test_distractor_reasons_required_for_llm_output(self):
        self.check_invalid("exactly 3 non-empty reasons", distractor_reasons=["only one"])
        from app.engine.question_policy import LLMQuestion
        q = LLMQuestion.model_validate(item(0, distractor_reasons=["for correct", "r1", "r2", "r3"]))
        self.assertEqual(q.distractor_reasons, ["r1", "r2", "r3"])
        self.check_invalid("distractor_reasons", distractor_reasons=None)
        q = dict(item(0))
        q.pop("distractor_reasons")
        GeneratedQuestion.model_validate(q)  # curated/fallback questions do not need them

    def test_self_admitted_second_correct_option_rejected(self):
        for text in ("Regularization is right. Collecting more data would also help variance.",
                     "Ridge works; early stopping is also a valid fix here.",
                     "Both options are correct in practice."):
            with self.subTest(text=text):
                self.check_invalid("admits another option would also work", explanation=text)
        valid, _, _ = validate_batch(llm_response(item(0, explanation="Only regularization addresses the gap; "
                                                                       "lowering the learning rate does not help.")),
                                     slots(1), [])
        self.assertEqual(len(valid), 1)

    def test_code_reference_without_code_rejected(self):
        self.check_invalid("refers to code", question="What does the following code print when run on the data?")
        valid, _, _ = validate_batch(llm_response(item(0, question="What does the following code print here?",
                                                       code_snippet="print(1)")), slots(1), [])
        self.assertEqual(len(valid), 1)

    def test_code_in_question_text_is_moved_to_code_snippet(self):
        q = GeneratedQuestion.model_validate(item(0, question="What prints?\n```python\nprint((2, 3))\n```\nPick one.",
                                                  code_snippet=None))
        self.assertEqual(q.code_snippet, "print((2, 3))")
        self.assertNotIn("```", q.question)
        self.assertIn("Pick one.", q.question)
        q = GeneratedQuestion.model_validate(item(0, question="What prints?\n```\nx = 1\n```", code_snippet="x = 1"))
        self.assertEqual((q.question, q.code_snippet), ("What prints?", "x = 1"))

    def test_missing_and_malformed(self):
        _, errors, failed = validate_batch({"items": []}, slots(2), [])
        self.assertEqual(len(failed), 2)
        self.assertIn('"questions" array', errors[0])
        valid, errors, failed = validate_batch(llm_response(item(0)), slots(2), [])
        self.assertEqual((len(valid), len(failed)), (1, 1))
        self.assertIn("missing", errors[0])

    def test_duplicate_rejection(self):
        recent = [STEMS[0].upper() + "!!"]  # same text modulo case/punctuation
        valid, errors, failed = validate_batch(llm_response(item(0), item(1)), slots(2), recent)
        self.assertEqual(len(valid), 1)
        self.assertIn("too similar", errors[0])
        self.assertIsNotNone(is_duplicate(STEMS[1] + " Explain.", [STEMS[1]]))
        self.assertIsNone(is_duplicate(STEMS[1], [STEMS[2]]))

    def test_duplicate_within_batch(self):
        valid, errors, _ = validate_batch(llm_response(item(0), item(0, title="Other title")), slots(2), [])
        self.assertEqual(len(valid), 1)
        self.assertIn("too similar", errors[0])


class TestShuffle(unittest.TestCase):
    def test_correct_answer_preserved(self):
        opts = ["right", "w1", "w2", "w3"]
        positions = set()
        for seed in range(60):
            new, idx = shuffle_options(opts, 0, seed=f"gen-{seed}")
            self.assertEqual(new[idx], "right")
            self.assertEqual(sorted(new), sorted(opts))
            positions.add(idx)
        self.assertEqual(positions, {0, 1, 2, 3})  # answer position is not stuck

    def test_deterministic(self):
        self.assertEqual(shuffle_options(list("abcd"), 2, "gen-x"), shuffle_options(list("abcd"), 2, "gen-x"))


class TestWhyThisQuestion(unittest.TestCase):
    STAY = {"difficulty": "Medium", "reason": "x", "rule_id": "R7_STAY"}

    def test_weak_with_misses(self):
        s = state_with("bias_variance", 49.1, "++-+--")
        self.assertEqual(
            why_this_question("weak", s, {"difficulty": "Easy", "rule_id": "R2_TWO_WRONG_MEDIUM"}),
            "Generated because you scored 49% on Bias vs Variance Tradeoff and missed 2 of your last 3 answers.",
        )

    def test_weak_without_misses(self):
        s = state_with("bias_variance", 54, "-+-+")
        self.assertEqual(why_this_question("weak", s, self.STAY),
                         "Generated because you scored 54% on Bias vs Variance Tradeoff.")

    def test_advanced_challenge(self):
        s = state_with("overfitting", 80, "-+++")
        nd = {"difficulty": "Hard", "reason": "x", "rule_id": "R4_FAST_STREAK"}
        self.assertEqual(why_this_question("lowest_practiced", s, nd),
                         "Advanced challenge: you answered 3 Medium questions on Overfitting & Underfitting "
                         "correctly in a row.")
        # An explicitly requested difficulty is not presented as an earned challenge.
        self.assertNotIn("Advanced challenge", why_this_question("lowest_practiced", s, nd, difficulty_requested=True))

    def test_new_topic(self):
        s = MasteryState(concept_id="neural_networks", category="Deep Learning", mastery=50)
        self.assertEqual(why_this_question("new_topic", s, self.STAY),
                         "New topic: you've mastered its prerequisites (Gradient Descent Optimization, "
                         "Matrix Multiplication & Shapes).")
        s = MasteryState(concept_id="probability_basics", category="Statistics", mastery=30)
        self.assertIn("no prerequisites", why_this_question("new_topic", s, self.STAY))

    def test_other_reasons(self):
        s = state_with("gradient_descent", 61, "+-+")
        self.assertEqual(why_this_question("requested", s, self.STAY),
                         "You asked to practice Gradient Descent Optimization; you scored 61% on "
                         "Gradient Descent Optimization.")
        self.assertIn("lowest-scoring practiced concept", why_this_question("lowest_practiced", s, self.STAY))
        untried = MasteryState(concept_id="cnns", category="Deep Learning", mastery=25)
        self.assertEqual(why_this_question("requested", untried, {"difficulty": "Easy", "rule_id": "R0_NO_ATTEMPTS"}),
                         "First practice on Convolutional Neural Networks: starting at Easy based on an "
                         "estimated 25% mastery.")
        self.assertIn("Next step in Deep Learning", why_this_question("next_in_category", untried, self.STAY))


class TestFallbackBank(unittest.TestCase):
    def test_bank_is_valid(self):
        bank = load_fallback_questions()
        self.assertEqual(len(bank), 12)
        self.assertEqual(len({q["id"] for q in bank}), 12)
        per_category = {c: 0 for c in CATEGORIES}
        for q in bank:
            with self.subTest(q=q["id"]):
                GeneratedQuestion.model_validate(q)
                self.assertIn(q["concept_id"], CONCEPTS)
                self.assertEqual(CONCEPTS[q["concept_id"]]["category"], q["category"])
                self.assertIn(q["difficulty"], ("Easy", "Medium", "Hard"))
                per_category[q["category"]] += 1
        self.assertEqual(set(per_category.values()), {2})


# =========================================================================== action (mocked LLM)

class GenerateApiCase(unittest.TestCase):
    def setUp(self):
        _support.client()
        for lid in (AKSHAT, ALEX):
            self.assertTrue(call("reset_learner", learner_id=lid)["success"])
        with app_db.session_scope() as s:  # isolate history and the shared pool between tests
            s.execute(delete(QuestionServe))
            s.execute(delete(Question).where(Question.id.not_in(["stored-q1", "seed-q-bias-variance"])))

    def generate(self, side_effect, payload=None, learner_id=AKSHAT, verify=approve_all):
        mock = AsyncMock(side_effect=side_effect)
        vmock = AsyncMock(side_effect=verify)
        with patch(MOCK_TARGET, mock), patch(VERIFY_TARGET, vmock):
            out = call("generate_questions", payload or {}, learner_id=learner_id)
        self.verify_mock = vmock
        return out, mock


class TestGenerateQuestions(GenerateApiCase):
    def test_success_adaptive(self):
        out, mock = self.generate([llm_response(item(0), item(1), item(2))])
        self.assertTrue(out["success"], out)
        d = out["data"]
        self.assertEqual((d["count"], d["concept"], d["selection_reason"]), (3, "bias_variance", "weak"))
        self.assertEqual(d["difficulty"], "Medium")  # next_difficulty for Akshat's seeded history (R7)
        self.assertEqual(mock.await_count, 1)
        kwargs = mock.await_args.kwargs
        self.assertTrue(kwargs["json_mode"])
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertEqual(self.verify_mock.await_count, 1)  # ONE verifier call for the whole batch
        self.assertEqual(out["data"]["sources"], {"llm": 3, "pool": 0, "fallback": 0})
        self.assertIsNone(out["error"])

        q = d["questions"][0]
        for key in ("id", "category", "difficulty", "conceptTested", "title", "question", "codeSnippet", "options",
                    "correctIndex", "explanation", "hint", "whyThisQuestion", "recommendedNextDifficulty",
                    "scenarioDomain", "questionAngle"):
            self.assertIn(key, q)
        self.assertTrue(q["id"].startswith("gen-"))
        self.assertEqual(q["options"][q["correctIndex"]], "Right conclusion 0")
        self.assertEqual((q["source"], q["verified"]), ("llm", True))
        self.assertNotIn("distractor_reasons", q)
        self.assertEqual((q["difficulty"], q["category"], q["conceptTested"]),
                         ("Medium", "Machine Learning", "Bias vs Variance Tradeoff"))
        self.assertEqual(q["whyThisQuestion"], "Generated because you scored 54% on Bias vs Variance Tradeoff.")
        # Domains/angles come from Python's assignment, not from the LLM output.
        self.assertEqual(len({x["scenarioDomain"] for x in d["questions"]}), 3)

        with app_db.session_scope() as s:
            rows = s.scalars(select(Question).where(Question.learner_id == AKSHAT)).all()
            self.assertEqual(len(rows), 3)
            self.assertEqual({r.concept_id for r in rows}, {"bias_variance"})

    def test_prompt_contents(self):
        _, mock = self.generate([llm_response(item(0))], {"count": 1}, learner_id=ALEX)
        messages = mock.await_args.args[0]
        system, user = messages[0]["content"], messages[1]["content"]
        self.assertIn("No raw math notation", system)
        self.assertIn("Python", system)  # code only in known languages
        self.assertIn('"name": "Alex"', user)
        self.assertNotIn("code_debugging", user)

    def test_difficulty_pass_through(self):
        out, _ = self.generate([llm_response(item(0), item(1))], {"difficulty": "hard", "count": 2})
        d = out["data"]
        self.assertEqual((d["difficulty"], d["difficulty_reason"]), ("Hard", "Hard requested explicitly."))
        self.assertTrue(all(q["difficulty"] == "Hard" for q in d["questions"]))
        self.assertTrue(all(q["recommendedNextDifficulty"] == "Medium" for q in d["questions"]))

    def test_target_and_category(self):
        out, _ = self.generate([llm_response(item(0))], {"target_concept": "ROC AUC", "count": 1})
        self.assertEqual((out["data"]["concept"], out["data"]["selection_reason"]), ("confusion_matrix_roc", "requested"))
        # A different stem: repeating item(0) would (correctly) be rejected as a duplicate.
        out, _ = self.generate([llm_response(item(1))], {"category": "Computer Vision", "count": 1})
        self.assertEqual(out["data"]["concept"], "neural_networks")
        self.assertIn("New topic", out["data"]["questions"][0]["whyThisQuestion"])

    def test_payload_errors(self):
        self.assertEqual(call("generate_questions", {"category": "Cooking"})["error"]["code"], "INVALID_PAYLOAD")
        self.assertEqual(call("generate_questions", {"count": 6})["error"]["code"], "INVALID_PAYLOAD")
        self.assertEqual(call("generate_questions", {"difficulty": "Extreme"})["error"]["code"], "INVALID_PAYLOAD")
        out = call("generate_questions", {"target_concept": "quantum knitting"})
        self.assertEqual(out["error"]["code"], "UNKNOWN_CONCEPT")
        self.assertEqual(call("generate_questions", learner_id="ghost")["error"]["code"], "LEARNER_NOT_FOUND")

    def test_validation_retry_repairs_failed_slot(self):
        first = llm_response(item(0), item(1, options=["a1", "b2", "c3"]), item(2))
        second = llm_response(item(3))
        out, mock = self.generate([first, second])
        self.assertTrue(out["success"], out)
        self.assertEqual(out["data"]["count"], 3)
        self.assertEqual(mock.await_count, 2)
        retry_messages = mock.await_args_list[1].args[0]
        self.assertIn("exactly 4 options", retry_messages[-1]["content"])
        self.assertEqual(out["data"]["generation"]["llm_calls"], 2)
        self.assertEqual(out["data"]["generation"]["rejected"], 1)
        # Repaired question keeps the failed slot's position (index 1).
        self.assertEqual(out["data"]["questions"][1]["title"], "Diagnosis case 3")

    def test_bad_json_counts_as_validation_failure(self):
        bad = LLMError("MODEL_ERROR", "bad", details={"reason": "bad_json"})
        out, mock = self.generate([bad, llm_response(item(0), item(1), item(2))])
        self.assertTrue(out["success"], out)
        self.assertEqual(mock.await_count, 2)
        self.assertIn("not valid JSON", mock.await_args_list[1].args[0][-1]["content"])

    def test_partial_success_is_topped_up(self):
        first = llm_response(item(0), item(1, correct_index=7), item(2))
        second = llm_response(item(3, hint="Right conclusion 3 is it"))
        out, _ = self.generate([first, second])
        self.assertTrue(out["success"], out)
        self.assertEqual((out["data"]["count"], out["data"]["requested_count"]), (3, 3))
        self.assertEqual(out["data"]["sources"], {"llm": 2, "pool": 0, "fallback": 1})
        self.assertEqual(out["error"]["code"], "GENERATION_FAILED")  # notice, not a failure

    def test_retry_llm_error_keeps_valid(self):
        first = llm_response(item(0), item(1, options=["x1"]))
        out, _ = self.generate([first, LLMError("MODEL_TIMEOUT", "slow")], {"count": 2})
        self.assertTrue(out["success"], out)
        self.assertEqual(out["data"]["sources"]["llm"], 1)
        self.assertEqual(out["error"]["code"], "MODEL_TIMEOUT")

    def test_total_failure_returns_fallback(self):
        broken = llm_response(item(0, options=["only one"]), item(1, options=["x1"]), item(2, correct_index=9))
        out, mock = self.generate([broken, broken])
        self.assertTrue(out["success"], out)  # count was met from the curated bank
        self.assertEqual(out["error"]["code"], "GENERATION_FAILED")
        self.assertEqual(mock.await_count, 2)
        fb = out["data"]
        self.assertEqual(fb["count"], 3)
        self.assertEqual(fb["sources"], {"llm": 0, "pool": 0, "fallback": 3})
        self.assertEqual(fb["questions"][0]["conceptId"], "bias_variance")  # same concept first
        self.assertTrue(all(q["category"] == "Machine Learning" for q in fb["questions"][:2]))
        for q in fb["questions"]:
            self.assertTrue(q["id"].startswith("fallback-"))
            GeneratedQuestion.model_validate({**q, "code_snippet": q["codeSnippet"], "correct_index": q["correctIndex"]})
        with app_db.session_scope() as s:
            self.assertEqual(len(s.scalars(select(Question).where(Question.learner_id == AKSHAT)).all()), 3)

    def test_llm_errors_return_fallback(self):
        for code in ("MODEL_RATE_LIMIT", "MODEL_TIMEOUT", "LLM_NOT_CONFIGURED"):
            with self.subTest(code=code):
                out, mock = self.generate([LLMError(code, "nope")], {"count": 2})
                self.assertTrue(out["success"], out)
                self.assertEqual(out["error"]["code"], code)
                self.assertEqual(out["data"]["count"], 2)
                self.assertTrue(all(q["source"] == "fallback" for q in out["data"]["questions"]))
                self.assertEqual(mock.await_count, 1)

    def test_rejects_repeat_of_recent_question(self):
        self.generate([llm_response(item(0), item(1))], {"count": 2})
        # Next batch: the model repeats a stem it already produced; the repair call fixes it.
        out, mock = self.generate([llm_response(item(0, title="Fresh title"), item(2)), llm_response(item(3))],
                                  {"count": 2})
        self.assertTrue(out["success"], out)
        self.assertEqual(out["data"]["generation"]["rejected"], 1)
        self.assertIn("Diagnosis case 0", mock.await_args_list[0].args[0][1]["content"])  # do-not-repeat titles

    def test_generated_question_graded_server_side(self):
        out, _ = self.generate([llm_response(item(0))], {"count": 1})
        q = out["data"]["questions"][0]
        wrong = (q["correctIndex"] + 1) % 4
        # Client claims a wrong option is correct; the stored answer wins.
        ev = call("evaluate", {"question_id": q["id"], "selected_option_index": wrong, "correct_index": wrong,
                               "is_correct": True, "time_taken_seconds": 30})
        self.assertTrue(ev["success"], ev)
        self.assertEqual(ev["data"]["results"][0], {"question_id": q["id"], "is_correct": False,
                                                    "correct_index": q["correctIndex"], "concept": "bias_variance"})
        ev = call("evaluate", {"question_id": q["id"], "selected_option_index": q["correctIndex"]})
        self.assertTrue(ev["data"]["results"][0]["is_correct"])

    def test_fallback_question_graded_server_side(self):
        out, _ = self.generate([LLMError("MODEL_RATE_LIMIT", "busy")], {"count": 1})
        q = out["data"]["questions"][0]
        ev = call("evaluate", {"question_id": q["id"], "selected_option_index": q["correctIndex"]})
        self.assertTrue(ev["data"]["results"][0]["is_correct"])
        self.assertEqual(ev["data"]["results"][0]["concept"], q["conceptId"])


# =========================================================================== Groq client (MockTransport)

class TestGroqClient(unittest.TestCase):
    def setUp(self):
        self.requests: list[dict] = []
        self.responses: list = []
        self._saved_client = groq_client._client
        self.settings = Settings(groq_api_key="test-key", groq_reasoning_effort="low")
        self.patches = [
            patch("app.llm.groq_client.get_settings", lambda: self.settings),
            patch("app.llm.groq_client.asyncio.sleep", AsyncMock()),
        ]
        for p in self.patches:
            p.start()

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            nxt = self.responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        groq_client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def tearDown(self):
        asyncio.run(groq_client._client.aclose())
        groq_client._client = self._saved_client
        for p in self.patches:
            p.stop()

    @staticmethod
    def ok(content: str | None, reasoning: str | None = "secret chain of thought") -> httpx.Response:
        msg = {"role": "assistant", "content": content}
        if reasoning is not None:
            msg["reasoning"] = reasoning
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    def run_chat(self, **kw):
        return asyncio.run(groq_client.chat_completion([{"role": "user", "content": "PROMPT-SECRET"}],
                                                       "test-model", **kw))

    def test_missing_key_makes_no_request(self):
        self.settings = Settings(groq_api_key="")
        with self.assertRaises(LLMError) as cm:
            self.run_chat()
        self.assertEqual(cm.exception.code, "LLM_NOT_CONFIGURED")
        self.assertEqual(self.requests, [])

    def test_content_used_reasoning_ignored_and_logged_without_prompt(self):
        self.responses = [self.ok('```json\n{"a": 1}\n```')]
        with self.assertLogs("learnai.llm", level="INFO") as logs:
            out = self.run_chat(json_mode=True)
        self.assertEqual(out, {"a": 1})
        body = self.requests[0]
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["reasoning_effort"], "low")
        log_text = "\n".join(logs.output)
        self.assertIn("model=test-model", log_text)
        self.assertIn("total_tokens=15", log_text)
        self.assertNotIn("PROMPT-SECRET", log_text)
        self.assertNotIn("secret chain", log_text)

    def test_reasoning_effort_not_sent_when_unset(self):
        self.settings = Settings(groq_api_key="k", groq_reasoning_effort="")
        self.responses = [self.ok("hi")]
        self.assertEqual(self.run_chat(), "hi")
        self.assertNotIn("reasoning_effort", self.requests[0])

    def test_400_on_reasoning_effort_retries_without_it(self):
        self.responses = [
            httpx.Response(400, json={"error": {"message": "unsupported parameter reasoning_effort"}}),
            self.ok("fine"),
        ]
        with self.assertLogs("learnai.llm", level="WARNING"):
            self.assertEqual(self.run_chat(), "fine")
        self.assertIn("reasoning_effort", self.requests[0])
        self.assertNotIn("reasoning_effort", self.requests[1])

    def test_rate_limit_fails_fast_without_short_retry_after(self):
        for headers in ({}, {"retry-after": "7"}):
            self.requests.clear()
            self.responses = [httpx.Response(429, json={}, headers=headers)]
            with self.assertRaises(LLMError) as cm:
                self.run_chat()
            self.assertEqual(cm.exception.code, "MODEL_RATE_LIMIT")
            self.assertEqual(len(self.requests), 1)

    def test_rate_limit_short_retry_after_waits_and_retries_once(self):
        self.responses = [httpx.Response(429, json={}, headers={"retry-after": "2"}), self.ok("after wait")]
        self.assertEqual(self.run_chat(), "after wait")
        self.assertEqual(len(self.requests), 2)
        groq_client.asyncio.sleep.assert_awaited_with(2.0)
        self.responses = [httpx.Response(429, json={}, headers={"retry-after": "1"}),
                          httpx.Response(429, json={}, headers={"retry-after": "1"})]
        with self.assertRaises(LLMError):
            self.run_chat()

    def test_reasoning_headroom_added_to_max_tokens(self):
        self.responses = [self.ok("x")]
        self.run_chat(max_tokens=300)
        self.assertEqual(self.requests[0]["max_tokens"], 300 + groq_client.REASONING_HEADROOM["low"])

    def test_concurrency_limited_to_two(self):
        active = {"now": 0, "peak": 0}

        async def slow_post(*args, **kwargs):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.01)
            active["now"] -= 1
            return self.ok("ok")

        async def run():
            with patch.object(groq_client._client, "post", side_effect=slow_post):
                await asyncio.gather(*[groq_client.chat_completion([{"role": "user", "content": "x"}], "m")
                                       for _ in range(6)])

        with patch("app.llm.groq_client.asyncio.sleep", new=asyncio.sleep.__wrapped__
                   if hasattr(asyncio.sleep, "__wrapped__") else _real_sleep):
            asyncio.run(run())
        self.assertEqual(active["peak"], 2)

    def test_5xx_retry_then_success(self):
        self.responses = [httpx.Response(503, json={}), self.ok("recovered")]
        self.assertEqual(self.run_chat(), "recovered")

    def test_timeout(self):
        self.responses = [httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")]
        with self.assertRaises(LLMError) as cm:
            self.run_chat()
        self.assertEqual(cm.exception.code, "MODEL_TIMEOUT")

    def test_other_failures_are_model_error(self):
        self.responses = [httpx.Response(500, json={}), httpx.Response(500, json={})]
        with self.assertRaises(LLMError) as cm:
            self.run_chat()
        self.assertEqual(cm.exception.code, "MODEL_ERROR")

        self.responses = [self.ok("   ")]
        with self.assertRaises(LLMError) as cm:
            self.run_chat()
        self.assertEqual((cm.exception.code, cm.exception.details["reason"]), ("MODEL_ERROR", "empty_content"))

        self.responses = [self.ok(None)]
        with self.assertRaises(LLMError):
            self.run_chat()

    def test_json_validate_failed_salvages_failed_generation(self):
        bad = '{"answer": "The update is \\eta \\nabla L and \\(x\\)"}'  # raw LaTeX backslashes
        self.responses = [httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "x",
                                                              "failed_generation": bad}})]
        with self.assertLogs("learnai.llm", level="WARNING"):
            out = self.run_chat(json_mode=True)
        self.assertEqual(out["answer"], "The update is \\eta \\nabla L and \\(x\\)")

    def test_json_validate_failed_is_bad_json_without_effort_retry(self):
        self.responses = [httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "x"}})]
        with self.assertRaises(LLMError) as cm:
            self.run_chat(json_mode=True)
        self.assertEqual(cm.exception.details, {"reason": "bad_json"})
        self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
