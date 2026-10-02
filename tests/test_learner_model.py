"""Unit tests for the pure learner model, concept DAG, and seed data (stdlib unittest)."""

import unittest

from app.engine.concepts import CATEGORIES, CONCEPTS, find_cycle, topological_order, validate_dag
from app.engine.learner_model import (
    MasteryState,
    build_profile_view,
    category_score,
    compute_confidence,
    compute_trend,
    update_mastery,
)
from app.llm.groq_client import LLMError, parse_json_content
from app.seed import DEMO_LEARNERS, demo_states


def state(mastery=50.0, attempts=0, consecutive_wrong=0, recent=None, category="Machine Learning", cid="bias_variance"):
    return MasteryState(
        concept_id=cid, category=category, mastery=mastery, attempts=attempts,
        correct=0, recent_results=list(recent or []), consecutive_wrong=consecutive_wrong,
    )


def results(*flags):
    return [{"correct": bool(f), "difficulty": "Medium", "time_taken": 30, "ts": "t"} for f in flags]


class TestUpdateMasteryCorrect(unittest.TestCase):
    def test_gain_by_difficulty_at_50(self):
        # gain = base * (1 - 0.5) * 1.5 = base * 0.75; slow answer -> no bonus
        for diff, expected in (("Easy", 52.25), ("Medium", 53.75), ("Hard", 56.0)):
            with self.subTest(diff=diff):
                new = update_mastery(state(50), True, diff, 60)
                self.assertAlmostEqual(new.mastery, expected, places=2)

    def test_fast_bonus(self):
        new = update_mastery(state(50), True, "Medium", 10)
        self.assertAlmostEqual(new.mastery, 54.75, places=2)

    def test_no_bonus_at_exactly_20s_or_unknown_time(self):
        self.assertAlmostEqual(update_mastery(state(50), True, "Medium", 20).mastery, 53.75, places=2)
        self.assertAlmostEqual(update_mastery(state(50), True, "Medium", None).mastery, 53.75, places=2)

    def test_minimum_gain_of_one(self):
        # Easy at 90: 3 * 0.1 * 1.5 = 0.45 -> floor to +1
        self.assertAlmostEqual(update_mastery(state(90), True, "Easy", 60).mastery, 91.0, places=2)

    def test_clamped_at_100(self):
        new = update_mastery(state(99.5), True, "Hard", 5)
        self.assertEqual(new.mastery, 100.0)

    def test_counters(self):
        s = update_mastery(state(50, attempts=3, consecutive_wrong=2), True, "easy", 30)
        self.assertEqual((s.attempts, s.correct, s.consecutive_wrong), (4, 1, 0))
        self.assertEqual(s.recent_results[-1]["difficulty"], "Easy")
        self.assertTrue(s.recent_results[-1]["correct"])

    def test_does_not_mutate_input(self):
        original = state(50, recent=results(1, 0))
        update_mastery(original, True, "Medium", 30)
        self.assertEqual(original.mastery, 50)
        self.assertEqual(len(original.recent_results), 2)


class TestUpdateMasteryWrong(unittest.TestCase):
    def test_loss_by_difficulty_at_50(self):
        # loss = base * 0.5 * 1.5 = base * 0.75
        for diff, expected in (("Easy", 44.0), ("Medium", 45.5), ("Hard", 47.75)):
            with self.subTest(diff=diff):
                self.assertAlmostEqual(update_mastery(state(50), False, diff, 30).mastery, expected, places=2)

    def test_minimum_loss_of_one(self):
        # Hard at 10: 3 * 0.1 * 1.5 = 0.45 -> floor to -1
        self.assertAlmostEqual(update_mastery(state(10), False, "Hard", 30).mastery, 9.0, places=2)

    def test_clamped_at_zero(self):
        self.assertEqual(update_mastery(state(0.5), False, "Easy", 30).mastery, 0.0)
        self.assertEqual(update_mastery(state(0), False, "Hard", 30).mastery, 0.0)

    def test_fast_wrong_gets_no_bonus(self):
        self.assertAlmostEqual(update_mastery(state(50), False, "Medium", 5).mastery, 45.5, places=2)

    def test_consecutive_wrong_increments(self):
        s = state(60)
        for _ in range(3):
            s = update_mastery(s, False, "Medium", 30)
        self.assertEqual((s.attempts, s.correct, s.consecutive_wrong), (3, 0, 3))

    def test_recent_results_capped_at_10(self):
        s = state(50)
        for i in range(15):
            s = update_mastery(s, i % 2 == 0, "Medium", 30, ts=f"t{i}")
        self.assertEqual(len(s.recent_results), 10)
        self.assertEqual(s.recent_results[0]["ts"], "t5")
        self.assertEqual(s.recent_results[-1]["ts"], "t14")

    def test_invalid_difficulty(self):
        with self.assertRaises(ValueError):
            update_mastery(state(50), True, "Impossible", 30)


class TestConfidence(unittest.TestCase):
    def test_values(self):
        self.assertEqual(compute_confidence(0, 0), 0.0)
        self.assertAlmostEqual(compute_confidence(5, 0), 0.5)
        self.assertAlmostEqual(compute_confidence(10, 0), 1.0)
        self.assertAlmostEqual(compute_confidence(25, 0), 1.0)
        self.assertAlmostEqual(compute_confidence(10, 1), 0.85)
        self.assertAlmostEqual(compute_confidence(10, 3), 0.55)
        self.assertAlmostEqual(compute_confidence(10, 7), 0.55)  # capped at 3 wrong
        self.assertAlmostEqual(compute_confidence(4, 2), 0.28)

    def test_update_sets_confidence(self):
        s = state(50, attempts=9)
        s = update_mastery(s, False, "Medium", 30)
        self.assertAlmostEqual(s.confidence, 0.85, places=3)


class TestTrend(unittest.TestCase):
    def test_fewer_than_four_is_stable(self):
        self.assertEqual(compute_trend([]), "stable")
        self.assertEqual(compute_trend(results(0, 0, 1)), "stable")

    def test_improving(self):
        self.assertEqual(compute_trend(results(0, 0, 0, 1, 1, 1)), "improving")
        self.assertEqual(compute_trend(results(0, 1, 1, 1)), "improving")  # prev = 1 result

    def test_declining(self):
        self.assertEqual(compute_trend(results(1, 1, 1, 0, 0, 1)), "declining")

    def test_stable_within_threshold(self):
        self.assertEqual(compute_trend(results(1, 0, 1, 1, 0, 1)), "stable")

    def test_uses_only_last_six(self):
        self.assertEqual(compute_trend(results(1, 1, 1, 1, 0, 0, 0, 1, 1, 1)), "improving")


class TestCategoryScore(unittest.TestCase):
    def test_attempt_weighted_with_unattempted_weight_one(self):
        states = [
            state(80, attempts=3, cid="a"),
            state(40, attempts=1, cid="b"),
            state(20, attempts=0, cid="c"),
            state(100, attempts=5, category="Python", cid="d"),
        ]
        # (80*3 + 40*1 + 20*1) / 5 = 60
        self.assertAlmostEqual(category_score("Machine Learning", states), 60.0)
        self.assertEqual(category_score("NLP", states), 0.0)


class TestConceptDag(unittest.TestCase):
    def test_catalog_is_valid_dag(self):
        validate_dag()
        self.assertGreaterEqual(len(CONCEPTS), 20)
        self.assertEqual({c["category"] for c in CONCEPTS.values()}, set(CATEGORIES))

    def test_required_concepts_present(self):
        for cid in ("matrix_shapes", "std_variance", "bias_variance", "gradient_descent", "confusion_matrix_roc",
                    "overfitting", "regularization", "decision_trees", "backpropagation",
                    "transformers_attention", "rag_chunking"):
            self.assertIn(cid, CONCEPTS)

    def test_topological_order_respects_prerequisites(self):
        order = topological_order()
        pos = {cid: i for i, cid in enumerate(order)}
        for cid, c in CONCEPTS.items():
            for pre in c["prerequisites"]:
                self.assertLess(pos[pre], pos[cid])

    def test_detects_cycle(self):
        bad = {
            "a": {"prerequisites": ["c"]},
            "b": {"prerequisites": ["a"]},
            "c": {"prerequisites": ["b"]},
        }
        with self.assertRaises(ValueError):
            validate_dag(bad)
        cycle = find_cycle({k: v["prerequisites"] for k, v in bad.items()})
        self.assertEqual(cycle[0], cycle[-1])

    def test_detects_self_loop_and_unknown_prereq(self):
        with self.assertRaises(ValueError):
            validate_dag({"a": {"prerequisites": ["a"]}})
        with self.assertRaises(ValueError):
            validate_dag({"a": {"prerequisites": ["zzz"]}})


class TestSeedMatchesSpec(unittest.TestCase):
    WEAKNESSES = {
        "alex-beginner": {"Matrix Multiplication & Shapes": 42, "Standard Deviation & Variance": 50},
        "akshat-intermediate": {"Bias vs Variance Tradeoff": 54, "Gradient Descent Optimization": 61,
                                "Confusion Matrix & ROC-AUC": 63},
        "elena-advanced": {"Vector DB Embedding Chunking & Semantic Drift": 68},
    }

    def test_skills_and_weaknesses(self):
        for learner_id, spec in DEMO_LEARNERS.items():
            with self.subTest(learner=learner_id):
                states = demo_states(learner_id)
                learner = {"id": learner_id, **spec["profile"], "questions_solved": 0, "accuracy_rate": 0}
                view = build_profile_view(learner, states)
                self.assertEqual(view["skills"], spec["skills"])
                got = {w["name"]: w["score"] for w in view["weaknesses"]}
                self.assertEqual(got, self.WEAKNESSES[learner_id])
                for s in states:
                    self.assertTrue(0 <= s.mastery <= 100)


class TestJsonParsing(unittest.TestCase):
    def test_fenced_and_plain(self):
        self.assertEqual(parse_json_content('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_content('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_content('Sure! {"a": 1} hope that helps'), {"a": 1})

    def test_bad_json_raises(self):
        with self.assertRaises(LLMError):
            parse_json_content("not json")
        with self.assertRaises(LLMError):
            parse_json_content("[1, 2]")


if __name__ == "__main__":
    unittest.main()
