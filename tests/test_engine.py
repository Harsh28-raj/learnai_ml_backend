"""Unit tests for Phase 2 engines: concept resolution, weakness, difficulty, adaptation."""

import unittest
from datetime import datetime, timedelta, timezone

from app.engine.adaptation import build_adaptive_decision, decide_action, human_timestamp, threshold_action
from app.engine.concepts import CONCEPTS, resolve_concept, suggest_concepts
from app.engine.difficulty import next_difficulty
from app.engine.learner_model import MasteryState
from app.engine.weakness import (
    build_weakness_report,
    classify,
    priority_label,
    priority_score,
    recommended_revision,
    tutor_alert,
    weakness_reason,
)


def ms(cid="bias_variance", mastery=50.0, attempts=5, recent=None, cw=0, trend="stable", correct=0):
    return MasteryState(
        concept_id=cid, category=CONCEPTS[cid]["category"], mastery=mastery, attempts=attempts,
        correct=correct, recent_results=list(recent or []), consecutive_wrong=cw, trend=trend,
    )


def r(spec: str, t: float = 45):
    """'M+' = Medium correct, 'H-' = Hard wrong, etc."""
    level = {"E": "Easy", "M": "Medium", "H": "Hard"}[spec[0]]
    return {"correct": spec[1] == "+", "difficulty": level, "time_taken": t, "ts": "t"}


def hist(*specs, t: float = 45):
    return [r(s, t) for s in specs]


class TestConceptResolution(unittest.TestCase):
    def test_exact_ids(self):
        for cid in CONCEPTS:
            self.assertEqual(resolve_concept(cid), cid)

    def test_every_alias_and_name_resolves_to_its_concept(self):
        for cid, c in CONCEPTS.items():
            for label in (c["name"], *c["aliases"]):
                with self.subTest(label=label):
                    self.assertEqual(resolve_concept(label), cid)

    def test_case_and_punctuation_insensitive(self):
        self.assertEqual(resolve_concept("Bias vs Variance"), "bias_variance")
        self.assertEqual(resolve_concept("BIAS-VARIANCE TRADEOFF"), "bias_variance")
        self.assertEqual(resolve_concept("bias versus variance"), "bias_variance")
        self.assertEqual(resolve_concept("confusion matrix and roc auc"), "confusion_matrix_roc")

    def test_token_overlap(self):
        self.assertEqual(resolve_concept("Decision Trees & Overfitting"), "decision_trees")
        self.assertEqual(resolve_concept("Ridge regularization penalty"), "regularization")
        self.assertEqual(resolve_concept("variance tradeoff"), "bias_variance")

    def test_typo_fallback(self):
        self.assertEqual(resolve_concept("Gradiant Descent"), "gradient_descent")

    def test_unknown(self):
        self.assertIsNone(resolve_concept("quantum knitting"))
        self.assertIsNone(resolve_concept(""))
        self.assertIsNone(resolve_concept(None))
        sugg = suggest_concepts("gradient boosting")
        self.assertEqual(len(sugg), 3)
        self.assertEqual(sugg[0]["concept_id"], "gradient_descent")


class TestWeakness(unittest.TestCase):
    def test_classify_thresholds(self):
        self.assertEqual(classify(75), "Strong")
        self.assertEqual(classify(74.9), "Average")
        self.assertEqual(classify(60), "Average")
        self.assertEqual(classify(59.9), "Weak")

    def test_classify_relative_rule(self):
        self.assertEqual(classify(70, learner_average=90), "Weak")
        self.assertEqual(classify(70, learner_average=80), "Average")
        self.assertEqual(classify(80, learner_average=99), "Strong")

    def test_priority_score_formula(self):
        # 0.45*60 + 20 + 0.15*60 + 10 + 0.10*50 = 71
        s = ms(mastery=40, attempts=5, cw=3, trend="declining", recent=hist("M-", "M+", "M-", "M+", "M-"))
        self.assertAlmostEqual(priority_score(s), 71.0)
        # 0.45*10 + 0 + 0 + 0 + 10 = 14.5
        s = ms(mastery=90, attempts=12, trend="improving", recent=hist(*["M+"] * 10))
        self.assertAlmostEqual(priority_score(s), 14.5)
        # 18 + 6.67 + 7.5 + 5 + 4 = 41.2
        s = ms(mastery=60, attempts=4, cw=1, recent=hist("M+", "M-", "M+", "M-"))
        self.assertAlmostEqual(priority_score(s), 41.2)

    def test_priority_labels(self):
        self.assertEqual(priority_label(55), "HIGH")
        self.assertEqual(priority_label(56.9), "HIGH")  # SPEC §11 single wrong answer
        self.assertEqual(priority_label(54.9), "MEDIUM")
        self.assertEqual(priority_label(40), "MEDIUM")
        self.assertEqual(priority_label(39.9), "LOW")

    def test_reason_and_alert(self):
        s = ms("overfitting", mastery=35, attempts=5, cw=3, trend="declining",
               recent=hist("M+", "M+", "M-", "M-", "M-"))
        reason = weakness_reason(s)
        for piece in ("5 attempts", "40% recent accuracy", "3 consecutive mistakes", "trend declining"):
            self.assertIn(piece, reason)
        self.assertEqual(tutor_alert(s, "HIGH"), "Your tutor noticed you're struggling with Overfitting & Underfitting.")
        self.assertIsNone(tutor_alert(s, "MEDIUM"))

    def test_untried_concepts_never_weak(self):
        states = [ms("bias_variance", mastery=50, attempts=6, recent=hist(*["M-"] * 6)),
                  ms("cnns", mastery=5, attempts=0)]
        report = build_weakness_report(states)
        self.assertEqual([e["concept"] for e in report["weak"]], ["bias_variance"])
        self.assertIn("cnns", report["untried"])
        self.assertNotIn("cnns", [e["concept"] for lst in ("strong", "average", "weak") for e in report[lst]])

    def test_weak_sorted_by_priority(self):
        states = [ms("bias_variance", mastery=55, attempts=6, recent=hist(*["M+"] * 6), trend="improving"),
                  ms("overfitting", mastery=30, attempts=6, cw=3, recent=hist(*["M-"] * 6), trend="declining")]
        weak = build_weakness_report(states)["weak"]
        self.assertEqual([e["concept"] for e in weak], ["overfitting", "bias_variance"])
        self.assertEqual(weak[0]["label"], "HIGH")
        self.assertIsNotNone(weak[0]["tutor_alert"])

    def test_recommended_revision_closest_prereq_gap(self):
        by_id = {s.concept_id: s for s in [
            ms("bias_variance", 50), ms("std_variance", 40), ms("linear_regression", 80),
            ms("probability_basics", 10),
        ]}
        rev = recommended_revision("bias_variance", by_id)
        self.assertEqual(rev["concept"], "std_variance")  # direct prereq beats deeper, weaker one
        self.assertTrue(rev["is_prerequisite"])

    def test_recommended_revision_deeper_layer_and_self(self):
        by_id = {s.concept_id: s for s in [
            ms("bias_variance", 50), ms("std_variance", 70), ms("linear_regression", 80), ms("matrix_shapes", 30),
        ]}
        self.assertEqual(recommended_revision("bias_variance", by_id)["concept"], "matrix_shapes")
        by_id["matrix_shapes"] = ms("matrix_shapes", 90)
        rev = recommended_revision("bias_variance", by_id)
        self.assertEqual(rev["concept"], "bias_variance")
        self.assertFalse(rev["is_prerequisite"])


class TestDifficulty(unittest.TestCase):
    def test_r0_no_attempts(self):
        self.assertEqual(next_difficulty(ms(mastery=30, attempts=0))["difficulty"], "Easy")
        self.assertEqual(next_difficulty(ms(mastery=60, attempts=0))["difficulty"], "Medium")
        out = next_difficulty(ms(mastery=80, attempts=0))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Hard", "R0_NO_ATTEMPTS"))

    def test_r1_spec_example(self):
        # SPEC: Easy+ Medium+ Medium+ Hard- Hard- -> Medium
        out = next_difficulty(ms(mastery=70, recent=hist("E+", "M+", "M+", "H-", "H-")))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Medium", "R1_TWO_WRONG_HARD"))
        self.assertIn("before retrying Hard", out["reason"])

    def test_r2_two_wrong_medium(self):
        out = next_difficulty(ms(recent=hist("M+", "M+", "M-", "M-")))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Easy", "R2_TWO_WRONG_MEDIUM"))

    def test_r3_two_of_three_wrong(self):
        out = next_difficulty(ms(recent=hist("M+", "M-", "M+", "H-")))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Medium", "R3_MOSTLY_WRONG"))
        out = next_difficulty(ms(recent=hist("E-", "E+", "E-")))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Easy", "R3_MOSTLY_WRONG"))
        out = next_difficulty(ms(recent=hist("E+", "E-", "E-")))  # two wrong at Easy is not R1/R2
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Easy", "R3_MOSTLY_WRONG"))

    def test_r4_fast_streak(self):
        recent = [r("M+", 20), r("M+", 25), r("M+", 30)]  # avg 25 < 0.6 * 60
        out = next_difficulty(ms(mastery=40, recent=recent))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Hard", "R4_FAST_STREAK"))

    def test_r5_streak_mastery_gate(self):
        recent = hist("E+", "E+", "E+", t=25)  # avg 25 > 0.6 * 30, so not R4
        out = next_difficulty(ms(mastery=65, recent=recent))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Medium", "R5_STREAK_MASTERY_GATE"))
        out = next_difficulty(ms(mastery=55, recent=recent))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Easy", "R5_STREAK_MASTERY_GATE"))
        out = next_difficulty(ms(mastery=74, recent=hist("M+", "M+", "M+", t=50)))
        self.assertEqual(out["difficulty"], "Medium")

    def test_r6_correct_but_slow(self):
        out = next_difficulty(ms(recent=[r("M-"), r("M+", 100)]))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Medium", "R6_CORRECT_BUT_SLOW"))

    def test_r7_default_stay(self):
        out = next_difficulty(ms(recent=hist("M+", "M-", "M+", t=40)))
        self.assertEqual((out["difficulty"], out["rule_id"]), ("Medium", "R7_STAY"))

    def test_only_last_five_considered(self):
        out = next_difficulty(ms(mastery=80, recent=hist("H-", "H-", "H-", "M+", "M+", "M+", "M+", "M+", t=50)))
        self.assertEqual(out["difficulty"], "Hard")


class TestAdaptation(unittest.TestCase):
    def test_threshold_mapping(self):
        self.assertEqual(threshold_action(0), "reduced")
        self.assertEqual(threshold_action(59.9), "reduced")
        self.assertEqual(threshold_action(60), "maintained")
        self.assertEqual(threshold_action(84.9), "maintained")
        self.assertEqual(threshold_action(85), "increased")

    def test_agreement(self):
        self.assertEqual(decide_action(0, "Medium", {"difficulty": "Easy"}), ("reduced", False))
        self.assertEqual(decide_action(100, "Medium", {"difficulty": "Hard"}), ("increased", False))

    def test_override_when_difficulty_disagrees(self):
        nd = {"difficulty": "Medium", "reason": "Correct but slow; staying at Medium.", "rule_id": "R6_CORRECT_BUT_SLOW"}
        out = build_adaptive_decision("bias_variance", 100, "Medium", nd, None)
        self.assertEqual(out["action"], "maintained")
        self.assertTrue(out["overridden"])
        self.assertEqual(out["threshold_action"], "increased")
        self.assertIn("difficulty engine", out["reason"])
        self.assertIn("slow", out["reason"])

    def test_no_override_at_floor_or_ceiling(self):
        self.assertEqual(decide_action(0, "Easy", {"difficulty": "Easy"}), ("reduced", False))
        self.assertEqual(decide_action(100, "Hard", {"difficulty": "Hard"}), ("increased", False))

    def test_reduced_injects_prerequisite(self):
        assessment = {"classification": "Weak", "label": "HIGH", "reason": "Mastery 49%",
                      "recommended_revision": {"concept": "std_variance", "name": "Standard Deviation & Variance",
                                               "mastery": 40, "is_prerequisite": True}}
        nd = {"difficulty": "Easy", "reason": "x", "rule_id": "R3_MOSTLY_WRONG"}
        out = build_adaptive_decision("bias_variance", 0, "Medium", nd, assessment)
        self.assertEqual(out["action"], "reduced")
        self.assertEqual(out["path_adjustment"],
                         "Prerequisite revision 'Standard Deviation & Variance' injected before 'Bias vs Variance Tradeoff'.")

    def test_override_step_up_while_weak_does_not_fast_track(self):
        assessment = {"classification": "Weak", "label": "LOW", "reason": "Mastery 58%",
                      "recommended_revision": {"concept": "bias_variance", "name": "Bias vs Variance Tradeoff",
                                               "mastery": 58, "is_prerequisite": False}}
        nd = {"difficulty": "Medium", "reason": "3 fast correct Easy answers.", "rule_id": "R4_FAST_STREAK"}
        out = build_adaptive_decision("bias_variance", 80, "Easy", nd, assessment)
        self.assertEqual((out["action"], out["overridden"]), ("increased", True))
        self.assertNotIn("Fast-tracked", out["path_adjustment"])
        self.assertNotIn("ready to start", out["recommendation"])

    def test_increased_unlocks_next_concept(self):
        nd = {"difficulty": "Hard", "reason": "x", "rule_id": "R4_FAST_STREAK"}
        out = build_adaptive_decision("bias_variance", 100, "Medium", nd, None)
        self.assertIn("'Overfitting & Underfitting' unlocked early", out["path_adjustment"])

    def test_increased_skips_mastered_unlock(self):
        nd = {"difficulty": "Hard", "reason": "x", "rule_id": "R4_FAST_STREAK"}
        out = build_adaptive_decision("bias_variance", 100, "Medium", nd, None, mastered={"overfitting"})
        self.assertNotIn("Overfitting", out["path_adjustment"])

    def test_human_timestamp(self):
        now = datetime(2026, 10, 2, 14, 15, tzinfo=timezone.utc)
        self.assertEqual(human_timestamp(now, now), "Today at 2:15 PM")
        self.assertEqual(human_timestamp(now - timedelta(days=1), now), "Yesterday at 2:15 PM")
        self.assertEqual(human_timestamp(now - timedelta(days=3), now), "Sep 29 at 2:15 PM")
        self.assertEqual(human_timestamp(now, now, utc_offset_minutes=330), "Today at 7:45 PM")


if __name__ == "__main__":
    unittest.main()
