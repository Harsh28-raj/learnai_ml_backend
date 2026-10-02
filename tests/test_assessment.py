"""Phase 5 tests: cold-start assessment (SPEC 7) and reset rules for assessed learners."""

import unittest

from sqlalchemy import delete, func, select

from tests import _support
from tests._support import app_db, call

from app.engine.cold_start import (
    EXPERIENCE_DEFAULT,
    baseline_masteries,
    level_from_assessment,
    resolve_known_topics,
)
from app.engine.concepts import CONCEPTS, concepts_in_category
from app.engine.recommender import ancestors
from app.models import ConceptState, LearningPath, Learner

PY = {c["id"] for c in concepts_in_category("Python")}
STATS = {c["id"] for c in concepts_in_category("Statistics")}


class TestLevelRule(unittest.TestCase):
    def test_spec_7_1(self):
        self.assertEqual(level_from_assessment("Beginner", ["a", "b", "c"]), "Beginner")
        self.assertEqual(level_from_assessment("Intermediate", ["NumPy"]), "Beginner")  # <= 1 topic
        self.assertEqual(level_from_assessment("Advanced", ["a", "b"]), "Advanced")
        self.assertEqual(level_from_assessment("Some Programming", list("abcdef")), "Advanced")  # >= 6 topics
        self.assertEqual(level_from_assessment("Some Programming", ["a", "b"]), "Intermediate")
        self.assertEqual(level_from_assessment("Intermediate", ["a", "b", "c"]), "Intermediate")


class TestBaselines(unittest.TestCase):
    def test_true_beginner(self):
        m = baseline_masteries("Beginner", [], [])
        self.assertTrue(all(m[c] == 15 for c in PY))
        self.assertTrue(all(v <= 15 for v in m.values()))

    def test_python_language(self):
        m = baseline_masteries("Some Programming", ["Python", "SQL"], [])
        self.assertEqual(m["python_basics"], 85)
        self.assertTrue(all(m[c] == 80 for c in PY - {"python_basics"}))
        self.assertEqual(m["linear_regression"], 25)

    def test_known_topics_and_ancestor_floor(self):
        m = baseline_masteries("Beginner", ["Python"], ["Neural Networks"])
        self.assertEqual(m["neural_networks"], 78)
        for anc in ancestors("neural_networks"):
            self.assertGreaterEqual(m[anc], 70, anc)
        self.assertEqual(resolve_known_topics(["Statistics", "Pandas", "unknown thing"]),
                         ["probability_basics", "std_variance", "distributions", "pandas_dataframes"])

    def test_defaults_capped_by_prerequisites(self):
        m = baseline_masteries("Advanced", ["None"], [])
        for cid, c in CONCEPTS.items():
            if cid not in PY:
                self.assertLessEqual(m[cid], min([EXPERIENCE_DEFAULT["Advanced"]] + [m[p] for p in c["prerequisites"]]))
        self.assertEqual(m["numpy_arrays"], 15)
        self.assertEqual(m["linear_regression"], 15)  # capped by matrix_shapes (Python, 15)

    def test_spec_gaps(self):
        m = baseline_masteries("Advanced", ["Python"], ["Machine Learning", "Deep Learning", "NLP"])
        # Statistics not known: capped at 50, except concepts implied by the known ML topics (>= 70).
        self.assertEqual({c: m[c] for c in STATS}, {"probability_basics": 70, "std_variance": 70,
                                                    "distributions": 50, "hypothesis_testing": 50})
        self.assertGreaterEqual(m["bias_variance"], 55)  # ML known: no bias_variance gap
        m = baseline_masteries("Advanced", ["Python"], ["Statistics"])
        self.assertEqual(m["bias_variance"], 55)  # ML not known
        self.assertEqual(m["std_variance"], 78)
        m = baseline_masteries("Advanced", ["Python"], ["Neural Networks"])
        self.assertEqual(m["probability_basics"], 70)  # implied by a known topic: the stats gap does not apply
        self.assertEqual(m["hypothesis_testing"], 50)  # not implied: gap applies


class AssessmentCase(unittest.TestCase):
    NEW = "test-learner-1"

    def setUp(self):
        _support.client()
        with app_db.session_scope() as s:
            for model in (ConceptState, LearningPath):
                s.execute(delete(model).where(model.learner_id.like("test-%")))
            s.execute(delete(Learner).where(Learner.id.like("test-%")))

    def assess(self, learner_id=NEW, **overrides):
        payload = {"name": "Riya", "experience_level": "Intermediate", "languages": ["Python"],
                   "topics_known": ["NumPy", "Pandas", "Statistics"], "goal": "ML Engineer", "pace": "Balanced",
                   "daily_minutes": 45}
        payload.update(overrides)
        return call("assessment", payload, learner_id=learner_id)


class TestAssessmentApi(AssessmentCase):
    def test_create_profile_path_and_snapshot(self):
        out = self.assess()
        self.assertTrue(out["success"], out)
        d = out["data"]
        self.assertEqual((d["level"], d["goal"]), ("Intermediate", "ML Engineer"))
        self.assertEqual(d["profile"]["name"], "Riya")
        self.assertEqual(d["profile"]["skills"]["Python"], 81)
        self.assertEqual(d["profile"]["weaknesses"], [])  # nothing practiced yet
        path = d["path"]
        skipped = {n["conceptId"] for n in path["nodes"] if n["skipped"]}
        self.assertTrue({"python_basics", "numpy_arrays", "pandas_dataframes", "std_variance"} <= skipped)
        self.assertEqual(path["beforeAfter"], f"Initial path created: start with {path['currentNode']['title']}.")
        fs = d["firstStep"]
        self.assertEqual((fs["type"], len(fs["conceptIds"])), ("diagnostic", 3))
        on_path = {n["conceptId"] for n in path["nodes"] if n["status"] != "completed"}
        self.assertTrue(set(fs["conceptIds"]) <= on_path)
        with app_db.session_scope() as s:
            self.assertEqual(s.scalar(select(func.count()).select_from(LearningPath)
                                      .where(LearningPath.learner_id == self.NEW)), 1)
            rows = s.scalars(select(ConceptState).where(ConceptState.learner_id == self.NEW)).all()
            self.assertEqual(len(rows), len(CONCEPTS))
            self.assertTrue(all(r.attempts == 0 and r.confidence == 0.1 for r in rows))
        self.assertEqual(call("get_profile", learner_id=self.NEW)["data"]["profile"]["level"], "Intermediate")

    def test_overwrite_and_protection(self):
        self.assertTrue(self.assess()["success"])
        self.assertEqual(self.assess()["error"]["code"], "LEARNER_EXISTS")
        redo = self.assess(experience_level="Beginner", topics_known=[], languages=[], overwrite=True)
        self.assertTrue(redo["success"])
        self.assertEqual(redo["data"]["level"], "Beginner")
        for lid in ("akshat-intermediate", "pool-beginner"):
            self.assertEqual(self.assess(learner_id=lid, overwrite=True)["error"]["code"], "DEMO_LEARNER_PROTECTED")
        self.assertEqual(self.assess(learner_id="Bad Id!")["error"]["code"], "INVALID_PAYLOAD")
        self.assertEqual(self.assess(experience_level="Expert")["error"]["code"], "INVALID_PAYLOAD")

    def test_reset_rules(self):
        self.assess()
        self.assertEqual(call("reset_learner", learner_id=self.NEW)["error"]["code"], "NOT_A_DEMO_LEARNER")
        self.assertEqual(call("reset_learner", learner_id="test-nobody")["error"]["code"], "LEARNER_NOT_FOUND")

    def test_beginner_and_intermediate_paths_differ(self):
        b = self.assess(learner_id="test-beginner", experience_level="Beginner", languages=[], topics_known=[])
        i = self.assess(learner_id="test-intermediate")
        bp, ip = b["data"]["path"], i["data"]["path"]
        self.assertEqual(b["data"]["level"], "Beginner")
        b_todo = {n["conceptId"] for n in bp["nodes"] if n["status"] != "completed"}
        i_todo = {n["conceptId"] for n in ip["nodes"] if n["status"] != "completed"}
        self.assertIn("python_basics", b_todo)
        self.assertNotIn("python_basics", i_todo)
        # No programming language: category order outranks goal relevance, so Python comes first.
        self.assertEqual(bp["currentNode"]["conceptId"], "python_basics")
        self.assertEqual([n["conceptId"] for n in bp["nextNodes"]], ["python_data_structures", "numpy_arrays"])
        self.assertGreater(bp["estimatedWeeksRemaining"], ip["estimatedWeeksRemaining"])

    def test_assessed_learner_can_evaluate_and_path_updates(self):
        self.assess()
        e = call("evaluate", {"concept_tested": "Bias vs Variance", "difficulty": "Medium", "is_correct": True},
                 learner_id=self.NEW)
        self.assertTrue(e["success"], e)
        self.assertTrue(e["data"]["adaptive_decision"]["pathAdjustment"])


if __name__ == "__main__":
    unittest.main()
