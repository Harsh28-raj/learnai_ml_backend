"""Demo learners from SPEC Section 6, seeded so category skills match the spec exactly.

Each category lists concepts as (mastery, attempts). Exactly one concept per category
has mastery=None: its mastery is solved so that the attempt-weighted category score
equals the SPEC skill number. Named weak concepts carry their exact SPEC scores.
Unattempted concepts (attempts=0) hold a prior estimate and never count as weaknesses.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.engine.concepts import CONCEPTS, concepts_in_category
from app.engine.learner_model import (
    RECENT_WINDOW,
    MasteryState,
    category_score,
    compute_confidence,
    compute_trend,
)
from app.models import AdaptiveEvent, Attempt, ChatMessage, ConceptState, LearningPath, Learner, QuestionServe

DEMO_LEARNERS: dict[str, dict] = {
    "alex-beginner": {
        "profile": {
            "name": "Alex",
            "level": "Beginner",
            "role": "Aspiring Tech Learner",
            "goal": "Learn ML from Scratch",
            "target_role": "Junior Data Analyst",
            "experience": "Basic Python syntax (variables, loops). No formal statistics or matrix algebra.",
            "known_languages": ["Python"],
            "known_topics": ["Python Basics"],
            "target_topics": ["Statistics", "NumPy", "Linear Regression", "Machine Learning Basics"],
            "learning_pace": "Relaxed",
            "daily_commitment_minutes": 30,
            "streak_days": 3,
        },
        "skills": {"Python": 65, "Statistics": 38, "Machine Learning": 25, "Deep Learning": 10, "NLP": 5, "Generative AI": 10},
        "concepts": {
            "Python": {"python_basics": (80, 10), "python_data_structures": (None, 8), "numpy_arrays": (45, 0),
                       "matrix_shapes": (42, 5), "pandas_dataframes": (30, 0)},
            "Statistics": {"probability_basics": (None, 0), "std_variance": (50, 4), "distributions": (20, 0),
                           "hypothesis_testing": (10, 0)},
            "Machine Learning": {"linear_regression": (40, 0), "gradient_descent": (25, 0), "bias_variance": (20, 0),
                                 "overfitting": (30, 0), "regularization": (15, 0), "decision_trees": (35, 0),
                                 "confusion_matrix_roc": (20, 0), "cross_validation": (None, 0)},
            "Deep Learning": {"neural_networks": (15, 0), "backpropagation": (8, 0), "cnns": (None, 0)},
            "NLP": {"tokenization": (10, 0), "word_embeddings": (3, 0), "transformers_attention": (None, 0)},
            "Generative AI": {"prompt_engineering": (20, 0), "rag_chunking": (5, 0), "llm_finetuning": (None, 0)},
        },
    },
    "akshat-intermediate": {
        "profile": {
            "name": "Akshat",
            "level": "Intermediate",
            "role": "Software Developer",
            "goal": "Become an ML Engineer",
            "target_role": "ML Engineer",
            "experience": "2+ years programming in Python, JavaScript and SQL. Knows NumPy, Pandas and basic linear regression.",
            "known_languages": ["Python", "JavaScript", "SQL"],
            "known_topics": ["Python", "NumPy", "Pandas", "Linear Regression"],
            "target_topics": ["Bias vs Variance", "Regularization", "Model Evaluation", "Deep Learning", "MLOps"],
            "learning_pace": "Balanced",
            "daily_commitment_minutes": 45,
            "streak_days": 12,
        },
        "skills": {"Python": 88, "Statistics": 62, "Machine Learning": 71, "Deep Learning": 35, "NLP": 20, "Generative AI": 15},
        "concepts": {
            "Python": {"python_basics": (92, 30), "python_data_structures": (90, 25), "numpy_arrays": (86, 20),
                       "matrix_shapes": (None, 12), "pandas_dataframes": (84, 18)},
            "Statistics": {"probability_basics": (70, 4), "std_variance": (66, 4), "distributions": (45, 0),
                           "hypothesis_testing": (None, 0)},
            "Machine Learning": {"linear_regression": (85, 15), "gradient_descent": (61, 8), "bias_variance": (54, 8),
                                 "overfitting": (72, 8), "regularization": (66, 6), "decision_trees": (78, 10),
                                 "confusion_matrix_roc": (63, 6), "cross_validation": (None, 5)},
            "Deep Learning": {"neural_networks": (50, 0), "backpropagation": (30, 0), "cnns": (None, 0)},
            "NLP": {"tokenization": (35, 0), "word_embeddings": (15, 0), "transformers_attention": (None, 0)},
            "Generative AI": {"prompt_engineering": (30, 0), "rag_chunking": (10, 0), "llm_finetuning": (None, 0)},
        },
    },
    "elena-advanced": {
        "profile": {
            "name": "Elena",
            "level": "Advanced",
            "role": "Senior Data Scientist",
            "goal": "Build GenAI Apps & Fine-tune LLMs",
            "target_role": "GenAI / LLM Engineer",
            "experience": "4+ years as a data scientist. Deep knowledge of PyTorch, CNNs, backpropagation and classical ML.",
            "known_languages": ["Python", "SQL", "C++"],
            "known_topics": ["Statistics", "Machine Learning", "Deep Learning", "PyTorch", "CNNs", "Transformers"],
            "target_topics": ["RAG Systems", "Vector Databases", "LLM Fine-Tuning", "LoRA", "Evaluation of LLMs"],
            "learning_pace": "Intensive",
            "daily_commitment_minutes": 90,
            "streak_days": 34,
        },
        "skills": {"Python": 98, "Statistics": 94, "Machine Learning": 92, "Deep Learning": 88, "NLP": 85, "Generative AI": 74},
        "concepts": {
            "Python": {"python_basics": (99, 40), "python_data_structures": (99, 35), "numpy_arrays": (98, 30),
                       "matrix_shapes": (None, 20), "pandas_dataframes": (97, 25)},
            "Statistics": {"probability_basics": (96, 20), "std_variance": (95, 18), "distributions": (93, 15),
                           "hypothesis_testing": (None, 12)},
            "Machine Learning": {"linear_regression": (95, 25), "gradient_descent": (93, 20), "bias_variance": (93, 18),
                                 "overfitting": (94, 18), "regularization": (91, 15), "decision_trees": (90, 15),
                                 "confusion_matrix_roc": (92, 15), "cross_validation": (None, 12)},
            "Deep Learning": {"neural_networks": (92, 20), "backpropagation": (90, 18), "cnns": (None, 12)},
            "NLP": {"tokenization": (90, 15), "word_embeddings": (86, 15), "transformers_attention": (None, 15)},
            "Generative AI": {"prompt_engineering": (80, 10), "rag_chunking": (68, 14), "llm_finetuning": (None, 10)},
        },
    },
}


def _solve_masteries(spec: dict) -> dict[str, tuple[float, int]]:
    """Fill in each category's balancing concept so its category score hits the target."""
    resolved: dict[str, tuple[float, int]] = {}
    for category, concepts in spec["concepts"].items():
        expected = {c["id"] for c in concepts_in_category(category)}
        if set(concepts) != expected:
            raise ValueError(f"Seed for {category} must cover exactly {sorted(expected)}")
        balance = [cid for cid, (m, _) in concepts.items() if m is None]
        if len(balance) != 1:
            raise ValueError(f"Seed for {category} needs exactly one balancing concept")
        weight = lambda a: a if a > 0 else 1  # noqa: E731  (mirrors category_score)
        total_w = sum(weight(a) for _, a in concepts.values())
        fixed = sum(m * weight(a) for m, a in concepts.values() if m is not None)
        bal_id = balance[0]
        bal_a = concepts[bal_id][1]
        bal_m = (spec["skills"][category] * total_w - fixed) / weight(bal_a)
        if not 0 <= bal_m <= 100:
            raise ValueError(f"Seed for {category} is infeasible (balancing mastery {bal_m:.1f})")
        for cid, (m, a) in concepts.items():
            resolved[cid] = (round(bal_m if cid == bal_id else m, 2), a)
    return resolved


def _synthetic_results(mastery: float, attempts: int, now: datetime) -> list[dict]:
    """Deterministic recent history whose accuracy roughly matches mastery."""
    n = min(attempts, RECENT_WINDOW)
    k = round(n * mastery / 100)
    results = []
    for i in range(n):
        # Bresenham-style spread so correct answers are evenly distributed.
        correct = (i + 1) * k // n > i * k // n if n else False
        results.append({
            "correct": correct,
            "difficulty": "Medium",
            "time_taken": 45,
            "ts": (now - timedelta(hours=(n - i) * 6)).isoformat(),
        })
    return results


def demo_states(learner_id: str, now: datetime | None = None) -> list[MasteryState]:
    """Pure: the seeded MasteryState list for a demo learner (catalog order)."""
    spec = DEMO_LEARNERS[learner_id]
    now = now or datetime.now(timezone.utc)
    resolved = _solve_masteries(spec)
    states = []
    for cid, concept in CONCEPTS.items():
        mastery, attempts = resolved[cid]
        recent = _synthetic_results(mastery, attempts, now)
        consecutive_wrong = 0
        for r in reversed(recent):
            if r["correct"]:
                break
            consecutive_wrong += 1
        states.append(MasteryState(
            concept_id=cid,
            category=concept["category"],
            mastery=mastery,
            attempts=attempts,
            correct=round(attempts * mastery / 100),
            recent_results=recent,
            consecutive_wrong=consecutive_wrong,
            confidence=round(compute_confidence(attempts, consecutive_wrong), 3),
            trend=compute_trend(recent),
        ))
    return states


def _seed_one(db: Session, learner_id: str) -> None:
    """Create the demo learner, or restore an existing row in place (keeps FK references valid)."""
    spec = DEMO_LEARNERS[learner_id]
    states = demo_states(learner_id)
    solved = sum(s.attempts for s in states)
    correct = sum(s.correct for s in states)
    fields = {
        **spec["profile"],
        "questions_solved": solved,
        "accuracy_rate": round(100 * correct / solved, 1) if solved else 0.0,
    }
    learner = db.get(Learner, learner_id)
    if learner is None:
        db.add(Learner(id=learner_id, **fields))
    else:
        for k, v in fields.items():
            setattr(learner, k, v)
    db.flush()
    for s in states:
        db.add(ConceptState(
            learner_id=learner_id,
            concept_id=s.concept_id,
            category=s.category,
            mastery=s.mastery,
            attempts=s.attempts,
            correct=s.correct,
            recent_results=s.recent_results,
            consecutive_wrong=s.consecutive_wrong,
            confidence=s.confidence,
            trend=s.trend,
        ))
    # Sanity check: seeded category scores must match the spec.
    for cat, target in spec["skills"].items():
        got = round(category_score(cat, states))
        if got != target:
            raise RuntimeError(f"Seed mismatch for {learner_id}/{cat}: {got} != {target}")


# Internal learners used only by scripts/warm_pool.py to pre-generate verified questions per level.
# They are hidden from every action except generate_questions (see router) and never seeded with history.
POOL_LEARNERS: dict[str, str] = {
    "pool-beginner": "Beginner",
    "pool-intermediate": "Intermediate",
    "pool-advanced": "Advanced",
}
RESERVED_LEARNER_IDS = frozenset(DEMO_LEARNERS) | frozenset(POOL_LEARNERS)


def ensure_pool_learners(db: Session) -> None:
    for lid, level in POOL_LEARNERS.items():
        if db.get(Learner, lid) is not None:
            continue
        db.add(Learner(id=lid, name=f"Question pool ({level})", level=level, role="internal", goal="ML Engineer",
                       target_role="", experience="internal pool learner", known_languages=["Python"],
                       known_topics=[], target_topics=[], learning_pace="Balanced", daily_commitment_minutes=30))
        db.flush()
        for cid, c in CONCEPTS.items():
            db.add(ConceptState(learner_id=lid, concept_id=cid, category=c["category"], mastery=50.0, attempts=0,
                                correct=0, recent_results=[], consecutive_wrong=0, confidence=0.0, trend="stable"))


# A curated question saved at startup so the Swagger examples (tutor hint, evaluate) work on a
# fresh database. source="fallback" keeps it out of the generated-question pool.
SEED_QUESTION_ID = "seed-q-bias-variance"


def ensure_seed_questions(db: Session) -> None:
    import json
    from pathlib import Path

    from app.engine.question_policy import shuffle_options
    from app.models import Question

    if db.get(Question, SEED_QUESTION_ID) is not None:
        return
    bank = json.loads((Path(__file__).resolve().parent / "data" / "fallback_questions.json").read_text(encoding="utf-8"))
    fb = next(q for q in bank if q["concept_id"] == "bias_variance")
    options, correct = shuffle_options(fb["options"], fb["correct_index"], seed=SEED_QUESTION_ID)
    db.add(Question(
        id=SEED_QUESTION_ID, learner_id=None, concept_id=fb["concept_id"], category=fb["category"],
        difficulty=fb["difficulty"], verified=True, level=None, source="fallback", times_served=0,
        payload={"id": SEED_QUESTION_ID, "category": fb["category"], "difficulty": fb["difficulty"],
                 "conceptId": fb["concept_id"], "conceptTested": CONCEPTS[fb["concept_id"]]["name"],
                 "title": fb["title"], "question": fb["question"], "codeSnippet": fb["code_snippet"],
                 "options": options, "correctIndex": correct, "explanation": fb["explanation"], "hint": fb["hint"],
                 "source": "fallback", "verified": True},
    ))


def seed_if_empty(db: Session) -> bool:
    if db.scalar(select(func.count()).select_from(Learner)):
        return False
    for learner_id in DEMO_LEARNERS:
        _seed_one(db, learner_id)
    return True


def reset_demo_learner(db: Session, learner_id: str) -> None:
    """Clear a demo learner's attempts, adaptive events, concept states, chat history and
    question-serve history, then re-seed.

    The learner row itself is updated in place, so questions that reference it survive.
    """
    if learner_id not in DEMO_LEARNERS:
        raise KeyError(learner_id)
    for model in (Attempt, AdaptiveEvent, ConceptState, ChatMessage, QuestionServe, LearningPath):
        db.execute(delete(model).where(model.learner_id == learner_id))
    db.flush()
    _seed_one(db, learner_id)
