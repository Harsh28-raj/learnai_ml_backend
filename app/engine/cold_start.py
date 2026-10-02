"""Cold-start profiling from the SPEC 7 assessment. Pure functions, no LLM.

Order of rules (later rules win where they overlap):
 1. experience default for every concept, in topological order, capped by its prerequisites
 2. languages: Python -> Python concepts 80 (python_basics 85); none -> Python concepts 15
 3. known topics -> their concepts 78, and every DAG ancestor at least 70
 4. SPEC 7.1 gaps: Statistics not known -> statistics concepts at most 50;
    ML not known -> bias_variance at most 55 (never applied to concepts implied by rule 3)
"""

from collections.abc import Iterable

from app.engine.concepts import CONCEPTS, concepts_in_category, normalize_category, resolve_concept, topological_order
from app.engine.learner_model import MasteryState
from app.engine.recommender import ancestors

EXPERIENCE_DEFAULT = {"Beginner": 15, "Some Programming": 25, "Intermediate": 40, "Advanced": 55}
KNOWN_TOPIC_MASTERY = 78
KNOWN_ANCESTOR_FLOOR = 70
PYTHON_KNOWN = 80
PYTHON_BASICS_KNOWN = 85  # SPEC 7.1: Python selected -> basic Python 85
NO_LANGUAGE = 15
STATS_GAP_CAP = 50  # SPEC 7.1 "Applied Statistics" gap
ML_GAP_CAP = 55  # SPEC 7.1 "Bias vs Variance" gap
BASELINE_CONFIDENCE = 0.1

# Broad assessment topics (SPEC 7 step 3) -> the core concepts they imply.
TOPIC_CONCEPTS: dict[str, list[str]] = {
    "Python": ["python_basics", "python_data_structures"],
    "Statistics": ["probability_basics", "std_variance", "distributions"],
    "Machine Learning": ["linear_regression", "gradient_descent", "overfitting", "decision_trees"],
    "Deep Learning": ["neural_networks", "backpropagation"],
    "NLP": ["tokenization", "word_embeddings"],
    "Generative AI": ["prompt_engineering"],
}


def level_from_assessment(experience_level: str, topics_known: Iterable[str]) -> str:
    """SPEC 7.1: Beginner if experience Beginner or <= 1 topic; Advanced if Advanced or >= 6 topics."""
    n = len([t for t in topics_known if str(t).strip()])
    if experience_level == "Beginner" or n <= 1:
        return "Beginner"
    if experience_level == "Advanced" or n >= 6:
        return "Advanced"
    return "Intermediate"


def resolve_known_topics(topics: Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in topics or []:
        cat = normalize_category(t)
        if cat:
            out += TOPIC_CONCEPTS.get(cat, [c["id"] for c in concepts_in_category(cat)][:2])
            continue
        cid = resolve_concept(t)
        if cid:
            out.append(cid)
    return list(dict.fromkeys(out))


def baseline_masteries(experience_level: str, languages: Iterable[str], topics_known: Iterable[str]) -> dict[str, float]:
    default = EXPERIENCE_DEFAULT.get(experience_level, 25)
    langs = {str(l).strip().lower() for l in languages or [] if str(l).strip()}
    knows_python = "python" in langs
    no_language = not langs or langs <= {"none"}
    python_ids = {c["id"] for c in concepts_in_category("Python")}
    known = resolve_known_topics(topics_known)

    explicit: dict[str, float] = {}
    if knows_python:
        explicit.update({cid: PYTHON_KNOWN for cid in python_ids})
        explicit["python_basics"] = PYTHON_BASICS_KNOWN
    elif no_language:
        explicit.update({cid: NO_LANGUAGE for cid in python_ids})

    m: dict[str, float] = {}
    for cid in topological_order():
        if cid in explicit:
            m[cid] = explicit[cid]
            continue
        prereqs = CONCEPTS[cid]["prerequisites"]
        m[cid] = min([default] + [m[p] for p in prereqs])

    implied: set[str] = set()
    for cid in known:
        m[cid] = max(m[cid], KNOWN_TOPIC_MASTERY)
        implied.add(cid)
        for anc in ancestors(cid):
            m[anc] = max(m[anc], KNOWN_ANCESTOR_FLOOR)
            implied.add(anc)

    # Gaps never override knowledge implied by a known topic (knowing neural nets implies variance).
    known_set = set(known)
    stats_ids = {c["id"] for c in concepts_in_category("Statistics")}
    ml_ids = {c["id"] for c in concepts_in_category("Machine Learning")}
    if not known_set & stats_ids:
        for cid in stats_ids - implied:
            m[cid] = min(m[cid], STATS_GAP_CAP)
    if not known_set & ml_ids and "bias_variance" not in implied:
        m["bias_variance"] = min(m["bias_variance"], ML_GAP_CAP)
    return m


def baseline_states(masteries: dict[str, float]) -> list[MasteryState]:
    return [MasteryState(concept_id=cid, category=CONCEPTS[cid]["category"], mastery=float(m), attempts=0,
                         correct=0, recent_results=[], consecutive_wrong=0, confidence=BASELINE_CONFIDENCE,
                         trend="stable")
            for cid, m in masteries.items()]
