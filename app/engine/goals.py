"""Learning goals (SPEC 7 step 4) -> target concepts, plus free-text goal resolution."""

import difflib
import re
from collections.abc import Iterable

from app.engine.concepts import CONCEPTS, concepts_in_category, normalize_category, resolve_concept

GOAL_TARGETS: dict[str, list[str]] = {
    "ML Engineer": [
        "pandas_dataframes", "linear_regression", "gradient_descent", "bias_variance", "overfitting",
        "regularization", "decision_trees", "confusion_matrix_roc", "cross_validation",
        "neural_networks", "backpropagation",
    ],
    "Interviews": [
        "bias_variance", "overfitting", "regularization", "gradient_descent", "confusion_matrix_roc",
        "decision_trees", "cross_validation", "backpropagation", "transformers_attention",
    ],
    "GenAI Apps": [
        "word_embeddings", "transformers_attention", "prompt_engineering", "rag_chunking", "llm_finetuning",
    ],
    "ML from Scratch": [
        "python_basics", "python_data_structures", "numpy_arrays", "std_variance", "distributions",
        "linear_regression", "gradient_descent", "bias_variance", "overfitting", "neural_networks",
    ],
}

GOAL_LABELS = {
    "ML Engineer": "become an ML Engineer",
    "Interviews": "prepare for ML interviews",
    "GenAI Apps": "build GenAI apps",
    "ML from Scratch": "learn ML from scratch",
}

DEFAULT_GOAL = "ML Engineer"

# Keyword aliases checked as whole words in the normalized goal text (most specific first).
_GOAL_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("GenAI Apps", ("genai", "gen ai", "generative", "llm", "llms", "rag", "fine tune", "finetune", "chatbot",
                    "gpt", "transformer apps")),
    ("Interviews", ("interview", "interviews", "job prep", "crack", "placement")),
    ("ML from Scratch", ("from scratch", "beginner", "basics", "start learning", "learn ml", "learn machine learning",
                         "new to")),
    ("ML Engineer", ("ml engineer", "machine learning engineer", "mle", "data scientist", "ml developer",
                     "production ml", "mlops")),
]


def _norm(text: str) -> str:
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text).lower().replace("-", " "))).strip() + " "


def resolve_goal(text: str | None) -> str:
    """Map a goal key or free text ("Become an ML Engineer") to a GOAL_TARGETS key."""
    if not text or not str(text).strip():
        return DEFAULT_GOAL
    if text in GOAL_TARGETS:
        return text
    norm = _norm(text)
    for key, words in _GOAL_KEYWORDS:
        if any(f" {w} " in norm for w in words):
            return key
    candidates = {_norm(k).strip(): k for k in GOAL_TARGETS} | {_norm(v).strip(): k for k, v in GOAL_LABELS.items()}
    close = difflib.get_close_matches(norm.strip(), list(candidates), n=1, cutoff=0.6)
    return candidates[close[0]] if close else DEFAULT_GOAL


def topic_targets(topics: Iterable[str]) -> list[str]:
    """Learner target_topics -> concept ids. A category name ("Deep Learning") adds its concepts."""
    out: list[str] = []
    for t in topics or []:
        cat = normalize_category(t)
        if cat:
            out += [c["id"] for c in concepts_in_category(cat)]
            continue
        cid = resolve_concept(t)
        if cid:
            out.append(cid)
    return list(dict.fromkeys(out))


def goal_targets(goal_key: str, target_topics: Iterable[str] = ()) -> list[str]:
    targets = list(GOAL_TARGETS.get(goal_key, GOAL_TARGETS[DEFAULT_GOAL])) + topic_targets(target_topics)
    return [c for c in dict.fromkeys(targets) if c in CONCEPTS]
