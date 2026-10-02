"""Concept catalog: fine-grained concepts, their skill category, aliases, and prerequisite DAG."""

import difflib
import re
from collections.abc import Iterable, Mapping
from functools import lru_cache

CATEGORIES: tuple[str, ...] = (
    "Python",
    "Statistics",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Generative AI",
)

_CATALOG: list[dict] = [
    # Python
    {"id": "python_basics", "name": "Python Basics (Variables, Loops, Functions)", "category": "Python",
     "prerequisites": [],
     "aliases": ["Python Basics", "Python Fundamentals", "Python Syntax", "Variables and Loops", "Python"]},
    {"id": "python_data_structures", "name": "Python Data Structures", "category": "Python",
     "prerequisites": ["python_basics"],
     "aliases": ["Data Structures", "Lists and Dictionaries", "Python Lists", "Dictionaries"]},
    {"id": "numpy_arrays", "name": "NumPy Arrays & Vectorization", "category": "Python",
     "prerequisites": ["python_data_structures"],
     "aliases": ["NumPy", "NumPy Arrays", "Vectorization", "ndarray"]},
    {"id": "matrix_shapes", "name": "Matrix Multiplication & Shapes", "category": "Python",
     "prerequisites": ["numpy_arrays"],
     "aliases": ["Matrix Multiplication", "Matrix Shapes", "Linear Algebra", "Matrix Operations", "Broadcasting"]},
    {"id": "pandas_dataframes", "name": "Pandas DataFrames", "category": "Python",
     "prerequisites": ["numpy_arrays"],
     "aliases": ["Pandas", "DataFrames", "Data Manipulation"]},
    # Statistics
    {"id": "probability_basics", "name": "Probability Fundamentals", "category": "Statistics",
     "prerequisites": [],
     "aliases": ["Probability", "Probability Basics", "Conditional Probability", "Bayes Theorem"]},
    {"id": "std_variance", "name": "Standard Deviation & Variance", "category": "Statistics",
     "prerequisites": ["probability_basics"],
     "aliases": ["Standard Deviation", "Variance", "Std Deviation & Variance", "Std Dev",
                 "Applied Statistics", "Descriptive Statistics"]},
    {"id": "distributions", "name": "Probability Distributions", "category": "Statistics",
     "prerequisites": ["std_variance"],
     "aliases": ["Distributions", "Normal Distribution", "Gaussian Distribution", "Binomial Distribution"]},
    {"id": "hypothesis_testing", "name": "Hypothesis Testing", "category": "Statistics",
     "prerequisites": ["distributions"],
     "aliases": ["p-values", "A/B Testing", "Statistical Significance", "t-test"]},
    # Machine Learning
    {"id": "linear_regression", "name": "Linear Regression", "category": "Machine Learning",
     "prerequisites": ["matrix_shapes", "std_variance"],
     "aliases": ["Least Squares", "OLS", "Regression"]},
    {"id": "gradient_descent", "name": "Gradient Descent Optimization", "category": "Machine Learning",
     "prerequisites": ["linear_regression"],
     "aliases": ["Gradient Descent", "SGD", "Stochastic Gradient Descent", "Learning Rate", "Optimizers"]},
    {"id": "bias_variance", "name": "Bias vs Variance Tradeoff", "category": "Machine Learning",
     "prerequisites": ["linear_regression", "std_variance"],
     "aliases": ["Bias vs Variance", "Bias-Variance Tradeoff", "bias variance", "Bias-Variance"]},
    {"id": "overfitting", "name": "Overfitting & Underfitting", "category": "Machine Learning",
     "prerequisites": ["bias_variance"],
     "aliases": ["Overfitting", "Underfitting", "Overfit", "Underfit", "Generalization", "Model Diagnostics",
                 "Learning Curves"]},
    {"id": "regularization", "name": "Regularization (L1/L2)", "category": "Machine Learning",
     "prerequisites": ["overfitting", "gradient_descent"],
     "aliases": ["Regularization", "L1 Regularization", "L2 Regularization", "Ridge", "Lasso", "Weight Decay",
                 "Regularization Hyperparameters"]},
    {"id": "decision_trees", "name": "Decision Trees", "category": "Machine Learning",
     "prerequisites": ["probability_basics", "pandas_dataframes"],
     "aliases": ["Decision Tree", "Random Forest", "Tree Pruning", "Gini Impurity", "Tree Regularization"]},
    {"id": "confusion_matrix_roc", "name": "Confusion Matrix & ROC-AUC", "category": "Machine Learning",
     "prerequisites": ["probability_basics", "decision_trees"],
     "aliases": ["Confusion Matrix", "ROC", "ROC-AUC", "AUC", "Precision and Recall", "F1 Score",
                 "Classification Metrics", "Model Evaluation"]},
    {"id": "cross_validation", "name": "Cross-Validation", "category": "Machine Learning",
     "prerequisites": ["overfitting"],
     "aliases": ["Cross Validation", "K-Fold", "K-Fold Cross Validation", "Train Test Split", "Validation Set"]},
    # Deep Learning
    {"id": "neural_networks", "name": "Neural Network Fundamentals", "category": "Deep Learning",
     "prerequisites": ["gradient_descent", "matrix_shapes"],
     "aliases": ["Neural Networks", "Neural Network", "Perceptron", "MLP", "Activation Functions"]},
    {"id": "backpropagation", "name": "Backpropagation", "category": "Deep Learning",
     "prerequisites": ["neural_networks"],
     "aliases": ["Backprop", "Chain Rule", "Autograd", "Vanishing Gradients"]},
    {"id": "cnns", "name": "Convolutional Neural Networks", "category": "Deep Learning",
     "prerequisites": ["backpropagation"],
     "aliases": ["CNN", "CNNs", "Convolution", "Computer Vision", "Pooling"]},
    # NLP
    {"id": "tokenization", "name": "Tokenization & Text Preprocessing", "category": "NLP",
     "prerequisites": ["python_data_structures"],
     "aliases": ["Tokenization", "Tokenizer", "Text Preprocessing", "BPE", "Byte Pair Encoding", "Lemmatization"]},
    {"id": "word_embeddings", "name": "Word Embeddings", "category": "NLP",
     "prerequisites": ["tokenization", "matrix_shapes"],
     "aliases": ["Embeddings", "Word2Vec", "GloVe", "Vector Embeddings", "Cosine Similarity"]},
    {"id": "transformers_attention", "name": "Transformers & Self-Attention", "category": "NLP",
     "prerequisites": ["word_embeddings", "backpropagation"],
     "aliases": ["Transformers", "Transformer", "Attention", "Self-Attention", "Multi-Head Attention",
                 "Attention Mechanism", "Positional Encoding", "FlashAttention"]},
    # Generative AI
    {"id": "prompt_engineering", "name": "Prompt Engineering", "category": "Generative AI",
     "prerequisites": ["tokenization"],
     "aliases": ["Prompting", "Few-Shot Prompting", "Chain of Thought", "System Prompts"]},
    {"id": "rag_chunking", "name": "Vector DB Embedding Chunking & Semantic Drift", "category": "Generative AI",
     "prerequisites": ["word_embeddings", "prompt_engineering"],
     "aliases": ["RAG", "Retrieval Augmented Generation", "RAG Chunking", "Chunking", "Vector DB",
                 "Vector Database", "Semantic Drift", "Vector DB Chunking", "Embedding Chunking"]},
    {"id": "llm_finetuning", "name": "LLM Fine-Tuning (LoRA)", "category": "Generative AI",
     "prerequisites": ["transformers_attention", "regularization"],
     "aliases": ["Fine-Tuning", "Fine Tuning", "LLM Fine-Tuning", "LoRA", "QLoRA", "PEFT", "Instruction Tuning"]},
]

# Two-line plain-English summaries, used as the tutor's offline fallback reply.
SUMMARIES: dict[str, str] = {
    "python_basics": "Python programs are built from variables, loops, conditionals and functions.\n"
                     "Mastering these lets you express any step-by-step data or ML workflow.",
    "python_data_structures": "Lists keep ordered items, dicts map keys to values, sets keep unique items.\n"
                              "Picking the right one makes data handling simpler and much faster.",
    "numpy_arrays": "NumPy arrays hold numbers in a fixed-type grid and apply operations to all of them at once.\n"
                    "This vectorization is what makes numerical Python fast enough for ML.",
    "matrix_shapes": "In A @ B the inner dimensions must match: (m, n) @ (n, p) gives (m, p).\n"
                     "Checking shapes before multiplying catches most bugs in ML code.",
    "pandas_dataframes": "A DataFrame is a table with labelled columns that you can filter, group and join.\n"
                         "It is the standard way to clean and explore data before modelling.",
    "probability_basics": "Probability measures how likely an event is, from 0 (never) to 1 (certain).\n"
                          "Conditional probability updates that likelihood once you know something else happened.",
    "std_variance": "Variance is the average squared distance from the mean; standard deviation is its square root.\n"
                    "Both tell you how spread out the data is, not how large it is.",
    "distributions": "A distribution describes how likely each possible value is, like the bell-shaped normal curve.\n"
                     "Knowing the shape of your data guides which models and tests make sense.",
    "hypothesis_testing": "A hypothesis test asks whether an observed effect is bigger than chance alone would produce.\n"
                          "The p-value is the chance of data this extreme if there were truly no effect.",
    "linear_regression": "Linear regression fits a straight line (or plane) that minimizes squared prediction errors.\n"
                         "Its coefficients say how much the prediction changes per unit of each feature.",
    "gradient_descent": "Gradient descent improves a model by repeatedly stepping its parameters downhill on the loss.\n"
                        "The learning rate sets the step size: too big overshoots, too small crawls.",
    "bias_variance": "Bias is error from a model too simple to capture the pattern; variance is error from fitting noise.\n"
                     "Good models balance the two so they generalize to new data.",
    "overfitting": "Overfitting means memorizing training data, so validation error is much worse than training error.\n"
                   "Underfitting means the model is too simple to do well on either.",
    "regularization": "Regularization adds a penalty on large weights so the model stays simpler.\n"
                      "L2 (Ridge) shrinks weights smoothly; L1 (Lasso) can push some exactly to zero.",
    "decision_trees": "A decision tree splits data with yes/no questions on features until the groups are pure.\n"
                      "Unlimited depth overfits, so depth limits, pruning or forests are used.",
    "confusion_matrix_roc": "A confusion matrix counts true/false positives and negatives for a classifier.\n"
                            "The ROC curve and its AUC show how well the model ranks positives above negatives.",
    "cross_validation": "Cross-validation trains and tests on several different splits of the data.\n"
                        "Averaging the scores gives a more reliable estimate of real-world performance.",
    "neural_networks": "A neural network stacks layers of weighted sums followed by non-linear activations.\n"
                       "The non-linearity is what lets it learn complex patterns.",
    "backpropagation": "Backpropagation uses the chain rule to compute how each weight affects the loss.\n"
                       "Those gradients tell gradient descent which way to adjust every weight.",
    "cnns": "Convolutional networks slide small learned filters over an image to detect local patterns.\n"
            "Stacking them builds up from edges to shapes to whole objects.",
    "tokenization": "Tokenization splits text into units (words or sub-words) that a model can turn into numbers.\n"
                    "Sub-word methods like BPE handle rare and new words gracefully.",
    "word_embeddings": "Embeddings map each token to a dense vector so similar meanings land close together.\n"
                       "Distances between vectors then capture relationships between words.",
    "transformers_attention": "Self-attention lets every token weigh every other token when building its representation.\n"
                              "Transformers stack attention layers, which is why they handle long context so well.",
    "prompt_engineering": "Prompt engineering means writing instructions and examples that steer an LLM's output.\n"
                          "Clear format specs and a few examples make answers far more consistent.",
    "rag_chunking": "RAG retrieves relevant text chunks from a vector database and feeds them to the LLM.\n"
                    "Chunk size and overlap decide whether each embedding captures one focused idea.",
    "llm_finetuning": "Fine-tuning continues training a pretrained LLM on your own examples.\n"
                      "LoRA trains small low-rank adapter matrices instead of all the weights, saving memory.",
}

for _c in _CATALOG:
    _c["summary"] = SUMMARIES[_c["id"]]

CONCEPTS: dict[str, dict] = {c["id"]: c for c in _CATALOG}


# Frontend categories (SPEC §10.1) and shorthands mapped onto the 6 skill categories.
_CATEGORY_ALIASES = {
    "python": "Python", "statistics": "Statistics", "stats": "Statistics",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning", "computer vision": "Deep Learning",
    "nlp": "NLP", "natural language processing": "NLP",
    "generative ai": "Generative AI", "genai": "Generative AI", "gen ai": "Generative AI", "llms": "Generative AI",
    "llm": "Generative AI",
}


def normalize_category(text: str | None) -> str | None:
    """Canonical category name, or None if unrecognised."""
    if not text:
        return None
    return _CATEGORY_ALIASES.get(re.sub(r"\s+", " ", str(text).strip().lower().replace("-", " ")))


def get_concept(concept_id: str) -> dict | None:
    return CONCEPTS.get(concept_id)


def concept_name(concept_id: str) -> str:
    c = CONCEPTS.get(concept_id)
    return c["name"] if c else concept_id


def concepts_in_category(category: str) -> list[dict]:
    return [c for c in _CATALOG if c["category"] == category]


def lesson_id_for(concept_id: str) -> str:
    """Frontend lesson ids are kebab-case (e.g. 'bias-variance')."""
    return concept_id.replace("_", "-")


def dependents_of(concept_id: str) -> list[str]:
    """Concepts that list `concept_id` as a direct prerequisite (catalog order)."""
    return [c["id"] for c in _CATALOG if concept_id in c["prerequisites"]]


# --------------------------------------------------------------------------- resolution

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "vs", "v",
    "basics", "basic", "fundamentals", "intro", "introduction", "concept", "concepts", "using",
})
_TOKEN_MATCH_THRESHOLD = 0.5
_FUZZY_CUTOFF = 0.9


def normalize_text(text: str) -> str:
    t = str(text).lower().replace("&", " and ")
    t = re.sub(r"[-_/]", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    t = re.sub(r"\bversus\b", "vs", t)
    return re.sub(r"\s+", " ", t).strip()


def _stem(token: str) -> str:
    # Light plural stripping, applied identically to input and labels.
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_stem(t) for t in normalize_text(text).split() if t not in _STOPWORDS)


@lru_cache(maxsize=1)
def _label_index() -> list[tuple[str, str, frozenset[str]]]:
    """(concept_id, normalized label, tokens) for every id, name and alias."""
    index = []
    for c in _CATALOG:
        for label in (c["id"], c["name"], *c["aliases"]):
            index.append((c["id"], normalize_text(label), _tokens(label)))
    return index


def _dice(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _scores(text: str) -> dict[str, float]:
    """Best token-overlap (Dice) score per concept."""
    tokens = _tokens(text)
    best: dict[str, float] = {}
    for cid, _, label_tokens in _label_index():
        s = _dice(tokens, label_tokens)
        if s > best.get(cid, 0.0):
            best[cid] = s
    return best


def resolve_concept(text_or_id: str | None, category_hint: str | None = None) -> str | None:
    """Map a concept id, name, alias or free-text topic to a concept id, or None.

    Order: exact id -> case-insensitive name/alias -> token overlap (with a strict
    fuzzy fallback for typos). `category_hint` breaks ties toward that category.
    """
    if not text_or_id or not str(text_or_id).strip():
        return None
    raw = str(text_or_id).strip()
    if raw in CONCEPTS:
        return raw

    norm = normalize_text(raw)
    exact = [cid for cid, label, _ in _label_index() if label == norm]
    if exact:
        return _prefer_category(dict.fromkeys(exact, 1.0), category_hint)

    scores = {cid: s for cid, s in _scores(raw).items() if s >= _TOKEN_MATCH_THRESHOLD}
    if scores:
        return _prefer_category(scores, category_hint)

    labels = [label for _, label, _ in _label_index()]
    close = difflib.get_close_matches(norm, labels, n=1, cutoff=_FUZZY_CUTOFF)
    if close:
        return next(cid for cid, label, _ in _label_index() if label == close[0])
    return None


def _prefer_category(scores: dict[str, float], category_hint: str | None) -> str:
    def key(cid: str) -> tuple[float, int]:
        in_hint = 1 if category_hint and CONCEPTS[cid]["category"].lower() == category_hint.strip().lower() else 0
        return (scores[cid], in_hint)

    return max(scores, key=key)


# Single generic words that would hijack chat messages ("give me python code for this").
_IN_TEXT_SKIP = frozenset({"python", "regression", "optimizers", "probability", "variance", "attention"})


def find_concept_in_text(text: str | None) -> str | None:
    """Best-effort: the longest concept name/alias that appears as a phrase inside free text."""
    if not text:
        return None
    norm = " " + " ".join(_stem(t) for t in normalize_text(text).split()) + " "
    best: tuple[int, str] | None = None
    for cid, label, _ in _label_index():
        if len(label) < 3 or label in _IN_TEXT_SKIP:
            continue
        stemmed = " ".join(_stem(t) for t in label.split())
        if f" {stemmed} " in norm and (best is None or len(stemmed) > best[0]):
            best = (len(stemmed), cid)
    return best[1] if best else None


def suggest_concepts(text: str, n: int = 3) -> list[dict]:
    """Closest concepts for an unresolvable input (for UNKNOWN_CONCEPT error details)."""
    norm = normalize_text(text or "")
    token_scores = _scores(text or "")
    ranked: dict[str, float] = {}
    for cid, label, _ in _label_index():
        fuzzy = difflib.SequenceMatcher(None, norm, label).ratio()
        ranked[cid] = max(ranked.get(cid, 0.0), fuzzy, token_scores.get(cid, 0.0))
    top = sorted(ranked.items(), key=lambda kv: -kv[1])[:n]
    return [{"concept_id": cid, "name": CONCEPTS[cid]["name"], "score": round(s, 2)} for cid, s in top]


# --------------------------------------------------------------------------- DAG

def find_cycle(graph: Mapping[str, Iterable[str]]) -> list[str] | None:
    """Return one cycle as a list of node ids (first == last), or None if acyclic.

    `graph` maps node -> prerequisites. Iterative DFS with white/grey/black colouring.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    for start in graph:
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, list[str]]] = [(start, list(graph[start]))]
        path = [start]
        color[start] = GREY
        while stack:
            node, pending = stack[-1]
            if not pending:
                color[node] = BLACK
                stack.pop()
                path.pop()
                continue
            nxt = pending.pop()
            state = color.get(nxt, WHITE)
            if state == GREY:
                return path[path.index(nxt):] + [nxt]
            if state == WHITE:
                color[nxt] = GREY
                stack.append((nxt, list(graph.get(nxt, ()))))
                path.append(nxt)
    return None


def validate_dag(concepts: Mapping[str, dict] = CONCEPTS) -> None:
    """Raise ValueError if any prerequisite is unknown or the graph has a cycle."""
    for cid, c in concepts.items():
        for pre in c["prerequisites"]:
            if pre not in concepts:
                raise ValueError(f"Concept '{cid}' has unknown prerequisite '{pre}'")
            if pre == cid:
                raise ValueError(f"Concept '{cid}' lists itself as a prerequisite")
    cycle = find_cycle({cid: c["prerequisites"] for cid, c in concepts.items()})
    if cycle:
        raise ValueError("Prerequisite cycle detected: " + " -> ".join(cycle))


def topological_order(concepts: Mapping[str, dict] = CONCEPTS) -> list[str]:
    """Concept ids ordered so every prerequisite comes before its dependents."""
    validate_dag(concepts)
    order: list[str] = []
    seen: set[str] = set()

    def visit(cid: str) -> None:
        if cid in seen:
            return
        seen.add(cid)
        for pre in concepts[cid]["prerequisites"]:
            visit(pre)
        order.append(cid)

    for cid in concepts:
        visit(cid)
    return order


def prerequisite_layers(concept_id: str, concepts: Mapping[str, dict] = CONCEPTS) -> list[list[str]]:
    """Ancestors grouped by distance: [direct prereqs, their prereqs, ...] (no duplicates)."""
    layers: list[list[str]] = []
    seen = {concept_id}
    frontier = [concept_id]
    while frontier:
        nxt: list[str] = []
        for cid in frontier:
            for pre in concepts[cid]["prerequisites"]:
                if pre not in seen:
                    seen.add(pre)
                    nxt.append(pre)
        if nxt:
            layers.append(nxt)
        frontier = nxt
    return layers
