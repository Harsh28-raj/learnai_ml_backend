"""Question-generation policy. Pure functions: everything except the LLM call itself.

Concept selection, scenario/angle rotation, output validation, duplicate rejection,
option shuffling and the user-facing "why this question" text all live here.
"""

import difflib
import random
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from app.engine.concepts import CONCEPTS, concept_name, topological_order
from app.engine.learner_model import DIFFICULTIES, MasteryState
from app.engine.weakness import build_weakness_report

DOMAINS: tuple[str, ...] = (
    "healthcare", "finance/credit", "e-commerce", "autonomous vehicles", "agriculture",
    "sports analytics", "social media", "manufacturing", "education", "cybersecurity",
)
ANGLES: tuple[str, ...] = (
    "concept_check", "formula_interpretation", "curve_or_metric_diagnosis",
    "code_debugging", "scenario_diagnosis", "hyperparameter_tuning",
)
PREREQ_MASTERED = 60
DUPLICATE_RATIO = 0.85
MAX_CODE_LINES = 15


# --------------------------------------------------------------------------- concept selection

def select_concept(
    states: Sequence[MasteryState],
    category: str | None = None,
    target_concept_id: str | None = None,
) -> tuple[str, str]:
    """Return (concept_id, selection_reason).

    Reasons: requested | weak | lowest_practiced | new_topic | next_in_category.
    """
    if target_concept_id:
        return target_concept_id, "requested"

    by_id = {s.concept_id: s for s in states}
    report = build_weakness_report(states)
    weak = [e for e in report["weak"] if category is None or e["category"] == category]
    if weak:
        return weak[0]["concept"], "weak"

    pool = [s for s in states if category is None or s.category == category]
    practiced = [s for s in pool if s.attempts > 0]
    if practiced:
        return min(practiced, key=lambda s: s.mastery).concept_id, "lowest_practiced"

    candidates = [cid for cid in topological_order() if category is None or CONCEPTS[cid]["category"] == category]
    candidates = [cid for cid in candidates if by_id.get(cid) is None or by_id[cid].attempts == 0]
    for cid in candidates:
        if all(_mastery(by_id, pre) >= PREREQ_MASTERED for pre in CONCEPTS[cid]["prerequisites"]):
            return cid, "new_topic"
    if candidates:
        # Nothing is fully unlocked: take the concept whose weakest prerequisite is strongest.
        best = max(candidates, key=lambda cid: min((_mastery(by_id, p) for p in CONCEPTS[cid]["prerequisites"]),
                                                   default=100.0))
        return best, "next_in_category"
    return "python_basics", "next_in_category"


def _mastery(by_id: dict[str, MasteryState], cid: str) -> float:
    s = by_id.get(cid)
    return s.mastery if s else 0.0


# --------------------------------------------------------------------------- domain / angle rotation

def allowed_angles(level: str) -> list[str]:
    if level == "Beginner":
        return [a for a in ANGLES if a != "code_debugging"]
    return list(ANGLES)


def _recency_rank(recent: Sequence[dict], field: str, value: str, concept_id: str | None) -> float:
    """Index of the most recent use (0 = newest); inf if never used."""
    for i, q in enumerate(recent):
        if q.get(field) == value and (concept_id is None or q.get("concept_id") == concept_id):
            return float(i)
    return float("inf")


def _rotate(pool: Sequence[str], count: int, recent: Sequence[dict], field: str, concept_id: str,
            rng: random.Random) -> list[str]:
    shuffled = list(pool)
    rng.shuffle(shuffled)  # random tie-break among equally fresh options
    ranked = sorted(
        shuffled,
        key=lambda v: (-_recency_rank(recent, field, v, concept_id), -_recency_rank(recent, field, v, None)),
    )
    picks = ranked[:count]
    while len(picks) < count:  # only if count > pool size
        picks.append(ranked[len(picks) % len(ranked)])
    return picks


def assign_slots(count: int, level: str, concept_id: str, recent: Sequence[dict],
                 rng: random.Random | None = None) -> list[dict]:
    """Pick a distinct scenario_domain and question_angle per question.

    `recent` is newest-first: [{concept_id, scenario_domain, question_angle}, ...].
    Values never used for this concept come first, then the least recently used.
    """
    rng = rng or random.Random()
    domains = _rotate(DOMAINS, count, recent, "scenario_domain", concept_id, rng)
    angles = _rotate(allowed_angles(level), count, recent, "question_angle", concept_id, rng)
    return [{"index": i, "scenario_domain": d, "question_angle": a} for i, (d, a) in enumerate(zip(domains, angles))]


# --------------------------------------------------------------------------- validation

def normalize_text(text: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


def code_line_count(code: str) -> int:
    """Lines that count toward the 15-line limit (blank lines are free)."""
    return sum(1 for line in code.splitlines() if line.strip())


_FENCED_CODE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)

# "...would also help", "is also a valid fix": the generator itself concedes a second correct option.
_SELF_ADMITTED_MULTIPLE = re.compile(
    r"\b(?:would|could|can|might|may|will)\s+also\s+(?:help|work|reduce|fix|address|improve|lower|solve|be\s+correct)\b"
    r"|\balso\s+(?:a\s+)?(?:valid|correct|reasonable|acceptable|viable)\b"
    r"|\bboth\s+(?:options|answers|choices)\s+(?:are|would\s+be)\s+(?:correct|valid)\b",
    re.IGNORECASE,
)

_CODE_REF = re.compile(
    r"\b(?:following|below|above|this|given)\s+(?:[\w-]+\s+){0,2}(?:code|snippet|program|pipeline|function|script)\b"
    r"|\b(?:code|snippet|program|script)\s+(?:below|above)\b",
    re.IGNORECASE,
)

# "option B", "Answer (2)", "the first option": wrong once options are shuffled. Letters must be
# capitals so ordinary prose like "answer a question" is not flagged.
_POSITION_REF = re.compile(
    r"\b(?i:option|answer|choice)\s*[\(\[]?\s*(?:[A-D]|[1-4])\s*[\)\]]?(?![A-Za-z0-9])"
    r"|\b(?i:first|second|third|fourth|last)\s+(?i:option|answer|choice)\b"
)


class GeneratedQuestion(BaseModel):
    """What the LLM must return per question (also used to validate fallback questions)."""

    title: str
    question: str
    code_snippet: str | None = None
    options: list[str]
    correct_index: int
    explanation: str
    hint: str
    scenario_domain: str | None = None
    question_angle: str | None = None

    @field_validator("title", "question", "explanation", "hint")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("code_snippet")
    @classmethod
    def _code(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        v = str(v).strip("\n")
        n = code_line_count(v)
        if n > MAX_CODE_LINES:
            raise ValueError(f"must be at most {MAX_CODE_LINES} non-blank lines (got {n}); "
                             "remove comments and non-essential lines")
        return v

    @field_validator("options")
    @classmethod
    def _options(cls, v: list[str]) -> list[str]:
        v = [str(o).strip() for o in v]
        if len(v) != 4:
            raise ValueError(f"must have exactly 4 options (got {len(v)})")
        if any(not o for o in v):
            raise ValueError("options must not be empty")
        if len({normalize_text(o) for o in v}) != 4:
            raise ValueError("options must be distinct")
        return v

    @field_validator("correct_index")
    @classmethod
    def _index(cls, v: int) -> int:
        if not 0 <= v <= 3:
            raise ValueError("must be between 0 and 3")
        return v

    @model_validator(mode="after")
    def _move_code_out_of_question(self) -> "GeneratedQuestion":
        """Models sometimes paste code into the question text as well; keep it in code_snippet only."""
        blocks = _FENCED_CODE.findall(self.question)
        if blocks:
            stripped = _FENCED_CODE.sub("", self.question)
            self.question = re.sub(r"\n{3,}", "\n\n", stripped).strip() or self.question
            if self.code_snippet is None:
                code = blocks[0].strip("\n")
                if code_line_count(code) > MAX_CODE_LINES:
                    raise ValueError(f"code_snippet must be at most {MAX_CODE_LINES} lines")
                self.code_snippet = code
        return self

    @model_validator(mode="after")
    def _cross_checks(self) -> "GeneratedQuestion":
        answer = normalize_text(self.options[self.correct_index])
        if len(answer) >= 4 and answer in normalize_text(self.hint):
            raise ValueError("hint must not contain the correct option's text")
        if _POSITION_REF.search(self.explanation) or _POSITION_REF.search(self.hint):
            raise ValueError("explanation/hint must not refer to options by letter or position "
                             "(options are shuffled)")
        if self.code_snippet is None and _CODE_REF.search(self.question):
            raise ValueError("question refers to code but code_snippet is null")
        if _SELF_ADMITTED_MULTIPLE.search(self.explanation):
            raise ValueError("the explanation admits another option would also work, so more than one option is "
                             "correct; make every distractor clearly wrong")
        return self


class LLMQuestion(GeneratedQuestion):
    """Generator output: additionally requires one distractor reason per wrong option.

    distractor_reasons are used for validation only and never sent to the frontend.
    """

    distractor_reasons: list[str]

    @model_validator(mode="after")
    def _reasons(self) -> "LLMQuestion":
        reasons = [str(r).strip() for r in self.distractor_reasons]
        if len(reasons) == 4:  # models sometimes include an entry for the correct option; drop it
            reasons.pop(self.correct_index)
        if len(reasons) != 3 or any(not r for r in reasons):
            raise ValueError("distractor_reasons must contain exactly 3 non-empty reasons, one per wrong option")
        self.distractor_reasons = reasons
        return self


def _short_errors(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "__root__")
        msg = err.get("msg", "").removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def is_duplicate(text: str, others: Sequence[str]) -> str | None:
    """Return the matching text if `text` is a near-duplicate (difflib ratio > 0.85)."""
    norm = normalize_text(text)
    for other in others:
        o = normalize_text(other)
        sm = difflib.SequenceMatcher(None, norm, o)
        if sm.quick_ratio() > DUPLICATE_RATIO and sm.ratio() > DUPLICATE_RATIO:
            return other
    return None


def validate_batch(raw: Any, slots: Sequence[dict], recent_texts: Sequence[str],
                   accepted_texts: Sequence[str] = (), model: type[GeneratedQuestion] = LLMQuestion,
                   ) -> tuple[list[tuple[GeneratedQuestion, dict]], list[str], list[dict]]:
    """Validate the LLM's {"questions": [...]} against the assigned slots (by position).

    Returns (valid [(question, slot)], error messages, failed slots).
    """
    items = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return [], ['Response must be a JSON object with a "questions" array.'], list(slots)

    valid: list[tuple[GeneratedQuestion, dict]] = []
    errors: list[str] = []
    failed: list[dict] = []
    seen = list(accepted_texts)
    for pos, slot in enumerate(slots):
        label = f"Question {pos + 1} ({slot['scenario_domain']}, {slot['question_angle']})"
        if pos >= len(items) or not isinstance(items[pos], dict):
            errors.append(f"{label}: missing.")
            failed.append(slot)
            continue
        try:
            q = model.model_validate(items[pos])
        except ValidationError as e:
            errors.append(f"{label}: {_short_errors(e)}.")
            failed.append(slot)
            continue
        dup = is_duplicate(q.question, list(recent_texts) + seen)
        if dup is not None:
            errors.append(f"{label}: too similar to a previous question; write a genuinely different one.")
            failed.append(slot)
            continue
        seen.append(q.question)
        valid.append((q, slot))
    return valid, errors, failed


# --------------------------------------------------------------------------- post-processing

def shuffle_options(options: Sequence[str], correct_index: int, seed: Any) -> tuple[list[str], int]:
    """Seeded shuffle that keeps track of the correct answer."""
    perm = list(range(len(options)))
    random.Random(str(seed)).shuffle(perm)
    return [options[i] for i in perm], perm.index(correct_index)


def _level_of_last(state: MasteryState) -> str | None:
    return state.recent_results[-1].get("difficulty") if state.recent_results else None


def why_this_question(selection_reason: str, state: MasteryState, next_diff: dict,
                      difficulty_requested: bool = False) -> str:
    """User-facing rationale built only from real learner facts (never by the LLM)."""
    name = concept_name(state.concept_id)
    last3 = state.recent_results[-3:]
    missed = sum(1 for r in last3 if not r.get("correct"))
    current = _level_of_last(state)
    stepped_up = (
        not difficulty_requested and current in DIFFICULTIES
        and next_diff.get("rule_id") in ("R4_FAST_STREAK", "R5_STREAK_MASTERY_GATE")
        and DIFFICULTIES.index(next_diff["difficulty"]) > DIFFICULTIES.index(current)
    )

    if stepped_up:
        return f"Advanced challenge: you answered 3 {current} questions on {name} correctly in a row."
    if selection_reason == "new_topic":
        prereqs = [concept_name(p) for p in CONCEPTS[state.concept_id]["prerequisites"]]
        if prereqs:
            return f"New topic: you've mastered its prerequisites ({', '.join(prereqs)})."
        return f"New topic: {name} is a foundation with no prerequisites, a good next step."
    if selection_reason == "next_in_category":
        return f"Next step in {state.category}: {name} is the most accessible concept you haven't practiced yet."
    if state.attempts == 0:
        return (f"First practice on {name}: starting at {next_diff['difficulty']} based on an estimated "
                f"{round(state.mastery)}% mastery.")

    text = f"you scored {round(state.mastery)}% on {name}"
    if missed >= 2 and len(last3) == 3:
        text += f" and missed {missed} of your last 3 answers"
    if selection_reason == "requested":
        return f"You asked to practice {name}; {text}."
    if selection_reason == "lowest_practiced":
        return f"Generated because {text}, your lowest-scoring practiced concept here."
    return f"Generated because {text}."
