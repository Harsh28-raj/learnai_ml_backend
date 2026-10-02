"""Tutor modes: auto-detection (pure Python) and the per-mode LLM config."""

import re

MODES: tuple[str, ...] = (
    "explanation", "simplify", "example", "code", "practice", "hint", "evaluate", "revision", "path",
)

# Single source of truth for sampling per mode. max_tokens is the budget for the visible
# answer; groq_client adds reasoning headroom on top for reasoning models.
MODE_CONFIG: dict[str, dict] = {
    "explanation": {"temperature": 0.5, "max_tokens": 900},
    "simplify": {"temperature": 0.6, "max_tokens": 600},
    "example": {"temperature": 0.7, "max_tokens": 800},
    "code": {"temperature": 0.2, "max_tokens": 1000},
    "practice": {"temperature": 0.4, "max_tokens": 700},
    "hint": {"temperature": 0.3, "max_tokens": 300},
    "evaluate": {"temperature": 0.1, "max_tokens": 700},
    "revision": {"temperature": 0.3, "max_tokens": 800},
    "path": {"temperature": 0.3, "max_tokens": 500},
}
REPAIR_TEMPERATURE = 0.2
CHECKPOINT_MODES = frozenset({"explanation", "simplify", "example", "practice"})


def _rx(*phrases: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(p) for p in phrases) + r")(?![a-z])")


_HINT = _rx("hint", "clue", "stuck")
_SIMPLIFY = _rx("simpler", "simple", "eli5", "don't understand", "dont understand", "do not understand",
                "samajh nahi", "samajh nahin", "confused", "aasan", "asaan")
_CODE = _rx("code", "implement", "python", "snippet", "example code")
_EXAMPLE = _rx("example", "examples", "real world", "real-world", "use case", "use cases", "udaharan")
_PRACTICE = _rx("quiz me", "test me", "practice", "question do", "questions do")
_REVISION = _rx("revise", "revision", "summary of my weak", "before exam", "before my exam")
_PATH = _rx("why am i learning", "what next", "what's next", "whats next", "learning path", "roadmap", "aage kya")


def detect_mode(message: str, question_id: str | None = None, student_answer: str | None = None) -> str:
    """First match wins, in the documented order."""
    m = message.lower().replace("’", "'")
    if question_id and _HINT.search(m):
        return "hint"
    for mode, pattern in (("simplify", _SIMPLIFY), ("code", _CODE), ("example", _EXAMPLE),
                          ("practice", _PRACTICE), ("revision", _REVISION), ("path", _PATH)):
        if pattern.search(m):
            return mode
    if student_answer and student_answer.strip():
        return "evaluate"
    return "explanation"
