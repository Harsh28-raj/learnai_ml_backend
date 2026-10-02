"""whyThisPath polish: GROQ_MODEL_FAST rewrites Python-built facts; numbers are validated against them."""

import logging
import re
from collections.abc import Sequence

from app import prompts
from app.config import get_settings
from app.llm.groq_client import LLMError, chat_completion

logger = logging.getLogger("learnai.why_path")

MAX_WORDS = 90
WORD_TOLERANCE = 1.2
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
               "eighty": "80", "ninety": "90", "hundred": "100", "dozen": "12"})
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


def numbers_in(text: str) -> set[str]:
    """Digits plus spelled-out numbers ("three weeks" counts as 3), so words can't bypass validation."""
    return set(_NUMBER.findall(text)) | {_WORDS[w.lower()] for w in _WORD_RE.findall(text)}


def validate_why(text: str, facts: Sequence[str]) -> str | None:
    """Return an issue string, or None if the polished text is acceptable."""
    if not text or not text.strip():
        return "empty"
    if len(text.split()) > MAX_WORDS * WORD_TOLERANCE:
        return "too long"
    invented = numbers_in(text) - set(_NUMBER.findall(" ".join(facts)))
    if invented:
        return f"numbers not in facts: {sorted(invented)}"
    return None


async def polish_why(name: str, facts: Sequence[str], template: str) -> tuple[str, str]:
    """Return (text, source) with source "llm" or "template". Never raises."""
    if not get_settings().llm_enabled:
        return template, "template"
    blk = prompts.blocks("why_this_path")
    messages = [
        {"role": "system", "content": prompts.render(blk["system"], {})},
        {"role": "user", "content": prompts.render(blk["user"], {"name": name, "facts": "\n".join(
            f"- {f}" for f in facts)})},
    ]
    try:
        raw = await chat_completion(messages, get_settings().groq_model_fast, json_mode=True, temperature=0.3,
                                    max_tokens=220)
    except LLMError as e:
        logger.warning("whyThisPath polish failed (%s); using template", e.code)
        return template, "template"
    text = str((raw or {}).get("why_this_path") or "").strip() if isinstance(raw, dict) else ""
    issue = validate_why(text, facts)
    if issue:
        logger.warning("whyThisPath polish rejected (%s); using template", issue)
        return template, "template"
    return text, "llm"
