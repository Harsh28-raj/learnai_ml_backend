"""Versioned prompt templates.

Each .txt file holds one or more blocks introduced by a line `### <name>`. Blocks are loaded
once (load_all() runs at startup) and rendered with str.format_map on a dict whose missing
keys render as "" so a forgotten value never crashes a request. Literal braces in templates
are written as {{ and }}.
"""

import re
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent
TUTOR_PROMPT_VERSION = "tutor-v1"
QUESTION_PROMPT_VERSION = "qgen-v2"

TEMPLATE_FILES = (
    "tutor_system", "tutor_levels", "tutor_modes", "tutor_guardrails", "tutor_output_schema",
    "question_generator", "question_verifier", "why_this_path",
)

_HEADER = re.compile(r"^### +(.+?) *$", re.MULTILINE)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _parse(text: str) -> dict[str, str]:
    parts = _HEADER.split(text)
    blocks = {"": parts[0].strip()} if parts[0].strip() else {}
    for name, body in zip(parts[1::2], parts[2::2]):
        blocks[name.strip()] = body.strip()
    return blocks


@lru_cache(maxsize=None)
def blocks(name: str) -> dict[str, str]:
    return _parse((PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8"))


def block(name: str, key: str = "") -> str:
    try:
        return blocks(name)[key]
    except KeyError:
        raise KeyError(f"Prompt block '{key}' not found in {name}.txt") from None


def render(template: str, values: dict) -> str:
    return template.format_map(_SafeDict(values))


def load_all() -> None:
    """Load and sanity-check every template (called once at startup)."""
    for name in TEMPLATE_FILES:
        if not blocks(name):
            raise RuntimeError(f"Prompt file {name}.txt is empty")
