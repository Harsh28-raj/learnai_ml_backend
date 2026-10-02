"""Question-pool warm-up planning (used by scripts/warm_pool.py). Pure functions.

A cell is (concept_id, level, difficulty). Cells that don't make sense are skipped:
  - Beginner: no Hard questions; Advanced: no Easy questions
  - foundational Python (python_basics, python_data_structures): no Hard at any level
"""

import math
from collections.abc import Iterable, Mapping, Sequence

from app.engine.concepts import CONCEPTS

LEVELS = ("Beginner", "Intermediate", "Advanced")
DIFFICULTIES = ("Easy", "Medium", "Hard")
FOUNDATIONAL = frozenset({"python_basics", "python_data_structures"})
MAX_PER_REQUEST = 5  # generate_questions count limit

Cell = tuple[str, str, str]


def cell_allowed(concept_id: str, level: str, difficulty: str) -> bool:
    if level == "Beginner" and difficulty == "Hard":
        return False
    if level == "Advanced" and difficulty == "Easy":
        return False
    if concept_id in FOUNDATIONAL and difficulty == "Hard":
        return False
    return True


def cells_for(concepts_by_level: Mapping[str, Iterable[str]]) -> list[Cell]:
    """{level: concept ids} -> allowed cells, deterministic order."""
    out: list[Cell] = []
    for level in LEVELS:
        for cid in sorted(set(concepts_by_level.get(level, ()))):
            for diff in DIFFICULTIES:
                if cid in CONCEPTS and cell_allowed(cid, level, diff):
                    out.append((cid, level, diff))
    return out


def all_cells() -> list[Cell]:
    return cells_for({level: list(CONCEPTS) for level in LEVELS})


def plan(cells: Sequence[Cell], counts: Mapping[Cell, int], per_cell: int) -> list[tuple[Cell, int]]:
    """Resumable: only cells below per_cell verified questions, with how many are still needed."""
    return [(c, per_cell - counts.get(c, 0)) for c in cells if counts.get(c, 0) < per_cell]


def estimate(todo: Sequence[tuple[Cell, int]], gap_seconds: float) -> dict:
    """Each request = 1 generate + 1 verify call best case, up to 2 + 2 with the regeneration round."""
    requests = sum(math.ceil(need / MAX_PER_REQUEST) for _, need in todo)
    return {
        "cells": len(todo),
        "questions": sum(need for _, need in todo),
        "requests": requests,
        "llm_calls_min": requests * 2,
        "llm_calls_max": requests * 4,
        "minutes_at_gap": round(requests * gap_seconds / 60, 1),
    }
