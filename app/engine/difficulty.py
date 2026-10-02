"""Adaptive difficulty: choose the next question difficulty for one concept. Pure."""

from app.engine.concepts import concept_name
from app.engine.learner_model import DIFFICULTIES, MasteryState

EXPECTED_SECONDS = {"Easy": 30, "Medium": 60, "Hard": 120}
FAST_RATIO = 0.6
SLOW_RATIO = 1.5
STEP_UP_MASTERY = {"Easy": 60, "Medium": 75}  # mastery needed to leave this level
WINDOW = 5


def step(difficulty: str, delta: int) -> str:
    i = DIFFICULTIES.index(difficulty) + delta
    return DIFFICULTIES[max(0, min(len(DIFFICULTIES) - 1, i))]


def _result(difficulty: str, reason: str, rule_id: str) -> dict:
    return {"difficulty": difficulty, "reason": reason, "rule_id": rule_id}


def next_difficulty(state: MasteryState) -> dict:
    """Return {difficulty, reason, rule_id}. Rules are evaluated in order; first match wins."""
    name = concept_name(state.concept_id)
    recent = list(state.recent_results)[-WINDOW:]

    # R0: no evidence yet -> start from the mastery estimate.
    if state.attempts == 0 or not recent:
        m = state.mastery
        level = "Easy" if m < 50 else "Medium" if m < 75 else "Hard"
        return _result(level, f"No attempts on {name} yet; starting at {level} based on an estimated "
                              f"{round(m)}% mastery.", "R0_NO_ATTEMPTS")

    current = recent[-1].get("difficulty", "Medium")
    if current not in DIFFICULTIES:
        current = "Medium"
    last2, last3 = recent[-2:], recent[-3:]

    # R1/R2: two misses in a row at the same high level.
    if len(last2) == 2 and not any(r["correct"] for r in last2):
        levels = {r.get("difficulty") for r in last2}
        if levels == {"Hard"}:
            return _result("Medium", f"Two Hard {name} questions missed in a row; reinforcing at Medium "
                                     f"before retrying Hard.", "R1_TWO_WRONG_HARD")
        if levels == {"Medium"}:
            return _result("Easy", f"Two Medium {name} questions missed in a row; rebuilding the "
                                   f"fundamentals at Easy.", "R2_TWO_WRONG_MEDIUM")

    # R3: 2+ misses in the last 3.
    if len(last3) >= 2 and sum(1 for r in last3 if not r["correct"]) >= 2:
        lower = step(current, -1)
        if lower == current:
            reason = f"{name}: 2 of the last 3 answers were wrong; staying at Easy until the basics click."
        else:
            reason = f"{name}: 2 of the last 3 answers were wrong; stepping down from {current} to {lower}."
        return _result(lower, reason, "R3_MOSTLY_WRONG")

    three_correct_same = (
        len(last3) == 3 and all(r["correct"] for r in last3)
        and len({r.get("difficulty") for r in last3}) == 1
    )

    # R4: 3 correct at the same level, and fast.
    if three_correct_same:
        times = [r.get("time_taken") for r in last3]
        if all(t is not None for t in times):
            avg = sum(times) / 3
            if avg < FAST_RATIO * EXPECTED_SECONDS[current]:
                higher = step(current, +1)
                if higher == current:
                    reason = (f"3 fast correct Hard answers on {name} (avg {round(avg)}s); "
                              f"already at the top level, keeping Hard.")
                else:
                    reason = (f"3 correct {current} answers on {name} in a row, averaging {round(avg)}s "
                              f"(well under {EXPECTED_SECONDS[current]}s); stepping up to {higher}.")
                return _result(higher, reason, "R4_FAST_STREAK")

    # R5: 3 correct at the same level, gated by mastery.
    if three_correct_same:
        if current == "Hard":
            return _result("Hard", f"3 correct Hard answers on {name}; staying at the top level.",
                           "R5_STREAK_MASTERY_GATE")
        needed = STEP_UP_MASTERY[current]
        higher = step(current, +1)
        if state.mastery >= needed:
            return _result(higher, f"3 correct {current} answers on {name} in a row and mastery is "
                                   f"{round(state.mastery)}%; stepping up to {higher}.", "R5_STREAK_MASTERY_GATE")
        return _result(current, f"3 correct {current} answers on {name}, but mastery is {round(state.mastery)}% "
                                f"(needs {needed}% for {higher}); staying at {current}.", "R5_STREAK_MASTERY_GATE")

    # R6: last answer correct but slow.
    last = recent[-1]
    t = last.get("time_taken")
    if last["correct"] and t is not None and t > SLOW_RATIO * EXPECTED_SECONDS[current]:
        return _result(current, f"Correct on {name} but slow ({round(t)}s vs ~{EXPECTED_SECONDS[current]}s "
                                f"expected); staying at {current} to build fluency.", "R6_CORRECT_BUT_SLOW")

    # R7: default.
    return _result(current, f"Mixed recent results on {name}; staying at {current}.", "R7_STAY")
