"""Question-generation and verification message builders (templates live in app/prompts/)."""

import json
from collections.abc import Sequence

from app import prompts

DIFFICULTY_RULES = {
    "Easy": "Easy: one reasoning step; tests core intuition or a definition applied to the scenario.",
    "Medium": "Medium: apply the concept to the scenario; requires interpreting numbers, code or behaviour.",
    "Hard": "Hard: multi-step reasoning, an edge case, or a trade-off where the obvious answer is wrong.",
}

ANGLE_GUIDE = {
    "concept_check": "Tests understanding of what the concept means and when it applies.",
    "formula_interpretation": "Learner interprets what a formula or quantity tells them about the model.",
    "curve_or_metric_diagnosis": "Learner is given metrics or a described curve and diagnoses what is happening.",
    "code_debugging": "Learner finds the bug or wrong behaviour in a short code_snippet (max 12 lines).",
    "scenario_diagnosis": "Learner reads a realistic production situation and identifies the root cause or best action.",
    "hyperparameter_tuning": "Learner picks the right hyperparameter change and justifies its effect.",
}


def _system_prompt(level: str, languages: Sequence[str]) -> str:
    gen = prompts.blocks("question_generator")
    level_rules = gen.get(f"level:{level}", gen["level:Intermediate"])
    if languages:
        code_rule = f"Any code must be written in one of the learner's known languages: {', '.join(languages)}."
    else:
        code_rule = "The learner knows no programming language: set code_snippet to null."
    formula_rule = ("\n" + gen[f"formula:{level}"]) if f"formula:{level}" in gen else ""
    return prompts.render(gen["system"], {
        "level": level, "level_rules": level_rules, "formula_rule": formula_rule, "code_rule": code_rule,
    })


def build_question_messages(
    *,
    learner: dict,
    concept: dict,
    prerequisite_names: Sequence[str],
    mastery: float,
    classification: str | None,
    difficulty: str,
    slots: Sequence[dict],
    recent_titles: Sequence[str],
    feedback: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Messages for ONE call that generates len(slots) questions."""
    level = learner.get("level") or "Intermediate"
    languages = list(learner.get("known_languages") or [])
    context = {
        "learner": {
            "name": learner.get("name"),
            "level": level,
            "goal": learner.get("goal"),
            "known_topics": learner.get("known_topics") or [],
            "known_languages": languages,
        },
        "concept": {
            "name": concept["name"],
            "category": concept["category"],
            "prerequisites": list(prerequisite_names),
            "learner_mastery_percent": round(mastery),
            "learner_status": classification or "untried",
        },
        "difficulty": difficulty,
        "difficulty_rule": DIFFICULTY_RULES[difficulty],
        "assignments": [
            {"position": i + 1, "scenario_domain": s["scenario_domain"], "question_angle": s["question_angle"],
             "angle_meaning": ANGLE_GUIDE[s["question_angle"]]}
            for i, s in enumerate(slots)
        ],
        "do_not_repeat_titles": list(recent_titles),
    }
    gen = prompts.blocks("question_generator")
    user = prompts.render(gen["user"], {
        "count": len(slots), "difficulty": difficulty, "concept_name": concept["name"],
        "context_json": json.dumps(context, ensure_ascii=False),
    })
    messages = [{"role": "system", "content": _system_prompt(level, languages)}, {"role": "user", "content": user}]
    if feedback:
        messages.append({"role": "user", "content": prompts.render(gen["feedback"], {
            "problems": "\n".join(f"- {f}" for f in feedback),
        })})
    return messages


def build_verifier_messages(questions: Sequence[dict]) -> list[dict[str, str]]:
    """questions: [{qid, question, code_snippet, options}] -- never the answer or explanation."""
    ver = prompts.blocks("question_verifier")
    return [
        {"role": "system", "content": prompts.render(ver["system"], {})},
        {"role": "user", "content": prompts.render(ver["user"], {
            "questions_json": json.dumps(list(questions), ensure_ascii=False, indent=1),
        })},
    ]
