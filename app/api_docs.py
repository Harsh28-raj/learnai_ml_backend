"""Swagger content: the frontend integration guide (markdown) and request examples per action."""

from app.seed import SEED_QUESTION_ID

DESCRIPTION = """
The AI/ML backend for **LearnAI**. Everything below is enough to integrate the React app, and every example in
the dropdown on `POST /api/v1/learnai` works with **Try it out**.

## 1. One endpoint, one envelope

All features go through **`POST /api/v1/learnai`** with:

```json
{ "action": "get_profile", "learner_id": "akshat-intermediate", "payload": {} }
```

Every response has the same shape:

```json
{ "success": true, "action": "get_profile", "data": { }, "error": null, "fallback_data": null }
```

* **Check `success`, not the HTTP status.** Action errors (unknown learner, bad payload, AI unavailable, rate
  limited, …) come back as **HTTP 200 with `success: false`** and `error: {code, message, details}`.
* HTTP **422** means the envelope itself was malformed, **413** means the body is over 32 KB, and **500** means
  an unexpected server error. All three still use the envelope.
* `fallback_data`, when present on a failure, is safe to render. For example, `tutor_chat` returns a short concept
  summary in `fallback_data.reply` if the AI is down.
* **`generate_questions` can return `success: true` together with an `error` notice.** This happens when some
  questions came from the verified pool or the curated bank (for example during an AI rate limit). Render the
  questions normally; the notice is informational.
* **Key casing:** learner profiles, paths, tutor replies and questions use **camelCase** (matching the TypeScript
  interfaces). The envelope fields of `evaluate` and `generate_questions` are snake_case (`mastery_updates`,
  `weakness_report`, `next_difficulty`, `adaptive_decision`, `requested_count`, `selection_reason`), but the
  objects inside them (questions, `adaptive_decision`) are camelCase.

## 2. Actions

| action | what it does | required payload | calls the AI? | typical latency |
|---|---|---|---|---|
| `get_profile` | Learner profile (SPEC §5) + weakness report + last 5 adaptive events | — | no | 0.1–0.5 s |
| `reset_learner` | Restore a demo learner to its seeded state | — | no | 0.2–1 s |
| `evaluate` | Grade one answer or a whole quiz → mastery update, weakness report, next difficulty, AdaptiveEvent, real path change | single: `concept_tested`, `difficulty` + (`question_id`+`selected_option_index` or `correct_index` or `is_correct`); quiz: `quiz: true`, `answers[]` | no | 0.2–1 s |
| `generate_questions` | Adaptive MCQs for the learner (independently verified) | — (optional `category`, `target_concept`, `difficulty`, `count` 1–5) | yes | 3–12 s |
| `tutor_chat` | Context-aware AI tutor, 9 modes, guardrails | `message` (+ `question_id` for hint, `student_answer` for evaluate) | yes (not for off-topic) | 1–6 s |
| `get_path` | Personalized learning path, milestones, daily plan, before/after change, why-this-path | — (optional `goal` for a what-if preview) | first time only (cached) | 0.05–0.8 s |
| `assessment` | Onboarding: create/overwrite a learner from the 6-step questionnaire and build their first path | `experience_level` (+ `languages`, `topics_known`, `goal`, `pace`, `daily_minutes`, `overwrite`) | small (path summary) | 0.5–2 s |

Tutor modes (`payload.mode`, auto-detected from the message if omitted, including Hinglish):
`explanation`, `simplify`, `example`, `code`, `practice` (always returns a verified checkpoint question), `hint`,
`evaluate`, `revision`, `path`.

## 3. Mapping from the current mock functions

| frontend today | replace with |
|---|---|
| `getTutorReply()` (mockTutorResponses.ts) | `tutor_chat` → render `data.message` (markdown + KaTeX), `followUpSuggestions` as pills, `checkpointQuestion` as the comprehension check |
| `recordQuizScore()` (LearnerContext) | `evaluate` with `quiz: true` → use `adaptive_decision` for the alert banner and notification feed, `mastery_updates` for the skill radar |
| `applyAssessment()` | `assessment` → `data.profile`, `data.path`, `data.firstStep` |
| `mockQuestions.ts` | `generate_questions` → `data.questions` already match `PracticeQuestion` (+ `whyThisQuestion`) |
| LearningPath page (mockLearningPath.ts) | `get_path` → `nodes` (status: completed/current/recommended/adapted/locked), `milestones`, `whyThisPath`, `dailyPlan` |
| Dashboard "Your tutor adapted your path" | `get_path` → `beforeAfter` + `changes` (or `adaptive_decision.pathAdjustment` right after `evaluate`) |
| Profile switcher (`switchDemoProfile`) | `get_profile` with the demo learner id |
| `resetToDefault()` | `reset_learner` |

When grading a generated or checkpoint question, send `question_id` + `selected_option_index`; the server
grades it with its stored answer.

## 4. Demo learners

| id | persona |
|---|---|
| `alex-beginner` | Beginner, goal "Learn ML from Scratch", weak on Matrix Shapes (42) and Std Deviation (50) |
| `akshat-intermediate` | Intermediate, goal "Become an ML Engineer", weak on Bias vs Variance (54) |
| `elena-advanced` | Advanced, goal "Build GenAI Apps", weak on RAG chunking (68) |

Call `reset_learner` for each before a demo.

## 5. Error codes and what the UI should do

| code | when | UI action |
|---|---|---|
| `INVALID_REQUEST` (HTTP 422) | envelope malformed | developer bug: fix the request |
| `UNKNOWN_ACTION` | action name typo | developer bug |
| `INVALID_PAYLOAD` | payload field wrong/missing (`details` lists them) | developer bug; show a generic error |
| `LEARNER_NOT_FOUND` | unknown `learner_id` | fall back to a demo learner / onboarding |
| `UNKNOWN_CONCEPT` | concept text not recognized | show `details.suggestions` or use mock data |
| `CANNOT_GRADE` | answer without any way to grade | send `question_id`+`selected_option_index`, or `correct_index`, or `is_correct` |
| `QUESTION_NOT_FOUND` | hint for an unknown `question_id` | retry the tutor without `question_id` |
| `EMPTY_MESSAGE`, `MESSAGE_TOO_LONG` | tutor input empty / > 2000 chars | inline input validation |
| `LEARNER_EXISTS` | assessment for an existing id | ask "start over?" → resend with `overwrite: true` |
| `DEMO_LEARNER_PROTECTED` | assessment on a demo id | use a different `learner_id` |
| `NOT_A_DEMO_LEARNER` | reset on an assessed learner | rerun `assessment` with `overwrite: true` |
| `RATE_LIMITED` | > 20 AI calls/min or > 120 other calls/min per IP | wait `details.retry_after_seconds`, show a "slow down" toast |
| `PAYLOAD_TOO_LARGE` (HTTP 413) | body > 32 KB | trim the request |
| `LLM_NOT_CONFIGURED`, `MODEL_RATE_LIMIT`, `MODEL_TIMEOUT`, `MODEL_ERROR` | AI unavailable | tutor: render `fallback_data.reply`; generate_questions: questions still arrive (notice only) |
| `GENERATION_FAILED`, `VERIFICATION_SHORTFALL` | notice on `generate_questions` (success stays true) | render questions normally |
| `INTERNAL_ERROR` (HTTP 500) | unexpected | use mock data + "local tutor mode" toast |

## 6. Cold start, timeouts and fallback

* The backend runs on the Render free tier and **sleeps when idle**: the first request can take **30–60 s**.
  Call `GET /health` as soon as the app opens (fire and forget) to wake it up.
* Use a **45 s client timeout**.
* **Fallback rule:** on a network error, timeout or HTTP 5xx, use the existing mock data
  (`mockTutorResponses.ts`, `mockQuestions.ts`, `mockLearner.ts`) and show the toast
  *"AI service momentarily unreachable. Running in local tutor mode."* (SPEC §13.2).

## 7. Minimal client (TypeScript)

```ts
const API = import.meta.env.VITE_LEARNAI_API; // e.g. https://learnai-ml-backend.onrender.com

export async function learnai<T = any>(action: string, learnerId: string, payload: object = {}): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 45_000);
  try {
    const res = await fetch(`${API}/api/v1/learnai`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, learner_id: learnerId, payload }),
      signal: ctrl.signal,
    });
    if (res.status >= 500) throw new Error("server");          // -> mock fallback
    const body = await res.json();
    if (!body.success) throw Object.assign(new Error(body.error?.message), body); // has error + fallback_data
    return body.data as T;
  } finally {
    clearTimeout(timer);
  }
}

// usage
const { profile } = await learnai("get_profile", "akshat-intermediate");
const tutor = await learnai("tutor_chat", "akshat-intermediate", { message: "Why does my model overfit?" });
```
"""

TAGS = [
    {"name": "LearnAI", "description": "The single action endpoint used by the frontend."},
    {"name": "System", "description": "Health and wake-up."},
]

AKSHAT, ALEX, ELENA = "akshat-intermediate", "alex-beginner", "elena-advanced"


def _ex(summary: str, action: str, learner_id: str, payload: dict, description: str = "") -> dict:
    out = {"summary": summary, "value": {"action": action, "learner_id": learner_id, "payload": payload}}
    if description:
        out["description"] = description
    return out


EXAMPLES: dict[str, dict] = {
    "get_profile": _ex("get_profile: Akshat (intermediate)", "get_profile", AKSHAT, {}),
    "reset_learner": _ex("reset_learner: restore Akshat", "reset_learner", AKSHAT, {}),
    "evaluate_single": _ex(
        "evaluate: one answer to a stored question (server grades)", "evaluate", AKSHAT,
        {"question_id": SEED_QUESTION_ID, "selected_option_index": 0, "time_taken_seconds": 38},
        "Grades with the server's stored answer and returns mastery update, weakness report, next difficulty and "
        "an AdaptiveEvent whose pathAdjustment is the real path change."),
    "evaluate_quiz": _ex(
        "evaluate: whole quiz (replaces recordQuizScore)", "evaluate", AKSHAT,
        {"quiz": True, "topic": "Bias vs Variance", "category": "Machine Learning", "answers": [
            {"question_id": "ml-q1", "concept_tested": "Bias vs Variance", "difficulty": "Medium",
             "selected_option_index": 1, "correct_index": 1, "time_taken_seconds": 40},
            {"question_id": "ml-q2", "concept_tested": "Overfitting", "difficulty": "Easy",
             "selected_option_index": 0, "correct_index": 2, "time_taken_seconds": 25},
            {"question_id": "ml-q3", "concept_tested": "Regularization", "difficulty": "Medium",
             "is_correct": True, "time_taken_seconds": 55}]}),
    "generate_adaptive": _ex(
        "generate_questions: adaptive (engine picks concept + difficulty)", "generate_questions", AKSHAT, {"count": 3}),
    "generate_targeted": _ex(
        "generate_questions: target concept + difficulty", "generate_questions", ALEX,
        {"target_concept": "Standard Deviation", "difficulty": "Easy", "count": 2}),
    "tutor_explanation": _ex("tutor_chat: explanation (beginner)", "tutor_chat", ALEX,
                             {"message": "What is gradient descent?"}),
    "tutor_simplify": _ex("tutor_chat: simplify (Hinglish)", "tutor_chat", ALEX,
                          {"message": "bias variance samajh nahi aaya, aasan bhasha mein batao"}),
    "tutor_code": _ex("tutor_chat: code (advanced)", "tutor_chat", ELENA,
                      {"message": "Show me PyTorch code for scaled dot-product attention", "mode": "code"}),
    "tutor_hint": _ex("tutor_chat: hint on a stored question", "tutor_chat", AKSHAT,
                      {"message": "I'm stuck, can you give me a hint?", "question_id": SEED_QUESTION_ID},
                      "Never reveals the answer. Uses a question saved at startup."),
    "tutor_evaluate": _ex(
        "tutor_chat: evaluate a free-text answer", "tutor_chat", AKSHAT,
        {"message": "Can you check my answer?", "current_topic": "Bias vs Variance",
         "student_answer": "High bias means the model memorizes noise, and high variance means it is too simple. "
                           "Adding more training data reduces variance."},
        "Returns evaluation.score and records the attempt (masteryUpdate + adaptiveDecision)."),
    "tutor_path": _ex("tutor_chat: path (why am I learning this next?)", "tutor_chat", AKSHAT,
                      {"message": "Why am I learning this next?"}),
    "get_path": _ex("get_path: learning path + daily plan", "get_path", AKSHAT, {}),
    "get_path_preview": _ex("get_path: what-if preview for another goal (not saved)", "get_path", AKSHAT,
                            {"goal": "GenAI Apps"}),
    "assessment_new": _ex(
        "assessment: new beginner (onboarding)", "assessment", "new-learner-01",
        {"name": "Riya", "experience_level": "Beginner", "languages": [], "topics_known": [],
         "goal": "Learn ML from Scratch", "pace": "Relaxed", "daily_minutes": 30},
        "Creates the learner. Running it a second time returns LEARNER_EXISTS: use the overwrite example."),
    "assessment_overwrite": _ex(
        "assessment: redo onboarding (overwrite)", "assessment", "new-learner-01",
        {"name": "Riya", "experience_level": "Intermediate", "languages": ["Python"],
         "topics_known": ["NumPy", "Pandas", "Statistics"], "goal": "Become an ML Engineer", "pace": "Balanced",
         "daily_minutes": 45, "overwrite": True}),
}

ENDPOINT_DESCRIPTION = (
    "Run one LearnAI action. Pick an example from the **Examples** dropdown, then **Try it out** → **Execute**. "
    "The `data` schema depends on the action (see the response schema below)."
)
