# LearnAI — ML/AI Backend (Phase 1)

FastAPI service that holds the learner knowledge model for LearnAI. It is built to fit the Render free tier: 512 MB RAM, a single worker, and no heavy ML libraries.

Phase 1 added the learner model, the concept DAG, three seeded demo learners, a single action endpoint, and a Groq client stub.

Phase 2 adds the core closed loop that runs on every answer:

```
answer → learner model update → weakness detection → adaptive difficulty → AdaptiveEvent
```

Phase 3 adds the first real LLM feature: `generate_questions`, an adaptive multiple-choice generator that runs on Groq and falls back to curated questions when the model is unavailable.

Phase 3.5 makes questions more trustworthy. An independent model checks every answer, and a pool of verified questions keeps practice working when Groq is rate limited.

Phase 4 adds `tutor_chat`, a context-aware tutor with 9 modes and layered guardrails.

Phase 5 adds the recommendation engine with personalized learning paths (`get_path`), cold-start profiling (`assessment`), the demo scripts, and a richer `/health`.

Phase 6 makes the backend deployment-ready:

- `/docs` doubles as the frontend integration guide, with a runnable example for every action.
- Per-IP rate limits and a request body limit.
- Verified on Neon Postgres, with a Render blueprint.

To deploy, see **[docs/DEPLOY.md](docs/DEPLOY.md)**.

`docs/SPEC.md` Section 20 is the source of truth wherever it conflicts with earlier sections of the spec.

## Setup

Requires Python 3.12 (Render uses `PYTHON_VERSION=3.12.7`).

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS/Linux:           source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in GROQ_API_KEY (optional in Phase 1)
```

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./learnai.db` | `postgres://` / `postgresql://` URLs (Neon, Supabase) are rewritten to `postgresql+psycopg://` automatically |
| `GROQ_API_KEY` | empty | The app starts without it; LLM calls will raise `LLM_NOT_CONFIGURED` |
| `GROQ_MODEL_MAIN` / `GROQ_MODEL_FAST` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | `generate_questions` uses the main model |
| `GROQ_REASONING_EFFORT` | `low` | Sent only when set. If Groq rejects it, the request is retried without it |
| `VERIFY_QUESTIONS` | `true` | Every generated question is checked by `GROQ_MODEL_FAST` before it is served |
| `TUTOR_SUPPORT_TEXT` | Tele-MANAS line | Support line the tutor adds when a learner seems distressed |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://localhost:8080` | Comma-separated CORS origins, or `*` for any origin (no credentials) |
| `ENABLE_DOCS` | `true` | Serves `/docs` (Swagger UI, which is also the integration guide), `/redoc` and `/openapi.json` |
| `RATE_LIMIT_LLM_PER_MIN` / `RATE_LIMIT_OTHER_PER_MIN` | `20` / `120` | Per-IP limits per minute for AI actions / other actions. `0` disables |

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

On startup the app checks that the concept DAG is valid and creates the tables. If the database has no learners, it seeds the three demo learners.

## Run tests

To run the same suite on the Postgres in `.env`, inside a throwaway schema that is dropped afterwards: `LEARNAI_TEST_PG=1 python -m unittest discover -s tests`.

Tests use stdlib `unittest` and never call the network: the LLM is mocked, and the Groq client is tested through `httpx.MockTransport`. The integration tests use FastAPI's `TestClient` against a throwaway SQLite file, so your local `learnai.db` is never touched:

```bash
python -m unittest discover -s tests -v
```

## API

The service exposes only two routes (plus `/docs` and `/openapi.json` when `ENABLE_DOCS=true`):

- `GET /health` returns `{"status": "ok", "db": "ok", "llm_configured": true, "pool_size": 42}`. It never calls the LLM.
- `POST /api/v1/learnai` handles every action

### Request / response contract

```jsonc
// Request
{ "action": "get_profile", "learner_id": "akshat-intermediate", "payload": {} }

// Response (always this shape, including on errors)
{
  "success": true,
  "action": "get_profile",
  "data": { ... } | null,
  "error": { "code": "...", "message": "...", "details": ... } | null,
  "fallback_data": { ... } | null
}
```

| Action | Status |
|---|---|
| `get_profile` | Working. Returns `data.profile` (the `LearnerProfile` from SPEC §5 in camelCase, plus `conceptStates` and `recentAdaptiveEvents`) and `data.weakness_report` |
| `reset_learner` | Working. Clears the demo learner's attempts and adaptive events, re-seeds its concept states, and returns the same data as `get_profile` |
| `evaluate` | Working. Grades one answer or a whole quiz and runs the closed loop. See below |
| `generate_questions` | Working. Adaptive MCQs from Groq, with a curated fallback. See below |
| `tutor_chat` | Working. The AI tutor; see below |
| `get_path` | Working. Personalized learning path; see below |
| `assessment` | Working. Cold-start profile from the onboarding questionnaire; see below |

`get_profile`, `reset_learner` and `evaluate` accept an optional `payload.utc_offset_minutes` (for example `330` for IST). It is used to format human-readable timestamps such as "Today at 2:15 PM". Every event also carries an ISO `createdAt`, so the frontend can format times itself if it prefers.

### `evaluate`

There are two payload shapes.

**Single answer:**

```json
{
  "question_id": "ml-q1",
  "concept_tested": "Bias vs Variance",
  "category": "Machine Learning",
  "difficulty": "Medium",
  "selected_option_index": 0,
  "time_taken_seconds": 38,
  "correct_index": 1
}
```

`correct_index` and `is_correct` are optional; see the grading order below.

**Whole quiz** (matches the frontend's `recordQuizScore`):

```json
{ "quiz": true, "topic": "Bias vs Variance", "category": "Machine Learning", "answers": [ /* single-answer objects */ ] }
```

Inside a quiz, an answer without `concept_tested` falls back to the quiz `topic`. A quiz takes at most 50 answers.

`concept_tested` can be a concept id, a name, an alias or free text. For example, "Bias vs Variance", "bias-variance tradeoff" and "bias_variance" all resolve to the same concept. If nothing matches, the response is `UNKNOWN_CONCEPT`, with the closest suggestions in `error.details.suggestions`.

**Grading order for each answer:**

1. If `question_id` is in the `questions` table, the server grades it with the stored `correct_index`, concept and difficulty, and ignores the payload's values.
2. Otherwise, if the payload has `correct_index` and `selected_option_index`, they are compared.
3. Otherwise, the payload's `is_correct` is used.
4. If none of these is available, the response is `CANNOT_GRADE`.

Client-side grading is allowed because the frontend's mock questions are not in the database.

**What happens on each call.** All of this runs in one database transaction, so a failing answer in a quiz writes nothing:

- Each answer is inserted into `attempts` and updates that concept's state through the learner model.
- `questionsSolved` goes up by the number of answers.
- `accuracyRate` uses the SPEC §8.3 cumulative formula. It is stored to one decimal place so rounding errors don't accumulate.
- Category scores are recomputed from concept states. The old exponential-moving-average formula from SPEC §8.3 is no longer used.
- An `AdaptiveEvent` is saved.

**Response `data`:**

```jsonc
{
  "results": [{ "question_id", "is_correct", "correct_index", "concept" }],
  "score": 0,                          // % correct in this request
  "mastery_updates": [{ "concept", "concept_name", "previous_score", "new_score", "category",
                        "previous_category_score", "new_category_score", "trend", "confidence" }],
  "weakness_report": { "strong": [], "average": [], "weak": [], "untried": [] },
  "next_difficulty": { "concept", "difficulty", "reason", "rule_id" },
  "adaptive_decision": { "id", "timestamp", "createdAt", "topic", "score", "action", "reason",
                         "recommendation", "pathAdjustment", "concept", "overridden", "thresholdAction" }
}
```

`adaptive_decision` follows the SPEC §8.2 `AdaptiveEvent` shape, with a few extra fields.

In quiz mode, `next_difficulty` and `adaptive_decision` are based on the quiz `topic` if it was answered. Otherwise they use the concept answered most often.

| Error code | When |
|---|---|
| `UNKNOWN_ACTION` | `action` is not in the registry |
| `LEARNER_NOT_FOUND` | No learner has this `learner_id` (for `reset_learner`, the id is not a demo learner) |
| `NOT_IMPLEMENTED` | The action is registered but not built yet |
| `UNKNOWN_CONCEPT` | `concept_tested` or the quiz topic does not match any concept. `details.suggestions` lists the 3 closest |
| `CANNOT_GRADE` | There is not enough information to grade an answer (see the grading order above) |
| `LLM_NOT_CONFIGURED`, `MODEL_RATE_LIMIT`, `MODEL_TIMEOUT`, `MODEL_ERROR`, `GENERATION_FAILED` | Question generation failed. `fallback_data` carries curated questions |
| `EMPTY_MESSAGE`, `MESSAGE_TOO_LONG`, `QUESTION_NOT_FOUND` | `tutor_chat` input problems |
| `VERIFICATION_SHORTFALL` | Notice: too few generated questions passed verification, so the pool or bank filled the gap |
| `LEARNER_EXISTS`, `DEMO_LEARNER_PROTECTED`, `NOT_A_DEMO_LEARNER` | `assessment` and `reset_learner` ownership rules |
| `RATE_LIMITED` | Too many requests from this IP. Wait `details.retry_after_seconds` |
| `PAYLOAD_TOO_LARGE` (HTTP 413) | Request body over 32 KB |
| `INVALID_PAYLOAD` | `payload` has the wrong shape, for example a missing or invalid `difficulty` or an empty quiz. `details` lists the problems; in quiz mode it includes `answer_index` |
| `INVALID_REQUEST` | The body does not match the contract (HTTP 422) |
| `INTERNAL_ERROR` | An unexpected exception. The server never returns a raw 500 body |

Business errors (everything except `INVALID_REQUEST` and `INTERNAL_ERROR`) return HTTP 200 with `success: false`.

### `generate_questions`

```json
{ "category": "Machine Learning", "target_concept": "Bias vs Variance", "difficulty": "Adaptive", "count": 3 }
```

Every field is optional:

- **`category`:** one of the 6 skill categories. The frontend's "Computer Vision" and "LLMs" are mapped onto them.
- **`difficulty`:** `Adaptive` (the default), `Easy`, `Medium` or `Hard`.
- **`count`:** 1–5, default 3.

Python picks the concept and difficulty, assigns a scenario domain and question angle to each question, validates and shuffles the options, and writes `whyThisQuestion`. The LLM only writes the question content.

Generated questions are saved server-side. When you grade them with `evaluate`, send `question_id` and `selected_option_index`; the server uses its stored answer.

**Response `data`:**

```jsonc
{
  "count": 3, "requested_count": 3,
  "concept": "bias_variance", "concept_name": "Bias vs Variance Tradeoff", "selection_reason": "weak",
  "difficulty": "Medium", "difficulty_reason": "...",
  "questions": [{
    "id": "gen-1d0e6051e6", "category": "Machine Learning", "difficulty": "Medium",
    "conceptId": "bias_variance", "conceptTested": "Bias vs Variance Tradeoff",
    "title": "...", "question": "...", "codeSnippet": null, "options": ["...", "...", "...", "..."],
    "correctIndex": 1, "explanation": "...", "hint": "...",
    "whyThisQuestion": "Generated because you scored 54% on Bias vs Variance Tradeoff.",
    "recommendedNextDifficulty": "Medium", "scenarioDomain": "e-commerce", "questionAngle": "curve_or_metric_diagnosis"
  }],
  "generation": { "model": "...", "llm_calls": 1, "rejected": 0, "latency_ms": 2400, "validation_errors": [] }
}
```

`count` can be lower than `requested_count` when some questions still fail validation after the one repair retry.

**Correctness and resilience (Phase 3.5).**

- **Verification:** a second model solves every generated question without seeing the answer. A question is rejected if that model picks a different answer, finds two defensible answers, flags a flaw, or has low confidence. Rejected questions are regenerated once. Unverified questions are never served.
- **When Groq can't deliver:** if Groq is rate limited, times out, or too few questions pass verification, the gap is filled from the **verified pool**, then from the curated bank.
  - The pool holds earlier verified questions at the same level that this learner hasn't seen.
  - Each question carries a `source` field: `llm`, `pool` or `fallback`.
  - The response stays `success: true` as long as the requested count was met. `error` then carries a notice explaining where the questions came from.

**When generation fails.** If the LLM is unavailable (`LLM_NOT_CONFIGURED`, `MODEL_RATE_LIMIT`, `MODEL_TIMEOUT` or `MODEL_ERROR`), or no question passes validation (`GENERATION_FAILED`), the response has `success: false`. `fallback_data.questions` then holds curated questions in the same shape, with `"fallback": true`. They are saved too, so `evaluate` can grade them.

**Live smoke test.** This is manual and calls the real Groq API through a running server:

```bash
uvicorn app.main:app --port 8000
python scripts/smoke_generate.py --base http://127.0.0.1:8000 --count 2 --gap 30   # gap: Groq free tier tokens/minute
```

### `tutor_chat`

```json
{ "message": "Why does my model overfit?", "conversation_id": "conv-…", "mode": "explanation",
  "current_topic": "Overfitting", "question_id": "gen-…", "student_answer": "…", "record": true }
```

- **`message`** is required: 1–2000 characters.
- **`conversation_id`** is optional. When it is missing, the server creates one and returns it as `conversationId`. Send it back to keep the conversation going.
- **`mode`** is optional. When it is missing, the mode is detected from the message.
  - The 9 modes are explanation, simplify, example, code, practice, hint, evaluate, revision and path.
  - Detection understands some Hinglish, such as "samajh nahi aaya" or "aage kya".
- **Hint mode** needs `question_id`.
- **Evaluate mode** needs `student_answer`. By default it records the result as a practice attempt; send `record: false` to skip that.

**Response `data`** (camelCase):

```jsonc
{
  "conversationId": "conv-…", "message": "markdown + KaTeX", "mode": "explanation", "modeSource": "auto",
  "concept": "overfitting", "conceptName": "Overfitting & Underfitting", "difficultyLevel": "Intermediate",
  "followUpSuggestions": ["…", "…", "…"],
  "checkpointQuestion": { "id": "chk-…", "question": "…", "options": ["…"], "correctIndex": 1, "explanation": "…", "hint": "…", "verified": true } | null,
  "recommendedNextAction": { "type": "practice|revise|advance|continue", "targetTopic": "…", "reason": "…" },
  "evaluation": { "score": 45, "correctPoints": [], "misconceptions": [], "correctedAnswer": "…", "recorded": true } | null,
  "masteryUpdate": { … } | null, "adaptiveDecision": { … } | null,
  "promptVersion": "tutor-v1",
  "guardrails": { "injectionAttempt": false, "distress": false, "offTopic": false, "violations": [], "fixes": [],
                  "repaired": false, "fallbackUsed": false, "checkpoint": "verified" },
  "generation": { "model": "…", "llmCalls": 1, "verifyCalls": 1, "latencyMs": 2900 }
}
```

- **Checkpoint questions** are independently verified and saved on the server. Grade them with `evaluate` using `question_id` and `selected_option_index`.
- **When the LLM is unavailable:** `success: false`, and `fallback_data` holds `{conversation_id, reply, follow_ups}`. Show that reply instead of an error. The learner's message is still saved.

**Guardrails.**

- **Scope:** the tutor stays on AI/ML topics. Off-topic messages get a template reply without calling the LLM.
- **Safety:** it refuses to reveal its prompt, and it detects distress and adds a support line.
- **Hints** never reveal the answer.
- **Links** are kept only for python, scikit-learn, pytorch, numpy and pandas documentation.
- **Length** is capped per level and per mode.
- **No duplicates:** checkpoint and follow-up text duplicated inside `message` is removed.
- **Hinglish** input gets a Hinglish reply.
- **Beginners** only see math that is explained in plain English.

Leaks and invalid output trigger one repair call, then a safe fallback reply.

**Prompts** live in `app/prompts/*.txt` and are versioned (`tutor-v1`, `qgen-v2`).

**Live tutor evaluation** is manual and calls the real Groq API: 13 scripted cases, scored by a temperature-0 judge model.

```bash
python scripts/eval_tutor.py --base http://127.0.0.1:8000 --gap 3
```

### `get_path`

The payload is `{}`. Send `{"goal": "GenAI Apps"}` to preview a different goal; a preview is computed but not saved.

The path covers the goal's target concepts and all their prerequisites, ordered by topological sort with these priorities:

1. weak concepts you've practiced, inserted as revision nodes
2. the foundations under those weak concepts
3. closeness to the goal
4. category

Concepts you already know (mastery 75 or above) are marked `completed`, with `skipped: true` if you never practiced them.

```jsonc
{
  "goal": "ML Engineer", "goalLabel": "become an ML Engineer", "targets": ["…"],
  "nodes": [{ "id": "node-bias-variance", "conceptId": "bias_variance", "title": "Bias vs Variance Tradeoff",
              "category": "Machine Learning", "status": "completed|current|recommended|adapted|locked",
              "progress": 54, "mastery": 54, "estimatedMinutes": 23, "reason": "Revision added: …",
              "isRevision": true, "skipped": false, "prerequisites": ["…"], "foundationFor": null, "order": 10 }],
  "milestones": [{ "id": "stage-machine-learning", "title": "Machine Learning", "status": "current", "nodeIds": [], "progress": 70 }],
  "currentNode": { … }, "nextNodes": [ … ],
  "changes": [{ "type": "inserted_revision|skipped|completed|unlocked|reordered", "conceptId": "…", "title": "…", "detail": "…" }],
  "changedNow": false, "beforeAfter": "Before: next up was X. After: Y revision added; current focus is Z.",
  "whyThisPath": "…", "whySource": "llm|template",
  "estimatedWeeksRemaining": 2, "totalEstimatedMinutes": 249,
  "dailyPlan": [{ "id": "plan-practice-…", "title": "…", "type": "revision|practice|lesson", "durationMinutes": 15, "completed": false, "conceptId": "…" }]
}
```

- **Dashboard:** `beforeAfter` is the line for the dashboard's "Your tutor adapted your path" banner.
- **After `evaluate`:** every evaluate call recomputes the path. The AdaptiveEvent's `pathAdjustment` then describes the real change, or says "Roadmap unchanged: …".
- **Caching:** `whyThisPath` is cached until the path changes, so repeated `get_path` calls are fast and make no LLM call.

### `assessment`

```json
{ "name": "Riya", "experience_level": "Beginner|Some Programming|Intermediate|Advanced",
  "languages": ["Python"], "topics_known": ["NumPy", "Statistics"], "goal": "ML Engineer",
  "pace": "Relaxed|Balanced|Intensive", "daily_minutes": 30, "overwrite": false }
```

- **New learners:** a new `learner_id` creates a learner.
- **Existing learners:** need `overwrite: true` to redo the assessment.
- **Protected ids:** demo and pool learners can't be assessed.
- **Baselines:** the learner's level and every concept's starting mastery are computed from the answers in Python, with no LLM call.
- **Response:** `{profile, weakness_report, path, level, goal, firstStep}`. `firstStep` suggests a 3-concept diagnostic.
- **Resetting an assessed learner:** use `assessment` with `overwrite: true`; `reset_learner` is only for the 3 demo learners.

**Demo helpers** (manual, live):

```bash
python scripts/demo_flow.py --base http://127.0.0.1:8000 --gap 20        # SPEC §11 loop for akshat, then reset
python scripts/warm_pool.py --only-demo --dry-run                         # how big a pool warm-up would be
python scripts/warm_pool.py --only-demo --per-cell 2 --gap 25             # pre-generate verified questions (resumable)
```

### curl examples

```bash
# Health
curl http://localhost:8000/health

# get_profile for each demo learner
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"get_profile","learner_id":"alex-beginner"}'
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"get_profile","learner_id":"akshat-intermediate"}'
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"get_profile","learner_id":"elena-advanced"}'

# reset_learner for each demo learner
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"reset_learner","learner_id":"alex-beginner"}'
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"reset_learner","learner_id":"akshat-intermediate"}'
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"reset_learner","learner_id":"elena-advanced"}'
```

```bash
# evaluate: a wrong Medium answer on Bias vs Variance
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"evaluate","learner_id":"akshat-intermediate","payload":{"question_id":"ml-q1","concept_tested":"Bias vs Variance","category":"Machine Learning","difficulty":"Medium","selected_option_index":0,"correct_index":1,"time_taken_seconds":38}}'

# evaluate: quiz mode
curl -s -X POST http://localhost:8000/api/v1/learnai -H "Content-Type: application/json" \
  -d '{"action":"evaluate","learner_id":"akshat-intermediate","payload":{"quiz":true,"topic":"Bias vs Variance","category":"Machine Learning","answers":[{"concept_tested":"Bias vs Variance","difficulty":"Easy","selected_option_index":1,"correct_index":1,"time_taken_seconds":15},{"concept_tested":"Overfitting","difficulty":"Medium","is_correct":false,"time_taken_seconds":50}]}}'
```

On Windows PowerShell, use `curl.exe` and escape the inner quotes. Alternatively, use `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/learnai -ContentType 'application/json' `
  -Body '{"action":"get_profile","learner_id":"akshat-intermediate"}'
```

## Note for the frontend team: single-endpoint design

Every ML feature goes through `POST /api/v1/learnai`, with a different `action` for each feature. This replaces the separate endpoints sketched in SPEC §12. The frontend needs one small client function:

```ts
async function learnai(action: string, learnerId: string, payload = {}) {
  const res = await fetch(`${API}/api/v1/learnai`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, learner_id: learnerId, payload }),
  });
  const body = await res.json();
  if (!body.success) throw Object.assign(new Error(body.error?.message), body);
  return body.data;
}
```

- Check the `success` field, not the HTTP status.
- When `success` is false, `fallback_data` may hold something worth rendering. If it is null, use the local mocks as SPEC §13.2 describes.
- New actions in later phases do not change the URL or the response shape.
- Demo learner ids map to the frontend personas: `beginner` → `alex-beginner`, `intermediate` → `akshat-intermediate`, `advanced` → `elena-advanced`.

## Learner model, in brief

The model lives in `app/engine/learner_model.py` and is made of pure functions, with no database access.

- **Mastery** is tracked per concept on a 0–100 scale. A correct answer adds Easy +3, Medium +5 or Hard +8, multiplied by `(1 − m/100)·1.5`. The minimum gain is +1, and an answer under 20 s earns +1 more. A wrong answer subtracts Easy −8, Medium −6 or Hard −3, multiplied by `(m/100)·1.5`. The minimum loss is −1.
- **Confidence** is `min(1, attempts/10) · (1 − 0.15·min(consecutive_wrong, 3))`.
- **Trend** compares accuracy on the last 3 results with the 3 before them. A difference above ±0.15 counts as improving or declining. With fewer than 4 results, the trend is stable.
- **Skills** are the attempt-weighted average mastery for each category. Concepts with no attempts count with weight 1.
- **Strengths** are practiced concepts at 75 or above.
- **Weaknesses** are practiced concepts below 65. A practiced concept below 75 also counts when it sits at least 15 points under the learner's own average. This relative rule keeps advanced learners from having no weaknesses: Elena's RAG chunking score of 68% (SPEC §6.3) is caught by it. Concepts with no attempts are never weaknesses. This is the `profile.weaknesses` list that the frontend's `LearnerProfile` expects.

### Weakness detector (`app/engine/weakness.py`)

This module produces `weakness_report`, which is separate from `profile.weaknesses`.

- **Classes:** Strong is 75 or above, Average is 60–74, and Weak is below 60. The relative rule above also applies here, and untried concepts are listed under `untried`.
  - Because of the different thresholds, Akshat's Gradient Descent (61) and ROC-AUC (63) count as Average in this report, while they still appear in `profile.weaknesses`, which uses the SPEC's 65 cut-off.
- **Priority:** a score from 0 to 100, computed as `0.45·(100−mastery) + 0.20·wrong-streak + 0.15·(100−recent accuracy) + 0.10·trend + 0.10·evidence`. HIGH is 55 or above, MEDIUM is 40–54 and LOW is below 40.
  - Only HIGH entries get a `tutor_alert`.
- **`recommended_revision`:** found by walking the concept's prerequisites one layer at a time, starting with its direct prerequisites. Within the closest layer that has a gap (mastery below 60), it picks the weakest concept. If no prerequisite has a gap, it recommends the concept itself.

### Adaptive difficulty (`app/engine/difficulty.py`)

The rules run in order and the first match wins. They look only at the last 5 results.

| Rule | Condition | Next difficulty |
|---|---|---|
| R0 | No attempts yet | Easy if mastery is below 50, Medium if below 75, otherwise Hard |
| R1 | Last two answers wrong at Hard | Medium |
| R2 | Last two answers wrong at Medium | Easy |
| R3 | 2 or more of the last 3 wrong | One level down |
| R4 | Last 3 correct at the same level, with average time under 60% of expected (Easy 30 s, Medium 60 s, Hard 120 s) | One level up |
| R5 | Last 3 correct at the same level | Up only if mastery is at least 60 (Easy → Medium) or 75 (Medium → Hard); otherwise stay |
| R6 | Last answer correct but slower than 1.5× expected | Stay |
| R7 | Anything else | Stay |

### Adaptive event (`app/engine/adaptation.py`)

- **Action from the score** (SPEC thresholds): below 60 is `reduced`, 60–84 is `maintained`, and 85 or above is `increased`.
- **Override:** if the difficulty engine disagrees, its decision wins and the `reason` explains why. For example, a 100% score with slow answers becomes `maintained`.
- **No override at the edges:** a learner who scores 0% while already at Easy stays `reduced`, because the engine cannot step lower. The same applies to `increased` at Hard.
- **`pathAdjustment`** describes the intended roadmap change, such as an injected prerequisite revision or an early unlock. The path engine that applies these changes comes in Phase 5.

## Deploy (Render)

`render.yaml` defines a free Python web service with one uvicorn worker and a health check on `/health`. Set `DATABASE_URL` (Neon or Supabase Postgres), `GROQ_API_KEY` and `ALLOWED_ORIGINS` in the Render dashboard. The free-tier disk is ephemeral, so use Postgres in production; SQLite data is lost on every redeploy.
