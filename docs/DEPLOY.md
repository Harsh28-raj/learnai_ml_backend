# Deploying the LearnAI ML Backend (Render + Neon)

This guide covers putting the backend on Render's free tier with Neon Postgres, warming it up before a demo, and handing it to the frontend.

## 1. Push the code to GitHub

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

`.env` and `*.db` files are gitignored. Never commit them.

## 2. Create the service on Render (Blueprint)

1. Go to <https://dashboard.render.com> → **New** → **Blueprint**, and connect the GitHub repo.
2. Render reads `render.yaml` and proposes one web service, **learnai-ml-backend**: Python 3.12.7, free plan, region **Singapore** (the same region as the Neon database, ap-southeast-1), one worker, health check on `/health`.
3. Fill in the secret variables (marked `sync: false`), then click **Apply**.

| Variable | Value |
|---|---|
| `DATABASE_URL` | The Neon connection string, as given (the `-pooler` host is fine). `postgresql://` is rewritten to the psycopg driver automatically. |
| `GROQ_API_KEY` | Your Groq API key (from console.groq.com). |
| `ALLOWED_ORIGINS` | `*` for the hackathon, so any frontend origin works. CORS is sent without credentials. Later, set it to the real frontend URL(s), comma-separated. |
| `GROQ_MODEL_MAIN` | `openai/gpt-oss-120b` (preset) |
| `GROQ_MODEL_FAST` | `openai/gpt-oss-20b` (preset) |
| `GROQ_REASONING_EFFORT` | `low` (preset) |
| `VERIFY_QUESTIONS` | `true` (preset): a second model checks every generated question's answer. |
| `ENABLE_DOCS` | `true` (preset): Swagger at `/docs`, ReDoc at `/redoc`. |
| `TUTOR_SUPPORT_TEXT` | Tele-MANAS support line (preset) |
| `RATE_LIMIT_LLM_PER_MIN` | `20` (preset): AI actions per IP per minute. |
| `RATE_LIMIT_OTHER_PER_MIN` | `120` (preset): other actions per IP per minute. |

The first build takes about 2–3 minutes.

**What startup does:**

- Creates any missing tables or columns.
- Seeds the 3 demo learners only if the database is empty.
- Logs one line like this:

```
LearnAI ML Backend v1.0.0 | db=postgresql | llm_configured=True | pool_size=… | prompts=tutor-v1,qgen-v2 | docs=True
```

Restarts are safe: nothing is wiped.

## 3. Verify

- `https://<service>.onrender.com/health` should return `{"status":"ok","db":"ok","llm_configured":true,"pool_size":N}`.
- Open `https://<service>.onrender.com/docs`. Expand **POST /api/v1/learnai**, pick an example from the dropdown, then click **Try it out** and **Execute**. Every example works on a fresh database.

## 4. Keep it awake (UptimeRobot)

Render's free tier sleeps after about 15 minutes idle, and a cold start takes 30–60 s. Set up a monitor:

1. Go to <https://uptimerobot.com> → **Add New Monitor**.
2. Choose type **HTTP(s)** and set the URL to `https://<service>.onrender.com/health`.
3. Set the interval to **10 minutes**.

`/health` costs one DB count and never calls the AI.

Neon also suspends its compute when idle. The first query after that takes a few seconds while it wakes. The app retries the connection, so the request still succeeds, just slower.

## 5. Warm the question pool before a demo (about 48 min of Groq calls)

When Groq rate limits generation, `generate_questions` serves verified questions from the pool instead. Fill the pool for every concept on the three demo learners' paths beforehand, running locally against the same Neon database:

```bash
# .env contains the Neon DATABASE_URL and GROQ_API_KEY
uvicorn app.main:app --port 8000                      # terminal 1
python scripts/warm_pool.py --only-demo --dry-run     # terminal 2: shows cells and calls (~114 cells, 228-456 LLM calls)
python scripts/warm_pool.py --only-demo --per-cell 2 --gap 25 --base http://127.0.0.1:8000
```

- **Resumable:** run it again after interruptions or rate limits; full cells are skipped.
- **Check progress:** `pool_size` in `/health` goes up as questions are added.

## 6. Pre-demo checklist

1. `GET /health` returns `db: ok`, `llm_configured: true`, and a `pool_size` of 150 or more (after warm-up).
2. Reset all three demo learners. In `/docs`, run the `reset_learner` example with each `learner_id`, or use curl:
   ```bash
   for id in alex-beginner akshat-intermediate elena-advanced; do
     curl -s -X POST https://<service>.onrender.com/api/v1/learnai -H "Content-Type: application/json" \
       -d "{\"action\":\"reset_learner\",\"learner_id\":\"$id\",\"payload\":{}}"; echo; done
   ```
3. Make one tutor call to warm everything up: the `tutor_chat: explanation (beginner)` example in `/docs`.
4. Optional: run `get_path` for each demo learner once, so `whyThisPath` is cached.
5. Optional: run the full SPEC §11 story locally with `python scripts/demo_flow.py --base https://<service>.onrender.com --gap 20`. It resets Akshat at the end.

## 7. Message for the frontend teammate

> Backend is live: **https://<service>.onrender.com/docs**. The page at the top is the full integration guide: every feature is `POST /api/v1/learnai` with `{action, learner_id, payload}`.
> 1. Call `GET /health` when the app opens (the server may take 30–60 s to wake up), and use a 45 s timeout on requests.
> 2. Try any example from the dropdown on `POST /api/v1/learnai` (Try it out → Execute). Demo learners are `alex-beginner`, `akshat-intermediate` and `elena-advanced`.
> 3. Always check `success` in the response (errors are HTTP 200 + `success:false`). On network errors or 5xx, fall back to the existing mocks and show the "local tutor mode" toast.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` shows `db: error` | Check `DATABASE_URL` in the Render dashboard. Neon may be waking up; retry after 10 s. |
| `llm_configured: false` | `GROQ_API_KEY` is missing in Render. |
| Many `MODEL_RATE_LIMIT` notices | This is Groq free-tier per-minute limits. Warm the pool (section 5) and space out demo calls. |
| CORS errors in the browser | Set `ALLOWED_ORIGINS` to `*` or to the exact frontend origin, then redeploy. |
| `RATE_LIMITED` responses | More than 20 AI calls a minute from one IP. Wait `retry_after_seconds`, or raise `RATE_LIMIT_LLM_PER_MIN`. |
