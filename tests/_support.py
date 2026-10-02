"""Shared test setup: one throwaway database and one started TestClient for all API tests.

Default: a temporary SQLite file. With LEARNAI_TEST_PG=1 the suite runs on the Postgres from
.env (Neon) inside a brand-new schema `test_<random>` that is dropped at exit; it never touches
the real tables. Neon's PgBouncer pooler rejects the `search_path` startup option, so tests use
the direct (non-pooler) host of the same database.
"""

import atexit
import logging
import os
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="learnai-test-")
_PG_SCHEMA: str | None = None
_PG_URL: str | None = None

if os.environ.get("LEARNAI_TEST_PG") == "1":
    from app.config import Settings, normalize_database_url

    _PG_URL = normalize_database_url(Settings().database_url).replace("-pooler.", ".")
    if not _PG_URL.startswith("postgresql"):
        raise RuntimeError("LEARNAI_TEST_PG=1 needs a Postgres DATABASE_URL in .env")
    _PG_SCHEMA = "test_" + uuid.uuid4().hex[:10]
    import psycopg

    with psycopg.connect(_PG_URL.replace("postgresql+psycopg://", "postgresql://"), autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{_PG_SCHEMA}"')
    os.environ["DATABASE_URL"] = _PG_URL
    os.environ["DB_SCHEMA"] = _PG_SCHEMA
else:
    os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
    os.environ["DB_SCHEMA"] = ""

# Tests never call the network: an unmocked LLM call fails fast with LLM_NOT_CONFIGURED.
os.environ["GROQ_API_KEY"] = ""
# Rate limiting is exercised explicitly in test_deploy.py; off for the rest of the suite.
os.environ["RATE_LIMIT_LLM_PER_MIN"] = "0"
os.environ["RATE_LIMIT_OTHER_PER_MIN"] = "0"
os.environ["ENABLE_DOCS"] = "true"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*httpx.*", category=DeprecationWarning)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import db as app_db  # noqa: E402

app_db.configure_engine()

if _PG_SCHEMA:  # hard safety check before any test can write
    with app_db.get_engine().connect() as _c:
        _current = _c.execute(text("SELECT current_schema()")).scalar()
    if _current != _PG_SCHEMA:
        raise RuntimeError(f"Refusing to run: test connection uses schema {_current!r}, expected {_PG_SCHEMA!r}")

from app.main import app  # noqa: E402

_client: TestClient | None = None
DB_KIND = "postgresql" if _PG_SCHEMA else "sqlite"


def client() -> TestClient:
    """Started TestClient (lifespan ran: tables created, demo learners seeded)."""
    global _client
    if _client is None:
        _client = TestClient(app)
        _client.__enter__()
    return _client


def call(action: str, payload: dict | None = None, learner_id: str = "akshat-intermediate") -> dict:
    resp = client().post("/api/v1/learnai",
                         json={"action": action, "learner_id": learner_id, "payload": payload or {}})
    return resp.json()


@atexit.register
def _cleanup() -> None:
    if _client is not None:
        _client.__exit__(None, None, None)
    app_db.get_engine().dispose()
    if _PG_SCHEMA and _PG_URL:
        import psycopg

        with psycopg.connect(_PG_URL.replace("postgresql+psycopg://", "postgresql://"), autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{_PG_SCHEMA}" CASCADE')
    shutil.rmtree(_TMP, ignore_errors=True)


def clear_generated_questions() -> None:
    """Remove generated/served questions so tests don't see each other's anti-repetition history.
    Keeps the startup seed question and test_api's stored question."""
    from sqlalchemy import delete

    from app.models import Question, QuestionServe
    from app.seed import SEED_QUESTION_ID

    with app_db.session_scope() as s:
        s.execute(delete(QuestionServe))
        s.execute(delete(Question).where(Question.id.not_in([SEED_QUESTION_ID, "stored-q1"])))
