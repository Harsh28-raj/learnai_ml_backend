"""Database engine and session factory (SQLite locally, Postgres in production).

The engine is created lazily on first use so tests can point DATABASE_URL at a
temporary database before anything connects.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import logging
import time

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


logger = logging.getLogger("learnai.db")


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_session_factory: sessionmaker | None = None


CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_SECONDS = 1.5


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    connect_args: dict = {
        "connect_timeout": 20,  # Neon computes can take a few seconds to wake up
        "prepare_threshold": None,  # no server-side prepared statements: safe behind PgBouncer poolers
    }
    schema = get_settings().db_schema.strip()
    if schema:
        connect_args["options"] = f"-csearch_path={schema}"
    # Small pool: Render free tier has 512 MB RAM and we run a single worker.
    engine = create_engine(url, pool_size=3, max_overflow=2, pool_pre_ping=True, pool_recycle=300,
                           connect_args=connect_args)

    @event.listens_for(engine, "do_connect")
    def _connect_with_retry(dialect, conn_rec, cargs, cparams):
        # A suspended Neon compute may refuse the very first connection while it wakes up;
        # retry so the first request after idle does not fail.
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            try:
                return dialect.connect(*cargs, **cparams)
            except Exception:
                if attempt == CONNECT_ATTEMPTS:
                    raise
                logger.warning("DB connect attempt %d failed; retrying", attempt)
                time.sleep(CONNECT_BACKOFF_SECONDS * attempt)

    return engine


def configure_engine(url: str | None = None) -> Engine:
    """(Re)build the engine. Defaults to the configured DATABASE_URL."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = _make_engine(url or get_settings().sqlalchemy_url)
    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    return _engine if _engine is not None else configure_engine()


@contextmanager
def session_scope() -> Iterator[Session]:
    """One transaction: commits on success, rolls back on any exception."""
    get_engine()
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Columns added after a table first shipped. create_all() never alters existing tables, so
# these are added in place (works on SQLite and Postgres). Additive only, never destructive.
_ADDED_COLUMNS: list[tuple[str, str, str]] = [
    ("questions", "verified", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("questions", "level", "VARCHAR(20)"),
    ("questions", "source", "VARCHAR(10) NOT NULL DEFAULT 'llm'"),
    ("questions", "times_served", "INTEGER NOT NULL DEFAULT 0"),
]


def _add_missing_columns(engine: Engine) -> None:
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl in _ADDED_COLUMNS:
            if table in tables and column not in {c["name"] for c in insp.get_columns(table)}:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def create_tables() -> None:
    from app import models  # noqa: F401  (register tables on Base.metadata)

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)
