"""FastAPI entrypoint. Two routes: GET /health and POST /api/v1/learnai (+ /docs and /redoc)."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api_docs import DESCRIPTION, TAGS
from app.api_models import HealthOut
from app.config import APP_VERSION, get_settings
from app.db import create_tables, get_engine, session_scope
from app.llm.groq_client import close_client, init_client
from app.models import Question
from app.prompts import QUESTION_PROMPT_VERSION, TUTOR_PROMPT_VERSION
from app.prompts import load_all as load_prompts
from app.protection import MAX_BODY_BYTES, BodySizeLimitMiddleware
from app.router import error_response, router
from app.seed import ensure_pool_learners, ensure_seed_questions, seed_if_empty

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("learnai")


def _pool_size() -> int:
    """One indexed count; doubles as the DB liveness check for /health."""
    with session_scope() as db:
        return int(db.scalar(select(func.count()).select_from(Question)
                             .where(Question.verified.is_(True), Question.source == "llm")) or 0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.engine.concepts import validate_dag

    settings = get_settings()
    validate_dag()
    load_prompts()
    # Idempotent on every (Render) restart: create missing tables/columns, seed only if empty.
    create_tables()
    with session_scope() as db:
        if seed_if_empty(db):
            logger.info("Seeded demo learners")
        ensure_pool_learners(db)
        ensure_seed_questions(db)
    logger.info("LearnAI ML Backend v%s | db=%s | llm_configured=%s | pool_size=%d | prompts=%s,%s | docs=%s",
                APP_VERSION, settings.db_type, settings.llm_enabled, _pool_size(), TUTOR_PROMPT_VERSION,
                QUESTION_PROMPT_VERSION, settings.enable_docs)
    if not settings.llm_enabled:
        logger.warning("GROQ_API_KEY is empty: LLM features will use fallbacks")
    init_client()
    try:
        yield
    finally:
        await close_client()


_docs = get_settings().enable_docs
app = FastAPI(
    title="LearnAI ML Backend",
    version=APP_VERSION,
    description=DESCRIPTION,
    openapi_tags=TAGS,
    lifespan=lifespan,
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
    swagger_ui_parameters={"defaultModelsExpandDepth": 0, "displayRequestDuration": True, "tryItOutEnabled": True},
)

app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,  # ["*"] when ALLOWED_ORIGINS="*" (no credentials)
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


async def _request_action(request: Request) -> str:
    try:
        body = await request.json()
        if isinstance(body, dict) and isinstance(body.get("action"), str):
            return body["action"]
    except Exception:
        pass
    return ""


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    resp = error_response(
        await _request_action(request), "INVALID_REQUEST", "Request body does not match the contract.",
        [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()],
    )
    return JSONResponse(status_code=422, content=resp.model_dump())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    resp = error_response("", code, str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=resp.model_dump())


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    resp = error_response("", "INTERNAL_ERROR", "Something went wrong while processing the request.")
    return JSONResponse(status_code=500, content=resp.model_dump())


def _health_checks() -> dict:
    """Autocommit read: no BEGIN/COMMIT round trips, so /health stays fast even over the network."""
    try:
        with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            pool = conn.execute(select(func.count()).select_from(Question)
                                .where(Question.verified.is_(True), Question.source == "llm")).scalar()
        return {"db": "ok", "pool_size": int(pool or 0)}
    except Exception:
        logger.exception("Health check DB query failed")
        return {"db": "error", "pool_size": 0}


@app.get("/health", response_model=HealthOut, tags=["System"], summary="Health + wake-up",
         description="Cheap liveness check (one DB count, no AI call). Call it when the app opens to wake the "
                     "free-tier server; the first call after idle can take 30-60 s.")
async def health() -> dict:
    checks = await run_in_threadpool(_health_checks)
    return {"status": "ok", "db": checks["db"], "llm_configured": get_settings().llm_enabled,
            "pool_size": checks["pool_size"]}


app.include_router(router)
