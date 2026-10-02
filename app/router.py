"""Single-endpoint action dispatcher: POST /api/v1/learnai."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from typing import Annotated

from fastapi import APIRouter, Body, Request

from app.actions.evaluate import evaluate
from app.actions.profile import get_profile, reset_learner
from app.actions.assessment import assessment
from app.actions.path import get_path
from app.actions.questions import generate_questions
from app.actions.tutor import tutor_chat
from app.api_docs import ENDPOINT_DESCRIPTION, EXAMPLES
from app.api_models import LearnAIResponseDoc
from app.config import get_settings
from app.protection import bucket_for, client_ip, limiter
from app.schemas import ActionError, ActionResult, ErrorInfo, LearnAIRequest, LearnAIResponse
from app.seed import POOL_LEARNERS

logger = logging.getLogger("learnai.router")

Handler = Callable[[str, dict[str, Any]], Awaitable[dict]]


ACTIONS: dict[str, Handler] = {
    "get_profile": get_profile,
    "reset_learner": reset_learner,
    "evaluate": evaluate,
    "generate_questions": generate_questions,
    "tutor_chat": tutor_chat,
    "get_path": get_path,
    "assessment": assessment,
}

# Internal pool learners (warm_pool.py) may only generate questions.
# ("assessment" is let through so it can answer DEMO_LEARNER_PROTECTED itself.)
POOL_ACTIONS = frozenset({"generate_questions", "assessment"})

router = APIRouter()


def error_response(action: str, code: str, message: str, details: Any = None,
                   fallback_data: dict | None = None) -> LearnAIResponse:
    return LearnAIResponse(
        success=False,
        action=action,
        error=ErrorInfo(code=code, message=message, details=details),
        fallback_data=fallback_data,
    )


@router.post(
    "/api/v1/learnai",
    response_model=None,  # documented via `responses`; the returned envelope is unchanged
    tags=["LearnAI"],
    summary="Run a LearnAI action (single endpoint)",
    description=ENDPOINT_DESCRIPTION,
    responses={
        200: {"model": LearnAIResponseDoc,
              "description": "Envelope. `success=false` action errors also use HTTP 200 (see the guide above)."},
        413: {"model": LearnAIResponseDoc, "description": "PAYLOAD_TOO_LARGE: body over 32 KB"},
        422: {"model": LearnAIResponseDoc, "description": "INVALID_REQUEST: malformed envelope"},
    },
)
async def learnai(request: Request,
                  req: Annotated[LearnAIRequest, Body(openapi_examples=EXAMPLES)]) -> LearnAIResponse:
    settings = get_settings()
    bucket = bucket_for(req.action)
    limit = settings.rate_limit_llm_per_min if bucket == "llm" else settings.rate_limit_other_per_min
    retry_after = limiter.check(client_ip(request), bucket, limit)
    if retry_after is not None:
        return error_response(
            req.action, "RATE_LIMITED", f"Too many requests; try again in {retry_after} s.",
            {"retry_after_seconds": retry_after, "limit_per_minute": limit,
             "bucket": "ai" if bucket == "llm" else "standard"},
        )
    handler = ACTIONS.get(req.action)
    if handler is None:
        return error_response(
            req.action, "UNKNOWN_ACTION", f"Unknown action '{req.action}'.",
            {"supported_actions": sorted(ACTIONS)},
        )
    if req.learner_id in POOL_LEARNERS and req.action not in POOL_ACTIONS:
        return error_response(req.action, "LEARNER_NOT_FOUND", f"No learner with id '{req.learner_id}'.",
                              {"learner_id": req.learner_id})
    try:
        data = await handler(req.learner_id, req.payload)
    except ActionError as e:
        return error_response(req.action, e.code, e.message, e.details, e.fallback_data)
    except Exception:
        logger.exception("Action %s failed for learner %s", req.action, req.learner_id)
        return error_response(req.action, "INTERNAL_ERROR", "Something went wrong while processing the request.")
    if isinstance(data, ActionResult):
        return LearnAIResponse(success=True, action=req.action, data=data.data, error=data.warning)
    return LearnAIResponse(success=True, action=req.action, data=data)
