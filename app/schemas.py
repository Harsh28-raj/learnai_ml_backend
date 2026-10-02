"""Request/response contract for the single /api/v1/learnai endpoint."""

from typing import Any

from pydantic import BaseModel, Field


class LearnAIRequest(BaseModel):
    action: str = Field(..., min_length=1, max_length=64)
    learner_id: str = Field(..., min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: Any = None


class LearnAIResponse(BaseModel):
    success: bool
    action: str
    data: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    fallback_data: dict[str, Any] | None = None


class ActionError(Exception):
    """Raised by action handlers to return a structured (non-500) error."""

    def __init__(self, code: str, message: str, details: Any = None, fallback_data: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.fallback_data = fallback_data


class ActionResult:
    """Successful result that still carries a non-fatal notice for the `error` field.

    Used when a request succeeded with degraded sources (e.g. questions served from the
    pool because the LLM was rate limited): success=true, error={code, message}.
    """

    def __init__(self, data: dict, warning: ErrorInfo | None = None):
        self.data = data
        self.warning = warning
