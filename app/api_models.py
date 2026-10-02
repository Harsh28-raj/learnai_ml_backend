"""Response schemas for Swagger/OpenAPI documentation.

These models describe what each action returns in `data`. They are used for documentation
only (`responses=` on the route); the endpoint itself returns the same envelope as before.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import ErrorInfo

Difficulty = Literal["Easy", "Medium", "Hard"]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- shared pieces

class AdaptiveEventOut(_Open):
    """SPEC 8.2 AdaptiveEvent (camelCase)."""
    id: str = Field(examples=["adapt-12"])
    timestamp: str = Field(examples=["Today at 2:15 PM"])
    createdAt: str
    topic: str
    score: int = Field(description="0-100: % correct in the request, or the graded score in tutor evaluate mode")
    action: Literal["reduced", "maintained", "increased"]
    reason: str
    recommendation: str
    pathAdjustment: str = Field(description="Real roadmap change caused by this answer (before/after line)")


class WeaknessEntry(_Open):
    concept: str
    name: str
    category: str
    mastery: float
    attempts: int
    classification: Literal["Strong", "Average", "Weak"]
    priority: float = Field(description="0-100")
    label: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str
    tutor_alert: str | None = Field(description="Only for HIGH priority")
    recommended_revision: dict[str, Any]


class WeaknessReport(_Open):
    strong: list[WeaknessEntry]
    average: list[WeaknessEntry]
    weak: list[WeaknessEntry]
    untried: list[str]


class ConceptStateOut(_Open):
    concept: str
    mastery: float
    attempts: int
    correct: int
    recentAccuracy: int | None
    confidence: float
    trend: Literal["improving", "stable", "declining"]


class LearnerProfileOut(_Open):
    """SPEC 5 LearnerProfile (camelCase)."""
    id: str
    name: str
    level: Literal["Beginner", "Intermediate", "Advanced"]
    role: str
    goal: str
    targetRole: str
    experience: str
    knownLanguages: list[str]
    knownTopics: list[str]
    targetTopics: list[str]
    learningPace: str
    dailyCommitmentMinutes: int
    streakDays: int
    questionsSolved: int
    accuracyRate: int
    conceptsMastered: int
    skills: dict[str, int] = Field(examples=[{"Python": 88, "Statistics": 62, "Machine Learning": 71}])
    strengths: list[dict[str, Any]]
    weaknesses: list[dict[str, Any]] = Field(description="[{name, score, reason, recommendedLessonId}], < 65")
    conceptStates: list[ConceptStateOut]
    recentAdaptiveEvents: list[AdaptiveEventOut]


class PracticeQuestionOut(_Open):
    """SPEC 10.1 PracticeQuestion (camelCase) plus LearnAI extras."""
    id: str = Field(examples=["gen-1d0e6051e6"])
    category: str
    difficulty: Difficulty
    conceptId: str
    conceptTested: str
    title: str
    question: str
    codeSnippet: str | None
    options: list[str] = Field(min_length=4, max_length=4)
    correctIndex: int
    explanation: str
    hint: str
    whyThisQuestion: str
    recommendedNextDifficulty: Difficulty
    scenarioDomain: str
    questionAngle: str
    source: Literal["llm", "pool", "fallback"]
    verified: bool


class PathNodeOut(_Open):
    id: str = Field(examples=["node-bias-variance"])
    conceptId: str
    title: str
    category: str
    status: Literal["completed", "current", "recommended", "adapted", "locked"]
    progress: int
    mastery: int
    estimatedMinutes: int
    reason: str
    isRevision: bool
    skipped: bool
    prerequisites: list[str]
    foundationFor: str | None
    order: int


class MilestoneOut(_Open):
    id: str
    title: str
    category: str
    status: Literal["completed", "current", "upcoming"]
    nodeIds: list[str]
    progress: int


class PathChangeOut(_Open):
    type: Literal["inserted_revision", "skipped", "reordered", "unlocked", "completed"]
    conceptId: str
    title: str
    detail: str


class DailyTaskOut(_Open):
    id: str
    title: str
    type: Literal["revision", "practice", "lesson"]
    durationMinutes: int
    completed: bool
    conceptId: str


# --------------------------------------------------------------------------- per-action data

class ProfileData(_Open):
    """get_profile"""
    profile: LearnerProfileOut
    weakness_report: WeaknessReport


class ResetData(ProfileData):
    """reset_learner"""
    reset: bool


class EvaluateData(_Open):
    """evaluate (single answer or quiz)"""
    results: list[dict[str, Any]] = Field(description="[{question_id, is_correct, correct_index, concept}]")
    score: int
    mastery_updates: list[dict[str, Any]] = Field(
        description="[{concept, concept_name, previous_score, new_score, category, previous_category_score, "
                    "new_category_score, trend, confidence}]")
    weakness_report: WeaknessReport
    next_difficulty: dict[str, Any] = Field(description="{concept, difficulty, reason, rule_id}")
    adaptive_decision: AdaptiveEventOut


class QuestionsData(_Open):
    """generate_questions"""
    count: int
    requested_count: int
    concept: str
    concept_name: str
    selection_reason: Literal["requested", "weak", "lowest_practiced", "new_topic", "next_in_category"]
    difficulty: Difficulty
    difficulty_reason: str
    questions: list[PracticeQuestionOut]
    sources: dict[str, int] = Field(examples=[{"llm": 3, "pool": 0, "fallback": 0}])
    generation: dict[str, Any] = Field(description="{model, verifier_model, llm_calls, verify_calls, verified, "
                                                   "rejected, rejection_issues, latency_ms}")


class TutorData(_Open):
    """tutor_chat"""
    conversationId: str
    message: str = Field(description="Markdown + KaTeX")
    mode: Literal["explanation", "simplify", "example", "code", "practice", "hint", "evaluate", "revision", "path"]
    modeSource: Literal["explicit", "auto"]
    concept: str | None
    conceptName: str | None
    difficultyLevel: str
    followUpSuggestions: list[str] = Field(min_length=3, max_length=3)
    checkpointQuestion: dict[str, Any] | None = Field(
        description="{id, category, difficulty, conceptId, conceptTested, question, codeSnippet, options, "
                    "correctIndex, explanation, hint, verified}; grade it with evaluate + question_id")
    recommendedNextAction: dict[str, Any] = Field(description="{type: practice|revise|advance|continue, targetTopic, reason}")
    evaluation: dict[str, Any] | None = Field(description="evaluate mode: {score, correctPoints, misconceptions, "
                                                          "correctedAnswer, recorded}")
    masteryUpdate: dict[str, Any] | None
    adaptiveDecision: dict[str, Any] | None
    promptVersion: str
    guardrails: dict[str, Any]
    generation: dict[str, Any]


class PathData(_Open):
    """get_path"""
    goal: str
    goalLabel: str
    targets: list[str]
    nodes: list[PathNodeOut]
    milestones: list[MilestoneOut]
    currentNode: PathNodeOut | None
    nextNodes: list[PathNodeOut]
    changes: list[PathChangeOut]
    changedNow: bool
    beforeAfter: str = Field(description="One line for the Dashboard 'tutor adapted your path' banner")
    whyThisPath: str
    whySource: Literal["llm", "template"]
    estimatedWeeksRemaining: int
    totalEstimatedMinutes: int
    dailyPlan: list[DailyTaskOut]
    pathHash: str
    preview: bool


class AssessmentData(ProfileData):
    """assessment"""
    path: PathData
    level: str
    goal: str
    firstStep: dict[str, Any] = Field(description="{type: 'diagnostic', conceptIds: [3], message}")


class LearnAIResponseDoc(BaseModel):
    """The envelope every action returns (HTTP 200 for success and for action errors)."""
    success: bool
    action: str
    data: (ProfileData | ResetData | EvaluateData | QuestionsData | TutorData | PathData | AssessmentData
           | None) = Field(description="Action-specific payload; null when success=false")
    error: ErrorInfo | None = Field(description="Set when success=false, or as a notice when generate_questions "
                                                "used pool/fallback questions but still succeeded")
    fallback_data: dict[str, Any] | None = Field(description="Something useful to show when the AI is unavailable")


class HealthOut(BaseModel):
    status: Literal["ok"]
    db: Literal["ok", "error"]
    llm_configured: bool
    pool_size: int = Field(description="Verified LLM questions available for rate-limit fallback")
