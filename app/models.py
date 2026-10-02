"""SQLAlchemy ORM models. JSON columns work on both SQLite and Postgres."""

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Learner(Base):
    __tablename__ = "learners"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    level: Mapped[str] = mapped_column(String(20))
    role: Mapped[str] = mapped_column(String(120), default="")
    goal: Mapped[str] = mapped_column(String(200), default="")
    target_role: Mapped[str] = mapped_column(String(120), default="")
    experience: Mapped[str] = mapped_column(Text, default="")
    known_languages: Mapped[list] = mapped_column(JSON, default=list)
    known_topics: Mapped[list] = mapped_column(JSON, default=list)
    target_topics: Mapped[list] = mapped_column(JSON, default=list)
    learning_pace: Mapped[str] = mapped_column(String(20), default="Balanced")
    daily_commitment_minutes: Mapped[int] = mapped_column(Integer, default=30)
    streak_days: Mapped[int] = mapped_column(Integer, default=0)
    questions_solved: Mapped[int] = mapped_column(Integer, default=0)
    accuracy_rate: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConceptState(Base):
    __tablename__ = "concept_states"
    __table_args__ = (UniqueConstraint("learner_id", "concept_id", name="uq_learner_concept"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.id", ondelete="CASCADE"), index=True)
    concept_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(40))
    mastery: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    recent_results: Mapped[list] = mapped_column(JSON, default=list)
    consecutive_wrong: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    trend: Mapped[str] = mapped_column(String(20), default="stable")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.id", ondelete="CASCADE"), index=True)
    concept_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(40))
    difficulty: Mapped[str] = mapped_column(String(10))
    is_correct: Mapped[bool] = mapped_column(Boolean)
    time_taken_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    question_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AdaptiveEvent(Base):
    __tablename__ = "adaptive_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.id", ondelete="CASCADE"), index=True)
    topic: Mapped[str] = mapped_column(String(120))
    score: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text, default="")
    recommendation: Mapped[str] = mapped_column(Text, default="")
    path_adjustment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Question(Base):
    """Generated/known questions. `payload` is the full PracticeQuestion incl. correct_index."""

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    learner_id: Mapped[str | None] = mapped_column(
        ForeignKey("learners.id", ondelete="SET NULL"), nullable=True, index=True
    )
    concept_id: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(40))
    difficulty: Mapped[str] = mapped_column(String(10))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    source: Mapped[str] = mapped_column(String(10), default="llm")  # "llm" | "fallback"
    times_served: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QuestionServe(Base):
    """Which questions a learner has already been shown (pool never re-serves them)."""

    __tablename__ = "question_serves"
    __table_args__ = (UniqueConstraint("learner_id", "question_id", name="uq_learner_question_serve"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.id", ondelete="CASCADE"), index=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), index=True)
    served_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(10))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    mode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    concept_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LearningPath(Base):
    """Snapshot of a learner's computed path; a new row is written only when the structure changes."""

    __tablename__ = "learning_paths"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(ForeignKey("learners.id", ondelete="CASCADE"), index=True)
    goal: Mapped[str] = mapped_column(String(40))
    nodes: Mapped[list] = mapped_column(JSON, default=list)
    path_hash: Mapped[str] = mapped_column(String(32), index=True)
    changes: Mapped[list] = mapped_column(JSON, default=list)
    before_after: Mapped[str] = mapped_column(Text, default="")
    why_this_path: Mapped[str] = mapped_column(Text, default="")
    why_source: Mapped[str] = mapped_column(String(10), default="template")  # "llm" | "template"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
