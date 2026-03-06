"""Session state and turn models."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from models.constants import InterviewPhase


class Turn(BaseModel):
    """Represents a single Q&A turn in the interview.
    
    A turn consists of a question asked by the interviewer and
    the candidate's answer.
    """

    question: Optional[str] = Field(
        default=None,
        description="The question asked by the interviewer"
    )
    answer: str = Field(
        description="The candidate's answer to the question"
    )
    timestamp: datetime = Field(
        description="When the turn occurred"
    )
    phase: InterviewPhase = Field(
        description="The interview phase when this turn occurred"
    )
    turn_number: int = Field(
        description="Sequential turn number in the session"
    )

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Validate that answer is not empty."""
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()

    @field_validator("turn_number")
    @classmethod
    def validate_turn_number(cls, v: int) -> int:
        """Validate that turn number is positive."""
        if v <= 0:
            raise ValueError("Turn number must be greater than 0")
        return v


class SessionState(BaseModel):
    """Session state model.
    
    Represents the complete state of an interview session,
    including current phase, question, turn history, and metadata.
    """

    session_id: str = Field(
        description="Unique session identifier"
    )
    phase: InterviewPhase = Field(
        description="Current interview phase"
    )
    current_question: Optional[str] = Field(
        default=None,
        description="Current question being asked to the candidate"
    )
    follow_up_count: int = Field(
        default=0,
        description="Number of follow-up questions asked for the current question"
    )
    turn_history: List[Turn] = Field(
        default_factory=list,
        description="History of Q&A turns in this session"
    )
    created_at: datetime = Field(
        description="Session creation timestamp"
    )
    updated_at: datetime = Field(
        description="Last update timestamp"
    )
    total_turns: int = Field(
        default=0,
        description="Total number of turns in the session"
    )
    session_start_time: datetime = Field(
        description="When the session started"
    )
    work_experience: str = Field(
        description="Candidate's work experience summary"
    )
    current_role: str = Field(
        description="Candidate's current job role"
    )
    target_role: Optional[str] = Field(
        default=None,
        description="Role candidate is applying for"
    )
    resume: str = Field(
        description="Resume text content"
    )
    interview_points: List[str] = Field(
        default_factory=list,
        description="Key discussion points extracted from resume"
    )
    years_of_experience: Optional[float] = Field(
        default=None,
        description="Total years of professional experience"
    )
    candidate_name: Optional[str] = Field(
        default=None,
        description="Candidate's full name"
    )
    current_company: Optional[str] = Field(
        default=None,
        description="Candidate's current employer"
    )
    initial_greeting_sent: bool = Field(
        default=False,
        description="Whether the initial greeting has been sent"
    )
    questions_in_current_phase: int = Field(
        default=0,
        description="Number of main questions asked in current phase"
    )
    last_activity_time: Optional[datetime] = Field(
        default=None,
        description="Last time there was activity in the session (for stuck session detection)"
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate that session_id is not empty."""
        if not v or not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()

    @field_validator("follow_up_count", "total_turns")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Validate that count is non-negative."""
        if v < 0:
            raise ValueError("Count must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_updated_at_after_created_at(self) -> "SessionState":
        """Validate that updated_at is not before created_at."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at")
        return self

