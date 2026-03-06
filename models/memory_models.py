"""Memory system data models.

This module defines data models for the memory system, including
log entries for conversation logging and memory structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LogEntryType(str, Enum):
    """Log entry type enumeration.
    
    Represents the different types of events that can be logged
    in the conversation log.
    """

    VAD_EVENT = "vad_event"
    STT_TRANSCRIPT = "stt_transcript"
    LLM_INPUT = "llm_input"
    LLM_OUTPUT = "llm_output"
    PLANNER_DECISION = "planner_decision"
    PHASE_TRANSITION = "phase_transition"
    TTS_START = "tts_start"
    TTS_STOP = "tts_stop"
    TURN_COMPLETE = "turn_complete"
    ERROR = "error"


class LogEntry(BaseModel):
    """Log entry for conversation logging.
    
    Represents a single log entry in the conversation log.
    All entries are append-only and immutable after creation.
    
    Attributes:
        entry_type: Type of log entry
        timestamp: When the event occurred
        session_id: Session identifier
        data: Flexible data structure for event-specific data
        metadata: Optional additional metadata
    """

    entry_type: LogEntryType = Field(
        description="Type of log entry"
    )
    timestamp: datetime = Field(
        description="When the event occurred"
    )
    session_id: str = Field(
        description="Session identifier"
    )
    data: Dict[str, Any] = Field(
        description="Event-specific data"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional metadata"
    )

