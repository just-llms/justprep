"""Enums and constants for the interview system."""

from enum import Enum


class InterviewPhase(str, Enum):
    """Interview phase enumeration.
    
    Represents the different phases of an interview session.
    """

    GREETING = "greeting"
    SMALL_TALK = "small_talk"  # Natural conversation, rapport building
    RESUME_DISCUSSION = "resume_discussion"  # High-level resume overview
    INTRODUCTION = "introduction"
    WARMUP = "warmup"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CLOSING = "closing"
    ENDED = "ended"


class LLMAction(str, Enum):
    """LLM action enumeration.
    
    Represents the actions that the LLM can suggest for the interviewer.
    These are the only valid actions the LLM can return.
    """

    ASK_FOLLOW_UP = "ask_follow_up"
    NEXT_QUESTION = "next_question"
    CLARIFY = "clarify"
    REPEAT_QUESTION = "repeat_question"
    END_PHASE = "end_phase"
    END_INTERVIEW = "end_interview"
    ACKNOWLEDGE = "acknowledge"


class VADEvent(str, Enum):
    """Voice Activity Detection event enumeration.
    
    Represents events emitted by the VAD system.
    """

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration.
    
    Represents the LLM's confidence in its response.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlannerOutputType(str, Enum):
    """Planner output type enumeration.
    
    Represents the type of output from the response planner.
    """

    SPEAK = "speak"
    SILENT = "silent"


# Event constants for utterance finalization
USER_RESPONSE_COMPLETE = "user_response_complete"
