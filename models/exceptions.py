"""Custom exceptions for the interview system."""


class SessionNotFoundError(Exception):
    """Exception raised when a session cannot be found.
    
    Attributes:
        message: Explanation of the error
        session_id: The session ID that was not found
    """

    def __init__(self, message: str = "Session not found", session_id: str | None = None) -> None:
        """Initialize SessionNotFoundError.
        
        Args:
            message: Error message
            session_id: The session ID that was not found (optional)
        """
        self.message = message
        self.session_id = session_id
        if session_id:
            self.message = f"Session not found: {session_id}"
        super().__init__(self.message)


class InvalidPhaseTransitionError(Exception):
    """Exception raised when a phase transition is not allowed.
    
    Attributes:
        message: Explanation of the error
        from_phase: The current phase
        to_phase: The attempted target phase
    """

    def __init__(
        self,
        message: str = "Invalid phase transition",
        from_phase: str | None = None,
        to_phase: str | None = None,
    ) -> None:
        """Initialize InvalidPhaseTransitionError.
        
        Args:
            message: Error message
            from_phase: The current phase (optional)
            to_phase: The attempted target phase (optional)
        """
        self.message = message
        self.from_phase = from_phase
        self.to_phase = to_phase
        if from_phase and to_phase:
            self.message = f"Invalid phase transition from {from_phase} to {to_phase}"
        super().__init__(self.message)


class InvalidActionError(Exception):
    """Exception raised when an action is not allowed in the current phase.
    
    Attributes:
        message: Explanation of the error
        action: The action that was attempted
        phase: The current phase
    """

    def __init__(
        self,
        message: str = "Invalid action for current phase",
        action: str | None = None,
        phase: str | None = None,
    ) -> None:
        """Initialize InvalidActionError.
        
        Args:
            message: Error message
            action: The action that was attempted (optional)
            phase: The current phase (optional)
        """
        self.message = message
        self.action = action
        self.phase = phase
        if action and phase:
            self.message = f"Action {action} is not allowed in phase {phase}"
        super().__init__(self.message)


class MaxFollowUpsExceededError(Exception):
    """Exception raised when the maximum number of follow-ups is exceeded.
    
    Attributes:
        message: Explanation of the error
        current_count: Current number of follow-ups
        max_count: Maximum allowed follow-ups
    """

    def __init__(
        self,
        message: str = "Maximum follow-ups exceeded",
        current_count: int | None = None,
        max_count: int | None = None,
    ) -> None:
        """Initialize MaxFollowUpsExceededError.
        
        Args:
            message: Error message
            current_count: Current number of follow-ups (optional)
            max_count: Maximum allowed follow-ups (optional)
        """
        self.message = message
        self.current_count = current_count
        self.max_count = max_count
        if current_count is not None and max_count is not None:
            self.message = (
                f"Maximum follow-ups exceeded: {current_count} >= {max_count}"
            )
        super().__init__(self.message)


class SessionTimeoutError(Exception):
    """Exception raised when a session exceeds time limits.
    
    Attributes:
        message: Explanation of the error
        session_id: The session ID that timed out
        duration: The duration of the session
        max_duration: The maximum allowed duration
    """

    def __init__(
        self,
        message: str = "Session timeout",
        session_id: str | None = None,
        duration: float | None = None,
        max_duration: float | None = None,
    ) -> None:
        """Initialize SessionTimeoutError.
        
        Args:
            message: Error message
            session_id: The session ID that timed out (optional)
            duration: The duration of the session in seconds (optional)
            max_duration: The maximum allowed duration in seconds (optional)
        """
        self.message = message
        self.session_id = session_id
        self.duration = duration
        self.max_duration = max_duration
        if duration is not None and max_duration is not None:
            self.message = (
                f"Session timeout: duration {duration}s exceeds maximum {max_duration}s"
            )
            if session_id:
                self.message = f"Session {session_id}: {self.message}"
        super().__init__(self.message)


# Component-specific error types

class VADError(Exception):
    """Base exception for VAD errors."""
    pass


class STTError(Exception):
    """Base exception for STT errors."""
    pass


class STTTimeoutError(STTError):
    """STT operation timed out."""
    
    def __init__(self, message: str = "STT operation timed out", timeout_seconds: float | None = None) -> None:
        self.message = message
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            self.message = f"STT operation timed out after {timeout_seconds} seconds"
        super().__init__(self.message)


class STTConnectionError(STTError):
    """STT connection error."""
    
    def __init__(self, message: str = "STT connection error", details: str | None = None) -> None:
        self.message = message
        self.details = details
        if details:
            self.message = f"STT connection error: {details}"
        super().__init__(self.message)


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMTimeoutError(LLMError):
    """LLM operation timed out."""
    
    def __init__(self, message: str = "LLM operation timed out", timeout_seconds: float | None = None) -> None:
        self.message = message
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            self.message = f"LLM operation timed out after {timeout_seconds} seconds"
        super().__init__(self.message)


class TTSError(Exception):
    """Base exception for TTS errors."""
    pass


class TTSTimeoutError(TTSError):
    """TTS operation timed out."""
    
    def __init__(self, message: str = "TTS operation timed out", timeout_seconds: float | None = None) -> None:
        self.message = message
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            self.message = f"TTS operation timed out after {timeout_seconds} seconds"
        super().__init__(self.message)


class WebSocketError(Exception):
    """Base exception for WebSocket errors."""
    
    def __init__(self, message: str = "WebSocket error", session_id: str | None = None) -> None:
        self.message = message
        self.session_id = session_id
        if session_id:
            self.message = f"WebSocket error for session {session_id}: {message}"
        super().__init__(self.message)


class SafetyLimitExceededError(Exception):
    """Exception raised when safety limit is exceeded.
    
    Attributes:
        limit_type: Type of limit exceeded (e.g., "max_duration_exceeded")
        message: Error message
        retry_after: Seconds until retry allowed (optional)
    """
    
    def __init__(
        self,
        limit_type: str,
        message: str,
        retry_after: float | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize SafetyLimitExceededError.
        
        Args:
            limit_type: Type of limit exceeded
            message: Error message
            retry_after: Seconds until retry allowed (optional)
            session_id: Session identifier (optional)
        """
        self.limit_type = limit_type
        self.retry_after = retry_after
        self.session_id = session_id
        self.message = message
        super().__init__(self.message)
