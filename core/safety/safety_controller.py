"""Safety controller for limits enforcement and abuse prevention.

This module implements centralized safety limits including session limits,
cost limits, rate limiting, and timeout handling to prevent abuse and
control costs.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from core.session_manager import SessionManager
from models.session_models import SessionState
from util.logger import get_component_logger
from util.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a safety limit check.
    
    Attributes:
        allowed: Whether the operation is allowed
        reason: Reason code if not allowed (e.g., "max_duration_exceeded")
        message: Human-readable message
        retry_after: Seconds until retry allowed (for rate limiting)
    """
    allowed: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    retry_after: Optional[float] = None


class SafetyController:
    """Centralized safety limits and abuse prevention.
    
    This class enforces all safety limits including:
    - Session limits (duration, turns)
    - Cost limits (LLM tokens, STT minutes)
    - Rate limiting (requests per minute)
    - Stuck session detection
    
    All limits are configurable and violations are logged.
    """
    
    # Configuration constants
    MAX_INTERVIEW_DURATION_MINUTES = 60
    MAX_TURNS_PER_SESSION = 100
    MAX_LLM_TOKENS_PER_SESSION = 100_000
    MAX_STT_MINUTES_PER_SESSION = 60
    RATE_LIMIT_REQUESTS_PER_MINUTE = 10
    STUCK_SESSION_TIMEOUT_MINUTES = 5
    
    _instance: Optional["SafetyController"] = None
    _lock = threading.Lock()  # Use threading.Lock for synchronous singleton pattern
    
    def __init__(self) -> None:
        """Initialize SafetyController."""
        self.session_manager = SessionManager.get_instance()
        self.metrics = MetricsCollector.get_instance()
        self.structured_logger = get_component_logger("safety")
        
        # Cost tracking per session
        self._session_tokens: Dict[str, int] = defaultdict(int)
        self._session_stt_minutes: Dict[str, float] = defaultdict(float)
        self._cost_lock = asyncio.Lock()
        
        # Rate limiting per user/IP
        self._rate_limit_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._rate_limit_lock = asyncio.Lock()
        
        # Stuck session detection task
        self._stuck_session_task: Optional[asyncio.Task] = None
        self._stuck_session_running = False
        
        logger.info("SafetyController initialized")
    
    @classmethod
    def get_instance(cls) -> "SafetyController":
        """Get singleton instance of SafetyController.
        
        Returns:
            SafetyController singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    async def check_session_duration(
        self, session_id: str, session: SessionState
    ) -> SafetyCheckResult:
        """Check if session duration exceeds limit.
        
        Args:
            session_id: Session identifier
            session: Session state
            
        Returns:
            SafetyCheckResult indicating if allowed
        """
        if session.session_start_time:
            duration_minutes = (
                (datetime.now() - session.session_start_time).total_seconds() / 60
            )
            if duration_minutes >= self.MAX_INTERVIEW_DURATION_MINUTES:
                self.structured_logger.warning(
                    "limit_violation",
                    "Session duration limit exceeded",
                    {
                        "session_id": session_id,
                        "duration_minutes": round(duration_minutes, 2),
                        "max_duration_minutes": self.MAX_INTERVIEW_DURATION_MINUTES,
                    },
                    session_id=session_id,
                )
                return SafetyCheckResult(
                    allowed=False,
                    reason="max_duration_exceeded",
                    message=(
                        f"Interview duration ({duration_minutes:.1f} min) exceeds "
                        f"maximum ({self.MAX_INTERVIEW_DURATION_MINUTES} min)"
                    ),
                )
        return SafetyCheckResult(allowed=True)
    
    async def check_turn_limit(
        self, session_id: str, session: SessionState
    ) -> SafetyCheckResult:
        """Check if turn count exceeds limit.
        
        Args:
            session_id: Session identifier
            session: Session state
            
        Returns:
            SafetyCheckResult indicating if allowed
        """
        if session.total_turns >= self.MAX_TURNS_PER_SESSION:
            self.structured_logger.warning(
                "limit_violation",
                "Session turn limit exceeded",
                {
                    "session_id": session_id,
                    "total_turns": session.total_turns,
                    "max_turns": self.MAX_TURNS_PER_SESSION,
                },
                session_id=session_id,
            )
            return SafetyCheckResult(
                allowed=False,
                reason="max_turns_exceeded",
                message=(
                    f"Turn count ({session.total_turns}) exceeds "
                    f"maximum ({self.MAX_TURNS_PER_SESSION})"
                ),
            )
        return SafetyCheckResult(allowed=True)
    
    async def check_session_limits(
        self, session_id: str
    ) -> SafetyCheckResult:
        """Check all session limits (duration and turns).
        
        Args:
            session_id: Session identifier
            
        Returns:
            SafetyCheckResult indicating if allowed
        """
        try:
            session = await self.session_manager.get_session_or_raise(session_id)
            
            # Check duration
            duration_result = await self.check_session_duration(session_id, session)
            if not duration_result.allowed:
                return duration_result
            
            # Check turn limit
            turn_result = await self.check_turn_limit(session_id, session)
            if not turn_result.allowed:
                return turn_result
            
            return SafetyCheckResult(allowed=True)
        except Exception as e:
            logger.error(f"Error checking session limits for {session_id}: {e}")
            # Allow on error to avoid blocking
            return SafetyCheckResult(allowed=True)
    
    async def record_llm_tokens(self, session_id: str, tokens: int) -> None:
        """Record LLM token usage for session.
        
        Args:
            session_id: Session identifier
            tokens: Number of tokens used
        """
        async with self._cost_lock:
            self._session_tokens[session_id] += tokens
            # Track tokens via counter (increment by tokens)
            for _ in range(tokens):
                self.metrics.increment_counter(
                    "safety.llm_tokens_used",
                    tags={"session_id": session_id},
                )
    
    async def record_stt_minutes(self, session_id: str, minutes: float) -> None:
        """Record STT minutes usage for session.
        
        Args:
            session_id: Session identifier
            minutes: Minutes of STT usage
        """
        async with self._cost_lock:
            self._session_stt_minutes[session_id] += minutes
            self.metrics.set_gauge(
                "safety.stt_minutes_used",
                self._session_stt_minutes[session_id],
                tags={"session_id": session_id},
            )
    
    async def check_llm_token_limit(self, session_id: str) -> SafetyCheckResult:
        """Check if LLM token limit exceeded.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SafetyCheckResult indicating if allowed
        """
        async with self._cost_lock:
            current_tokens = self._session_tokens.get(session_id, 0)
            if current_tokens >= self.MAX_LLM_TOKENS_PER_SESSION:
                self.structured_logger.warning(
                    "limit_violation",
                    "LLM token limit exceeded",
                    {
                        "session_id": session_id,
                        "current_tokens": current_tokens,
                        "max_tokens": self.MAX_LLM_TOKENS_PER_SESSION,
                    },
                    session_id=session_id,
                )
                return SafetyCheckResult(
                    allowed=False,
                    reason="max_tokens_exceeded",
                    message=(
                        f"LLM tokens ({current_tokens}) exceed "
                        f"maximum ({self.MAX_LLM_TOKENS_PER_SESSION})"
                    ),
                )
            return SafetyCheckResult(allowed=True)
    
    async def check_stt_minute_limit(self, session_id: str) -> SafetyCheckResult:
        """Check if STT minute limit exceeded.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SafetyCheckResult indicating if allowed
        """
        async with self._cost_lock:
            current_minutes = self._session_stt_minutes.get(session_id, 0.0)
            if current_minutes >= self.MAX_STT_MINUTES_PER_SESSION:
                self.structured_logger.warning(
                    "limit_violation",
                    "STT minute limit exceeded",
                    {
                        "session_id": session_id,
                        "current_minutes": round(current_minutes, 2),
                        "max_minutes": self.MAX_STT_MINUTES_PER_SESSION,
                    },
                    session_id=session_id,
                )
                return SafetyCheckResult(
                    allowed=False,
                    reason="max_stt_minutes_exceeded",
                    message=(
                        f"STT minutes ({current_minutes:.1f}) exceed "
                        f"maximum ({self.MAX_STT_MINUTES_PER_SESSION})"
                    ),
                )
            return SafetyCheckResult(allowed=True)
    
    async def check_cost_limits(
        self, session_id: str, tokens: Optional[int] = None, stt_minutes: Optional[float] = None
    ) -> SafetyCheckResult:
        """Check if cost limits exceeded.
        
        Args:
            session_id: Session identifier
            tokens: Optional tokens to check (if None, checks current usage)
            stt_minutes: Optional STT minutes to check (if None, checks current usage)
            
        Returns:
            SafetyCheckResult indicating if allowed
        """
        # Check LLM token limit
        if tokens is not None:
            async with self._cost_lock:
                current_tokens = self._session_tokens.get(session_id, 0)
                if current_tokens + tokens > self.MAX_LLM_TOKENS_PER_SESSION:
                    return await self.check_llm_token_limit(session_id)
        else:
            token_result = await self.check_llm_token_limit(session_id)
            if not token_result.allowed:
                return token_result
        
        # Check STT minute limit
        if stt_minutes is not None:
            async with self._cost_lock:
                current_minutes = self._session_stt_minutes.get(session_id, 0.0)
                if current_minutes + stt_minutes > self.MAX_STT_MINUTES_PER_SESSION:
                    return await self.check_stt_minute_limit(session_id)
        else:
            stt_result = await self.check_stt_minute_limit(session_id)
            if not stt_result.allowed:
                return stt_result
        
        return SafetyCheckResult(allowed=True)
    
    async def check_rate_limit(self, user_id: str) -> SafetyCheckResult:
        """Check rate limit for user (10 requests/minute).
        
        Args:
            user_id: User identifier (can be session_id or IP address)
            
        Returns:
            SafetyCheckResult indicating if allowed, with retry_after if exceeded
        """
        now = time.time()
        async with self._rate_limit_lock:
            window = self._rate_limit_windows[user_id]
            # Remove requests older than 1 minute
            while window and (now - window[0]) > 60:
                window.popleft()
            
            if len(window) >= self.RATE_LIMIT_REQUESTS_PER_MINUTE:
                oldest_request = window[0] if window else now
                retry_after = max(0, 60 - (now - oldest_request))
                
                self.structured_logger.warning(
                    "limit_violation",
                    "Rate limit exceeded",
                    {
                        "user_id": user_id,
                        "requests_in_window": len(window),
                        "max_requests": self.RATE_LIMIT_REQUESTS_PER_MINUTE,
                        "retry_after": round(retry_after, 2),
                    },
                )
                
                return SafetyCheckResult(
                    allowed=False,
                    reason="rate_limit_exceeded",
                    message=(
                        f"Rate limit exceeded: {len(window)} requests in last minute. "
                        f"Please wait {retry_after:.0f} seconds."
                    ),
                    retry_after=retry_after,
                )
            
            window.append(now)
            return SafetyCheckResult(allowed=True)
    
    async def enforce_limits(self, session_id: str, user_id: Optional[str] = None) -> SafetyCheckResult:
        """Enforce all limits and return result.
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier for rate limiting (defaults to session_id)
            
        Returns:
            SafetyCheckResult indicating if all limits are satisfied
        """
        # Check session limits
        session_result = await self.check_session_limits(session_id)
        if not session_result.allowed:
            return session_result
        
        # Check cost limits
        cost_result = await self.check_cost_limits(session_id)
        if not cost_result.allowed:
            return cost_result
        
        # Check rate limit (use session_id as user_id if not provided)
        rate_result = await self.check_rate_limit(user_id or session_id)
        if not rate_result.allowed:
            return rate_result
        
        return SafetyCheckResult(allowed=True)
    
    async def _handle_stuck_session(self, session_id: str) -> None:
        """Handle stuck session (inactivity timeout).
        
        Args:
            session_id: Session identifier
        """
        try:
            session = await self.session_manager.get_session(session_id)
            if not session:
                return
            
            self.structured_logger.warning(
                "stuck_session_detected",
                "Stuck session detected and terminated",
                {
                    "session_id": session_id,
                    "inactivity_minutes": self.STUCK_SESSION_TIMEOUT_MINUTES,
                },
                session_id=session_id,
            )
            
            # Send notification to user
            try:
                from routes.websocket_routes import connection_manager
                await connection_manager.send_control_message(
                    session_id,
                    {
                        "type": "SESSION_TIMEOUT",
                        "message": (
                            "Your session has been inactive for too long. "
                            "The interview session has ended."
                        ),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to send timeout notification for {session_id}: {e}")
            
            # Optionally terminate session
            # For now, just log - actual termination can be handled by cleanup task
        except Exception as e:
            logger.error(f"Error handling stuck session {session_id}: {e}")
    
    async def _check_stuck_sessions(self) -> None:
        """Background task to detect and clean up stuck sessions."""
        while self._stuck_session_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                now = datetime.now()
                
                # Get all sessions (need to access the sessions dict directly)
                sessions = self.session_manager.sessions
                
                for session_id, session in sessions.items():
                    # Check if session has last_activity_time
                    activity_time = session.last_activity_time if hasattr(session, "last_activity_time") and session.last_activity_time else session.updated_at
                    if activity_time:
                        inactivity_minutes = (
                            (now - activity_time).total_seconds() / 60
                        )
                        if inactivity_minutes >= self.STUCK_SESSION_TIMEOUT_MINUTES:
                            await self._handle_stuck_session(session_id)
            except Exception as e:
                logger.error(f"Error in stuck session check: {e}")
    
    def start_stuck_session_detection(self) -> None:
        """Start background task to detect stuck sessions."""
        if self._stuck_session_running:
            return
        
        self._stuck_session_running = True
        self._stuck_session_task = asyncio.create_task(self._check_stuck_sessions())
        logger.info("Started stuck session detection")
    
    def stop_stuck_session_detection(self) -> None:
        """Stop stuck session detection."""
        self._stuck_session_running = False
        if self._stuck_session_task:
            self._stuck_session_task.cancel()
            self._stuck_session_task = None
        logger.info("Stopped stuck session detection")
    
    async def get_session_costs(self, session_id: str) -> Dict[str, float]:
        """Get cost metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with cost metrics
        """
        async with self._cost_lock:
            return {
                "llm_tokens": self._session_tokens.get(session_id, 0),
                "stt_minutes": self._session_stt_minutes.get(session_id, 0.0),
            }
    
    async def reset_session_costs(self, session_id: str) -> None:
        """Reset cost tracking for a session (for testing or cleanup).
        
        Args:
            session_id: Session identifier
        """
        async with self._cost_lock:
            if session_id in self._session_tokens:
                del self._session_tokens[session_id]
            if session_id in self._session_stt_minutes:
                del self._session_stt_minutes[session_id]

