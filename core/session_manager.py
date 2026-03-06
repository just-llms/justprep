"""Session manager for in-memory session state."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from models import SessionState, InterviewPhase, SessionNotFoundError
from util.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Global singleton instance
_session_manager_instance: Optional['SessionManager'] = None


class SessionManager:
    """Manages interview session state in memory.
    
    This class handles creation, retrieval, update, and deletion
    of interview sessions. All state is stored in memory.
    
    Uses singleton pattern to ensure all components share the same instance.
    """

    def __init__(self) -> None:
        """Initialize SessionManager with empty sessions dict."""
        self.sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self.metrics = MetricsCollector.get_instance()
    
    @classmethod
    def get_instance(cls) -> 'SessionManager':
        """Get the global singleton SessionManager instance.
        
        Returns:
            The global SessionManager instance
        """
        global _session_manager_instance
        if _session_manager_instance is None:
            _session_manager_instance = cls()
            logger.debug("Created global SessionManager singleton instance")
        return _session_manager_instance

    def _generate_session_id(self) -> str:
        """Generate a unique session ID using UUID v4.
        
        Returns:
            Unique session identifier
        """
        return str(uuid.uuid4())

    async def create_session(
        self,
        session_id: Optional[str] = None,
        work_experience: Optional[str] = None,
        current_role: Optional[str] = None,
        target_role: Optional[str] = None,
        resume: Optional[str] = None,
        interview_points: Optional[list[str]] = None,
        years_of_experience: Optional[float] = None,
        candidate_name: Optional[str] = None,
        current_company: Optional[str] = None,
    ) -> SessionState:
        """Create a new interview session.
        
        Args:
            session_id: Optional session ID. If not provided, generates a new UUID.
            work_experience: Candidate's work experience summary (required)
            current_role: Candidate's current job role (required)
            target_role: Role candidate is applying for (optional)
            resume: Resume text content (required)
            interview_points: Key discussion points from resume (optional)
            years_of_experience: Total years of professional experience (optional)
            candidate_name: Candidate's full name (optional)
            current_company: Candidate's current employer (optional)
            
        Returns:
            Newly created SessionState instance
            
        Raises:
            ValueError: If session_id is provided and already exists, or if required fields are missing
        """
        # Validate required fields
        if work_experience is None or not work_experience.strip():
            raise ValueError("work_experience is required")
        if current_role is None or not current_role.strip():
            raise ValueError("current_role is required")
        if resume is None or not resume.strip():
            raise ValueError("resume is required")
        
        async with self._lock:
            if session_id is None:
                session_id = self._generate_session_id()
            elif session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            now = datetime.now()
            session = SessionState(
                session_id=session_id,
                phase=InterviewPhase.GREETING,
                current_question=None,
                follow_up_count=0,
                turn_history=[],
                created_at=now,
                updated_at=now,
                total_turns=0,
                session_start_time=now,
                work_experience=work_experience,
                current_role=current_role,
                target_role=target_role,
                resume=resume,
                interview_points=interview_points or [],
                years_of_experience=years_of_experience,
                candidate_name=candidate_name,
                current_company=current_company,
                initial_greeting_sent=False,
                last_activity_time=now,
            )
            
            self.sessions[session_id] = session
            # #region agent log
            import json
            import time
            with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_session_created","timestamp":int(time.time()*1000),"location":"session_manager.py:65","message":"Session created in SessionManager","data":{"sessionId":session_id,"instanceId":str(id(self)),"sessionsCount":len(self.sessions),"sessionIds":list(self.sessions.keys())},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
            # #endregion
            logger.info(f"Created new session {session_id} with phase {InterviewPhase.GREETING.value}")
            return session

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            SessionState if found, None otherwise
        """
        async with self._lock:
            return self.sessions.get(session_id)

    async def get_session_or_raise(self, session_id: str) -> SessionState:
        """Get a session by ID, raising exception if not found.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            SessionState instance
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        # #region agent log
        import json
        import time
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_get_or_raise","timestamp":int(time.time()*1000),"location":"session_manager.py:81","message":"get_session_or_raise called","data":{"sessionId":session_id,"instanceId":str(id(self)),"sessionsCount":len(self.sessions),"sessionIds":list(self.sessions.keys())},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
        # #endregion
        session = await self.get_session(session_id)
        if session is None:
            # #region agent log
            with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_not_found","timestamp":int(time.time()*1000),"location":"session_manager.py:95","message":"Session not found - raising SessionNotFoundError","data":{"sessionId":session_id,"instanceId":str(id(self)),"sessionsCount":len(self.sessions),"sessionIds":list(self.sessions.keys())},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
            # #endregion
            raise SessionNotFoundError(session_id=session_id)
        return session

    async def update_session(
        self, session_id: str, updates: Dict[str, Any]
    ) -> SessionState:
        """Update a session with new data.
        
        Args:
            session_id: Unique identifier for the session
            updates: Dictionary of fields to update
            
        Returns:
            Updated SessionState instance
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        async with self._lock:
            if session_id not in self.sessions:
                raise SessionNotFoundError(session_id=session_id)
            
            session = self.sessions[session_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
                else:
                    logger.warning(f"Attempted to update unknown field {key} in session {session_id}")
            
            # Always update updated_at timestamp
            now = datetime.now()
            session.updated_at = now
            # Update last_activity_time on any session update
            session.last_activity_time = now
            
            logger.debug(f"Updated session {session_id}: {list(updates.keys())}")
            return session

    async def delete_session(self, session_id: str) -> None:
        """Delete a session.
        
        Args:
            session_id: Unique identifier for the session
        """
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                # Update active sessions metric
                self.metrics.set_gauge("system.active_sessions", len(self.sessions))
                
                # Clean up cost tracking
                try:
                    from core.safety import SafetyController
                    safety_controller = SafetyController.get_instance()
                    await safety_controller.reset_session_costs(session_id)
                except Exception as e:
                    logger.debug(f"Error resetting session costs: {e}")
                
                logger.info(f"Deleted session {session_id}")

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            True if session exists, False otherwise
        """
        async with self._lock:
            return session_id in self.sessions

    def get_all_sessions(self) -> Dict[str, SessionState]:
        """Get all active sessions (for debugging).
        
        Returns:
            Dictionary of all session_id -> SessionState mappings
        """
        return self.sessions.copy()

