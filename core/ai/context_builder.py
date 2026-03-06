"""Inference context builder for LLM prompts.

This module implements the context builder that gathers minimal context
from session state and formats it for LLM inference.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from models.constants import InterviewPhase
from models.exceptions import SessionNotFoundError
from models.session_models import Turn
from core.session_manager import SessionManager
from core.ai.fsm_controller import FSMController

if TYPE_CHECKING:
    from core.memory import MemoryManager

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds minimal context for LLM inference.
    
    This class gathers essential information from session state and
    formats it efficiently for LLM prompts. It maintains a minimal
    context by using a sliding window of recent turns.
    
    The context builder:
    - Gathers current phase, question, and recent turns
    - Formats context as structured text for LLM
    - Caches context where possible
    - Keeps context minimal (no full transcript)
    
    Attributes:
        session_manager: Reference to SessionManager for state access
        fsm_controller: Reference to FSMController for phase access
        _context_cache: Optional cache for built contexts
        DEFAULT_SLIDING_WINDOW_SIZE: Default number of recent turns to include
    """

    # Default sliding window size for recent turns
    DEFAULT_SLIDING_WINDOW_SIZE: int = 5

    def __init__(
        self,
        session_manager: SessionManager,
        fsm_controller: FSMController,
        memory_manager: Optional["MemoryManager"] = None,
    ) -> None:
        """Initialize ContextBuilder with dependencies.
        
        Args:
            session_manager: SessionManager instance for state access
            fsm_controller: FSMController instance for phase access
            memory_manager: Optional MemoryManager instance for short-term memory access
        """
        self.session_manager = session_manager
        self.fsm_controller = fsm_controller
        self.memory_manager = memory_manager
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        logger.debug("ContextBuilder initialized")

    async def _get_current_phase(self, session_id: str) -> InterviewPhase:
        """Get current phase for a session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Current InterviewPhase for the session
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        phase = await self.fsm_controller.get_current_phase(session_id)
        logger.debug(f"Retrieved current phase {phase.value} for session {session_id}")
        return phase

    async def _get_current_question(self, session_id: str) -> Optional[str]:
        """Get current question from session state.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Current question string, or None if no question set
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        session = await self.session_manager.get_session_or_raise(session_id)
        current_question = session.current_question
        logger.debug(
            f"Retrieved current question for session {session_id}: "
            f"{'None' if current_question is None else 'Set'}"
        )
        return current_question

    async def _get_recent_turns(
        self, session_id: str, n: int = DEFAULT_SLIDING_WINDOW_SIZE
    ) -> List[Turn]:
        """Get last N turns from short-term memory or session's turn history.
        
        Implements sliding window logic to get the most recent turns.
        Prefers short-term memory if available, falls back to session.turn_history.
        
        Args:
            session_id: Unique identifier for the session
            n: Number of recent turns to retrieve (default: 5)
            
        Returns:
            List of Turn objects, most recent first (last N in history).
            Returns empty list if turn_history is empty or session not found.
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        # Try to use short-term memory first if available
        if self.memory_manager is not None:
            try:
                recent_turns = await self.memory_manager.short_term_memory.get_recent_turns(
                    session_id, n
                )
                if recent_turns:
                    logger.debug(
                        f"Retrieved {len(recent_turns)} recent turns from short-term memory "
                        f"for session {session_id} (requested {n})"
                    )
                    return recent_turns
            except Exception as e:
                logger.warning(
                    f"Error retrieving turns from short-term memory for session {session_id}: {e}. "
                    f"Falling back to session.turn_history"
                )
        
        # Fallback to session.turn_history
        session = await self.session_manager.get_session_or_raise(session_id)
        turn_history = session.turn_history
        
        # Get last N turns using list slicing
        # If turn_history has fewer than N turns, return all available
        recent_turns = turn_history[-n:] if len(turn_history) > 0 else []
        
        logger.debug(
            f"Retrieved {len(recent_turns)} recent turns from session.turn_history "
            f"for session {session_id} (requested {n}, available {len(turn_history)})"
        )
        
        return recent_turns

    def _format_context_for_llm(self, context_data: Dict[str, Any]) -> str:
        """Format context dictionary into text string for LLM prompt.
        
        Creates a readable, structured text format that includes all
        context components.
        
        Args:
            context_data: Dictionary containing context components:
                - phase: InterviewPhase
                - current_question: Optional[str]
                - recent_turns: List[Turn]
                - current_answer: str
                - follow_up_count: int
                - candidate_info: Optional[Dict] with work_experience, current_role, target_role, resume
                
        Returns:
            Formatted text string ready for LLM prompt
        """
        phase = context_data.get("phase")
        current_question = context_data.get("current_question")
        recent_turns = context_data.get("recent_turns", [])
        current_answer = context_data.get("current_answer", "")
        follow_up_count = context_data.get("follow_up_count", 0)
        candidate_info = context_data.get("candidate_info")
        
        # Build formatted context string
        lines = []
        
        # Add candidate profile section if available
        if candidate_info:
            lines.append("=== CANDIDATE PROFILE ===")
            if candidate_info.get("candidate_name"):
                lines.append(f"Name: {candidate_info['candidate_name']}")
            if candidate_info.get("current_role"):
                role_text = candidate_info['current_role']
                if candidate_info.get("current_company"):
                    role_text += f" at {candidate_info['current_company']}"
                lines.append(f"Current Role: {role_text}")
            if candidate_info.get("years_of_experience") is not None:
                lines.append(f"Years of Experience: {candidate_info['years_of_experience']}")
            if candidate_info.get("target_role"):
                lines.append(f"Target Role: {candidate_info['target_role']}")
            lines.append("")
            
            # Add interview points as key discussion points
            interview_points = candidate_info.get("interview_points", [])
            if interview_points:
                lines.append("KEY DISCUSSION POINTS (use these to ask specific questions):")
                for i, point in enumerate(interview_points, 1):
                    lines.append(f"  {i}. {point}")
                lines.append("")
            
            if candidate_info.get("resume"):
                # Truncate resume if too long (keep first 1500 chars for more context)
                resume_text = candidate_info['resume']
                if len(resume_text) > 1500:
                    resume_text = resume_text[:1500] + "..."
                lines.append(f"Resume Summary: {resume_text}")
            lines.append("")
        
        lines.append(f"Interview Phase: {phase.value if phase else 'Unknown'}")
        lines.append(
            f"Current Question: {current_question if current_question else 'None'}"
        )
        lines.append("")
        
        if recent_turns:
            lines.append("Recent Conversation:")
            for turn in recent_turns:
                question_text = turn.question if turn.question else "None"
                lines.append(f"  Q: {question_text}")
                lines.append(f"  A: {turn.answer}")
                lines.append(
                    f"  (Phase: {turn.phase.value}, Turn #{turn.turn_number})"
                )
                lines.append("")
            
            # Explicitly list questions already asked to prevent repetition
            asked_questions = [turn.question for turn in recent_turns if turn.question]
            if asked_questions:
                lines.append("=== DO NOT REPEAT THESE QUESTIONS ===")
                for i, q in enumerate(asked_questions, 1):
                    lines.append(f"  {i}. {q}")
                lines.append("")
        else:
            lines.append("Recent Conversation: (No previous turns)")
            lines.append("")
        
        # Handle empty user utterance for initial greeting
        if current_answer == "" or current_answer == "[INITIAL_GREETING]":
            lines.append("Candidate's Current Answer: [No answer yet - this is the initial greeting]")
        else:
            lines.append(f"Candidate's Current Answer: {current_answer}")
        lines.append("")
        lines.append(f"Follow-up Count: {follow_up_count}")
        
        formatted_context = "\n".join(lines)
        
        logger.debug(
            f"Formatted context: {len(formatted_context)} characters, "
            f"{len(recent_turns)} turns"
        )
        
        return formatted_context

    async def build_context(
        self, session_id: str, user_utterance: str
    ) -> Dict[str, Any]:
        """Build context for LLM inference.
        
        Gathers all context components and formats them for LLM prompt.
        This is the main method used by the AI Engine before calling LLM.
        
        Args:
            session_id: Unique identifier for the session
            user_utterance: The candidate's current answer/utterance
            
        Returns:
            Dictionary containing:
                - phase: InterviewPhase
                - current_question: Optional[str]
                - recent_turns: List[Turn]
                - current_answer: str (user_utterance)
                - follow_up_count: int
                - formatted_context: str (ready-to-use text for LLM)
                
        Raises:
            SessionNotFoundError: If session does not exist
        """
        logger.debug(f"Building context for session {session_id}")
        
        # Check cache first (optional optimization)
        # Cache key could include session_id + hash of relevant state
        # For now, we'll skip caching as session state changes frequently
        
        # Get session for additional context
        # #region agent log
        import json
        import time
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_build_ctx","timestamp":int(time.time()*1000),"location":"context_builder.py:215","message":"Before get_session_or_raise in context_builder","data":{"sessionId":session_id,"instanceId":str(id(self.session_manager)),"sessionsCount":len(self.session_manager.sessions),"sessionIds":list(self.session_manager.sessions.keys())},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
        # #endregion
        session = await self.session_manager.get_session_or_raise(session_id)
        
        # Gather all context components
        phase = await self._get_current_phase(session_id)
        current_question = await self._get_current_question(session_id)
        recent_turns = await self._get_recent_turns(
            session_id, self.DEFAULT_SLIDING_WINDOW_SIZE
        )
        follow_up_count = session.follow_up_count
        
        # Handle empty user utterance for initial greeting
        if not user_utterance or user_utterance.strip() == "":
            current_answer = "[INITIAL_GREETING]"
        else:
            current_answer = user_utterance
        
        # Gather candidate information including extracted resume data
        candidate_info = {
            "candidate_name": session.candidate_name,
            "current_role": session.current_role,
            "current_company": session.current_company,
            "target_role": session.target_role,
            "years_of_experience": session.years_of_experience,
            "work_experience": session.work_experience,
            "resume": session.resume,
            "interview_points": session.interview_points,
        }
        
        # Build context dictionary
        context_data = {
            "phase": phase,
            "current_question": current_question,
            "recent_turns": recent_turns,
            "current_answer": current_answer,
            "follow_up_count": follow_up_count,
            "candidate_info": candidate_info,
        }
        
        # Format context for LLM
        formatted_context = self._format_context_for_llm(context_data)
        context_data["formatted_context"] = formatted_context
        
        logger.info(
            f"Context built for session {session_id}: "
            f"phase={phase.value}, turns={len(recent_turns)}, "
            f"context_size={len(formatted_context)} chars"
        )
        
        return context_data

    def clear_cache(self, session_id: Optional[str] = None) -> None:
        """Clear context cache for a session or all sessions.
        
        Useful when session state changes significantly (e.g., new turn
        added, phase transition, question changed).
        
        Args:
            session_id: Optional session ID to clear cache for.
                       If None, clears cache for all sessions.
        """
        if session_id is None:
            self._context_cache.clear()
            logger.debug("Cleared context cache for all sessions")
        else:
            if session_id in self._context_cache:
                del self._context_cache[session_id]
                logger.debug(f"Cleared context cache for session {session_id}")
            else:
                logger.debug(
                    f"No cache entry found for session {session_id} to clear"
                )

