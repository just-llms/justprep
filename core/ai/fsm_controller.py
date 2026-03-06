"""Finite State Machine controller for interview phases.

This module implements the FSM controller that manages interview phase
transitions and validates allowed actions per phase.
"""

import logging
from typing import Dict, Set

from models.constants import InterviewPhase, LLMAction
from models.exceptions import (
    InvalidActionError,
    InvalidPhaseTransitionError,
    SessionNotFoundError,
)
from core.session_manager import SessionManager
from util.logger import get_fsm_logger

logger = logging.getLogger(__name__)


class FSMController:
    """Finite State Machine controller for interview phases.
    
    This class manages interview phase transitions and validates allowed
    actions per phase. It enforces interview flow rules and integrates
    with SessionManager for state persistence.
    
    The FSM controller acts as a rule table that:
    - Validates phase transitions
    - Determines allowed actions per phase
    - Blocks invalid transitions and actions
    - Stores phase state per session
    
    Attributes:
        session_manager: Reference to SessionManager for state persistence
        _transition_table: Dictionary mapping phases to valid next phases
        _allowed_actions: Dictionary mapping phases to allowed actions
    """

    # Phase transition rules
    # Maps each phase to the set of valid next phases
    PHASE_TRANSITIONS: Dict[InterviewPhase, Set[InterviewPhase]] = {
        InterviewPhase.GREETING: {InterviewPhase.SMALL_TALK},
        InterviewPhase.SMALL_TALK: {InterviewPhase.RESUME_DISCUSSION},
        InterviewPhase.RESUME_DISCUSSION: {InterviewPhase.INTRODUCTION},
        InterviewPhase.INTRODUCTION: {InterviewPhase.WARMUP},
        InterviewPhase.WARMUP: {InterviewPhase.TECHNICAL},
        InterviewPhase.TECHNICAL: {InterviewPhase.BEHAVIORAL, InterviewPhase.CLOSING},
        InterviewPhase.BEHAVIORAL: {InterviewPhase.CLOSING},
        InterviewPhase.CLOSING: {InterviewPhase.ENDED},
        InterviewPhase.ENDED: set(),  # Terminal state, no transitions
    }

    # Allowed actions per phase
    # Maps each phase to the set of allowed LLM actions
    ALLOWED_ACTIONS: Dict[InterviewPhase, Set[LLMAction]] = {
        InterviewPhase.GREETING: {LLMAction.NEXT_QUESTION},
        InterviewPhase.SMALL_TALK: {
            LLMAction.NEXT_QUESTION,
            LLMAction.ASK_FOLLOW_UP,
            LLMAction.ACKNOWLEDGE,
        },
        InterviewPhase.RESUME_DISCUSSION: {
            LLMAction.NEXT_QUESTION,
            LLMAction.ASK_FOLLOW_UP,
            LLMAction.CLARIFY,
        },
        InterviewPhase.INTRODUCTION: {LLMAction.NEXT_QUESTION, LLMAction.CLARIFY},
        InterviewPhase.WARMUP: {LLMAction.NEXT_QUESTION, LLMAction.CLARIFY, LLMAction.ASK_FOLLOW_UP},
        InterviewPhase.TECHNICAL: {
            LLMAction.ASK_FOLLOW_UP,
            LLMAction.NEXT_QUESTION,
            LLMAction.CLARIFY,
            LLMAction.REPEAT_QUESTION,
        },
        InterviewPhase.BEHAVIORAL: {
            LLMAction.ASK_FOLLOW_UP,
            LLMAction.NEXT_QUESTION,
            LLMAction.CLARIFY,
        },
        InterviewPhase.CLOSING: {LLMAction.NEXT_QUESTION, LLMAction.END_INTERVIEW},
        InterviewPhase.ENDED: set(),  # No actions allowed in ended state
    }

    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize FSMController with SessionManager dependency.
        
        Args:
            session_manager: SessionManager instance for state persistence
        """
        self.session_manager = session_manager
        self._transition_table = self.PHASE_TRANSITIONS.copy()
        self._allowed_actions = self.ALLOWED_ACTIONS.copy()
        self.structured_logger = get_fsm_logger()
        logger.debug("FSMController initialized")

    async def get_current_phase(self, session_id: str) -> InterviewPhase:
        """Get the current phase for a session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Current InterviewPhase for the session
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        session = await self.session_manager.get_session_or_raise(session_id)
        logger.debug(f"Retrieved current phase {session.phase.value} for session {session_id}")
        return session.phase

    def get_allowed_actions(self, phase: InterviewPhase) -> Set[LLMAction]:
        """Get the set of allowed actions for a given phase.
        
        Args:
            phase: The interview phase to check
            
        Returns:
            Set of allowed LLMAction values for the phase.
            Returns empty set if phase not found (defensive).
        """
        allowed = self._allowed_actions.get(phase, set())
        logger.debug(f"Allowed actions for phase {phase.value}: {[action.value for action in allowed]}")
        return allowed

    def can_transition(
        self, from_phase: InterviewPhase, to_phase: InterviewPhase
    ) -> bool:
        """Check if a phase transition is valid.
        
        Args:
            from_phase: The current phase
            to_phase: The target phase to transition to
            
        Returns:
            True if transition is valid according to transition table,
            False otherwise
        """
        valid_next_phases = self._transition_table.get(from_phase, set())
        is_valid = to_phase in valid_next_phases
        
        if not is_valid:
            logger.warning(
                f"Invalid transition attempted: {from_phase.value} → {to_phase.value}"
            )
        else:
            logger.debug(
                f"Valid transition: {from_phase.value} → {to_phase.value}"
            )
        
        return is_valid

    async def transition_to_phase(
        self, session_id: str, new_phase: InterviewPhase
    ) -> InterviewPhase:
        """Transition a session to a new phase.
        
        This method validates the transition and updates the session state
        if the transition is valid.
        
        Args:
            session_id: Unique identifier for the session
            new_phase: The target phase to transition to
            
        Returns:
            The new phase after successful transition
            
        Raises:
            SessionNotFoundError: If session does not exist
            InvalidPhaseTransitionError: If transition is not allowed
        """
        # Get current phase from session
        current_phase = await self.get_current_phase(session_id)
        
        # Validate transition
        if not self.can_transition(current_phase, new_phase):
            raise InvalidPhaseTransitionError(
                from_phase=current_phase.value,
                to_phase=new_phase.value,
            )
        
        # Update session phase
        await self.session_manager.update_session(
            session_id, {"phase": new_phase}
        )
        
        logger.info(
            f"Phase transition: {session_id} {current_phase.value} → {new_phase.value}"
        )
        
        # Log phase transition
        try:
            import asyncio
            from core.memory import MemoryManager, create_log_entry, LogEntryType
            memory_manager = MemoryManager.get_instance_sync()
            if memory_manager:
                log_entry = create_log_entry(
                    LogEntryType.PHASE_TRANSITION,
                    session_id,
                    {
                        "from_phase": current_phase.value,
                        "to_phase": new_phase.value,
                    },
                )
                asyncio.create_task(
                    memory_manager.conversation_log.append_log(log_entry)
                )
        except Exception as e:
            logger.debug(f"Failed to log phase transition: {e}")
        
        # Trigger long-term memory phase summary for the completed phase
        try:
            from core.memory import MemoryManager
            from models.session_models import Turn
            memory_manager = MemoryManager.get_instance_sync()
            if memory_manager and memory_manager.long_term_memory:
                # Get recent turns from the completed phase
                session = await self.session_manager.get_session_or_raise(session_id)
                recent_turns = [
                    turn for turn in session.turn_history
                    if turn.phase == current_phase
                ]
                
                # Trigger async phase summary (non-blocking)
                asyncio.create_task(
                    memory_manager.long_term_memory.trigger_phase_summary(
                        session_id, current_phase, recent_turns
                    )
                )
        except Exception as e:
            logger.debug(f"Failed to trigger phase summary: {e}")
        
        return new_phase

    async def is_action_allowed(
        self, session_id: str, action: LLMAction
    ) -> bool:
        """Check if an action is allowed in the current phase for a session.
        
        Args:
            session_id: Unique identifier for the session
            action: The LLM action to validate
            
        Returns:
            True if action is allowed, False otherwise
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        current_phase = await self.get_current_phase(session_id)
        allowed_actions = self.get_allowed_actions(current_phase)
        is_allowed = action in allowed_actions
        
        logger.debug(
            f"Action validation for session {session_id}: "
            f"action={action.value}, phase={current_phase.value}, allowed={is_allowed}"
        )
        
        return is_allowed

    async def validate_action(self, session_id: str, action: LLMAction) -> None:
        """Validate that an action is allowed in the current phase.
        
        This method is used by the Response Planner to validate LLM actions
        before they are executed.
        
        Args:
            session_id: Unique identifier for the session
            action: The LLM action to validate
            
        Raises:
            SessionNotFoundError: If session does not exist
            InvalidActionError: If action is not allowed in current phase
        """
        current_phase = await self.get_current_phase(session_id)
        
        if not await self.is_action_allowed(session_id, action):
            logger.warning(
                f"Invalid action attempted: session={session_id}, "
                f"action={action.value}, phase={current_phase.value}"
            )
            raise InvalidActionError(
                action=action.value,
                phase=current_phase.value,
            )
        
        logger.debug(
            f"Action validated: session={session_id}, "
            f"action={action.value}, phase={current_phase.value}"
        )

