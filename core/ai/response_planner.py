"""Response planner for validating LLM outputs and enforcing rules.

This module implements the response planner that validates LLM outputs,
enforces FSM rules, applies safety limits, and generates speakable text.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

from models.constants import InterviewPhase, LLMAction, PlannerOutputType
from models.exceptions import InvalidActionError, MaxFollowUpsExceededError
from models.response_models import LLMResponse, PlannerOutput
from core.session_manager import SessionManager
from core.ai.fsm_controller import FSMController
from util.logger import get_planner_logger

logger = logging.getLogger(__name__)


class ResponsePlanner:
    """Response planner for validating LLM outputs and enforcing rules.
    
    This class acts as a critical safety layer between the LLM and the
    rest of the system. It validates LLM outputs, enforces FSM rules,
    applies safety limits, and generates speakable text.
    
    The response planner:
    - Validates LLM response structure
    - Enforces FSM rules (action allowed in phase, max follow-ups)
    - Applies safety limits (question length, etc.)
    - Generates speakable text from actions
    - Updates session state
    - Handles phase transitions
    - Logs all overrides and fallbacks
    
    Attributes:
        session_manager: Reference to SessionManager for state access
        fsm_controller: Reference to FSMController for rule validation
        MAX_FOLLOW_UPS: Maximum follow-ups per question
        MAX_QUESTION_LENGTH: Maximum question text length
        MIN_QUESTION_LENGTH: Minimum question text length
    """

    # Configuration constants
    MAX_FOLLOW_UPS: int = 3
    MAX_QUESTION_LENGTH: int = 500
    MIN_QUESTION_LENGTH: int = 10

    def __init__(
        self,
        session_manager: SessionManager,
        fsm_controller: FSMController,
    ) -> None:
        """Initialize ResponsePlanner with dependencies.
        
        Args:
            session_manager: SessionManager instance for state access
            fsm_controller: FSMController instance for rule validation
        """
        self.session_manager = session_manager
        self.fsm_controller = fsm_controller
        self.structured_logger = get_planner_logger()
        logger.debug("ResponsePlanner initialized")

    def _validate_llm_response(self, llm_response: LLMResponse) -> None:
        """Validate LLM response structure.
        
        Double-checks that the LLM response is valid, even though
        LLMEngine already validates it. This provides an extra
        safety layer.
        
        Args:
            llm_response: LLMResponse to validate
            
        Raises:
            ValueError: If response is invalid
        """
        # Action should already be validated by LLMEngine, but check anyway
        if not isinstance(llm_response.action, LLMAction):
            raise ValueError(f"Invalid action type: {type(llm_response.action)}")
        
        # Check question is provided for actions that require it
        requires_question = llm_response.action in [
            LLMAction.ASK_FOLLOW_UP,
            LLMAction.NEXT_QUESTION,
            LLMAction.CLARIFY,
            LLMAction.REPEAT_QUESTION,
        ]
        
        if requires_question and (not llm_response.question or not llm_response.question.strip()):
            raise ValueError(
                f"Question is required for action {llm_response.action.value}"
            )
        
        logger.debug(f"LLM response validated: action={llm_response.action.value}")

    async def _check_action_allowed(
        self, session_id: str, action: LLMAction
    ) -> bool:
        """Check if action is allowed in current phase.
        
        Args:
            session_id: Unique identifier for the session
            action: The action to check
            
        Returns:
            True if action is allowed, False otherwise
        """
        try:
            await self.fsm_controller.validate_action(session_id, action)
            return True
        except InvalidActionError:
            return False

    async def _check_follow_up_limit(self, session_id: str) -> bool:
        """Check if max follow-ups limit is not exceeded.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            True if limit not exceeded, False if exceeded
        """
        session = await self.session_manager.get_session_or_raise(session_id)
        follow_up_count = session.follow_up_count
        
        is_within_limit = follow_up_count < self.MAX_FOLLOW_UPS
        
        if not is_within_limit:
            logger.warning(
                f"Max follow-ups exceeded for session {session_id}: "
                f"{follow_up_count} >= {self.MAX_FOLLOW_UPS}"
            )
        
        return is_within_limit

    async def _should_auto_transition(
        self, session_id: str, action: LLMAction
    ) -> bool:
        """Check if phase should auto-transition based on heuristics.
        
        Args:
            session_id: Unique identifier for the session
            action: The action being taken
            
        Returns:
            True if phase should auto-transition, False otherwise
        """
        if action != LLMAction.NEXT_QUESTION:
            return False
        
        session = await self.session_manager.get_session_or_raise(session_id)
        phase = session.phase
        questions_in_phase = session.questions_in_current_phase
        
        # Auto-transition rules based on phase and question count
        auto_transition_rules = {
            InterviewPhase.GREETING: {
                "min_questions": 1,  # After first exchange (greeting + response)
                "max_questions": 1,  # Don't stay in greeting phase
            },
            InterviewPhase.SMALL_TALK: {
                "min_questions": 2,  # At least 2 exchanges for rapport
                "max_questions": 3,  # Don't drag on too long
            },
            InterviewPhase.RESUME_DISCUSSION: {
                "min_questions": 2,  # Discuss resume/responsibilities
                "max_questions": 3,
            },
            InterviewPhase.INTRODUCTION: {
                "min_questions": 1,
                "max_questions": 2,
                # Transition after good introduction (no follow-ups needed)
                "require_no_followups": True,
            },
            InterviewPhase.WARMUP: {
                "min_questions": 2,
                "max_questions": 3,
            },
            InterviewPhase.TECHNICAL: {
                "min_questions": 4,
                "max_questions": 6,
            },
            InterviewPhase.BEHAVIORAL: {
                "min_questions": 2,
                "max_questions": 3,
            },
        }
        
        rule = auto_transition_rules.get(phase)
        if not rule:
            return False  # No auto-transition for this phase
        
        # Check if we've reached minimum questions
        if questions_in_phase < rule.get("min_questions", 0):
            return False
        
        # Check if we've exceeded maximum questions
        if questions_in_phase >= rule.get("max_questions", float("inf")):
            return True
        
        # For INTRODUCTION phase, also check if no follow-ups were needed
        if phase == InterviewPhase.INTRODUCTION and rule.get("require_no_followups"):
            if session.follow_up_count == 0 and questions_in_phase >= rule["min_questions"]:
                return True
        
        # Conservative approach: only auto-transition if we've reached max
        # This prevents premature transitions
        return False

    def _apply_question_limits(self, question: str) -> str:
        """Apply question length limits.
        
        Truncates question if it exceeds max length and validates
        minimum length.
        
        Args:
            question: Question text to validate
            
        Returns:
            Validated and trimmed question text
        """
        if not question:
            return question
        
        # Trim whitespace
        question = question.strip()
        
        # Truncate if too long
        if len(question) > self.MAX_QUESTION_LENGTH:
            logger.warning(
                f"Question truncated from {len(question)} to {self.MAX_QUESTION_LENGTH} characters"
            )
            question = question[: self.MAX_QUESTION_LENGTH].rstrip()
        
        # Check minimum length (but allow if empty for actions that don't need questions)
        if question and len(question) < self.MIN_QUESTION_LENGTH:
            logger.warning(
                f"Question too short ({len(question)} chars), but allowing it"
            )
        
        return question

    def _generate_speakable_text(
        self,
        action: LLMAction,
        question: Optional[str],
        current_question: Optional[str],
    ) -> Optional[str]:
        """Generate text to be spoken based on action.
        
        Args:
            action: The action being taken
            question: Question text from LLM (if provided)
            current_question: Current question from session state
            
        Returns:
            Text string to be spoken, or None for SILENT actions
        """
        if action == LLMAction.ASK_FOLLOW_UP:
            return question if question else "Can you tell me more about that?"
        
        if action == LLMAction.NEXT_QUESTION:
            return question if question else "Let's move on to the next question."
        
        if action == LLMAction.CLARIFY:
            return question if question else "Could you clarify that?"
        
        if action == LLMAction.REPEAT_QUESTION:
            # Prefer provided question, fallback to current question
            if question:
                return question
            if current_question:
                return current_question
            return "Could you repeat that?"
        
        if action == LLMAction.ACKNOWLEDGE:
            # Generate simple acknowledgment
            acknowledgments = ["I see.", "Got it.", "Understood.", "Okay."]
            return acknowledgments[0]  # Simple default
        
        if action == LLMAction.END_PHASE:
            return "Let's move on to the next phase."
        
        if action == LLMAction.END_INTERVIEW:
            return "Thank you for your time. The interview is now complete."
        
        # Default fallback
        return None

    def _determine_output_type(self, action: LLMAction) -> PlannerOutputType:
        """Determine if output should be SPEAK or SILENT.
        
        Args:
            action: The action being taken
            
        Returns:
            PlannerOutputType (SPEAK or SILENT)
        """
        # Most actions result in speech
        # Only internal/system actions might be SILENT
        # For now, all actions result in SPEAK
        return PlannerOutputType.SPEAK

    async def _handle_fallback(
        self, session_id: str, reason: str, original_action: LLMAction
    ) -> PlannerOutput:
        """Create fallback response when LLM output is invalid.
        
        Args:
            session_id: Unique identifier for the session
            reason: Reason for fallback
            original_action: Original action from LLM
            
        Returns:
            PlannerOutput with safe fallback response
        """
        logger.warning(
            f"Using fallback response for session {session_id}: {reason}"
        )
        
        # Use ACKNOWLEDGE as safe fallback
        fallback_action = LLMAction.ACKNOWLEDGE
        fallback_text = self._generate_speakable_text(
            fallback_action, None, None
        )
        
        session = await self.session_manager.get_session_or_raise(session_id)
        
        return PlannerOutput(
            type=PlannerOutputType.SPEAK,
            text=fallback_text,
            action=fallback_action,
            metadata={
                "phase": session.phase.value,
                "follow_up_count": session.follow_up_count,
            },
            was_overridden=True,
            override_reason=f"Fallback: {reason}",
        )

    async def _handle_action_override(
        self,
        session_id: str,
        original_action: LLMAction,
        new_action: LLMAction,
        reason: str,
    ) -> PlannerOutput:
        """Handle case where action is not allowed and override is needed.
        
        Args:
            session_id: Unique identifier for the session
            original_action: Original action from LLM
            new_action: Alternative action to use
            reason: Reason for override
            
        Returns:
            PlannerOutput with overridden action
        """
        logger.warning(
            f"Action override for session {session_id}: "
            f"{original_action.value} -> {new_action.value}, reason: {reason}"
        )
        
        session = await self.session_manager.get_session_or_raise(session_id)
        
        # Generate text for new action
        text = self._generate_speakable_text(
            new_action, None, session.current_question
        )
        
        return PlannerOutput(
            type=self._determine_output_type(new_action),
            text=text,
            action=new_action,
            metadata={
                "phase": session.phase.value,
                "follow_up_count": session.follow_up_count,
            },
            was_overridden=True,
            override_reason=reason,
        )

    async def _update_session_state(
        self, session_id: str, action: LLMAction, question: Optional[str]
    ) -> None:
        """Update session state based on action.
        
        Args:
            session_id: Unique identifier for the session
            action: The action being taken
            question: Question text (if applicable)
        """
        session = await self.session_manager.get_session_or_raise(session_id)
        updates: Dict[str, Any] = {}
        
        # Handle follow-up count
        if action == LLMAction.ASK_FOLLOW_UP:
            updates["follow_up_count"] = session.follow_up_count + 1
        elif action in [LLMAction.NEXT_QUESTION, LLMAction.END_PHASE]:
            updates["follow_up_count"] = 0
        
        # Handle question progression tracking
        if action == LLMAction.NEXT_QUESTION:
            updates["questions_in_current_phase"] = session.questions_in_current_phase + 1
        elif action == LLMAction.END_PHASE:
            updates["questions_in_current_phase"] = 0  # Reset on phase transition
        
        # Handle current question update
        question_actions = [
            LLMAction.ASK_FOLLOW_UP,
            LLMAction.NEXT_QUESTION,
            LLMAction.CLARIFY,
            LLMAction.REPEAT_QUESTION,
        ]
        if action in question_actions and question:
            updates["current_question"] = question
        
        # Apply updates if any
        if updates:
            await self.session_manager.update_session(session_id, updates)
            logger.debug(
                f"Updated session state for {session_id}: {list(updates.keys())}"
            )

    def _get_next_phase(
        self, current_phase: InterviewPhase, session_id: Optional[str] = None
    ) -> Optional[InterviewPhase]:
        """Get next phase for END_PHASE action.
        
        Args:
            current_phase: Current interview phase
            session_id: Optional session ID to check for BEHAVIORAL phase configuration
            
        Returns:
            Next phase if valid transition exists, None otherwise
        """
        valid_next_phases = self.fsm_controller.PHASE_TRANSITIONS.get(
            current_phase, set()
        )
        
        if not valid_next_phases:
            return None
        
        # If multiple options, prefer the first one (or could use logic)
        # For TECHNICAL phase, prefer BEHAVIORAL if available, else CLOSING
        # TODO: Could add session-level config for BEHAVIORAL phase (enabled/disabled)
        if current_phase == InterviewPhase.TECHNICAL:
            if InterviewPhase.BEHAVIORAL in valid_next_phases:
                # For now, always prefer BEHAVIORAL if available
                # In future, could check session config
                return InterviewPhase.BEHAVIORAL
            return InterviewPhase.CLOSING
        
        # For other phases, return the first (and usually only) option
        return next(iter(valid_next_phases))

    async def plan_response(
        self, session_id: str, llm_response: LLMResponse
    ) -> PlannerOutput:
        """Plan response from LLM output.
        
        Main method that validates LLM response, enforces FSM rules,
        applies limits, generates speakable text, and updates session state.
        
        Args:
            session_id: Unique identifier for the session
            llm_response: LLMResponse from LLM engine
            
        Returns:
            PlannerOutput with validated and planned response
        """
        logger.debug(
            f"Planning response for session {session_id}, "
            f"action={llm_response.action.value}"
        )
        
        # Step 1: Validate LLM response structure
        try:
            self._validate_llm_response(llm_response)
        except ValueError as e:
            logger.warning(
                f"LLM response validation failed for session {session_id}: {e}"
            )
            return await self._handle_fallback(
                session_id, f"Invalid LLM response: {e}", llm_response.action
            )
        
        # Step 2: Check FSM rules - action allowed in current phase?
        action_allowed = await self._check_action_allowed(
            session_id, llm_response.action
        )
        if not action_allowed:
            # Choose alternative action
            session = await self.session_manager.get_session_or_raise(session_id)
            allowed_actions = self.fsm_controller.get_allowed_actions(
                session.phase
            )
            
            # Prefer NEXT_QUESTION as safe alternative
            if LLMAction.NEXT_QUESTION in allowed_actions:
                alternative_action = LLMAction.NEXT_QUESTION
            elif allowed_actions:
                alternative_action = next(iter(allowed_actions))
            else:
                alternative_action = LLMAction.ACKNOWLEDGE
            
            # Log override
            try:
                import asyncio
                from core.memory import MemoryManager, create_log_entry, LogEntryType
                memory_manager = MemoryManager.get_instance_sync()
                if memory_manager:
                    log_entry = create_log_entry(
                        LogEntryType.PLANNER_DECISION,
                        session_id,
                        {
                            "event": "action_override",
                            "original_action": llm_response.action.value,
                            "new_action": alternative_action.value,
                            "reason": f"Action {llm_response.action.value} not allowed in phase {session.phase.value}",
                        },
                    )
                    asyncio.create_task(
                        memory_manager.conversation_log.append_log(log_entry)
                    )
            except Exception:
                pass  # Don't fail on logging errors
            
            self.structured_logger.warning(
                "planner_action_override",
                "Action not allowed in phase, using alternative",
                {
                    "original_action": llm_response.action.value,
                    "alternative_action": alternative_action.value,
                    "phase": session.phase.value
                },
                session_id=session_id
            )
            return await self._handle_action_override(
                session_id,
                llm_response.action,
                alternative_action,
                f"Action {llm_response.action.value} not allowed in phase {session.phase.value}",
            )
        
        # Step 3: Check limits - max follow-ups
        if llm_response.action == LLMAction.ASK_FOLLOW_UP:
            follow_up_limit_ok = await self._check_follow_up_limit(session_id)
            if not follow_up_limit_ok:
                # Override to NEXT_QUESTION
                session = await self.session_manager.get_session_or_raise(
                    session_id
                )
                
                # Log override
                try:
                    import asyncio
                    from core.memory import MemoryManager, create_log_entry, LogEntryType
                    memory_manager = MemoryManager.get_instance_sync()
                    if memory_manager:
                        log_entry = create_log_entry(
                            LogEntryType.PLANNER_DECISION,
                            session_id,
                            {
                                "event": "follow_up_limit_exceeded",
                                "original_action": llm_response.action.value,
                                "new_action": LLMAction.NEXT_QUESTION.value,
                                "reason": f"Max follow-ups ({self.MAX_FOLLOW_UPS}) exceeded",
                            },
                        )
                        asyncio.create_task(
                            memory_manager.conversation_log.append_log(log_entry)
                        )
                except Exception:
                    pass  # Don't fail on logging errors
                
                return await self._handle_action_override(
                    session_id,
                    llm_response.action,
                    LLMAction.NEXT_QUESTION,
                    f"Max follow-ups ({self.MAX_FOLLOW_UPS}) exceeded",
                )
        
        # Step 4: Apply question limits
        question = llm_response.question
        if question:
            question = self._apply_question_limits(question)
        
        # Step 5: Get current question for text generation
        session = await self.session_manager.get_session_or_raise(session_id)
        current_question = session.current_question
        
        # Step 6: Generate speakable text
        text = self._generate_speakable_text(
            llm_response.action, question, current_question
        )
        
        # Step 7: Determine output type
        output_type = self._determine_output_type(llm_response.action)
        
        # Step 8: Handle automatic phase transitions
        # Check if we should auto-transition before handling manual END_PHASE
        if llm_response.action == LLMAction.NEXT_QUESTION:
            should_auto_transition = await self._should_auto_transition(session_id, llm_response.action)
            if should_auto_transition:
                logger.info(
                    f"Auto-transitioning phase for session {session_id} "
                    f"from {session.phase.value} after {session.questions_in_current_phase} questions"
                )
                # Override action to END_PHASE for auto-transition
                llm_response.action = LLMAction.END_PHASE
        
        # Step 9: Handle phase transitions (manual or auto)
        if llm_response.action == LLMAction.END_PHASE:
            current_phase = session.phase
            next_phase = self._get_next_phase(current_phase, session_id)
            if next_phase:
                try:
                    # Get recent turns from current phase before transition
                    recent_turns = [
                        turn for turn in session.turn_history
                        if turn.phase == current_phase
                    ]
                    
                    await self.fsm_controller.transition_to_phase(
                        session_id, next_phase
                    )
                    logger.info(
                        f"Phase transition for session {session_id}: "
                        f"{current_phase.value} -> {next_phase.value}"
                    )
                    
                    # Trigger long-term memory phase summary (non-blocking)
                    try:
                        import asyncio
                        from core.memory import MemoryManager
                        memory_manager = MemoryManager.get_instance_sync()
                        if memory_manager and memory_manager.long_term_memory:
                            asyncio.create_task(
                                memory_manager.long_term_memory.trigger_phase_summary(
                                    session_id, current_phase, recent_turns
                                )
                            )
                    except Exception as e:
                        logger.debug(f"Failed to trigger phase summary: {e}")
                except Exception as e:
                    logger.error(
                        f"Failed to transition phase for session {session_id}: {e}"
                    )
            else:
                logger.warning(
                    f"No valid next phase for session {session_id} "
                    f"in phase {session.phase.value}"
                )
        
        if llm_response.action == LLMAction.END_INTERVIEW:
            try:
                await self.fsm_controller.transition_to_phase(
                    session_id, InterviewPhase.ENDED
                )
                logger.info(
                    f"Interview ended for session {session_id}"
                )
                # Send completion notification (non-blocking)
                try:
                    from routes.websocket_routes import connection_manager
                    completion_message = {
                        "type": "INTERVIEW_COMPLETE",
                        "message": "The interview has been completed. Thank you for your time!",
                        "timestamp": datetime.now().isoformat()
                    }
                    asyncio.create_task(
                        connection_manager.send_control_message(session_id, completion_message)
                    )
                except Exception as e:
                    logger.warning(f"Failed to send completion notification: {e}")
            except Exception as e:
                logger.error(
                    f"Failed to end interview for session {session_id}: {e}"
                )
        
        # Step 10: Update session state
        await self._update_session_state(
            session_id, llm_response.action, question
        )
        
        # Step 11: Get updated session for metadata
        updated_session = await self.session_manager.get_session_or_raise(
            session_id
        )
        
        # Step 12: Create and return PlannerOutput
        planner_output = PlannerOutput(
            type=output_type,
            text=text,
            action=llm_response.action,
            metadata={
                "phase": updated_session.phase.value,
                "follow_up_count": updated_session.follow_up_count,
            },
            was_overridden=False,
            override_reason=None,
        )
        
        logger.info(
            f"Response planned for session {session_id}: "
            f"action={llm_response.action.value}, "
            f"type={output_type.value}, "
            f"text_length={len(text) if text else 0}"
        )
        
        # Log planner decision (if not already logged)
        try:
            import asyncio
            from core.memory import MemoryManager, create_log_entry, LogEntryType
            memory_manager = MemoryManager.get_instance_sync()
            if memory_manager:
                log_entry = create_log_entry(
                    LogEntryType.PLANNER_DECISION,
                    session_id,
                    {
                        "event": "response_planned",
                        "action": planner_output.action.value,
                        "type": planner_output.type.value,
                        "was_overridden": planner_output.was_overridden,
                        "override_reason": planner_output.override_reason,
                    },
                )
                asyncio.create_task(
                    memory_manager.conversation_log.append_log(log_entry)
                )
        except Exception:
            pass  # Don't fail on logging errors
        
        return planner_output

