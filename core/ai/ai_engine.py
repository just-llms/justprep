"""AI engine orchestration for processing user responses.

This module implements the AI engine that orchestrates all AI components
to process complete user utterances and generate planned responses.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from datetime import datetime
from models.constants import InterviewPhase, LLMAction, PlannerOutputType
from models.response_models import LLMResponse, PlannerOutput
from models.session_models import Turn
from core.session_manager import SessionManager
from core.ai.fsm_controller import FSMController
from core.ai.context_builder import ContextBuilder
from core.ai.llm_engine import LLMEngine
from core.ai.response_planner import ResponsePlanner

logger = logging.getLogger(__name__)


class AIEngine:
    """AI engine for orchestrating AI component processing.
    
    This class coordinates FSMController, ContextBuilder, LLMEngine, and
    ResponsePlanner to process complete user utterances and generate
    planned responses ready for TTS.
    
    The AI engine:
    - Orchestrates the complete processing pipeline
    - Prevents double-processing with per-session locks
    - Handles errors at each step with graceful fallbacks
    - Logs the entire processing flow
    - Emits events for memory system
    
    Attributes:
        session_manager: Reference to SessionManager for state access
        fsm_controller: Reference to FSMController for phase management
        context_builder: Reference to ContextBuilder for context building
        llm_engine: Reference to LLMEngine for LLM calls
        response_planner: Reference to ResponsePlanner for response planning
        _processing_locks: Dictionary of per-session processing locks
    """

    def __init__(
        self,
        session_manager: SessionManager,
        fsm_controller: FSMController,
        context_builder: ContextBuilder,
        llm_engine: LLMEngine,
        response_planner: ResponsePlanner,
    ) -> None:
        """Initialize AIEngine with all component dependencies.
        
        Args:
            session_manager: SessionManager instance for state access
            fsm_controller: FSMController instance for phase management
            context_builder: ContextBuilder instance for context building
            llm_engine: LLMEngine instance for LLM calls
            response_planner: ResponsePlanner instance for response planning
        """
        self.session_manager = session_manager
        self.fsm_controller = fsm_controller
        self.context_builder = context_builder
        self.llm_engine = llm_engine
        self.response_planner = response_planner
        self._processing_locks: Dict[str, asyncio.Lock] = {}
        logger.info("AIEngine initialized with all components")

    def _get_or_create_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create processing lock for a session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            asyncio.Lock for the session
        """
        if session_id not in self._processing_locks:
            self._processing_locks[session_id] = asyncio.Lock()
        return self._processing_locks[session_id]

    async def _acquire_processing_lock(self, session_id: str) -> bool:
        """Acquire processing lock for session.
        
        Attempts to acquire the lock. If lock is already held,
        returns False immediately (non-blocking).
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            True if lock acquired, False if already processing
        """
        lock = self._get_or_create_lock(session_id)
        
        # Try to acquire lock (non-blocking)
        if lock.locked():
            logger.warning(
                f"Processing already in progress for session {session_id}, "
                "skipping duplicate request"
            )
            return False
        
        await lock.acquire()
        logger.debug(f"Acquired processing lock for session {session_id}")
        return True

    async def _release_processing_lock(self, session_id: str) -> None:
        """Release processing lock for session.
        
        Args:
            session_id: Unique identifier for the session
        """
        if session_id in self._processing_locks:
            lock = self._processing_locks[session_id]
            if lock.locked():
                lock.release()
                logger.debug(f"Released processing lock for session {session_id}")

    async def _get_fsm_state(self, session_id: str) -> InterviewPhase:
        """Get current FSM state (phase) for session.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Current InterviewPhase for the session
        """
        # #region agent log
        import json
        import time
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_get_fsm","timestamp":int(time.time()*1000),"location":"ai_engine.py:122","message":"Before get_current_phase in AI engine","data":{"sessionId":session_id,"instanceId":str(id(self.session_manager)),"sessionsCount":len(self.session_manager.sessions),"sessionIds":list(self.session_manager.sessions.keys())},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
        # #endregion
        phase = await self.fsm_controller.get_current_phase(session_id)
        logger.debug(f"Retrieved FSM state for session {session_id}: {phase.value}")
        return phase

    async def _build_context(
        self, session_id: str, user_utterance: str
    ) -> Dict[str, Any]:
        """Build inference context for LLM.
        
        Args:
            session_id: Unique identifier for the session
            user_utterance: The candidate's current answer/utterance
            
        Returns:
            Context dictionary with formatted_context string
        """
        context_data = await self.context_builder.build_context(
            session_id, user_utterance
        )
        logger.debug(
            f"Built context for session {session_id}: "
            f"phase={context_data['phase'].value}, "
            f"turns={len(context_data['recent_turns'])}, "
            f"context_size={len(context_data['formatted_context'])} chars"
        )
        return context_data

    async def _call_llm(
        self, context: str, user_answer: str, phase: InterviewPhase, session_id: str
    ) -> LLMResponse:
        """Call LLM with context and user answer.
        
        Args:
            context: Formatted context string from ContextBuilder
            user_answer: The candidate's current answer/utterance
            phase: Current interview phase for phase-specific guidance
            session_id: Session identifier for token tracking
            
        Returns:
            LLMResponse with structured output from LLM
        """
        # Log LLM input
        try:
            from core.memory import MemoryManager, create_log_entry, LogEntryType
            memory_manager = MemoryManager.get_instance_sync()
            if memory_manager:
                # Get session_id from context if available, or use a placeholder
                # Note: We don't have session_id here, so we'll log it later in process_user_response
                pass
        except Exception:
            pass  # Don't fail on logging errors
        
        llm_response = await self.llm_engine.call_llm(context, user_answer, phase, session_id)
        logger.debug(
            f"LLM response received: action={llm_response.action.value}, "
            f"confidence={llm_response.confidence.value}"
        )
        return llm_response

    async def _plan_response(
        self, session_id: str, llm_response: LLMResponse
    ) -> PlannerOutput:
        """Plan response from LLM output.
        
        Args:
            session_id: Unique identifier for the session
            llm_response: LLMResponse from LLM engine
            
        Returns:
            PlannerOutput with validated and planned response
        """
        planner_output = await self.response_planner.plan_response(
            session_id, llm_response
        )
        logger.debug(
            f"Response planned for session {session_id}: "
            f"action={planner_output.action.value}, "
            f"type={planner_output.type.value}, "
            f"was_overridden={planner_output.was_overridden}"
        )
        return planner_output

    async def _handle_processing_error(
        self, session_id: str, step: str, error: Exception
    ) -> PlannerOutput:
        """Handle errors during processing and create fallback response.
        
        Implements graceful degradation: continues session on non-critical errors,
        sends user notifications, and provides appropriate fallback responses.
        
        Args:
            session_id: Unique identifier for the session
            step: Name of the processing step that failed
            error: The exception that occurred
            
        Returns:
            PlannerOutput with safe fallback response
        """
        from models.exceptions import (
            LLMTimeoutError, LLMError, STTError, STTTimeoutError,
            TTSError, TTSTimeoutError
        )
        
        logger.error(
            f"Error in {step} for session {session_id}: {type(error).__name__}: {error}"
        )
        
        # Determine error severity and appropriate response
        is_critical = False
        fallback_text = "Sorry, could you say that again? I want to make sure I catch everything."
        
        if isinstance(error, LLMTimeoutError):
            fallback_text = "Hmm, I lost my train of thought there for a second. Anyway, tell me more about your recent work!"
            # Send notification to user
            await self._notify_user_error(
                session_id, 
                "I'm experiencing some delays. Let me continue with the next question."
            )
        elif isinstance(error, LLMError):
            fallback_text = "Sorry, I got distracted for a moment. So, what else would you like to tell me about your experience?"
            await self._notify_user_error(
                session_id,
                "I encountered an error. Let me continue with the next question."
            )
        elif isinstance(error, (STTError, STTTimeoutError)):
            fallback_text = "I didn't catch that clearly. Could you please repeat?"
            await self._notify_user_error(
                session_id,
                "I'm having trouble hearing you. Could you please repeat?"
            )
        elif isinstance(error, (TTSError, TTSTimeoutError)):
            # TTS errors are non-critical - we can send text-only response
            fallback_text = "I apologize, but I'm having trouble speaking. Let me continue with the next question."
            await self._notify_user_error(
                session_id,
                "I'm experiencing audio issues. The interview will continue."
            )
        else:
            # Unknown error - treat as non-critical but notify user
            await self._notify_user_error(
                session_id,
                "I encountered an unexpected error. Let me continue."
            )
        
        # Create safe fallback response
        try:
            session = await self.session_manager.get_session_or_raise(session_id)
            phase = session.phase
        except Exception:
            phase = InterviewPhase.GREETING
        
        fallback_output = PlannerOutput(
            type=PlannerOutputType.SPEAK,
            text=fallback_text,
            action=LLMAction.ACKNOWLEDGE,
            metadata={
                "phase": phase.value,
                "follow_up_count": 0,
                "error_step": step,
                "error_type": type(error).__name__,
                "is_critical": is_critical,
            },
            was_overridden=True,
            override_reason=f"Processing error in {step}: {error}",
        )
        
        return fallback_output
    
    async def _notify_user_error(self, session_id: str, message: str) -> None:
        """Send error notification to user via control message.
        
        Args:
            session_id: Session identifier
            message: Error message to send to user
        """
        try:
            from routes.websocket_routes import connection_manager
            from datetime import datetime
            await connection_manager.send_control_message(
                session_id,
                {
                    "type": "ERROR",
                    "error_type": "processing_error",
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to send error notification to user for session {session_id}: {e}")

    async def _create_and_store_turn(
        self, session_id: str, user_utterance: str, phase: InterviewPhase,
        question: Optional[str] = None
    ) -> None:
        """Create turn from user response and add to session and short-term memory.
        
        Creates a turn if there's both a question and answer, then adds it to
        session.turn_history and short-term memory.
        
        Args:
            session_id: Unique identifier for the session
            user_utterance: The candidate's answer/utterance
            phase: Current interview phase
            question: The question being answered (if None, uses session.current_question)
        """
        try:
            # Get session to check for current question
            session = await self.session_manager.get_session_or_raise(session_id)
            
            # Use provided question or fall back to session's current question
            turn_question = question if question is not None else session.current_question
            
            # Only create turn if there's both a question and answer
            # Skip for initial greeting (no question yet)
            if not turn_question or not user_utterance.strip():
                logger.debug(
                    f"Skipping turn creation for session {session_id}: "
                    f"question={turn_question is not None}, "
                    f"answer={bool(user_utterance.strip())}"
                )
                return
            
            # Create turn
            turn = Turn(
                question=turn_question,
                answer=user_utterance.strip(),
                timestamp=datetime.now(),
                phase=phase,
                turn_number=session.total_turns + 1,
            )
            
            # Add to session.turn_history
            session.turn_history.append(turn)
            
            # Update total_turns
            try:
                await self.session_manager.update_session(
                    session_id, {"total_turns": session.total_turns + 1}
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update total_turns for session {session_id}: {e}"
                )
                # Continue - turn is still in history
            
            # Add to short-term memory (non-blocking, fire-and-forget)
            try:
                from core.memory import MemoryManager
                memory_manager = MemoryManager.get_instance_sync()
                if memory_manager:
                    # Fire-and-forget: don't await to avoid blocking
                    asyncio.create_task(
                        memory_manager.short_term_memory.add_turn(session_id, turn)
                    )
                    logger.debug(
                        f"Added turn to short-term memory for session {session_id} "
                        f"(turn_number={turn.turn_number})"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to add turn to short-term memory for session {session_id}: {e}"
                )
            
            # Log turn completion
            try:
                from core.memory import MemoryManager, create_log_entry, LogEntryType
                memory_manager = MemoryManager.get_instance_sync()
                if memory_manager:
                    log_entry = create_log_entry(
                        LogEntryType.TURN_COMPLETE,
                        session_id,
                        {
                            "turn_number": turn.turn_number,
                            "question": turn.question,
                            "answer": turn.answer,
                            "phase": turn.phase.value,
                        },
                    )
                    asyncio.create_task(
                        memory_manager.conversation_log.append_log(log_entry)
                    )
            except Exception as e:
                logger.debug(f"Failed to log turn completion: {e}")
            
            logger.info(
                f"Created turn for session {session_id}: "
                f"turn_number={turn.turn_number}, phase={phase.value}"
            )
        except Exception as e:
            logger.error(
                f"Error creating/storing turn for session {session_id}: {e}",
                exc_info=True
            )
            # Don't raise - turn creation failure shouldn't block response

    async def _emit_memory_event(
        self, session_id: str, event_type: str, data: Dict[str, Any]
    ) -> None:
        """Emit event for memory system (placeholder for future integration).
        
        Args:
            session_id: Unique identifier for the session
            event_type: Type of event (e.g., USER_RESPONSE_PROCESSED)
            data: Event data dictionary
        """
        # Placeholder for future memory system integration
        # For now, just log the event
        logger.debug(
            f"Memory event emitted for session {session_id}: "
            f"type={event_type}, data_keys={list(data.keys())}"
        )
        
        # Future: Integrate with memory system
        # await memory_system.handle_event(session_id, event_type, data)

    async def start_interview(self, session_id: str) -> PlannerOutput:
        """Start the interview by generating initial greeting.
        
        This method generates a personalized greeting based on the candidate's
        resume, work experience, and target role. It should be called immediately
        after session creation.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            PlannerOutput with initial greeting ready for TTS/text response
            
        Raises:
            SessionNotFoundError: If session does not exist
        """
        start_time = time.time()
        logger.info(f"Starting interview for session {session_id}")
        
        try:
            # Check if greeting already sent
            session = await self.session_manager.get_session_or_raise(session_id)
            if session.initial_greeting_sent:
                logger.warning(
                    f"Initial greeting already sent for session {session_id}, "
                    "returning existing greeting"
                )
                # Return a simple acknowledgment if already sent
                return PlannerOutput(
                    type=PlannerOutputType.SPEAK,
                    text="Welcome back! Let's continue the interview.",
                    action=LLMAction.ACKNOWLEDGE,
                    metadata={
                        "phase": session.phase.value,
                        "follow_up_count": 0,
                    },
                    was_overridden=False,
                    override_reason=None,
                )
            
            # Step 1: Build context with empty user utterance (for initial greeting)
            try:
                context_data = await self._build_context(session_id, "")
                formatted_context = context_data["formatted_context"]
                logger.debug(
                    f"Context built for initial greeting: {len(formatted_context)} chars"
                )
            except Exception as e:
                logger.error(
                    f"Failed to build context for initial greeting in session {session_id}: {e}"
                )
                # Use minimal context as fallback
                formatted_context = f"Interview Phase: {session.phase.value}\nCandidate Information: Current Role: {session.current_role}, Target Role: {session.target_role}"
                logger.warning(f"Using minimal context fallback for initial greeting in session {session_id}")
            
            # Step 2: Call LLM with empty user answer (triggers initial greeting logic)
            try:
                llm_response = await self._call_llm(formatted_context, "[INITIAL_GREETING]", session.phase, session_id)
                logger.debug(
                    f"LLM response for initial greeting: action={llm_response.action.value}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to call LLM for initial greeting in session {session_id}: {e}"
                )
                # Return fallback greeting
                return await self._handle_processing_error(
                    session_id, "llm_call_initial_greeting", e
                )
            
            # Step 3: Plan response
            try:
                planner_output = await self._plan_response(session_id, llm_response)
                logger.debug(
                    f"Initial greeting planned: action={planner_output.action.value}, "
                    f"type={planner_output.type.value}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to plan initial greeting for session {session_id}: {e}"
                )
                # Return fallback greeting
                return await self._handle_processing_error(
                    session_id, "response_planning_initial_greeting", e
                )
            
            # Step 4: Mark initial greeting as sent
            try:
                await self.session_manager.update_session(
                    session_id, {"initial_greeting_sent": True}
                )
                logger.debug(f"Marked initial greeting as sent for session {session_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to update initial_greeting_sent flag for session {session_id}: {e}"
                )
                # Continue anyway, this is not critical
            
            # Step 5: Update current question if LLM provided one
            if planner_output.text:
                try:
                    await self.session_manager.update_session(
                        session_id, {"current_question": planner_output.text}
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to update current_question for session {session_id}: {e}"
                    )
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Initial greeting generated for session {session_id}: "
                f"action={planner_output.action.value}, "
                f"text_length={len(planner_output.text) if planner_output.text else 0}, "
                f"time={elapsed_time:.2f}s"
            )
            
            return planner_output
            
        except Exception as e:
            # Catch-all for any unexpected errors
            logger.exception(
                f"Unexpected error generating initial greeting for session {session_id}: {e}"
            )
            return await self._handle_processing_error(
                session_id, "unexpected_error_initial_greeting", e
            )

    async def process_user_response(
        self, session_id: str, user_utterance: str
    ) -> PlannerOutput:
        """Process complete user utterance through AI pipeline.
        
        Main method that orchestrates all AI components to process
        a complete user utterance and return a planned response.
        
        Processing flow:
        1. Acquire processing lock (prevent double-processing)
        2. Get FSM state (current phase)
        3. Build inference context
        4. Call LLM
        5. Plan response
        6. Emit memory events
        7. Return planner output
        
        Args:
            session_id: Unique identifier for the session
            user_utterance: The candidate's complete answer/utterance
            
        Returns:
            PlannerOutput ready for TTS generation
            
        Raises:
            Exception: Only if all fallbacks fail (should not happen)
        """
        start_time = time.time()
        # Extract trace ID from log context if available (for flow tracing)
        # For now, log without trace ID - it will be added by audio processor
        logger.info(
            f"[TIMING] Processing user response for session {session_id}, "
            f"utterance_length={len(user_utterance)} chars"
        )
        
        # Step 0: Check if interview has ended
        session = await self.session_manager.get_session_or_raise(session_id)
        if session.phase == InterviewPhase.ENDED:
            logger.info(
                f"Interview already ended for session {session_id}, "
                "returning completion message"
            )
            return PlannerOutput(
                type=PlannerOutputType.SPEAK,
                text="The interview has been completed. Thank you for your time!",
                action=LLMAction.ACKNOWLEDGE,
                metadata={
                    "phase": InterviewPhase.ENDED.value,
                    "follow_up_count": 0,
                },
                was_overridden=False,
                override_reason=None,
            )
        
        # Step 0.5: Check safety limits before processing
        try:
            from core.safety import SafetyController
            from models.exceptions import SafetyLimitExceededError
            
            safety_controller = SafetyController.get_instance()
            safety_result = await safety_controller.enforce_limits(session_id)
            
            if not safety_result.allowed:
                logger.warning(
                    f"Safety limit exceeded for session {session_id}: {safety_result.reason}"
                )
                # Send notification to user
                try:
                    from routes.websocket_routes import connection_manager
                    await connection_manager.send_control_message(
                        session_id,
                        {
                            "type": "LIMIT_EXCEEDED",
                            "reason": safety_result.reason,
                            "message": safety_result.message,
                            "retry_after": safety_result.retry_after,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to send limit notification: {e}")
                
                # Return completion message if limits exceeded
                from models.constants import PlannerOutputType, LLMAction
                return PlannerOutput(
                    type=PlannerOutputType.SPEAK,
                    text=(
                        "I'm sorry, but we've reached the session limits. "
                        "Thank you for your time!"
                    ),
                    action=LLMAction.END_INTERVIEW,
                    metadata={"limit_reason": safety_result.reason},
                    was_overridden=False,
                    override_reason=None,
                )
        except Exception as e:
            logger.error(f"Error checking safety limits for {session_id}: {e}")
            # Continue processing on safety check error to avoid blocking
        
        # Step 0.6: Acquire processing lock (prevent double-processing)
        lock_acquired = await self._acquire_processing_lock(session_id)
        if not lock_acquired:
            # Already processing, return fallback
            logger.warning(
                f"Duplicate processing request for session {session_id}, "
                "returning fallback response"
            )
            return await self._handle_processing_error(
                session_id, "lock_acquisition", Exception("Already processing")
            )
        
        try:
            # Step 1: Get FSM state
            try:
                phase = await self._get_fsm_state(session_id)
                logger.debug(f"Step 1 complete: FSM state retrieved - {phase.value}")
            except Exception as e:
                logger.warning(
                    f"Failed to get FSM state for session {session_id}: {e}. "
                    "Continuing with context building."
                )
                phase = InterviewPhase.GREETING  # Default fallback
            
            # Step 2: Build inference context
            try:
                context_data = await self._build_context(session_id, user_utterance)
                formatted_context = context_data["formatted_context"]
                logger.debug(
                    f"Step 2 complete: Context built - {len(formatted_context)} chars"
                )
                
                # Log LLM input
                try:
                    from core.memory import MemoryManager, create_log_entry, LogEntryType
                    memory_manager = MemoryManager.get_instance_sync()
                    if memory_manager:
                        log_entry = create_log_entry(
                            LogEntryType.LLM_INPUT,
                            session_id,
                            {
                                "context_length": len(formatted_context),
                                "user_utterance": user_utterance,
                                "phase": phase.value,
                            },
                        )
                        asyncio.create_task(
                            memory_manager.conversation_log.append_log(log_entry)
                        )
                except Exception as e:
                    logger.debug(f"Failed to log LLM input: {e}")
            except Exception as e:
                logger.error(
                    f"Failed to build context for session {session_id}: {e}"
                )
                # Use minimal context as fallback
                formatted_context = f"Interview Phase: {phase.value}\nCandidate's Current Answer: {user_utterance}"
                logger.warning(f"Using minimal context fallback for session {session_id}")
            
            # Step 2.5: Check cost limits before LLM call
            try:
                from core.safety import SafetyController
                safety_controller = SafetyController.get_instance()
                cost_result = await safety_controller.check_cost_limits(session_id)
                if not cost_result.allowed:
                    logger.warning(
                        f"Cost limit exceeded for session {session_id}: {cost_result.reason}"
                    )
                    # Send notification
                    try:
                        from routes.websocket_routes import connection_manager
                        await connection_manager.send_control_message(
                            session_id,
                            {
                                "type": "LIMIT_EXCEEDED",
                                "reason": cost_result.reason,
                                "message": cost_result.message,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                    except Exception:
                        pass
                    # Use fallback response
                    from models.constants import PlannerOutputType, LLMAction
                    return PlannerOutput(
                        type=PlannerOutputType.SPEAK,
                        text=(
                            "I'm sorry, but we've reached the cost limits for this session. "
                            "Thank you for your time!"
                        ),
                        action=LLMAction.END_INTERVIEW,
                        metadata={"limit_reason": cost_result.reason},
                        was_overridden=False,
                        override_reason=None,
                    )
            except Exception as e:
                logger.debug(f"Error checking cost limits: {e}")
                # Continue on error to avoid blocking
            
            # Step 3: Call LLM
            try:
                llm_response = await self._call_llm(formatted_context, user_utterance, phase, session_id)
                logger.debug(
                    f"Step 3 complete: LLM called - action={llm_response.action.value}"
                )
                
                # Log LLM output
                try:
                    from core.memory import MemoryManager, create_log_entry, LogEntryType
                    memory_manager = MemoryManager.get_instance_sync()
                    if memory_manager:
                        log_entry = create_log_entry(
                            LogEntryType.LLM_OUTPUT,
                            session_id,
                            {
                                "action": llm_response.action.value,
                                "question": llm_response.question,
                                "confidence": llm_response.confidence.value,
                                "reasoning": llm_response.reasoning,
                            },
                        )
                        asyncio.create_task(
                            memory_manager.conversation_log.append_log(log_entry)
                        )
                except Exception as e:
                    logger.debug(f"Failed to log LLM output: {e}")
            except Exception as e:
                logger.error(
                    f"Failed to call LLM for session {session_id}: {e}"
                )
                # LLMEngine should have returned fallback, but handle if it didn't
                return await self._handle_processing_error(
                    session_id, "llm_call", e
                )
            
            # Step 3.5: Capture current question BEFORE planning (for turn history)
            # This is the question the user was responding to
            try:
                pre_plan_session = await self.session_manager.get_session_or_raise(session_id)
                question_being_answered = pre_plan_session.current_question
            except Exception:
                question_being_answered = None
            
            # Step 4: Plan response
            try:
                planner_output = await self._plan_response(session_id, llm_response)
                logger.debug(
                    f"Step 4 complete: Response planned - "
                    f"type={planner_output.type.value}, "
                    f"action={planner_output.action.value}"
                )
                
                # Log planner decision
                try:
                    from core.memory import MemoryManager, create_log_entry, LogEntryType
                    memory_manager = MemoryManager.get_instance_sync()
                    if memory_manager:
                        log_entry = create_log_entry(
                            LogEntryType.PLANNER_DECISION,
                            session_id,
                            {
                                "action": planner_output.action.value,
                                "type": planner_output.type.value,
                                "text": planner_output.text,
                                "was_overridden": planner_output.was_overridden,
                                "override_reason": planner_output.override_reason,
                                "metadata": planner_output.metadata,
                            },
                        )
                        asyncio.create_task(
                            memory_manager.conversation_log.append_log(log_entry)
                        )
                except Exception as e:
                    logger.debug(f"Failed to log planner decision: {e}")
            except Exception as e:
                logger.error(
                    f"Failed to plan response for session {session_id}: {e}"
                )
                # ResponsePlanner should have returned fallback, but handle if it didn't
                return await self._handle_processing_error(
                    session_id, "response_planning", e
                )
            
            # Step 5: Create turn and add to short-term memory (if applicable)
            try:
                await self._create_and_store_turn(
                    session_id, user_utterance, phase, question_being_answered
                )
            except Exception as e:
                # Don't fail on turn creation errors
                logger.warning(
                    f"Failed to create/store turn for session {session_id}: {e}"
                )
            
            # Step 6: Emit memory events (non-blocking)
            try:
                await self._emit_memory_event(
                    session_id,
                    "USER_RESPONSE_PROCESSED",
                    {
                        "user_utterance": user_utterance,
                        "llm_action": llm_response.action.value,
                        "planner_action": planner_output.action.value,
                        "was_overridden": planner_output.was_overridden,
                    },
                )
            except Exception as e:
                # Don't fail on memory event errors
                logger.warning(
                    f"Failed to emit memory event for session {session_id}: {e}"
                )
            
            # Processing complete
            elapsed_time = time.time() - start_time
            
            # Record end-to-end latency
            try:
                from util.metrics import MetricsCollector
                metrics = MetricsCollector.get_instance()
                metrics.record_latency(
                    "pipeline.end_to_end_latency",
                    elapsed_time,
                    {"session_id": session_id}
                )
            except Exception:
                pass  # Don't fail on metrics errors
            
            logger.info(
                f"User response processed for session {session_id}: "
                f"action={planner_output.action.value}, "
                f"type={planner_output.type.value}, "
                f"text_length={len(planner_output.text) if planner_output.text else 0}, "
                f"time={elapsed_time:.2f}s"
            )
            
            return planner_output
            
        except Exception as e:
            # Catch-all for any unexpected errors
            logger.exception(
                f"Unexpected error processing user response for session {session_id}: {e}"
            )
            
            # Check if this is a fatal error that requires session cleanup
            is_fatal = self._is_fatal_error(e)
            if is_fatal:
                logger.error(
                    f"Fatal error detected for session {session_id}, initiating cleanup"
                )
                try:
                    await self._cleanup_session_on_fatal_error(session_id, e)
                except Exception as cleanup_error:
                    logger.error(
                        f"Error during session cleanup for {session_id}: {cleanup_error}"
                    )
            
            return await self._handle_processing_error(
                session_id, "unexpected_error", e
            )
            
        finally:
            # Always release lock, even on error
            await self._release_processing_lock(session_id)
    
    def _is_fatal_error(self, error: Exception) -> bool:
        """Check if error is fatal and requires session cleanup.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is fatal
        """
        # For now, most errors are non-fatal
        # Fatal errors would be things like:
        # - Database corruption
        # - Memory corruption
        # - Critical system failures
        # Most component errors (LLM, STT, TTS) are non-fatal
        fatal_error_types = [
            "MemoryError",
            "SystemError",
            "KeyboardInterrupt",  # Not really fatal, but handled separately
        ]
        return type(error).__name__ in fatal_error_types
    
    async def _cleanup_session_on_fatal_error(
        self, session_id: str, error: Exception
    ) -> None:
        """Clean up session on fatal error.
        
        Args:
            session_id: Session identifier
            error: The fatal error that occurred
        """
        try:
            # Send notification to user
            await self._notify_user_error(
                session_id,
                "A critical error occurred. The session will be terminated. Please reconnect."
            )
            
            # Mark session for cleanup (don't delete immediately to allow logging)
            # The session will be cleaned up by the normal cleanup process
            logger.warning(
                f"Session {session_id} marked for cleanup due to fatal error: {error}"
            )
        except Exception as e:
            logger.error(f"Error in session cleanup for {session_id}: {e}")

