"""Utterance finalizer for combining VAD and STT signals.

This module implements the UtteranceFinalizer class that combines signals from
VAD (Voice Activity Detection) and STT (Speech-to-Text) to determine when a user
has completed their utterance. This is the only component that triggers the AI engine.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Callable, Optional

from models.constants import VADEvent

logger = logging.getLogger(__name__)

# Default timeout for waiting for STT final transcript (in milliseconds)
DEFAULT_STT_TIMEOUT_MS = 3000


class UtteranceFinalizer:
    """Utterance finalizer that combines VAD and STT signals.
    
    This component is responsible for determining when a user has finished speaking
    by combining signals from VAD (speech_end event) and STT (final transcript).
    It emits USER_RESPONSE_COMPLETE events to trigger the AI engine.
    
    Attributes:
        session_id: Unique identifier for the session
        vad_speech_end_received: Boolean flag indicating speech_end was received
        stt_final_transcript: Final transcript from STT (or None)
        speech_end_timestamp: Timestamp when speech_end was received
        stt_timeout_ms: Maximum time to wait for STT final transcript
        completion_callback: Callback function to emit USER_RESPONSE_COMPLETE
        _timeout_task: Background task for timeout handling
        _is_waiting: Whether we're currently waiting for finalization
    """

    def __init__(
        self,
        session_id: str,
        completion_callback: Optional[Callable[[str, str], None]] = None,
        stt_timeout_ms: Optional[int] = None,
    ) -> None:
        """Initialize utterance finalizer.
        
        Args:
            session_id: Unique identifier for the session
            completion_callback: Optional callback function called when utterance is complete.
                Signature: async def callback(session_id: str, transcript: str) -> None
            stt_timeout_ms: Maximum time to wait for STT final transcript (default: 3000ms)
        """
        self.session_id = session_id
        self.completion_callback = completion_callback
        
        # Get timeout from environment or use default
        if stt_timeout_ms is None:
            stt_timeout_ms = int(
                os.getenv("UTTERANCE_FINALIZER_TIMEOUT_MS", DEFAULT_STT_TIMEOUT_MS)
            )
        self.stt_timeout_ms = stt_timeout_ms
        
        # State tracking
        self.vad_speech_end_received = False
        self.stt_final_transcript: Optional[str] = None
        self.speech_end_timestamp: Optional[datetime] = None
        self._is_waiting = False
        self._timeout_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"UtteranceFinalizer initialized for session {session_id} "
            f"(timeout: {stt_timeout_ms}ms)"
        )

    async def handle_vad_event(self, event: VADEvent, timestamp: datetime) -> None:
        """Handle VAD event.
        
        Listens for SPEECH_END events and triggers finalization check.
        Resets state on SPEECH_START to prepare for new utterance.
        
        Args:
            event: VAD event (SPEECH_START or SPEECH_END)
            timestamp: Timestamp when event occurred
        """
        try:
            if event == VADEvent.SPEECH_START:
                # Reset state for new utterance
                self._reset_state()
                logger.debug(
                    f"Speech start detected for session {self.session_id}, "
                    "resetting finalizer state"
                )
            
            elif event == VADEvent.SPEECH_END:
                # Record speech_end event
                if self.vad_speech_end_received:
                    # Duplicate event - ignore
                    logger.debug(
                        f"Duplicate speech_end event for session {self.session_id}, ignoring"
                    )
                    return
                
                self.vad_speech_end_received = True
                self.speech_end_timestamp = timestamp
                self._is_waiting = True
                
                logger.info(
                    f"Speech end detected for session {self.session_id} at {timestamp}"
                )
                
                # Start timeout timer
                self._start_timeout()
                
                # Check if we can finalize (if transcript already received)
                await self._check_finalization()
            
        except Exception as e:
            logger.error(
                f"Error handling VAD event for session {self.session_id}: {e}",
                exc_info=True
            )

    async def handle_stt_final_transcript(self, transcript: Optional[str]) -> None:
        """Handle final transcript from STT.
        
        Receives final transcript from STT engine and triggers finalization check.
        
        Args:
            transcript: Final transcript text (may be None or empty)
        """
        try:
            if transcript is None:
                transcript = ""
            
            # Store transcript (even if empty)
            self.stt_final_transcript = transcript.strip() if transcript else ""
            
            logger.info(
                f"Final transcript received for session {self.session_id}: "
                f"'{self.stt_final_transcript}' (length: {len(self.stt_final_transcript)})"
            )
            
            # Cancel timeout if transcript received
            self._cancel_timeout()
            
            # Check if we can finalize (if speech_end already received)
            await self._check_finalization()
            
        except Exception as e:
            logger.error(
                f"Error handling STT transcript for session {self.session_id}: {e}",
                exc_info=True
            )

    async def _check_finalization(self) -> None:
        """Check if finalization conditions are met and emit event if so.
        
        Both conditions must be true:
        1. VAD speech_end event received
        2. STT final transcript available (even if empty)
        
        When both conditions are met, emits USER_RESPONSE_COMPLETE event.
        """
        finalization_start_time = time.time()
        
        # Check if both signals are received
        if not self.vad_speech_end_received:
            logger.debug(
                f"Waiting for speech_end for session {self.session_id}"
            )
            return
        
        if self.stt_final_transcript is None:
            logger.debug(
                f"Waiting for STT transcript for session {self.session_id}"
            )
            return
        
        # Both conditions met - emit completion event
        transcript = self.stt_final_transcript
        
        # Calculate time from speech_end to finalization
        finalization_delay = 0.0
        if self.speech_end_timestamp:
            finalization_delay = (datetime.now() - self.speech_end_timestamp).total_seconds()
        
        logger.info(
            f"Utterance finalization complete for session {self.session_id}: "
            f"'{transcript}' (delay={finalization_delay*1000:.2f}ms from speech_end)"
        )
        
        # Emit completion event via callback
        if self.completion_callback:
            try:
                if asyncio.iscoroutinefunction(self.completion_callback):
                    # Add timeout to callback execution to prevent hanging
                    try:
                        await asyncio.wait_for(
                            self.completion_callback(self.session_id, transcript),
                            timeout=30.0  # 30 second timeout for AI processing
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Completion callback timeout for session {self.session_id}. "
                            "AI processing took too long."
                        )
                        # Reset state to allow retry
                        self._reset_state()
                else:
                    # Sync callback - execute in thread to avoid blocking
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: self.completion_callback(self.session_id, transcript)
                        )
                    except Exception as executor_error:
                        logger.error(
                            f"Error executing sync callback in executor for session {self.session_id}: {executor_error}",
                            exc_info=True
                        )
            except Exception as e:
                logger.error(
                    f"Error in completion callback for session {self.session_id}: {e}",
                    exc_info=True
                )
                # Reset state to allow retry on next utterance
                self._reset_state()
        else:
            logger.warning(
                f"No completion callback registered for session {self.session_id}"
            )
        
        # Reset state for next utterance
        self._reset_state()

    def _start_timeout(self) -> None:
        """Start timeout timer for STT response.
        
        If STT doesn't provide transcript within timeout, will trigger
        finalization with empty transcript.
        """
        # Cancel any existing timeout
        self._cancel_timeout()
        
        # Create timeout task
        async def timeout_handler() -> None:
            try:
                await asyncio.sleep(self.stt_timeout_ms / 1000.0)
                
                # Timeout reached - check if we still need transcript
                if self._is_waiting and self.stt_final_transcript is None:
                    logger.warning(
                        f"STT timeout ({self.stt_timeout_ms}ms) for session {self.session_id}, "
                        "finalizing with empty transcript"
                    )
                    
                    # Set empty transcript and finalize
                    self.stt_final_transcript = ""
                    await self._check_finalization()
            
            except asyncio.CancelledError:
                # Timeout was cancelled (transcript received)
                logger.debug(
                    f"Timeout cancelled for session {self.session_id} "
                    "(transcript received)"
                )
            except Exception as e:
                logger.error(
                    f"Error in timeout handler for session {self.session_id}: {e}",
                    exc_info=True
                )
        
        self._timeout_task = asyncio.create_task(timeout_handler())
        logger.debug(
            f"Started timeout timer ({self.stt_timeout_ms}ms) for session {self.session_id}"
        )

    def _cancel_timeout(self) -> None:
        """Cancel timeout timer if running."""
        if self._timeout_task is not None and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None
            logger.debug(f"Timeout cancelled for session {self.session_id}")

    def _reset_state(self) -> None:
        """Reset state for new utterance."""
        self.vad_speech_end_received = False
        self.stt_final_transcript = None
        self.speech_end_timestamp = None
        self._is_waiting = False
        self._cancel_timeout()
        logger.debug(f"State reset for session {self.session_id}")

    async def reset(self) -> None:
        """Reset finalizer state (async version for cleanup).
        
        Cancels any pending timeouts and resets all state.
        """
        self._reset_state()
        logger.debug(f"UtteranceFinalizer reset for session {self.session_id}")

