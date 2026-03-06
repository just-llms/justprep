"""Turn controller for managing turn-taking between user and assistant.

This module implements turn-taking logic with hard rules to prevent double-speaking,
enable interruptions (barge-in), and ensure TTS stops immediately when the user
starts speaking.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Optional, Union

from models.constants import VADEvent
from util.logger import get_turn_controller_logger

logger = logging.getLogger(__name__)


class TurnOwner(str, Enum):
    """Turn ownership enumeration.
    
    Represents who currently has the turn (microphone ownership).
    """

    USER = "user"
    ASSISTANT = "assistant"
    IDLE = "idle"


# Type alias for TTS stop callback (can be sync or async)
TTSStopCallback = Union[Callable[[], None], Callable[[], Awaitable[None]]]


class TurnController:
    """Turn controller for managing turn-taking between user and assistant.
    
    Enforces hard rules:
    1. User speech always wins (immediate interruption)
    2. TTS stops immediately on user speech
    3. Assistant never speaks while user speaks
    4. Double-responses are impossible (idempotent operations)
    
    Attributes:
        _turn_owner: Current turn owner (USER, ASSISTANT, or IDLE)
        _lock: Async lock for thread-safe operations
        _tts_stop_callback: Callback function to stop TTS
        _tts_is_speaking: Flag to track if TTS is currently speaking
    """

    def __init__(self, tts_stop_callback: Optional[TTSStopCallback] = None) -> None:
        """Initialize turn controller.
        
        Args:
            tts_stop_callback: Optional callback function to stop TTS.
                Can be sync or async function. Called when user speech detected.
        """
        self._turn_owner = TurnOwner.IDLE
        self._lock = asyncio.Lock()
        self._tts_stop_callback = tts_stop_callback
        self._tts_is_speaking = False
        
        logger.info("Turn controller initialized")

    async def handle_speech_start(self) -> None:
        """Handle user speech start event from VAD.
        
        Enforces Rule 1: User speech always wins.
        - If turn is ASSISTANT → immediately stop TTS, transition to USER
        - If turn is IDLE → transition to USER
        - If turn is USER → no-op (idempotent)
        
        This method is thread-safe and idempotent.
        """
        async with self._lock:
            previous_owner = self._turn_owner
            
            # Idempotent: if already USER, no-op
            if self._turn_owner == TurnOwner.USER:
                logger.debug("Turn already belongs to user, skipping transition")
                return
            
            # Rule 1: User speech always wins
            # If assistant is speaking, stop TTS immediately
            if self._turn_owner == TurnOwner.ASSISTANT:
                logger.info(
                    f"User interrupting assistant. Stopping TTS and transitioning "
                    f"{previous_owner.value} -> {TurnOwner.USER.value}"
                )
                await self._stop_tts_internal()
            else:
                # IDLE -> USER
                logger.info(
                    f"User starting speech. Transitioning "
                    f"{previous_owner.value} -> {TurnOwner.USER.value}"
                )
            
            self._turn_owner = TurnOwner.USER

    async def handle_speech_end(self) -> None:
        """Handle user speech end event from VAD.
        
        Transitions from USER to IDLE when user finishes speaking.
        Does NOT trigger assistant to speak (that's handled elsewhere).
        
        This method is thread-safe and idempotent.
        """
        async with self._lock:
            previous_owner = self._turn_owner
            
            # Idempotent: if already IDLE or ASSISTANT, log warning but don't crash
            if self._turn_owner != TurnOwner.USER:
                logger.warning(
                    f"Received speech_end event but turn is {self._turn_owner.value}, "
                    f"expected USER. Ignoring."
                )
                return
            
            # USER -> IDLE
            logger.info(
                f"User finished speaking. Transitioning "
                f"{previous_owner.value} -> {TurnOwner.IDLE.value}"
            )
            self._turn_owner = TurnOwner.IDLE

    async def stop_tts(self) -> None:
        """Atomically stop TTS if assistant is speaking.
        
        Enforces Rule 2: TTS stops immediately on user speech.
        This is a public method that can be called externally.
        
        This method is thread-safe and idempotent.
        """
        async with self._lock:
            await self._stop_tts_internal()

    async def _stop_tts_internal(self) -> None:
        """Internal method to stop TTS.
        
        Must be called while holding self._lock.
        """
        # Idempotent: if TTS not speaking, no-op
        if not self._tts_is_speaking:
            logger.debug("TTS not speaking, skipping stop")
            return
        
        # Call TTS stop callback if provided
        if self._tts_stop_callback:
            try:
                logger.info("Executing TTS stop callback due to user speech interruption")
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(self._tts_stop_callback):
                    # Async callback
                    await self._tts_stop_callback()  # type: ignore
                else:
                    # Sync callback - run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._tts_stop_callback)  # type: ignore
                
                self._tts_is_speaking = False
                logger.info("TTS stop callback executed successfully - TTS interrupted by user speech")
            except Exception as e:
                # Log error but continue with state transition
                # TTS stop is best-effort, state change is mandatory
                logger.error(
                    f"Error in TTS stop callback during interruption: {e}",
                    exc_info=True
                )
                self._tts_is_speaking = False
        else:
            # No callback registered, just update flag
            self._tts_is_speaking = False
            logger.warning(
                "No TTS stop callback registered - TTS may not stop on user speech"
            )

    async def can_assistant_speak(self) -> bool:
        """Check if assistant is allowed to speak.
        
        Enforces Rule 3: Assistant never speaks while user speaks.
        
        Returns:
            True if assistant can speak (turn is IDLE or ASSISTANT),
            False if user has the turn.
        
        This method is thread-safe.
        """
        async with self._lock:
            can_speak = self._turn_owner in (TurnOwner.IDLE, TurnOwner.ASSISTANT)
            logger.debug(
                f"can_assistant_speak() called. Turn: {self._turn_owner.value}, "
                f"Result: {can_speak}"
            )
            return can_speak

    async def get_turn_owner(self) -> TurnOwner:
        """Get current turn owner.
        
        Returns:
            Current turn owner (USER, ASSISTANT, or IDLE)
        
        This method is thread-safe.
        """
        async with self._lock:
            return self._turn_owner

    async def set_tts_stop_callback(self, callback: TTSStopCallback) -> None:
        """Set or update TTS stop callback.
        
        Allows registering the callback after turn controller initialization.
        This is useful when TTS engine is created after turn controller.
        
        Args:
            callback: Callback function to stop TTS (sync or async).
                Called when user speech is detected to interrupt TTS.
        
        This method is thread-safe and idempotent.
        """
        async with self._lock:
            self._tts_stop_callback = callback
            logger.info("TTS stop callback registered/updated in turn controller")

    async def set_assistant_speaking(self, is_speaking: bool) -> None:
        """Set assistant speaking state.
        
        Called by TTS engine to indicate when it starts/stops speaking.
        This helps track TTS state for idempotent stop operations.
        
        Args:
            is_speaking: True if assistant is speaking, False otherwise
        """
        async with self._lock:
            previous_state = self._tts_is_speaking
            self._tts_is_speaking = is_speaking
            
            if is_speaking and not previous_state:
                # Assistant starting to speak
                if self._turn_owner != TurnOwner.ASSISTANT:
                    logger.info(
                        f"Assistant starting to speak. Transitioning "
                        f"{self._turn_owner.value} -> {TurnOwner.ASSISTANT.value}"
                    )
                    self._turn_owner = TurnOwner.ASSISTANT
            elif not is_speaking and previous_state:
                # Assistant finished speaking
                logger.debug("Assistant finished speaking")
                # Don't change turn owner here - let it stay as ASSISTANT
                # or transition to IDLE based on other events

    async def handle_vad_event(self, event: VADEvent, timestamp: datetime) -> None:
        """Handle VAD event from VAD processor.
        
        Routes VAD events to appropriate handlers.
        This is the callback function to register with VAD processor.
        
        Args:
            event: VAD event (SPEECH_START or SPEECH_END)
            timestamp: Timestamp when event occurred
        """
        if event == VADEvent.SPEECH_START:
            await self.handle_speech_start()
        elif event == VADEvent.SPEECH_END:
            await self.handle_speech_end()
        else:
            logger.warning(f"Unknown VAD event received: {event}")

    async def reset(self) -> None:
        """Reset turn state to IDLE.
        
        Clears turn ownership and TTS state.
        Useful when starting a new session or after errors.
        """
        async with self._lock:
            previous_owner = self._turn_owner
            self._turn_owner = TurnOwner.IDLE
            self._tts_is_speaking = False
            logger.info(
                f"Turn controller reset. Previous owner: {previous_owner.value}, "
                f"New owner: {TurnOwner.IDLE.value}"
            )

