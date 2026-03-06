"""Cartesia AI TTS engine implementation.

This module implements the CartesiaTTSEngine class that integrates with
Cartesia AI's Sonic-3 model for real-time text-to-speech conversion.
"""

import asyncio
import logging
import os
import time
from typing import Optional

from cartesia import Cartesia

from core.connection_manager import ConnectionManager
from core.audio.turn_controller import TurnController
from util.audio_config import SAMPLE_RATE
from util.logger import get_tts_logger

logger = logging.getLogger(__name__)


class CartesiaTTSEngine:
    """Cartesia AI TTS engine for real-time text-to-speech conversion.
    
    This engine uses Cartesia's Sonic-3 model to convert text to streaming
    audio with low latency and interruptibility support.
    
    Attributes:
        session_id: Unique identifier for the session
        api_key: Cartesia API key for authentication
        client: Cartesia client instance
        connection_manager: ConnectionManager for sending audio chunks
        turn_controller: Optional TurnController for interruption handling
        voice_id: Voice ID for consistent voice throughout session
        model_id: Cartesia model ID (default: sonic-3)
        is_streaming: Whether TTS is currently active
        _streaming_task: Background task for streaming
        _stop_event: Event to signal stop request
    """

    def __init__(
        self,
        session_id: str,
        connection_manager: ConnectionManager,
        api_key: Optional[str] = None,
        turn_controller: Optional[TurnController] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        """Initialize Cartesia TTS engine.
        
        Args:
            session_id: Unique identifier for the session
            connection_manager: ConnectionManager for sending audio chunks
            api_key: Cartesia API key (if None, will load from CARTESIA_API_KEY env var)
            turn_controller: Optional TurnController for interruption handling
            voice_id: Voice ID for consistent voice (if None, will load from CARTESIA_VOICE_ID env var)
            model_id: Cartesia model ID (if None, will load from CARTESIA_MODEL_ID env var, defaults to sonic-3)
            
        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        self.session_id = session_id
        self.connection_manager = connection_manager
        self.turn_controller = turn_controller
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("CARTESIA_API_KEY")
            if not api_key:
                raise ValueError(
                    "Cartesia API key not provided. Set CARTESIA_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        self.api_key = api_key
        
        # Get voice ID from parameter or environment, with default stable voice
        if voice_id is None:
            voice_id = os.getenv("CARTESIA_VOICE_ID")
            if not voice_id:
                # Default to Katie (stable, realistic voice for voice agents)
                voice_id = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
        
        self.voice_id = voice_id
        
        # Get model ID from parameter or environment, default to sonic-3
        if model_id is None:
            model_id = os.getenv("CARTESIA_MODEL_ID", "sonic-3")
        
        self.model_id = model_id
        
        # Initialize Cartesia client
        self.client: Optional[Cartesia] = None
        
        # Streaming state
        self.is_streaming = False
        self._streaming_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._callback_registered = False  # Track if callback is registered with turn controller
        self._stream_start_time: Optional[float] = None  # Track when streaming started for interruption timing
        
        # Initialize structured logger
        self.structured_logger = get_tts_logger(session_id=session_id)
        
        logger.info(
            f"CartesiaTTSEngine initialized for session {self.session_id} "
            f"(voice_id={self.voice_id}, model_id={self.model_id})"
        )

    async def speak(self, text: str) -> None:
        """Start streaming TTS for given text.
        
        Converts text to speech and streams audio chunks immediately to the client.
        Updates turn controller state to indicate assistant is speaking.
        
        Args:
            text: Text to convert to speech
            
        Raises:
            RuntimeError: If already streaming or connection fails
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for TTS in session {self.session_id}")
            return
        
        if self.is_streaming:
            logger.warning(
                f"TTS already streaming for session {self.session_id}, stopping previous stream"
            )
            await self.stop()
        
        try:
            # Register stop callback with turn controller if not already registered
            if self.turn_controller is not None and not self._callback_registered:
                await self.turn_controller.set_tts_stop_callback(self.stop)
                self._callback_registered = True
                logger.info(
                    f"TTS stop callback registered with turn controller for session {self.session_id}"
                )
            
            # Initialize client if not already done
            if self.client is None:
                self.client = Cartesia(api_key=self.api_key)
            
            # Reset stop event
            self._stop_event.clear()
            
            # Track stream start time for interruption timing
            self._stream_start_time = time.time()
            
            # Update turn controller state
            if self.turn_controller is not None:
                await self.turn_controller.set_assistant_speaking(True)
            
            # Check if assistant can speak (user might have started speaking)
            if self.turn_controller is not None:
                can_speak = await self.turn_controller.can_assistant_speak()
                if not can_speak:
                    logger.info(
                        f"User has turn, cancelling TTS for session {self.session_id}"
                    )
                    if self.turn_controller is not None:
                        await self.turn_controller.set_assistant_speaking(False)
                    self._stream_start_time = None
                    return
            
            self.is_streaming = True
            
            # Start streaming in background task
            self._streaming_task = asyncio.create_task(self._stream_audio(text))
            
            logger.info(
                f"TTS streaming started for session {self.session_id} "
                f"(text_length={len(text)})"
            )
            
            # Log TTS start
            try:
                from core.memory import MemoryManager, create_log_entry, LogEntryType
                memory_manager = MemoryManager.get_instance_sync()
                if memory_manager:
                    log_entry = create_log_entry(
                        LogEntryType.TTS_START,
                        self.session_id,
                        {
                            "text_length": len(text),
                            "text_preview": text[:100] if len(text) > 100 else text,
                        },
                    )
                    asyncio.create_task(
                        memory_manager.conversation_log.append_log(log_entry)
                    )
            except Exception:
                pass  # Don't fail on logging errors
            
        except Exception as e:
            logger.error(
                f"Error starting TTS streaming for session {self.session_id}: {e}",
                exc_info=True
            )
            self.is_streaming = False
            if self.turn_controller is not None:
                await self.turn_controller.set_assistant_speaking(False)
            raise RuntimeError(f"Failed to start TTS streaming: {e}") from e

    async def stop(self) -> None:
        """Stop TTS immediately (interruptible).
        
        Cancels streaming and flushes any in-flight chunks.
        Idempotent - safe to call multiple times.
        """
        was_streaming = self.is_streaming
        
        if not was_streaming:
            logger.debug(
                f"TTS not streaming for session {self.session_id}, skipping stop"
            )
            return
        
        try:
            # Calculate interruption duration if streaming was active
            interruption_duration = 0.0
            if self._stream_start_time is not None:
                import time
                interruption_duration = time.time() - self._stream_start_time
                self._stream_start_time = None
            
            # Log interruption if it was due to user speech
            was_interrupted = interruption_duration > 0
            if was_interrupted:
                logger.warning(
                    f"TTS interrupted by user speech for session {self.session_id} "
                    f"(was_speaking_for={interruption_duration:.2f}s)"
                )
            
            # Log TTS stop
            try:
                from core.memory import MemoryManager, create_log_entry, LogEntryType
                memory_manager = MemoryManager.get_instance_sync()
                if memory_manager:
                    log_entry = create_log_entry(
                        LogEntryType.TTS_STOP,
                        self.session_id,
                        {
                            "was_interrupted": was_interrupted,
                            "duration": interruption_duration,
                        },
                    )
                    asyncio.create_task(
                        memory_manager.conversation_log.append_log(log_entry)
                    )
            except Exception:
                pass  # Don't fail on logging errors
            
            # Signal stop atomically
            self._stop_event.set()
            
            # Cancel streaming task if running
            if self._streaming_task is not None and not self._streaming_task.done():
                self._streaming_task.cancel()
                try:
                    await self._streaming_task
                except asyncio.CancelledError:
                    logger.debug(
                        f"TTS streaming task cancelled for session {self.session_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error cancelling TTS streaming task for session {self.session_id}: {e}"
                    )
                finally:
                    self._streaming_task = None
            
            # Update turn controller state
            if self.turn_controller is not None:
                await self.turn_controller.set_assistant_speaking(False)
            
            self.is_streaming = False
            
            logger.info(f"TTS streaming stopped for session {self.session_id}")
            
        except Exception as e:
            logger.error(
                f"Error stopping TTS streaming for session {self.session_id}: {e}",
                exc_info=True
            )
            self.is_streaming = False
            self._stream_start_time = None
            if self.turn_controller is not None:
                await self.turn_controller.set_assistant_speaking(False)

    async def _stream_audio(self, text: str) -> None:
        """Internal method to handle Cartesia API streaming and chunk processing.
        
        Streams audio from Cartesia and sends chunks immediately to the connection manager.
        Checks stop event before each chunk to support interruption.
        Implements retry logic for connection failures.
        
        Args:
            text: Text to convert to speech
            
        Raises:
            RuntimeError: If streaming fails after retries
        """
        max_retries = 2
        retry_count = 0
        base_delay = 0.5  # Base delay in seconds for exponential backoff
        
        while retry_count <= max_retries:
            try:
                # Configure output format to match system requirements
                # System expects: 16kHz, 16-bit PCM little-endian
                # Cartesia supports various formats - request closest match
                output_format = {
                    "container": "wav",  # WAV container
                    "sample_rate": SAMPLE_RATE,  # 16000 Hz to match system
                    "encoding": "pcm_s16le",  # 16-bit PCM little-endian
                }
                
                # Get streaming iterator from Cartesia
                # Note: Cartesia returns a synchronous iterator, so we can't easily timeout the initial call
                # Instead, we'll timeout the chunk processing loop
                TTS_TIMEOUT_SECONDS = 10.0
                try:
                    chunk_iter = self.client.tts.bytes(
                        model_id=self.model_id,
                        transcript=text,
                        voice={
                            "mode": "id",
                            "id": self.voice_id,
                        },
                        output_format=output_format,
                    )
                except Exception as api_error:
                    # Check if it's a connection/API error that we should retry
                    error_str = str(api_error).lower()
                    is_retryable = any(
                        keyword in error_str
                        for keyword in ["connection", "timeout", "network", "temporary"]
                    )
                    
                    if is_retryable and retry_count < max_retries:
                        retry_count += 1
                        delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.warning(
                            f"TTS API error (retryable) for session {self.session_id}: {api_error}. "
                            f"Retrying in {delay}s (attempt {retry_count}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Not retryable or max retries reached
                        logger.error(
                            f"TTS API error for session {self.session_id}: {api_error}",
                            exc_info=True
                        )
                        raise RuntimeError(f"TTS API call failed: {api_error}") from api_error
            
                # Process chunks - Cartesia returns a synchronous iterator
                # We'll iterate synchronously but make async calls for sending chunks
                chunk_count = 0
                total_bytes = 0
                chunks_dropped_after_stop = 0
                stream_start_time = time.time()
                
                # Process chunks from Cartesia iterator with timeout
                # Note: chunk_iter is a synchronous iterator, so we iterate normally
                # but make async calls for sending and checking state
                # We'll check timeout before processing each chunk
                for chunk in chunk_iter:
                    # Check timeout (10 seconds total)
                    elapsed_time = time.time() - stream_start_time
                    if elapsed_time > TTS_TIMEOUT_SECONDS:
                        logger.warning(
                            f"TTS streaming timeout after {elapsed_time:.2f}s for session {self.session_id}"
                        )
                        self._stop_event.set()
                        from models.exceptions import TTSTimeoutError
                        raise TTSTimeoutError(
                            f"TTS operation timed out after {elapsed_time:.2f} seconds",
                            timeout_seconds=elapsed_time
                        )
                    # Check if stop was requested - atomic check before sending
                    if self._stop_event.is_set():
                        # Count chunks that would have been sent but are dropped due to stop
                        chunks_dropped_after_stop += 1
                        logger.info(
                            f"TTS streaming stopped due to interruption for session {self.session_id} "
                            f"(chunks_sent={chunk_count}, chunks_dropped={chunks_dropped_after_stop}, "
                            f"total_bytes={total_bytes})"
                        )
                        # Break immediately - no more chunks will be sent
                        break
                    
                    # Check if assistant can still speak (user might have started)
                    if self.turn_controller is not None:
                        can_speak = await self.turn_controller.can_assistant_speak()
                        if not can_speak:
                            logger.info(
                                f"User has turn, stopping TTS streaming for session {self.session_id}"
                            )
                            self._stop_event.set()
                            break
                    
                    # Process audio chunk (may need format conversion)
                    processed_chunk = self._process_audio_chunk(chunk)
                    
                    if processed_chunk:
                        # Double-check stop event before sending (atomic operation)
                        if self._stop_event.is_set():
                            chunks_dropped_after_stop += 1
                            logger.debug(
                                f"Dropping audio chunk due to stop event for session {self.session_id} "
                                f"(chunk_size={len(processed_chunk)} bytes)"
                            )
                            # Break immediately - no more chunks will be sent
                            break
                        
                        # Send chunk immediately to connection manager
                        success = await self.connection_manager.send_audio_chunk(
                            self.session_id, processed_chunk
                        )
                        
                        if not success:
                            logger.warning(
                                f"Failed to send audio chunk to connection manager "
                                f"for session {self.session_id}"
                            )
                            # Continue streaming - connection manager handles disconnect
                        
                        chunk_count += 1
                        total_bytes += len(processed_chunk)
                        
                        logger.debug(
                            f"Sent audio chunk {chunk_count} ({len(processed_chunk)} bytes) "
                            f"for session {self.session_id}"
                        )
                
                # Log completion or interruption
                if chunks_dropped_after_stop > 0:
                    logger.info(
                        f"TTS streaming interrupted for session {self.session_id} "
                        f"(chunks_sent={chunk_count}, chunks_dropped={chunks_dropped_after_stop}, "
                        f"total_bytes={total_bytes})"
                    )
                    self.structured_logger.info(
                        "tts_interrupted",
                        "TTS streaming interrupted",
                        {
                            "chunks_sent": chunk_count,
                            "chunks_dropped": chunks_dropped_after_stop,
                            "total_bytes": total_bytes
                        },
                        session_id=self.session_id
                    )
                else:
                    logger.info(
                        f"TTS streaming completed for session {self.session_id} "
                        f"(chunks={chunk_count}, total_bytes={total_bytes})"
                    )
                    self.structured_logger.info(
                        "tts_completed",
                        "TTS streaming completed",
                        {
                            "chunks_sent": chunk_count,
                            "total_bytes": total_bytes
                        },
                        session_id=self.session_id
                    )
                
                # Update state after streaming completes successfully
                self.is_streaming = False
                self._stream_start_time = None
                if self.turn_controller is not None:
                    await self.turn_controller.set_assistant_speaking(False)
                
                # Success - break out of retry loop
                return
                
            except asyncio.CancelledError:
                logger.debug(
                    f"TTS streaming cancelled for session {self.session_id}"
                )
                self.is_streaming = False
                self._stream_start_time = None
                if self.turn_controller is not None:
                    await self.turn_controller.set_assistant_speaking(False)
                raise
            
            except RuntimeError as runtime_error:
                # Re-raise RuntimeError (API errors after retries)
                # But check if we should retry first
                error_str = str(runtime_error).lower()
                is_retryable = any(
                    keyword in error_str
                    for keyword in ["connection", "timeout", "network", "temporary"]
                )
                
                if is_retryable and retry_count < max_retries:
                    retry_count += 1
                    delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.warning(
                        f"TTS streaming error (retryable) for session {self.session_id}: {runtime_error}. "
                        f"Retrying in {delay}s (attempt {retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable or max retries reached
                    logger.error(
                        f"TTS streaming failed after retries for session {self.session_id}: {runtime_error}",
                        exc_info=True
                    )
                    # Raise TTS error for error handling
                    from models.exceptions import TTSError
                    raise TTSError(f"TTS streaming failed after retries: {runtime_error}") from runtime_error
                    self.is_streaming = False
                    if self.turn_controller is not None:
                        await self.turn_controller.set_assistant_speaking(False)
                    # Don't raise - allow graceful degradation
                    return
            
            except Exception as e:
                # Check if we should retry
                error_str = str(e).lower()
                is_retryable = any(
                    keyword in error_str
                    for keyword in ["connection", "timeout", "network", "temporary"]
                )
                
                if is_retryable and retry_count < max_retries:
                    retry_count += 1
                    delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    logger.warning(
                        f"TTS streaming error (retryable) for session {self.session_id}: {e}. "
                        f"Retrying in {delay}s (attempt {retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable or max retries reached
                    logger.error(
                        f"Error in TTS streaming for session {self.session_id}: {e}",
                        exc_info=True
                    )
                    self.is_streaming = False
                    self._stream_start_time = None
                    if self.turn_controller is not None:
                        await self.turn_controller.set_assistant_speaking(False)
                    # Don't raise - allow graceful degradation
                    return

    def _process_audio_chunk(self, audio_chunk: bytes) -> Optional[bytes]:
        """Process audio chunk from Cartesia to match system format.
        
        Currently, Cartesia is configured to output in the correct format (16kHz, 16-bit PCM).
        If format conversion is needed in the future, it can be added here.
        
        Args:
            audio_chunk: Raw audio chunk from Cartesia
            
        Returns:
            Processed audio chunk, or None if chunk should be skipped
        """
        if not audio_chunk:
            return None
        
        # For now, Cartesia is configured to output in the correct format
        # If format conversion is needed, add it here
        # Example: Convert from WAV container to raw PCM, resample, etc.
        
        # If chunk is WAV format, we may need to extract PCM data
        # For v1, assume Cartesia outputs raw PCM or WAV that browser can handle
        # If issues arise, we can add WAV header parsing here
        
        return audio_chunk

