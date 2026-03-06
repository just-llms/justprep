"""Deepgram Flux STT engine implementation.

This module implements the FluxSTTEngine class that integrates with Deepgram's
Flux API for real-time speech-to-text conversion using the v2/listen endpoint.
"""

import asyncio
import logging
import os
from typing import Optional

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV2SocketClientResponse

from util.audio_config import DEEPGRAM_CHUNK_SIZE_BYTES, DEEPGRAM_ENCODING, SAMPLE_RATE
from util.logger import get_stt_logger

logger = logging.getLogger(__name__)


class FluxSTTEngine:
    """Deepgram Flux STT engine for real-time speech-to-text conversion.
    
    This engine uses Deepgram's Flux model with the v2/listen endpoint to provide
    conversational speech recognition with built-in turn detection.
    
    Attributes:
        session_id: Unique identifier for the session
        api_key: Deepgram API key for authentication
        client: AsyncDeepgramClient instance
        connection: Active Deepgram connection (when streaming)
        is_streaming: Whether streaming is currently active
        final_transcript: Final transcript text (set after stop_streaming)
        _audio_buffer: Internal buffer for accumulating audio chunks
        _listening_task: Background task for listening to Deepgram responses
    """

    def __init__(self, session_id: str, api_key: Optional[str] = None) -> None:
        """Initialize Flux STT engine.
        
        Args:
            session_id: Unique identifier for the session
            api_key: Deepgram API key (if None, will load from DEEPGRAM_API_KEY env var)
            
        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        self.session_id = session_id
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("DEEPGRAM_API_KEY")
            if not api_key:
                raise ValueError(
                    "Deepgram API key not provided. Set DEEPGRAM_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        self.api_key = api_key
        
        # Initialize Deepgram client
        self.client: Optional[AsyncDeepgramClient] = None
        self.connection = None
        self._connection_context = None  # Store context manager for proper cleanup
        self.is_streaming = False
        self.final_transcript: Optional[str] = None
        self._latest_transcript: Optional[str] = None  # Store latest transcript from Update events
        
        # Audio buffering: accumulate 20ms chunks to send 80ms chunks to Deepgram
        self._audio_buffer = bytearray()
        
        # Background task for listening to responses
        self._listening_task: Optional[asyncio.Task] = None
        
        # Track total audio bytes sent for cost calculation
        self._total_audio_bytes_sent = 0
        
        # Initialize structured logger
        self.structured_logger = get_stt_logger(session_id=session_id)
        
        logger.info(f"FluxSTTEngine initialized for session {session_id}")

    async def start_streaming(self) -> None:
        """Start STT streaming session.
        
        Establishes connection to Deepgram Flux API using v2/listen endpoint.
        Sets up event handlers for receiving transcripts.
        Includes retry logic with exponential backoff and timeout handling.
        
        Raises:
            RuntimeError: If already streaming
            STTTimeoutError: If operation times out after retries
            STTConnectionError: If connection fails after retries
        """
        if self.is_streaming:
            logger.warning(
                f"STT streaming already active for session {self.session_id}, skipping start"
            )
            return

        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Attempt to start streaming with timeout
                await asyncio.wait_for(self._start_streaming_internal(), timeout=30.0)
                return
            except asyncio.TimeoutError:
                retry_count += 1
                last_error = asyncio.TimeoutError("STT start_streaming timed out after 30 seconds")
                if retry_count < max_retries:
                    delay = 2 ** retry_count  # Exponential backoff
                    logger.warning(
                        f"STT timeout for session {self.session_id}, "
                        f"retrying in {delay}s (attempt {retry_count}/{max_retries})"
                    )
                    self.structured_logger.warning(
                        "stt_timeout_retry",
                        "STT timeout, retrying",
                        {
                            "retry_count": retry_count,
                            "max_retries": max_retries,
                            "delay_seconds": delay
                        },
                        session_id=self.session_id
                    )
                    await asyncio.sleep(delay)
                else:
                    from models.exceptions import STTTimeoutError
                    raise STTTimeoutError(
                        "STT operation timed out after all retries",
                        timeout_seconds=30.0
                    ) from last_error
            except Exception as e:
                retry_count += 1
                last_error = e
                if retry_count < max_retries and self._is_retryable_error(e):
                    delay = 2 ** retry_count
                    logger.warning(
                        f"STT error for session {self.session_id}, "
                        f"retrying in {delay}s: {e}"
                    )
                    self.structured_logger.warning(
                        "stt_error_retry",
                        "STT error, retrying",
                        {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "retry_count": retry_count,
                            "max_retries": max_retries,
                            "delay_seconds": delay
                        },
                        session_id=self.session_id
                    )
                    await asyncio.sleep(delay)
                else:
                    from models.exceptions import STTConnectionError
                    raise STTConnectionError(
                        f"STT operation failed: {e}",
                        details=str(e)
                    ) from e
        
        # Should not reach here, but handle just in case
        if last_error:
            from models.exceptions import STTError
            raise STTError(f"STT operation failed after {max_retries} retries") from last_error
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is retryable
        """
        error_str = str(error).lower()
        retryable_keywords = ["connection", "timeout", "network", "temporary", "rate limit"]
        return any(keyword in error_str for keyword in retryable_keywords)
    
    async def _start_streaming_internal(self) -> None:
        """Internal method to start STT streaming (without retry logic)."""
        try:
            # Initialize client if not already done
            if self.client is None:
                self.client = AsyncDeepgramClient(api_key=self.api_key)
            
            # Connect to Deepgram Flux using v2/listen endpoint
            # Model: flux-general-en (required for Flux)
            # Encoding: linear16 (16-bit PCM little-endian)
            # Sample rate: 16000 Hz
            #
            # Flux turn-taking can be tuned via:
            # - eot_threshold
            # - eager_eot_threshold
            # - eot_timeout_ms
            # See: https://developers.deepgram.com/docs/flux/quickstart
            # Note: connect() returns an async context manager, so we need to enter it
            eot_timeout_ms = os.getenv("FLUX_EOT_TIMEOUT_MS")
            eot_threshold = os.getenv("FLUX_EOT_THRESHOLD")
            eager_eot_threshold = os.getenv("FLUX_EAGER_EOT_THRESHOLD")

            connect_kwargs = {
                "model": "flux-general-en",
                "encoding": DEEPGRAM_ENCODING,
                "sample_rate": str(SAMPLE_RATE),
            }
            # Add optional Flux tuning params if provided
            if eot_timeout_ms:
                connect_kwargs["eot_timeout_ms"] = int(eot_timeout_ms)
            if eot_threshold:
                connect_kwargs["eot_threshold"] = float(eot_threshold)
            if eager_eot_threshold:
                connect_kwargs["eager_eot_threshold"] = float(eager_eot_threshold)

            connection_context = self.client.listen.v2.connect(
                **connect_kwargs
            )
            # Enter the context manager to get the connection object
            self.connection = await connection_context.__aenter__()
            # Store context for cleanup
            self._connection_context = connection_context
            
            # Set up event handlers
            self.connection.on(EventType.OPEN, self._on_open)
            self.connection.on(EventType.MESSAGE, self._on_message)
            self.connection.on(EventType.CLOSE, self._on_close)
            self.connection.on(EventType.ERROR, self._on_error)
            
            # Start listening in background
            self._listening_task = asyncio.create_task(self.connection.start_listening())
            
            self.is_streaming = True
            self.final_transcript = None  # Clear previous transcript
            self._latest_transcript = None  # Clear latest transcript
            self._audio_buffer.clear()  # Clear buffer
            self._total_audio_bytes_sent = 0  # Reset audio tracking
            
            logger.info(f"STT streaming started for session {self.session_id}")
            self.structured_logger.info(
                "stt_streaming_started",
                "STT streaming started",
                {"session_id": self.session_id}
            )
            
        except Exception as e:
            logger.error(
                f"Error starting STT streaming for session {self.session_id}: {e}",
                exc_info=True
            )
            self.is_streaming = False
            # Try to clean up on error
            try:
                if self._connection_context is not None:
                    await self._connection_context.__aexit__(None, None, None)
                    self._connection_context = None
                if self.connection is not None:
                    self.connection = None
            except Exception as cleanup_error:
                logger.warning(f"Error during STT cleanup: {cleanup_error}")
            # Don't raise - allow flow to continue without STT
            logger.warning(
                f"STT streaming failed for session {self.session_id}. "
                "Flow will continue without STT."
            )

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio chunk to STT.
        
        Buffers incoming 20ms chunks and sends 80ms chunks to Deepgram when buffer
        reaches the recommended size for optimal performance.
        
        Args:
            audio_data: Binary audio data (16-bit PCM, 16kHz, mono)
            
        Raises:
            RuntimeError: If not streaming or connection is not available
        """
        if not self.is_streaming:
            logger.warning(
                f"STT not streaming for session {self.session_id}, cannot send audio"
            )
            return
        
        if self.connection is None:
            logger.error(
                f"STT connection not available for session {self.session_id}"
            )
            return

        try:
            # Add audio data to buffer
            self._audio_buffer.extend(audio_data)
            
            # Send buffered data when we have enough for 80ms chunk (2560 bytes)
            # This optimizes performance as recommended by Deepgram
            while len(self._audio_buffer) >= DEEPGRAM_CHUNK_SIZE_BYTES:
                # Extract 80ms chunk from buffer
                chunk_to_send = bytes(self._audio_buffer[:DEEPGRAM_CHUNK_SIZE_BYTES])
                self._audio_buffer = self._audio_buffer[DEEPGRAM_CHUNK_SIZE_BYTES:]
                
                # Send to Deepgram
                # The connection object should have a send method for binary data
                try:
                    # Check if still streaming before sending
                    if not self.is_streaming:
                        logger.debug(
                            f"STT no longer streaming for session {self.session_id}, "
                            f"dropping {len(chunk_to_send)} bytes"
                        )
                        return
                    
                    if hasattr(self.connection, 'send'):
                        await self.connection.send(chunk_to_send)
                    elif hasattr(self.connection, '_send'):
                        await self.connection._send(chunk_to_send)
                    else:
                        logger.error(
                            f"No send method found on connection for session {self.session_id}"
                        )
                        return
                    
                    # Track audio bytes sent for cost calculation
                    self._total_audio_bytes_sent += len(chunk_to_send)
                    
                    logger.debug(
                        f"Sent {len(chunk_to_send)} bytes audio chunk to Deepgram "
                        f"for session {self.session_id}"
                    )
                except Exception as e:
                    # Check if it's a connection closed error (expected when stopping)
                    # Check both exception type and message
                    from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
                    
                    is_connection_closed = (
                        isinstance(e, (ConnectionClosedOK, ConnectionClosedError)) or
                        "ConnectionClosed" in str(type(e).__name__) or
                        "ConnectionClosed" in str(e) or
                        "connection closed" in str(e).lower()
                    )
                    
                    if is_connection_closed:
                        # Connection was closed (likely by stop_streaming), this is expected
                        logger.debug(
                            f"Connection closed while sending audio for session {self.session_id}, "
                            f"stopping audio send"
                        )
                        self.is_streaming = False  # Mark as not streaming to prevent further sends
                        return
                    else:
                        # Unexpected error, log it and try to recover
                        logger.error(
                            f"Error sending audio chunk to Deepgram for session {self.session_id}: {e}",
                            exc_info=True
                        )
                        # Mark as not streaming to prevent further sends
                        # The audio processor will handle restarting if needed
                        self.is_streaming = False
        
        except Exception as e:
            logger.error(
                f"Error sending audio to STT for session {self.session_id}: {e}",
                exc_info=True
            )
            # Don't raise - allow streaming to continue

    async def stop_streaming(self) -> None:
        """Stop STT streaming and prepare for final transcript.
        
        Closes the connection to Deepgram and waits for final transcript.
        Sends any remaining buffered audio before closing.
        
        Raises:
            RuntimeError: If not streaming
        """
        if not self.is_streaming:
            logger.debug(
                f"STT not streaming for session {self.session_id}, skipping stop"
            )
            return

        try:
            # Send any remaining buffered audio
            if self.connection is not None and len(self._audio_buffer) > 0:
                try:
                    buffer_data = bytes(self._audio_buffer)
                    if hasattr(self.connection, "send"):
                        await self.connection.send(buffer_data)
                    else:
                        await self.connection._send(buffer_data)
                    # Track remaining audio bytes
                    self._total_audio_bytes_sent += len(buffer_data)
                    logger.debug(
                        f"Sent remaining {len(self._audio_buffer)} bytes to Deepgram "
                        f"for session {self.session_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error sending remaining buffer for session {self.session_id}: {e}"
                    )
                finally:
                    self._audio_buffer.clear()

            # Ask Deepgram to finalize the stream before closing.
            # Some SDK versions expose this as `finish()`.
            if self.connection is not None and hasattr(self.connection, "finish"):
                try:
                    await self.connection.finish()
                    logger.debug(f"Requested Deepgram finish() for session {self.session_id}")
                except Exception as e:
                    logger.warning(f"Error calling finish() for session {self.session_id}: {e}")
            
            # Close connection using context manager exit
            if self._connection_context is not None:
                try:
                    # Exit the context manager to close connection gracefully
                    await self._connection_context.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(
                        f"Error closing connection context for session {self.session_id}: {e}"
                    )
                finally:
                    self._connection_context = None
                    self.connection = None
            
            # Wait for listening task to complete (with timeout)
            # Give Flux a moment to send EndOfTurn event
            if self._listening_task is not None:
                try:
                    await asyncio.wait_for(self._listening_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.debug(
                        f"Listening task timeout for session {self.session_id}, cancelling"
                    )
                    self._listening_task.cancel()
                    try:
                        await self._listening_task
                    except asyncio.CancelledError:
                        pass
                except Exception as e:
                    logger.error(
                        f"Error waiting for listening task for session {self.session_id}: {e}"
                    )
                finally:
                    self._listening_task = None
            
            # If we didn't get a final transcript from EndOfTurn, use the latest transcript
            if self.final_transcript is None and self._latest_transcript is not None:
                self.final_transcript = self._latest_transcript
                logger.info(
                    f"Using latest transcript as final (no EndOfTurn received) for session {self.session_id}: "
                    f"'{self.final_transcript}'"
                )
            
            # Record STT minutes for cost tracking
            if self._total_audio_bytes_sent > 0:
                try:
                    from util.audio_config import SAMPLE_RATE
                    from core.safety import SafetyController
                    
                    # Calculate duration in minutes
                    # 16-bit PCM = 2 bytes per sample
                    # Total samples = bytes / 2
                    # Duration seconds = samples / sample_rate
                    total_samples = self._total_audio_bytes_sent // 2
                    duration_seconds = total_samples / SAMPLE_RATE
                    duration_minutes = duration_seconds / 60.0
                    
                    safety_controller = SafetyController.get_instance()
                    await safety_controller.record_stt_minutes(self.session_id, duration_minutes)
                    logger.debug(
                        f"Recorded {duration_minutes:.3f} STT minutes "
                        f"({self._total_audio_bytes_sent} bytes, {duration_seconds:.2f}s) "
                        f"for session {self.session_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to record STT minutes for session {self.session_id}: {e}")
            
            self.is_streaming = False
            
            logger.info(f"STT streaming stopped for session {self.session_id}")
            
        except Exception as e:
            logger.error(
                f"Error stopping STT streaming for session {self.session_id}: {e}",
                exc_info=True
            )
            self.is_streaming = False
            # Don't raise - ensure cleanup happens

    async def get_final_transcript(self) -> Optional[str]:
        """Get final transcript after stop_streaming.
        
        Returns the final transcript that was captured during the streaming session.
        Only final transcripts are returned (partial/interim transcripts are ignored).
        
        Returns:
            Final transcript text, or None if not available
        """
        return self.final_transcript

    def _on_open(self, event) -> None:
        """Handle connection open event.
        
        Args:
            event: Connection open event
        """
        logger.info(f"Deepgram connection opened for session {self.session_id}")

    def _on_message(self, message: ListenV2SocketClientResponse) -> None:
        """Handle message from Deepgram.
        
        Processes transcripts and filters out partial/interim transcripts.
        Only final transcripts are stored for retrieval.
        
        Args:
            message: Deepgram response message
        """
        try:
            # Track message count for debugging
            if not hasattr(self, '_message_count'):
                self._message_count = 0
            self._message_count += 1
            
            # Log first few messages to understand structure
            if self._message_count <= 5:
                attrs = [attr for attr in dir(message) if not attr.startswith('_')]
                logger.info(
                    f"Deepgram message #{self._message_count} for session {self.session_id}: "
                    f"type={type(message).__name__}, attrs={attrs[:15]}"
                )
                # Try to log message content
                try:
                    if hasattr(message, '__dict__'):
                        logger.debug(f"Message dict keys: {list(message.__dict__.keys())[:10]}")
                except:
                    pass
            
            # Check if message has transcript
            # Deepgram messages can have transcript in different attributes
            transcript_text = None
            is_final = False
            
            # Get message type name for better logging
            msg_type_name = type(message).__name__
            
            # Check for transcript in various possible attributes
            # Deepgram Flux may send transcripts in different message types
            if "Transcript" in msg_type_name:
                logger.info(
                    f"Transcript message type detected for session {self.session_id}: {msg_type_name}"
                )
            
            if hasattr(message, 'transcript') and message.transcript:
                transcript_text = message.transcript
            elif hasattr(message, 'channel') and hasattr(message.channel, 'alternatives'):
                # Check channel.alternatives[0].transcript
                if message.channel.alternatives and len(message.channel.alternatives) > 0:
                    transcript_text = getattr(message.channel.alternatives[0], 'transcript', None)
            elif hasattr(message, 'sentence'):
                # Check sentence attribute (may be a Sentence object)
                sentence_obj = getattr(message, 'sentence', None)
                if sentence_obj:
                    transcript_text = getattr(sentence_obj, 'text', None) or getattr(sentence_obj, 'transcript', None)
            elif hasattr(message, 'text'):
                # Direct text attribute
                transcript_text = message.text
            
            # Log what we found
            if transcript_text:
                logger.info(
                    f"Found transcript text for session {self.session_id}: '{transcript_text}' "
                    f"(from {msg_type_name})"
                )
            elif self._message_count <= 10:
                # Log message structure for first 10 messages to debug
                logger.debug(
                    f"No transcript found in message #{self._message_count} "
                    f"(type={msg_type_name}) for session {self.session_id}"
                )
            
            if transcript_text:
                # Store the latest transcript (even if not marked as final)
                # This ensures we capture the transcript even if EndOfTurn doesn't arrive
                normalized_text = self._normalize_transcript(transcript_text)
                if normalized_text:
                    self._latest_transcript = normalized_text
                
                # Check if this is a final transcript
                # Flux/Deepgram indicates final transcripts via is_final flag
                # or by checking if speech_final is True
                is_final = getattr(message, 'is_final', False)
                if not is_final:
                    # Also check speech_final attribute (alternative indicator)
                    is_final = getattr(message, 'speech_final', False)
                if not is_final and hasattr(message, 'channel'):
                    # Check channel-level flags
                    is_final = getattr(message.channel, 'is_final', False)
                
                # For Flux, we want to capture final transcripts only
                # Partial/interim transcripts are ignored per task requirements
                if is_final:
                    if normalized_text:
                        self.final_transcript = normalized_text
                        logger.info(
                            f"Final transcript received for session {self.session_id}: "
                            f"{normalized_text}"
                        )
                else:
                    # Partial transcript - store as latest but don't mark as final yet
                    # We'll use it as final if EndOfTurn doesn't arrive
                    logger.debug(
                        f"Partial transcript stored (latest) for session {self.session_id}: "
                        f"{transcript_text[:50]}..." if len(transcript_text) > 50 else transcript_text
                    )
            
            # Handle Flux-specific events (EndOfTurn, EagerEndOfTurn, TurnResumed)
            # ListenV2TurnInfoEvent has an 'event' attribute that indicates the event type
            # Check if this is a TurnInfoEvent with an event attribute
            if msg_type_name == "ListenV2TurnInfoEvent":
                event_type = getattr(message, 'event', None)
                if event_type:
                    logger.info(
                        f"Flux TurnInfoEvent received for session {self.session_id}: event={event_type}"
                    )
                    
                    if event_type == "EndOfTurn":
                        logger.info(
                            f"EndOfTurn event received for session {self.session_id}"
                        )
                        # When EndOfTurn is received, we should have a final transcript
                        # Try to get transcript from the message if available
                        if not transcript_text:
                            # Check if TurnInfoEvent has transcript data
                            if hasattr(message, 'transcript'):
                                transcript_text = message.transcript
                            elif hasattr(message, 'sentence'):
                                transcript_text = getattr(message.sentence, 'text', None)
                            
                            if transcript_text:
                                logger.info(
                                    f"Transcript from EndOfTurn event for session {self.session_id}: "
                                    f"'{transcript_text}'"
                                )
                                normalized_text = self._normalize_transcript(transcript_text)
                                if normalized_text:
                                    self.final_transcript = normalized_text
                                    logger.info(
                                        f"Final transcript stored from EndOfTurn: '{normalized_text}'"
                                    )
                        else:
                            # We already have transcript_text from earlier checks
                            # Mark it as final since EndOfTurn was received
                            normalized_text = self._normalize_transcript(transcript_text)
                            if normalized_text:
                                self.final_transcript = normalized_text
                                logger.info(
                                    f"Final transcript stored (EndOfTurn received): '{normalized_text}'"
                                )
                    elif event_type == "EagerEndOfTurn":
                        logger.debug(
                            f"EagerEndOfTurn event received for session {self.session_id}"
                        )
                    elif event_type == "TurnResumed":
                        logger.debug(
                            f"TurnResumed event received for session {self.session_id}"
                        )
            
            # Also check for generic message type attribute
            msg_type = getattr(message, 'type', None)
            if msg_type and msg_type != event_type if 'event_type' in locals() else None:
                logger.debug(
                    f"Message type attribute for session {self.session_id}: {msg_type}"
                )
        
        except Exception as e:
            logger.error(
                f"Error processing Deepgram message for session {self.session_id}: {e}",
                exc_info=True
            )

    def _on_close(self, event) -> None:
        """Handle connection close event.
        
        Args:
            event: Connection close event
        """
        logger.info(f"Deepgram connection closed for session {self.session_id}")
        self.is_streaming = False

    def _on_error(self, error) -> None:
        """Handle connection error event.
        
        Args:
            error: Error event or exception
        """
        logger.error(
            f"Deepgram connection error for session {self.session_id}: {error}",
            exc_info=True
        )
        self.is_streaming = False

    def _normalize_transcript(self, transcript: str) -> str:
        """Normalize transcript text.
        
        Applies normalization to transcript:
        - Strip whitespace
        - Ensure proper capitalization (first letter capitalized)
        - Preserve punctuation from Deepgram
        
        Args:
            transcript: Raw transcript text from Deepgram
            
        Returns:
            Normalized transcript text
        """
        if not transcript:
            return ""
        
        # Strip leading/trailing whitespace
        normalized = transcript.strip()
        
        # Capitalize first letter if transcript is not empty
        if normalized:
            # Only capitalize if the first character is a letter
            if normalized[0].isalpha():
                normalized = normalized[0].upper() + normalized[1:]
        
        return normalized

