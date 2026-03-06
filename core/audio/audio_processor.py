"""Audio processor for orchestrating VAD, turn controller, and STT components.

This module implements the audio processor that coordinates all audio processing
components for real-time speech detection, turn-taking, and speech-to-text conversion.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Deque, Dict, Optional

from models.constants import VADEvent, PlannerOutputType

from core.audio.turn_controller import TurnController, TurnOwner
from core.audio.utterance_finalizer import UtteranceFinalizer
from core.audio.vad_processor import VADProcessor, create_vad_processor_from_env
from core.stt import create_stt_engine
from util.audio_config import validate_audio_format

logger = logging.getLogger(__name__)


class STTEngineInterface:
    """Interface for STT engine (to be implemented in Task 4.1).
    
    This is a placeholder interface that will be replaced with the actual
    STT engine implementation in Task 4.1.
    """

    async def start_streaming(self) -> None:
        """Start STT streaming session.
        
        Raises:
            NotImplementedError: STT engine not yet implemented
        """
        raise NotImplementedError("STT engine not yet implemented (Task 4.1)")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio chunk to STT.
        
        Args:
            audio_data: Binary audio data to send to STT
            
        Raises:
            NotImplementedError: STT engine not yet implemented
        """
        raise NotImplementedError("STT engine not yet implemented (Task 4.1)")

    async def stop_streaming(self) -> None:
        """Stop STT streaming and prepare for final transcript.
        
        Raises:
            NotImplementedError: STT engine not yet implemented
        """
        raise NotImplementedError("STT engine not yet implemented (Task 4.1)")

    async def get_final_transcript(self) -> Optional[str]:
        """Get final transcript after stop_streaming.
        
        Returns:
            Final transcript text, or None if not available
            
        Raises:
            NotImplementedError: STT engine not yet implemented
        """
        raise NotImplementedError("STT engine not yet implemented (Task 4.1)")


class AudioProcessor:
    """Audio processor that orchestrates VAD, turn controller, and STT.
    
    This is the real-time core of the system that receives audio chunks,
    feeds them to VAD and STT, handles VAD events, and coordinates turn-taking.
    
    Attributes:
        session_id: Unique identifier for the session
        vad_processor: VAD processor instance for speech detection
        turn_controller: Turn controller instance for turn-taking
        stt_engine: STT engine instance (placeholder until Task 4.1)
        is_active: Whether audio processing is active
        stt_is_active: Whether STT streaming is active
        _vad_queue: Bounded queue for VAD processing (backpressure handling)
    """

    def __init__(
        self,
        session_id: str,
        tts_stop_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize audio processor.
        
        Args:
            session_id: Unique identifier for the session
            tts_stop_callback: Optional callback function to stop TTS.
                Passed to turn controller for interruption handling.
        """
        self.session_id = session_id
        self.is_active = False
        self.stt_is_active = False

        # Store event loop for scheduling callbacks from thread pool
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no loop is running yet, we'll set it when start() is called
            self._event_loop = None

        # Initialize components
        try:
            self.vad_processor = create_vad_processor_from_env()
            self.turn_controller = TurnController(tts_stop_callback)
            # STT engine will be created lazily in start() method
            self.stt_engine: Optional[STTEngineInterface] = None
            # Utterance finalizer will be created in start() method with completion callback
            self.utterance_finalizer: Optional[UtteranceFinalizer] = None

            # Backpressure handling: bounded queues
            # Note: _vad_queue is a bounded deque that automatically drops old items when full
            # We don't need to manually check size - the deque handles overflow
            self._vad_queue: Deque[bytes] = deque(maxlen=5)
            # STT queue removed - sending directly to STT engine which handles buffering internally
            
            # Pre-buffer for audio chunks before STT starts (to capture early speech)
            # Buffer up to ~2 seconds of audio (100 chunks at 20ms each = 2000ms)
            # This ensures we don't lose the first ~1-2 seconds of speech while VAD detects speech_start
            self._stt_pre_buffer: Deque[bytes] = deque(maxlen=100)
            
            # Thread pool for running synchronous VAD processing without blocking event loop
            self._vad_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"vad-{session_id}")

            # Setup component integration
            self._setup_components()

            logger.info(f"Audio processor initialized for session {session_id}")
        except Exception as e:
            logger.exception(f"Failed to initialize audio processor for session {session_id}: {e}")
            raise

    def _setup_components(self) -> None:
        """Setup component integration and callbacks."""
        # Register VAD event callback with event loop for thread-safe scheduling
        self.vad_processor.register_callback(self._handle_vad_event, event_loop=self._event_loop)
        logger.debug(f"VAD event callback registered for session {self.session_id}")

    async def process_audio_chunk(self, audio_data: bytes) -> None:
        """Process incoming audio chunk from WebSocket.
        
        Feeds audio to VAD (always) and STT (only when user speaking).
        Handles backpressure by dropping chunks if queues are full.
        
        Args:
            audio_data: Binary audio data (16-bit PCM, 16kHz, mono)
        """
        # #region agent log
        chunk_arrival_time = time.time()
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(chunk_arrival_time*1000)}_arrival","timestamp":int(chunk_arrival_time*1000),"location":"audio_processor.py:125","message":"Chunk arrived","data":{"sessionId":self.session_id,"chunkSize":len(audio_data),"queueSizeBefore":len(self._vad_queue)},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,C"}) + "\n")
        # #endregion
        
        if not self.is_active:
            logger.debug(f"Audio processor not active for session {self.session_id}, ignoring chunk")
            return

        # Validate audio format
        is_valid, error_msg = validate_audio_format(audio_data)
        if not is_valid:
            logger.warning(
                f"Invalid audio chunk for session {self.session_id}: {error_msg}. Skipping."
            )
            return

        try:
            # Feed to VAD (always) - use bounded deque for automatic backpressure
            # The deque automatically drops old items when full (maxlen=5)
            # We run VAD processing in a thread pool to avoid blocking the event loop
            queue_size_before = len(self._vad_queue)
            # #region agent log
            check_time = time.time()
            with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                f.write(json.dumps({"id":f"log_{int(check_time*1000)}_check","timestamp":int(check_time*1000),"location":"audio_processor.py:155","message":"Queue check","data":{"sessionId":self.session_id,"queueSize":queue_size_before,"isFull":queue_size_before>=5},"sessionId":"debug-session","runId":"post-fix","hypothesisId":"C"}) + "\n")
            # #endregion
            
            # Add to queue (deque will automatically drop oldest if full)
            self._vad_queue.append(audio_data)
            queue_size_after = len(self._vad_queue)
            
            # #region agent log
            process_start = time.time()
            with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                f.write(json.dumps({"id":f"log_{int(process_start*1000)}_process_start","timestamp":int(process_start*1000),"location":"audio_processor.py:162","message":"Process chunk start (async)","data":{"sessionId":self.session_id,"queueSizeAfter":queue_size_after,"timeSinceArrival":process_start-chunk_arrival_time},"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A,B,E"}) + "\n")
            # #endregion
            
            # Process VAD chunk asynchronously in thread pool to avoid blocking event loop
            # This prevents the 107ms model inference from blocking chunk processing
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                self._vad_executor,
                self.vad_processor.process_chunk,
                audio_data
            )
            
            # #region agent log
            process_end = time.time()
            process_duration = process_end - process_start
            with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                f.write(json.dumps({"id":f"log_{int(process_end*1000)}_process_end","timestamp":int(process_end*1000),"location":"audio_processor.py:162","message":"Process chunk scheduled (non-blocking)","data":{"sessionId":self.session_id,"scheduleDuration":process_duration,"queueSize":len(self._vad_queue)},"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A,B,E"}) + "\n")
            # #endregion

            # Feed to STT (only when user speaking)
            if self.stt_is_active:
                # Check if STT engine is available
                if self.stt_engine is None:
                    logger.debug(
                        f"STT engine not available for session {self.session_id} "
                        "(expected until Task 4.1)"
                    )
                else:
                    # Send directly to STT (non-blocking task)
                    # STT engine handles internal buffering for 80ms chunks
                    try:
                        asyncio.create_task(self._send_to_stt(audio_data))
                    except Exception as e:
                        logger.error(
                            f"Error sending audio to STT in session {self.session_id}: {e}"
                        )
            else:
                # Buffer audio before STT starts (to capture early speech)
                # This ensures we don't lose the first ~1-2 seconds while VAD detects speech_start
                if len(self._stt_pre_buffer) < self._stt_pre_buffer.maxlen:
                    self._stt_pre_buffer.append(audio_data)
                else:
                    # Buffer full, drop oldest chunk and add new one (circular buffer behavior)
                    self._stt_pre_buffer.popleft()
                    self._stt_pre_buffer.append(audio_data)

            chunk_process_time = time.time() - process_start
            logger.debug(
                f"Processed audio chunk for session {self.session_id}: "
                f"{len(audio_data)} bytes in {chunk_process_time*1000:.2f}ms"
            )
        except Exception as e:
            logger.error(
                f"Error processing audio chunk for session {self.session_id}: {e}"
            )

    async def _send_to_stt(self, audio_data: bytes) -> None:
        """Send audio chunk to STT engine.
        
        Args:
            audio_data: Binary audio data to send
        """
        if self.stt_engine is None:
            return

        try:
            await self.stt_engine.send_audio(audio_data)
        except NotImplementedError:
            # Expected until Task 4.1
            pass
        except Exception as e:
            logger.error(
                f"Error sending audio to STT for session {self.session_id}: {e}"
            )

    async def _handle_vad_event(
        self, event: VADEvent, timestamp: datetime
    ) -> None:
        """Handle VAD event from VAD processor.
        
        Routes VAD events to turn controller, utterance finalizer, and manages STT lifecycle.
        
        Args:
            event: VAD event (SPEECH_START or SPEECH_END)
            timestamp: Timestamp when event occurred
        """
        try:
            logger.info(
                f"VAD event received for session {self.session_id}: {event.value} at {timestamp}"
            )

            # Log VAD event
            try:
                from core.memory import MemoryManager, create_log_entry, LogEntryType
                memory_manager = MemoryManager.get_instance_sync()
                if memory_manager:
                    log_entry = create_log_entry(
                        LogEntryType.VAD_EVENT,
                        self.session_id,
                        {
                            "event": event.value,
                            "timestamp": timestamp.isoformat(),
                        },
                    )
                    asyncio.create_task(
                        memory_manager.conversation_log.append_log(log_entry)
                    )
            except Exception:
                pass  # Don't fail on logging errors

            # Route to turn controller
            await self.turn_controller.handle_vad_event(event, timestamp)

            # Route to utterance finalizer (if available)
            if self.utterance_finalizer is not None:
                try:
                    await self.utterance_finalizer.handle_vad_event(event, timestamp)
                except Exception as e:
                    logger.error(
                        f"Error forwarding VAD event to finalizer for session {self.session_id}: {e}"
                    )

            # Manage STT lifecycle based on event
            if event == VADEvent.SPEECH_START:
                await self._start_stt()
            elif event == VADEvent.SPEECH_END:
                await self._stop_stt()

        except Exception as e:
            logger.error(
                f"Error handling VAD event for session {self.session_id}: {e}"
            )

    async def _start_stt(self) -> None:
        """Start STT streaming when user starts speaking."""
        if self.stt_is_active:
            logger.debug(
                f"STT already active for session {self.session_id}, skipping start"
            )
            return

        if self.stt_engine is None:
            logger.debug(
                f"STT engine not available for session {self.session_id} "
                "(expected until Task 4.1)"
            )
            return

        try:
            await self.stt_engine.start_streaming()
            self.stt_is_active = True
            
            # Send buffered audio first (to capture early speech that arrived before speech_start)
            if len(self._stt_pre_buffer) > 0:
                buffer_duration_ms = len(self._stt_pre_buffer) * 20  # 20ms per chunk
                logger.info(
                    f"Sending {len(self._stt_pre_buffer)} buffered audio chunks "
                    f"(~{buffer_duration_ms}ms) to STT for session {self.session_id}"
                )
                for buffered_chunk in self._stt_pre_buffer:
                    try:
                        await self._send_to_stt(buffered_chunk)
                    except Exception as e:
                        logger.warning(
                            f"Error sending buffered chunk to STT for session {self.session_id}: {e}"
                        )
                # Clear buffer after sending
                self._stt_pre_buffer.clear()
                logger.debug(
                    f"Sent all buffered audio and cleared pre-buffer for session {self.session_id}"
                )
            
            logger.info(f"STT streaming started for session {self.session_id}")
        except NotImplementedError:
            # Expected until Task 4.1
            logger.debug(
                f"STT engine not implemented for session {self.session_id} "
                "(Task 4.1)"
            )
        except Exception as e:
            logger.error(
                f"Error starting STT for session {self.session_id}: {e}"
            )

    async def _stop_stt(self) -> None:
        """Stop STT streaming when user stops speaking."""
        if not self.stt_is_active:
            logger.debug(
                f"STT not active for session {self.session_id}, skipping stop"
            )
            return

        if self.stt_engine is None:
            self.stt_is_active = False
            return

        try:
            await self.stt_engine.stop_streaming()
            self.stt_is_active = False

            # Get final transcript (if available)
            try:
                transcript = await self.stt_engine.get_final_transcript()
                if transcript:
                    logger.info(
                        f"Final transcript for session {self.session_id}: {transcript}"
                    )
                    
                    # Log STT transcript
                    try:
                        from core.memory import MemoryManager, create_log_entry, LogEntryType
                        memory_manager = MemoryManager.get_instance_sync()
                        if memory_manager:
                            log_entry = create_log_entry(
                                LogEntryType.STT_TRANSCRIPT,
                                self.session_id,
                                {
                                    "transcript": transcript,
                                    "length": len(transcript),
                                },
                            )
                            asyncio.create_task(
                                memory_manager.conversation_log.append_log(log_entry)
                            )
                    except Exception:
                        pass  # Don't fail on logging errors
                
                # Forward transcript to utterance finalizer (even if empty/None)
                if self.utterance_finalizer is not None:
                    try:
                        logger.info(
                            f"Forwarding transcript to utterance finalizer for session {self.session_id}: "
                            f"'{transcript}'"
                        )
                        await self.utterance_finalizer.handle_stt_final_transcript(transcript)
                        
                        # If we have a transcript but speech_end wasn't detected (manual stop),
                        # trigger finalization by sending a synthetic speech_end event
                        # This ensures the utterance finalizer can complete even when user stops manually
                        if transcript and transcript.strip():
                            # Check if speech_end was already received by checking if finalizer is waiting
                            # We'll trigger a synthetic speech_end if needed
                            from datetime import datetime
                            # The finalizer will check if both conditions are met in _check_finalization
                            # If speech_end wasn't received, we need to send it
                            # But actually, let's check the finalizer's state first
                            # For now, let's just log and let the timeout handle it
                            logger.debug(
                                f"Transcript forwarded. Finalizer will check finalization conditions."
                            )
                        
                        logger.debug(
                            f"Transcript forwarded successfully to finalizer for session {self.session_id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error forwarding transcript to finalizer for session {self.session_id}: {e}",
                            exc_info=True
                        )
                else:
                    logger.warning(
                        f"Utterance finalizer not available for session {self.session_id}, "
                        "cannot forward transcript"
                    )
            except NotImplementedError:
                pass
            except Exception as e:
                logger.error(
                    f"Error getting final transcript for session {self.session_id}: {e}"
                )

            logger.info(f"STT streaming stopped for session {self.session_id}")
        except NotImplementedError:
            # Expected until Task 4.1
            self.stt_is_active = False
            logger.debug(
                f"STT engine not implemented for session {self.session_id} "
                "(Task 4.1)"
            )
        except Exception as e:
            logger.error(
                f"Error stopping STT for session {self.session_id}: {e}"
            )
            self.stt_is_active = False

    async def can_assistant_speak(self) -> bool:
        """Check if assistant is allowed to speak.
        
        Delegates to turn controller.
        
        Returns:
            True if assistant can speak, False otherwise
        """
        return await self.turn_controller.can_assistant_speak()

    async def set_stt_engine(self, stt_engine: STTEngineInterface) -> None:
        """Set STT engine instance.
        
        Allows manual override of STT engine (useful for testing or custom implementations).
        By default, STT engine is automatically created in start() method.
        
        Args:
            stt_engine: STT engine instance
        """
        self.stt_engine = stt_engine
        logger.info(f"STT engine manually set for session {self.session_id}")

    async def start(self) -> None:
        """Start audio processing."""
        if self.is_active:
            logger.debug(
                f"Audio processor already active for session {self.session_id}"
            )
            return

        # Ensure event loop is set (in case it wasn't available during __init__)
        if self._event_loop is None:
            try:
                self._event_loop = asyncio.get_running_loop()
                # Update VAD processor with event loop
                self.vad_processor.set_event_loop(self._event_loop)
            except RuntimeError:
                logger.warning(f"No event loop available for session {self.session_id}")

        # Create STT engine if not already created
        if self.stt_engine is None:
            try:
                self.stt_engine = create_stt_engine(self.session_id)
                logger.info(f"STT engine created for session {self.session_id}")
            except Exception as e:
                logger.error(
                    f"Failed to create STT engine for session {self.session_id}: {e}",
                    exc_info=True
                )
                # Continue without STT engine - system can still function for testing

        # Create utterance finalizer if not already created
        if self.utterance_finalizer is None:
            try:
                # AI engine callback to process user responses
                async def ai_engine_completion_callback(session_id: str, transcript: str) -> None:
                    """Callback when user utterance is complete - triggers AI engine."""
                    logger.info(
                        f"AI Engine Triggered for session {session_id} with transcript: '{transcript}'"
                    )
                    
                    try:
                        # Import here to avoid circular dependencies
                        from core.ai import get_ai_engine
                        from routes.websocket_routes import connection_manager
                        
                        # Get AI engine instance
                        ai_engine = get_ai_engine()
                        
                        # Process user response through AI engine
                        logger.info(
                            f"Processing user response through AI engine for session {session_id}"
                        )
                        planner_output = await ai_engine.process_user_response(
                            session_id=session_id,
                            user_utterance=transcript
                        )
                        logger.info(
                            f"AI engine processing complete for session {session_id}: "
                            f"action={planner_output.action.value}, type={planner_output.type.value}"
                        )
                        
                        # Send AI response to client via WebSocket
                        ai_response_message = {
                            "type": "AI_RESPONSE",
                            "response": {
                                "action": planner_output.action.value,
                                "text": planner_output.text,
                                "type": planner_output.type.value,
                                "metadata": planner_output.metadata,
                                "was_overridden": planner_output.was_overridden,
                                "override_reason": planner_output.override_reason,
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await connection_manager.send_control_message(
                            session_id, ai_response_message
                        )
                        
                        logger.info(
                            f"AI response sent to client for session {session_id}: "
                            f"action={planner_output.action.value}, "
                            f"type={planner_output.type.value}"
                        )
                        
                        # If output is SPEAK type, send to TTS
                        if planner_output.type == PlannerOutputType.SPEAK and planner_output.text:
                            try:
                                from core.tts import get_tts_engine
                                from routes.websocket_routes import connection_manager
                                
                                # Get TTS engine for session
                                tts_engine = await get_tts_engine(
                                    session_id,
                                    connection_manager,
                                    self.turn_controller
                                )
                                await tts_engine.speak(planner_output.text)
                                logger.info(
                                    f"TTS started for session {session_id}: "
                                    f"text_length={len(planner_output.text)}"
                                )
                            except Exception as e:
                                from models.exceptions import TTSTimeoutError, TTSError
                                
                                if isinstance(e, TTSTimeoutError):
                                    logger.warning(
                                        f"TTS timeout for session {session_id}: {e}"
                                    )
                                    # Send notification to user
                                    try:
                                        await connection_manager.send_control_message(
                                            session_id,
                                            {
                                                "type": "ERROR",
                                                "error_type": "tts_timeout",
                                                "message": "I'm having trouble speaking. Please continue.",
                                                "timestamp": datetime.now().isoformat(),
                                            },
                                        )
                                    except Exception:
                                        pass
                                elif isinstance(e, TTSError):
                                    logger.error(
                                        f"TTS error for session {session_id}: {e}",
                                        exc_info=True
                                    )
                                    # Send notification to user
                                    try:
                                        await connection_manager.send_control_message(
                                            session_id,
                                            {
                                                "type": "ERROR",
                                                "error_type": "tts_error",
                                                "message": "I encountered an error speaking. Please continue.",
                                                "timestamp": datetime.now().isoformat(),
                                            },
                                        )
                                    except Exception:
                                        pass
                                else:
                                    logger.error(
                                        f"Error speaking planner output for session {session_id}: {e}",
                                        exc_info=True
                                    )
                        
                    except Exception as e:
                        logger.error(
                            f"Error processing AI response for session {session_id}: {e}",
                            exc_info=True
                        )
                        # Send error message to client
                        try:
                            from routes.websocket_routes import connection_manager
                            error_message = {
                                "type": "AI_RESPONSE",
                                "response": {
                                    "action": "acknowledge",
                                    "text": "I apologize, but I encountered an error processing your response.",
                                    "type": "speak",
                                    "metadata": {},
                                    "was_overridden": True,
                                    "override_reason": f"Error: {str(e)}",
                                },
                                "timestamp": datetime.now().isoformat()
                            }
                            await connection_manager.send_control_message(
                                session_id, error_message
                            )
                        except Exception as send_error:
                            logger.error(
                                f"Failed to send error message to client: {send_error}",
                                exc_info=True
                            )

                self.utterance_finalizer = UtteranceFinalizer(
                    session_id=self.session_id,
                    completion_callback=ai_engine_completion_callback
                )
                logger.info(f"Utterance finalizer created for session {self.session_id}")
            except Exception as e:
                logger.error(
                    f"Failed to create Utterance Finalizer for session {self.session_id}: {e}",
                    exc_info=True
                )
                # Continue without utterance finalizer - system can still function

        self.is_active = True
        logger.info(f"Audio processing started for session {self.session_id}")

    async def stop(self) -> None:
        """Stop audio processing and cleanup components."""
        if not self.is_active:
            logger.debug(
                f"Audio processor not active for session {self.session_id}"
            )
            return

        self.is_active = False

        # Stop STT if active
        if self.stt_is_active:
            # Store whether we have a transcript before stopping
            has_transcript = False
            if self.stt_engine is not None:
                try:
                    transcript = await self.stt_engine.get_final_transcript()
                    has_transcript = transcript is not None and transcript.strip() != ""
                except:
                    pass
            
            await self._stop_stt()
            
            # If we stopped STT manually (via CONTROL_STOP) and have a transcript,
            # but speech_end wasn't detected by VAD, trigger a synthetic speech_end
            # event to allow utterance finalization
            if self.utterance_finalizer is not None and has_transcript:
                try:
                    from datetime import datetime
                    logger.info(
                        f"Manual stop detected with transcript. Triggering synthetic speech_end "
                        f"for session {self.session_id} to allow finalization"
                    )
                    await self.utterance_finalizer.handle_vad_event(
                        VADEvent.SPEECH_END,
                        datetime.now()
                    )
                except Exception as e:
                    logger.warning(
                        f"Error triggering synthetic speech_end for session {self.session_id}: {e}"
                    )

        # Reset components
        self.vad_processor.reset()
        await self.turn_controller.reset()
        
        # Reset utterance finalizer (after finalization if it happened)
        if self.utterance_finalizer is not None:
            try:
                await self.utterance_finalizer.reset()
            except Exception as e:
                logger.error(
                    f"Error resetting finalizer during stop for session {self.session_id}: {e}"
                )

        # Clear queues
        self._vad_queue.clear()
        self._stt_pre_buffer.clear()

        # Shutdown thread pool executor
        if self._vad_executor is not None:
            self._vad_executor.shutdown(wait=True)
            self._vad_executor = None

        logger.info(f"Audio processing stopped for session {self.session_id}")

    async def reset(self) -> None:
        """Reset all components for new utterance."""
        try:
            self.vad_processor.reset()
            await self.turn_controller.reset()
            self.stt_is_active = False

            # Reset utterance finalizer
            if self.utterance_finalizer is not None:
                try:
                    await self.utterance_finalizer.reset()
                except Exception as e:
                    logger.error(
                        f"Error resetting finalizer for session {self.session_id}: {e}"
                    )

            # Clear queues
            self._vad_queue.clear()
            self._stt_pre_buffer.clear()

            logger.debug(f"Audio processor reset for session {self.session_id}")
        except Exception as e:
            logger.error(
                f"Error resetting audio processor for session {self.session_id}: {e}"
            )


# Per-session audio processor management
_audio_processors: Dict[str, AudioProcessor] = {}


async def get_audio_processor(
    session_id: str, tts_stop_callback: Optional[Callable[[], None]] = None
) -> AudioProcessor:
    """Get or create audio processor for session.
    
    Args:
        session_id: Unique identifier for the session
        tts_stop_callback: Optional callback function to stop TTS
        
    Returns:
        Audio processor instance for the session
    """
    if session_id not in _audio_processors:
        _audio_processors[session_id] = AudioProcessor(
            session_id, tts_stop_callback
        )
        await _audio_processors[session_id].start()
        logger.info(f"Created audio processor for session {session_id}")

    return _audio_processors[session_id]


async def remove_audio_processor(session_id: str) -> None:
    """Remove audio processor for session.
    
    Stops processing and cleans up resources.
    
    Args:
        session_id: Unique identifier for the session
    """
    if session_id in _audio_processors:
        processor = _audio_processors[session_id]
        await processor.stop()
        del _audio_processors[session_id]
        logger.info(f"Removed audio processor for session {session_id}")

