"""Silero VAD (Voice Activity Detection) processor for real-time speech detection.

This module implements a VAD processor using Silero VAD library to detect
when users start and stop speaking. It processes audio chunks in real-time,
buffers them for analysis, and emits events via callbacks.

Reference: https://github.com/snakers4/silero-vad
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch

from models.constants import VADEvent
from util.audio_config import (
    BROWSER_CHUNK_SIZE_BYTES,
    SAMPLE_RATE,
    calculate_audio_duration,
    validate_audio_format,
)
from util.logger import get_vad_logger

logger = logging.getLogger(__name__)


class VADState(str, Enum):
    """Internal VAD state enumeration."""

    SILENT = "silent"
    SPEECH_STARTING = "speech_starting"
    SPEAKING = "speaking"
    SPEECH_ENDING = "speech_ending"


class VADProcessor:
    """Voice Activity Detection processor using Silero VAD.

    Processes audio chunks in real-time to detect speech start and end events.
    Uses a buffering strategy to handle streaming audio with Silero VAD model.

    Attributes:
        model: Silero VAD model instance
        speech_start_threshold: Threshold for detecting speech start (0.0-1.0)
        silence_threshold: Threshold for detecting silence (0.0-1.0)
        min_speech_duration_ms: Minimum speech duration before emitting start event
        min_silence_duration_ms: Minimum silence duration before emitting end event
        buffer_duration_ms: Duration of audio buffer for processing
        process_interval_chunks: Number of chunks to accumulate before processing
        audio_buffer: Rolling buffer for audio chunks
        state: Current VAD state
        speech_start_time: Timestamp when speech started
        silence_start_time: Timestamp when silence started
        chunk_count: Counter for chunks processed
        callback: Registered callback function for VAD events
    """

    def __init__(
        self,
        speech_start_threshold: float = 0.5,
        silence_threshold: float = 0.3,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        buffer_duration_ms: int = 1500,
        process_interval_chunks: int = 5,
    ) -> None:
        """Initialize VAD processor with configuration.

        Args:
            speech_start_threshold: Threshold for detecting speech start (0.0-1.0)
            silence_threshold: Threshold for detecting silence (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration before emitting start (ms)
            min_silence_duration_ms: Minimum silence duration before emitting end (ms)
            buffer_duration_ms: Duration of audio buffer for processing (ms)
            process_interval_chunks: Number of chunks to accumulate before processing

        Raises:
            RuntimeError: If model loading fails
        """
        # Validate thresholds
        if not 0.0 <= speech_start_threshold <= 1.0:
            raise ValueError(
                f"speech_start_threshold must be between 0.0 and 1.0, got {speech_start_threshold}"
            )
        if not 0.0 <= silence_threshold <= 1.0:
            raise ValueError(
                f"silence_threshold must be between 0.0 and 1.0, got {silence_threshold}"
            )

        self.speech_start_threshold = speech_start_threshold
        self.silence_threshold = silence_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.buffer_duration_ms = buffer_duration_ms
        self.process_interval_chunks = process_interval_chunks

        # Initialize audio buffer (store samples, not bytes)
        # Silero VAD requires exactly 512 samples (32ms at 16kHz) per inference
        max_buffer_samples = (SAMPLE_RATE * buffer_duration_ms) // 1000
        self.audio_buffer_samples: np.ndarray = np.zeros(max_buffer_samples, dtype=np.float32)
        self.buffer_write_pos = 0  # Position to write next samples
        self.buffer_filled = False  # Whether buffer has enough samples

        # Silero VAD requires exactly 512 samples at 16kHz (32ms)
        self.VAD_FRAME_SIZE = 512  # Samples required by Silero VAD at 16kHz

        # Initialize state
        self.state = VADState.SILENT
        self.speech_start_time: Optional[datetime] = None
        self.silence_start_time: Optional[datetime] = None
        self.chunk_count = 0
        self.callback: Optional[Callable[[VADEvent, datetime], None]] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize structured logger
        self.structured_logger = get_vad_logger()

        # Load Silero VAD model
        try:
            logger.info("Loading Silero VAD model...")
            self.model = self._load_model()
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load Silero VAD model: {e}")
            # Set model to None to indicate failure - processing will use fallback
            self.model = None
            logger.warning(
                "VAD model failed to load. VAD processing will use fallback heuristics. "
                "Audio pipeline will continue to function."
            )

    def _load_model(self) -> torch.nn.Module:
        """Load Silero VAD model.

        Returns:
            Loaded Silero VAD model

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Try using silero_vad package first
            from silero_vad import load_silero_vad

            model = load_silero_vad()
            logger.debug("Loaded Silero VAD model using silero_vad package")
            return model
        except ImportError:
            # Fallback to torch.hub
            try:
                logger.debug("Trying torch.hub to load Silero VAD model")
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                )
                logger.debug("Loaded Silero VAD model using torch.hub")
                return model
            except Exception as e:
                logger.exception(f"Failed to load model via torch.hub: {e}")
                raise RuntimeError(f"Failed to load Silero VAD model: {e}") from e

    def _bytes_to_audio_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to normalized float32 numpy array.

        Args:
            audio_bytes: Binary audio data (16-bit PCM, little-endian)

        Returns:
            Normalized float32 numpy array (-1.0 to 1.0 range)
        """
        # Convert bytes to int16 array
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        # Normalize to float32 (-1.0 to 1.0)
        audio_array = samples.astype(np.float32) / 32768.0
        return audio_array

    def _process_buffered_audio(self) -> Optional[float]:
        """Process buffered audio through VAD model.

        Silero VAD requires exactly 512 samples (32ms at 16kHz) per inference.
        We take the most recent 512 samples from the buffer for processing.

        Returns:
            Speech probability (0.0-1.0) if processing successful, None otherwise
        """
        # Check if we have enough samples (need at least 512)
        if not self.buffer_filled:
            # Buffer not filled yet - need at least 512 samples written
            if self.buffer_write_pos < self.VAD_FRAME_SIZE:
                return None
            # We have enough samples, but buffer not wrapped yet
            # Take the last 512 samples from the filled portion
            audio_array = self.audio_buffer_samples[
                self.buffer_write_pos - self.VAD_FRAME_SIZE : self.buffer_write_pos
            ].copy()
        else:
            # Buffer is filled (circular), get most recent 512 samples
            buffer_size = len(self.audio_buffer_samples)
            audio_array = np.zeros(self.VAD_FRAME_SIZE, dtype=np.float32)
            
            # Extract last 512 samples from circular buffer
            # Start position: (write_pos - 512) mod buffer_size
            start_pos = (self.buffer_write_pos - self.VAD_FRAME_SIZE) % buffer_size
            
            if start_pos + self.VAD_FRAME_SIZE <= buffer_size:
                # No wrap-around, can copy directly
                audio_array = self.audio_buffer_samples[start_pos:start_pos + self.VAD_FRAME_SIZE].copy()
            else:
                # Wrap-around case: copy in two parts
                first_part_size = buffer_size - start_pos
                second_part_size = self.VAD_FRAME_SIZE - first_part_size
                audio_array[:first_part_size] = self.audio_buffer_samples[start_pos:].copy()
                audio_array[first_part_size:] = self.audio_buffer_samples[:second_part_size].copy()

        try:
            # Check if model is available
            if self.model is None:
                # Model failed to load - use fallback heuristic
                audio_energy = np.abs(audio_array).mean()
                speech_prob = min(1.0, audio_energy * 10.0)
                logger.debug(
                    "VAD model not available, using fallback heuristic. "
                    f"Audio energy: {audio_energy:.4f}, speech_prob: {speech_prob:.3f}"
                )
                return speech_prob

            # Ensure we have exactly 512 samples
            if len(audio_array) != self.VAD_FRAME_SIZE:
                logger.warning(
                    f"Expected {self.VAD_FRAME_SIZE} samples, got {len(audio_array)}"
                )
                return None

            # Convert to torch tensor (shape: [1, 512])
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)

            # Process through model
            with torch.no_grad():
                # Silero VAD model forward pass
                # Model expects (batch, samples) tensor with exactly 512 samples
                try:
                    # Try direct model call (for newer API)
                    output = self.model(audio_tensor, SAMPLE_RATE)
                    # Extract probability (model may return tensor or float)
                    if isinstance(output, torch.Tensor):
                        speech_prob = output.item()
                    else:
                        speech_prob = float(output)
                except (TypeError, AttributeError):
                    # Fallback: try using model's forward method
                    try:
                        output = self.model.forward(audio_tensor, SAMPLE_RATE)
                        if isinstance(output, torch.Tensor):
                            speech_prob = output.item()
                        else:
                            speech_prob = float(output)
                    except Exception as e:
                        logger.warning(f"Error in model forward pass: {e}, using fallback")
                        # Fallback: use audio energy as heuristic
                        audio_energy = np.abs(audio_array).mean()
                        speech_prob = min(1.0, audio_energy * 10.0)
                        logger.debug(
                            "Using fallback speech probability calculation. "
                            f"Audio energy: {audio_energy:.4f}, speech_prob: {speech_prob:.3f}"
                        )

            return speech_prob

        except Exception as e:
            logger.error(f"Error processing buffered audio: {e}", exc_info=True)
            # Return fallback probability instead of None to keep pipeline running
            try:
                audio_energy = np.abs(audio_array).mean()
                speech_prob = min(1.0, audio_energy * 10.0)
                logger.warning(
                    f"VAD processing failed, using fallback. "
                    f"Audio energy: {audio_energy:.4f}, speech_prob: {speech_prob:.3f}"
                )
                return speech_prob
            except Exception as fallback_error:
                logger.error(f"Fallback calculation also failed: {fallback_error}")
                return None

    def _update_state(self, speech_prob: float, current_time: datetime) -> None:
        """Update VAD state based on speech probability.

        Args:
            speech_prob: Speech probability from model (0.0-1.0)
            current_time: Current timestamp
        """
        if self.state == VADState.SILENT:
            if speech_prob >= self.speech_start_threshold:
                # Transition to SPEECH_STARTING
                self.state = VADState.SPEECH_STARTING
                self.speech_start_time = current_time
                logger.debug(f"State: SILENT -> SPEECH_STARTING (prob: {speech_prob:.3f})")

        elif self.state == VADState.SPEECH_STARTING:
            if speech_prob >= self.speech_start_threshold:
                # Check if min duration elapsed
                if self.speech_start_time:
                    elapsed_ms = (current_time - self.speech_start_time).total_seconds() * 1000
                    if elapsed_ms >= self.min_speech_duration_ms:
                        # Transition to SPEAKING and emit event
                        self.state = VADState.SPEAKING
                        self._emit_event(VADEvent.SPEECH_START, current_time)
                        logger.info(f"State: SPEECH_STARTING -> SPEAKING (prob: {speech_prob:.3f})")
            else:
                # Probability dropped, go back to SILENT
                self.state = VADState.SILENT
                self.speech_start_time = None
                logger.debug(f"State: SPEECH_STARTING -> SILENT (prob: {speech_prob:.3f})")

        elif self.state == VADState.SPEAKING:
            if speech_prob < self.silence_threshold:
                # Transition to SPEECH_ENDING
                self.state = VADState.SPEECH_ENDING
                self.silence_start_time = current_time
                logger.debug(f"State: SPEAKING -> SPEECH_ENDING (prob: {speech_prob:.3f})")
                self.structured_logger.debug(
                    "vad_state_transition",
                    "VAD state transition: SPEAKING -> SPEECH_ENDING",
                    {"speech_prob": round(speech_prob, 3), "silence_threshold": self.silence_threshold}
                )

        elif self.state == VADState.SPEECH_ENDING:
            if speech_prob < self.silence_threshold:
                # Check if min silence duration elapsed
                if self.silence_start_time:
                    elapsed_ms = (current_time - self.silence_start_time).total_seconds() * 1000
                    if elapsed_ms >= self.min_silence_duration_ms:
                        # Transition to SILENT and emit event
                        self.state = VADState.SILENT
                        self._emit_event(VADEvent.SPEECH_END, current_time)
                        self.speech_start_time = None
                        self.silence_start_time = None
                        logger.info(f"State: SPEECH_ENDING -> SILENT (prob: {speech_prob:.3f})")
            else:
                # Probability increased, go back to SPEAKING
                self.state = VADState.SPEAKING
                self.silence_start_time = None
                logger.debug(f"State: SPEECH_ENDING -> SPEAKING (prob: {speech_prob:.3f})")

    def _emit_event(self, event: VADEvent, timestamp: datetime) -> None:
        """Emit VAD event via callback.

        Args:
            event: VAD event to emit
            timestamp: Timestamp when event occurred
        """
        if self.callback:
            try:
                # Call callback (may be async or sync)
                if asyncio.iscoroutinefunction(self.callback):
                    # Check if we're in a thread (no running event loop)
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in the main event loop, can use create_task
                        asyncio.create_task(self.callback(event, timestamp))
                    except RuntimeError:
                        # No running loop - we're in a thread, need to schedule to main loop
                        if self.event_loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                self.callback(event, timestamp),
                                self.event_loop
                            )
                        else:
                            logger.error(
                                "Cannot schedule async callback: no event loop available. "
                                "VAD processor may be running in a thread without event loop reference."
                            )
                else:
                    # Call sync callback
                    self.callback(event, timestamp)
            except Exception as e:
                logger.error(f"Error in VAD event callback: {e}")

    def process_chunk(self, audio_data: bytes) -> None:
        """Process single audio chunk.

        Adds chunk to buffer and processes buffered audio if needed.
        Silero VAD requires exactly 512 samples per inference, so we buffer
        samples until we have enough, then process the most recent 512 samples.

        Args:
            audio_data: Binary audio data (16-bit PCM, 16kHz, mono)

        Raises:
            ValueError: If audio data is invalid
        """
        # #region agent log
        vad_start = time.time()
        # #endregion
        
        # Validate audio format
        is_valid, error_msg = validate_audio_format(audio_data)
        if not is_valid:
            logger.warning(f"Invalid audio chunk: {error_msg}")
            return

        # Convert bytes to audio array
        # #region agent log
        convert_start = time.time()
        # #endregion
        audio_array = self._bytes_to_audio_array(audio_data)
        num_samples = len(audio_array)
        # #region agent log
        convert_end = time.time()
        # #endregion

        # Add samples to buffer (circular buffer)
        # #region agent log
        buffer_start = time.time()
        # #endregion
        buffer_size = len(self.audio_buffer_samples)
        for i, sample in enumerate(audio_array):
            pos = (self.buffer_write_pos + i) % buffer_size
            self.audio_buffer_samples[pos] = sample

        # Update write position
        self.buffer_write_pos = (self.buffer_write_pos + num_samples) % buffer_size

        # Mark buffer as filled once we've written at least one full cycle
        if not self.buffer_filled and self.buffer_write_pos >= self.VAD_FRAME_SIZE:
            self.buffer_filled = True
        # #region agent log
        buffer_end = time.time()
        # #endregion

        self.chunk_count += 1

        # Process buffered audio periodically (only if we have enough samples)
        will_process = (self.chunk_count % self.process_interval_chunks == 0)
        # #region agent log
        model_start = None
        model_end = None
        if will_process:
            model_start = time.time()
        # #endregion
        
        if will_process:
            try:
                current_time = datetime.now()
                speech_prob = self._process_buffered_audio()
                # #region agent log
                model_end = time.time()
                # #endregion

                if speech_prob is not None:
                    try:
                        self._update_state(speech_prob, current_time)
                        logger.debug(
                            f"Processed chunk {self.chunk_count}, speech_prob: {speech_prob:.3f}"
                        )
                    except Exception as state_error:
                        logger.error(
                            f"Error updating VAD state: {state_error}",
                            exc_info=True
                        )
                        # Continue processing - state update failure shouldn't crash pipeline
            except Exception as e:
                logger.error(
                    f"Error in VAD processing for chunk {self.chunk_count}: {e}",
                    exc_info=True
                )
                # Continue processing - VAD errors shouldn't crash audio pipeline
                # #region agent log
                model_end = time.time()
                # #endregion
        
        # #region agent log
        vad_end = time.time()
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(vad_end*1000)}_vad","timestamp":int(vad_end*1000),"location":"vad_processor.py:333","message":"VAD process_chunk complete","data":{"chunkCount":self.chunk_count,"totalDuration":vad_end-vad_start,"convertDuration":convert_end-convert_start,"bufferDuration":buffer_end-buffer_start,"modelDuration":(model_end-model_start) if model_start and model_end else None,"willProcess":will_process},"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,E"}) + "\n")
        # #endregion

    def register_callback(
        self, 
        callback: Callable[[VADEvent, datetime], None],
        event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Register callback function for VAD events.

        Args:
            callback: Callback function that receives (event, timestamp)
                Can be async or sync function
            event_loop: Optional event loop for scheduling async callbacks from threads
        """
        self.callback = callback
        self.event_loop = event_loop
        logger.debug("VAD event callback registered")
    
    def set_event_loop(self, event_loop: asyncio.AbstractEventLoop) -> None:
        """Set event loop for scheduling async callbacks from threads.
        
        Args:
            event_loop: Event loop to use for thread-safe callback scheduling
        """
        self.event_loop = event_loop
        logger.debug("VAD processor event loop set")

    def reset(self) -> None:
        """Reset VAD state and buffer.

        Clears audio buffer and resets state machine to SILENT.
        Useful when starting a new session or utterance.
        """
        self.audio_buffer_samples.fill(0.0)
        self.buffer_write_pos = 0
        self.buffer_filled = False
        self.state = VADState.SILENT
        self.speech_start_time = None
        self.silence_start_time = None
        self.chunk_count = 0
        logger.debug("VAD state and buffer reset")

    def cleanup(self) -> None:
        """Clean up resources.

        Clears buffer and resets state. Model remains loaded for reuse.
        """
        self.reset()
        logger.debug("VAD processor cleaned up")


def create_vad_processor_from_env() -> VADProcessor:
    """Create VAD processor with configuration from environment variables.

    Reads configuration from environment variables with defaults:
    - VAD_SPEECH_START_THRESHOLD (default: 0.5)
    - VAD_SILENCE_THRESHOLD (default: 0.3)
    - VAD_MIN_SPEECH_DURATION_MS (default: 250)
    - VAD_MIN_SILENCE_DURATION_MS (default: 500)

    Returns:
        Configured VADProcessor instance
    """
    speech_start_threshold = float(
        os.getenv("VAD_SPEECH_START_THRESHOLD", "0.5")
    )
    silence_threshold = float(os.getenv("VAD_SILENCE_THRESHOLD", "0.3"))
    min_speech_duration_ms = int(
        os.getenv("VAD_MIN_SPEECH_DURATION_MS", "250")
    )
    min_silence_duration_ms = int(
        os.getenv("VAD_MIN_SILENCE_DURATION_MS", "500")
    )

    return VADProcessor(
        speech_start_threshold=speech_start_threshold,
        silence_threshold=silence_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

