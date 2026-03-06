"""Audio format configuration constants, validation, and utility functions.

This module defines the audio format standards used throughout the interview system,
including constants for sample rates, formats, chunk sizes, and helper functions
for audio validation and calculations.

Audio Format Specification:
    - Format: PCM (Pulse Code Modulation)
    - Bit depth: 16-bit
    - Endianness: Little-endian (linear16)
    - Sample rate: 16000 Hz (fixed, no resampling)
    - Channels: Mono (1 channel)
    - Chunk size: 20ms from browser (640 bytes), 80ms for Deepgram (2560 bytes)

Requirements:
    - All audio chunks must be 16-bit PCM little-endian format
    - Sample rate must be 16000 Hz (no resampling in v1)
    - Browser sends 20ms chunks (640 bytes at 16kHz)
    - Deepgram recommends 80ms chunks (2560 bytes) for optimal performance
    - Audio data must be non-empty and size must be multiple of 2 (16-bit samples)

Validation Rules:
    - Chunk size must match expected size (with small tolerance for network variations)
    - Data must not be empty
    - Size must be multiple of 2 (16-bit requirement)
    - Size must be within reasonable bounds (MIN_CHUNK_SIZE_MS to MAX_CHUNK_SIZE_MS)

Deepgram Compatibility:
    - Encoding: "linear16" (Deepgram's name for 16-bit PCM little-endian)
    - Sample rate: 16000 Hz (recommended by Deepgram)
    - Chunk size: 80ms recommended for optimal performance
    - Format: Mono audio only

Usage Examples:
    >>> from util.audio_config import validate_audio_chunk, BROWSER_CHUNK_SIZE_BYTES
    >>> audio_data = b'...'  # 640 bytes of PCM audio
    >>> is_valid, error = validate_audio_chunk(audio_data, BROWSER_CHUNK_SIZE_BYTES)
    >>> if not is_valid:
    ...     print(f"Invalid audio: {error}")
    
    >>> from util.audio_config import bytes_to_samples, calculate_audio_duration
    >>> samples = bytes_to_samples(audio_data)  # 320 samples
    >>> duration_ms = calculate_audio_duration(audio_data)  # 20.0 ms
"""

from typing import Optional, Tuple, List

# ============================================================================
# Audio Format Constants
# ============================================================================

# Core audio format constants
SAMPLE_RATE: int = 16000  # Hz (Deepgram recommended)
AUDIO_FORMAT: str = "pcm_s16le"  # 16-bit PCM little endian (internal reference)
BYTES_PER_SAMPLE: int = 2  # 16-bit = 2 bytes per sample

# Deepgram-specific encoding
DEEPGRAM_ENCODING: str = "linear16"  # Deepgram name for 16-bit PCM little-endian
DEEPGRAM_CHANNELS: int = 1  # Mono audio

# Supported sample rates (for future flexibility)
SUPPORTED_SAMPLE_RATES: List[int] = [8000, 16000, 24000, 44100, 48000]

# Chunk size constants
# Browser sends 20ms chunks, Deepgram recommends 80ms chunks for optimal performance
BROWSER_CHUNK_SIZE_MS: int = 20  # milliseconds (what browser sends)
BROWSER_CHUNK_SIZE_SAMPLES: int = (SAMPLE_RATE * BROWSER_CHUNK_SIZE_MS) // 1000
BROWSER_CHUNK_SIZE_BYTES: int = (SAMPLE_RATE * BROWSER_CHUNK_SIZE_MS * 2) // 1000  # 640 bytes

DEEPGRAM_CHUNK_SIZE_MS: int = 80  # milliseconds (Deepgram recommended)
DEEPGRAM_CHUNK_SIZE_SAMPLES: int = (SAMPLE_RATE * DEEPGRAM_CHUNK_SIZE_MS) // 1000
DEEPGRAM_CHUNK_SIZE_BYTES: int = (SAMPLE_RATE * DEEPGRAM_CHUNK_SIZE_MS * 2) // 1000  # 2560 bytes

# Chunk size limits (safety bounds)
MIN_CHUNK_SIZE_MS: int = 10  # Minimum chunk size in milliseconds
MAX_CHUNK_SIZE_MS: int = 100  # Maximum chunk size in milliseconds (safety limit)

# Legacy constants (for backward compatibility)
CHUNK_SIZE_MS: int = BROWSER_CHUNK_SIZE_MS  # Alias for browser chunk size
CHUNK_SIZE_SAMPLES: int = BROWSER_CHUNK_SIZE_SAMPLES

# ============================================================================
# Validation Functions
# ============================================================================

def validate_audio_chunk(
    audio_data: bytes,
    expected_size_bytes: Optional[int] = None,
    tolerance_percent: float = 5.0,
) -> Tuple[bool, Optional[str]]:
    """Validate audio chunk format and size.
    
    Performs basic validation on audio chunk data:
    - Checks if data is not empty
    - Validates size matches expected (with tolerance for network variations)
    - Ensures size is multiple of 2 (16-bit samples requirement)
    - Validates size is within reasonable bounds
    
    Args:
        audio_data: Binary audio data to validate
        expected_size_bytes: Expected size in bytes (defaults to BROWSER_CHUNK_SIZE_BYTES)
        tolerance_percent: Percentage tolerance for size mismatch (default: 5.0%)
        
    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if chunk is valid, False otherwise
        - error_message: None if valid, descriptive error message if invalid
        
    Example:
        >>> audio_data = b'...'  # 640 bytes
        >>> is_valid, error = validate_audio_chunk(audio_data, 640)
        >>> if not is_valid:
        ...     print(f"Invalid: {error}")
    """
    if expected_size_bytes is None:
        expected_size_bytes = BROWSER_CHUNK_SIZE_BYTES
    
    # Check if data is not empty
    if not audio_data:
        return False, "Audio data is empty"
    
    data_size = len(audio_data)
    
    # Check if size is multiple of 2 (16-bit samples requirement)
    if data_size % 2 != 0:
        return False, f"Audio data size ({data_size} bytes) must be multiple of 2 (16-bit samples)"
    
    # Check if size matches expected (with tolerance)
    if expected_size_bytes > 0:
        tolerance_bytes = int(expected_size_bytes * tolerance_percent / 100.0)
        min_size = expected_size_bytes - tolerance_bytes
        max_size = expected_size_bytes + tolerance_bytes
        
        if data_size < min_size or data_size > max_size:
            return (
                False,
                f"Audio chunk size ({data_size} bytes) does not match expected "
                f"({expected_size_bytes} bytes, tolerance: ±{tolerance_percent}%)"
            )
    
    # Check if size is within reasonable bounds
    min_bytes = get_expected_chunk_size_bytes(MIN_CHUNK_SIZE_MS)
    max_bytes = get_expected_chunk_size_bytes(MAX_CHUNK_SIZE_MS)
    
    if data_size < min_bytes:
        return (
            False,
            f"Audio chunk size ({data_size} bytes) is below minimum "
            f"({min_bytes} bytes, {MIN_CHUNK_SIZE_MS}ms)"
        )
    
    if data_size > max_bytes:
        return (
            False,
            f"Audio chunk size ({data_size} bytes) exceeds maximum "
            f"({max_bytes} bytes, {MAX_CHUNK_SIZE_MS}ms)"
        )
    
    return True, None


def validate_audio_format(audio_data: bytes) -> Tuple[bool, Optional[str]]:
    """Validate basic audio format requirements.
    
    Performs basic format validation:
    - Checks if data is not empty
    - Ensures size is multiple of 2 (16-bit samples)
    - Validates size is within reasonable bounds
    
    Note: This function does not validate actual PCM format, sample rate, or endianness.
    Full format validation would require audio processing libraries (numpy, wave, etc.).
    For v1, basic size validation is sufficient as browser is expected to send correct format.
    
    Args:
        audio_data: Binary audio data to validate
        
    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if format appears valid, False otherwise
        - error_message: None if valid, descriptive error message if invalid
        
    Example:
        >>> audio_data = b'...'  # PCM audio bytes
        >>> is_valid, error = validate_audio_format(audio_data)
        >>> if not is_valid:
        ...     print(f"Invalid format: {error}")
    """
    if not audio_data:
        return False, "Audio data is empty"
    
    data_size = len(audio_data)
    
    # Check if size is multiple of 2 (16-bit samples requirement)
    if data_size % 2 != 0:
        return False, f"Audio data size ({data_size} bytes) must be multiple of 2 (16-bit samples)"
    
    # Check if size is within reasonable bounds
    min_bytes = get_expected_chunk_size_bytes(MIN_CHUNK_SIZE_MS)
    max_bytes = get_expected_chunk_size_bytes(MAX_CHUNK_SIZE_MS)
    
    if data_size < min_bytes:
        return (
            False,
            f"Audio data size ({data_size} bytes) is below minimum "
            f"({min_bytes} bytes, {MIN_CHUNK_SIZE_MS}ms)"
        )
    
    if data_size > max_bytes:
        return (
            False,
            f"Audio data size ({data_size} bytes) exceeds maximum "
            f"({max_bytes} bytes, {MAX_CHUNK_SIZE_MS}ms)"
        )
    
    return True, None


def validate_sample_rate(sample_rate: int) -> bool:
    """Validate that sample rate is supported.
    
    Args:
        sample_rate: Sample rate in Hz to validate
        
    Returns:
        True if sample rate is supported, False otherwise
        
    Example:
        >>> validate_sample_rate(16000)  # True
        >>> validate_sample_rate(22050)  # False
    """
    return sample_rate in SUPPORTED_SAMPLE_RATES


def validate_chunk_size(chunk_size_ms: int) -> Tuple[bool, Optional[str]]:
    """Validate chunk size is within acceptable range.
    
    Args:
        chunk_size_ms: Chunk size in milliseconds to validate
        
    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if chunk size is valid, False otherwise
        - error_message: None if valid, descriptive error message if invalid
        
    Example:
        >>> is_valid, error = validate_chunk_size(20)
        >>> if not is_valid:
        ...     print(f"Invalid chunk size: {error}")
    """
    if chunk_size_ms < MIN_CHUNK_SIZE_MS:
        return (
            False,
            f"Chunk size ({chunk_size_ms}ms) is below minimum ({MIN_CHUNK_SIZE_MS}ms)"
        )
    
    if chunk_size_ms > MAX_CHUNK_SIZE_MS:
        return (
            False,
            f"Chunk size ({chunk_size_ms}ms) exceeds maximum ({MAX_CHUNK_SIZE_MS}ms)"
        )
    
    return True, None


# ============================================================================
# Utility Functions
# ============================================================================

def bytes_to_samples(audio_data: bytes) -> int:
    """Convert audio bytes to sample count.
    
    Args:
        audio_data: Binary audio data
        
    Returns:
        Number of samples in the audio data
        
    Example:
        >>> audio_data = b'...'  # 640 bytes
        >>> samples = bytes_to_samples(audio_data)  # 320 samples
    """
    return len(audio_data) // BYTES_PER_SAMPLE


def samples_to_bytes(sample_count: int) -> int:
    """Convert sample count to byte count.
    
    Args:
        sample_count: Number of audio samples
        
    Returns:
        Number of bytes required for the samples
        
    Example:
        >>> samples = 320
        >>> bytes_count = samples_to_bytes(samples)  # 640 bytes
    """
    return sample_count * BYTES_PER_SAMPLE


def ms_to_samples(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> int:
    """Convert milliseconds to sample count.
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz (default: SAMPLE_RATE)
        
    Returns:
        Number of samples for the given duration
        
    Example:
        >>> samples = ms_to_samples(20)  # 320 samples at 16kHz
    """
    return (sample_rate * duration_ms) // 1000


def samples_to_ms(sample_count: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Convert sample count to milliseconds.
    
    Args:
        sample_count: Number of audio samples
        sample_rate: Sample rate in Hz (default: SAMPLE_RATE)
        
    Returns:
        Duration in milliseconds (as float for precision)
        
    Example:
        >>> duration_ms = samples_to_ms(320)  # 20.0 ms at 16kHz
    """
    return (sample_count * 1000.0) / sample_rate


def calculate_audio_duration(audio_data: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    """Calculate audio duration in milliseconds from audio data.
    
    Args:
        audio_data: Binary audio data
        sample_rate: Sample rate in Hz (default: SAMPLE_RATE)
        
    Returns:
        Duration in milliseconds (as float for precision)
        
    Example:
        >>> audio_data = b'...'  # 640 bytes
        >>> duration_ms = calculate_audio_duration(audio_data)  # 20.0 ms
    """
    sample_count = bytes_to_samples(audio_data)
    return samples_to_ms(sample_count, sample_rate)


def get_expected_chunk_size_bytes(
    duration_ms: int,
    sample_rate: int = SAMPLE_RATE,
) -> int:
    """Get expected byte size for a chunk of given duration.
    
    Args:
        duration_ms: Chunk duration in milliseconds
        sample_rate: Sample rate in Hz (default: SAMPLE_RATE)
        
    Returns:
        Expected byte size for the chunk
        
    Example:
        >>> bytes_size = get_expected_chunk_size_bytes(20)  # 640 bytes at 16kHz
    """
    samples = ms_to_samples(duration_ms, sample_rate)
    return samples_to_bytes(samples)

