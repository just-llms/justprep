"""Speech-to-text engine.

This module provides the FluxSTTEngine for Deepgram Flux API integration
and a factory function to create STT engine instances.
"""

import logging
import os
from typing import Optional

from core.stt.stt_engine import FluxSTTEngine

logger = logging.getLogger(__name__)


def create_stt_engine(session_id: str, api_key: Optional[str] = None) -> FluxSTTEngine:
    """Create a Flux STT engine instance for the given session.
    
    Factory function that creates a FluxSTTEngine instance with proper
    configuration. Loads API key from environment if not provided.
    
    Args:
        session_id: Unique identifier for the session
        api_key: Optional Deepgram API key. If not provided, will load from
            DEEPGRAM_API_KEY environment variable.
            
    Returns:
        FluxSTTEngine instance configured for the session
        
    Raises:
        ValueError: If API key is not provided and not found in environment
        
    Example:
        >>> stt_engine = create_stt_engine("session-123")
        >>> await stt_engine.start_streaming()
    """
    # Load API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "Deepgram API key not found. Set DEEPGRAM_API_KEY environment variable "
                "or pass api_key parameter to create_stt_engine()."
            )
    
    try:
        engine = FluxSTTEngine(session_id=session_id, api_key=api_key)
        logger.info(f"Created FluxSTTEngine for session {session_id}")
        return engine
    except Exception as e:
        logger.error(f"Failed to create STT engine for session {session_id}: {e}")
        raise


__all__ = ["FluxSTTEngine", "create_stt_engine"]
