"""Text-to-speech engine."""

import logging
from typing import Dict, Optional

from core.tts.tts_engine import CartesiaTTSEngine
from core.connection_manager import ConnectionManager
from core.audio.turn_controller import TurnController

logger = logging.getLogger(__name__)

# Per-session TTS engine management
_tts_engines: Dict[str, CartesiaTTSEngine] = {}


async def get_tts_engine(
    session_id: str,
    connection_manager: ConnectionManager,
    turn_controller: TurnController,
) -> CartesiaTTSEngine:
    """Get or create TTS engine for session.
    
    Args:
        session_id: Unique identifier for the session
        connection_manager: ConnectionManager for sending audio chunks
        turn_controller: TurnController for interruption handling
        
    Returns:
        TTS engine instance for the session
    """
    if session_id not in _tts_engines:
        _tts_engines[session_id] = CartesiaTTSEngine(
            session_id=session_id,
            connection_manager=connection_manager,
            turn_controller=turn_controller,
        )
        logger.info(f"Created TTS engine for session {session_id}")
    
    return _tts_engines[session_id]


async def remove_tts_engine(session_id: str) -> None:
    """Remove TTS engine for session.
    
    Cleans up resources when session ends.
    
    Args:
        session_id: Unique identifier for the session
    """
    if session_id in _tts_engines:
        # Stop any active TTS streaming
        tts_engine = _tts_engines[session_id]
        try:
            await tts_engine.stop()
        except Exception as e:
            logger.warning(f"Error stopping TTS engine for session {session_id}: {e}")
        
        del _tts_engines[session_id]
        logger.info(f"Removed TTS engine for session {session_id}")


__all__ = ["CartesiaTTSEngine", "get_tts_engine", "remove_tts_engine"]
