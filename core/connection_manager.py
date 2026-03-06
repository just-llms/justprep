import asyncio
import logging
from typing import Dict, Optional

from fastapi import WebSocket

from util.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for the interview system."""

    def __init__(self) -> None:
        """Initialize ConnectionManager with empty connections dict."""
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
        self.metrics = MetricsCollector.get_instance()

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept and store a WebSocket connection."""
        async with self._lock:
            if session_id in self.active_connections:
                logger.warning(f"Session {session_id} already has an active connection")
                try:
                    await self.active_connections[session_id].close()
                except Exception as e:
                    logger.error(f"Error closing existing connection for {session_id}: {e}")

            await websocket.accept()
            self.active_connections[session_id] = websocket
            self.metrics.set_gauge("system.websocket_connections", len(self.active_connections))
            logger.info(f"WebSocket connected for session {session_id}")

    async def disconnect(self, session_id: str, websocket: Optional[WebSocket] = None) -> None:
        """Remove a WebSocket connection.

        If websocket is provided, disconnect only when it matches the currently
        active connection for the session. This prevents stale handlers from
        closing a newer connection.
        """
        async with self._lock:
            active = self.active_connections.get(session_id)
            if active is None:
                return

            if websocket is not None and active is not websocket:
                return

            try:
                await active.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for {session_id}: {e}")

            del self.active_connections[session_id]
            self.metrics.set_gauge("system.websocket_connections", len(self.active_connections))
            logger.info(f"WebSocket disconnected for session {session_id}")

    async def is_connected(self, session_id: str) -> bool:
        """Check if a session has an active connection."""
        async with self._lock:
            return session_id in self.active_connections

    async def is_current_connection(self, session_id: str, websocket: WebSocket) -> bool:
        """Check whether websocket is the active connection for the session."""
        async with self._lock:
            return self.active_connections.get(session_id) is websocket

    async def send_audio_chunk(self, session_id: str, audio_data: bytes) -> bool:
        """Send binary audio data to a connected client."""
        async with self._lock:
            websocket = self.active_connections.get(session_id)
            if websocket is None:
                logger.warning(f"Cannot send audio: session {session_id} not connected")
                return False

            try:
                await websocket.send_bytes(audio_data)
                return True
            except Exception as e:
                logger.error(f"Error sending audio to {session_id}: {e}")
                return False

    async def send_control_message(self, session_id: str, message: dict) -> bool:
        """Send a JSON control message to a connected client."""
        async with self._lock:
            websocket = self.active_connections.get(session_id)
            if websocket is None:
                logger.warning(f"Cannot send control message: session {session_id} not connected")
                return False

            try:
                await websocket.send_json(message)
                return True
            except Exception as e:
                logger.error(f"Error sending control message to {session_id}: {e}")
                return False

    def get_connection_count(self) -> int:
        """Get the total number of active WebSocket connections."""
        return len(self.active_connections)
