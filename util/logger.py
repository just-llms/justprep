"""Logging configuration and utilities."""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Union
from pathlib import Path
from contextlib import contextmanager

# Standard logger pattern per coding standards
logger = logging.getLogger(__name__)


class AppLogger:
    """Session-scoped logger with session ID support.
    
    This class provides session-scoped logging with session ID support,
    following the coding standards pattern for structured logging.
    """

    def __init__(self) -> None:
        """Initialize AppLogger instance."""
        self.logger = logging.getLogger(__name__)
        self._session_id: Optional[str] = None

    def set_session_id(self, session_id: str) -> None:
        """Set session ID for session-scoped logging.
        
        Args:
            session_id: Unique identifier for the current session
        """
        self._session_id = session_id

    def clear_session_id(self) -> None:
        """Clear session ID."""
        self._session_id = None

    def _format_message(self, message: str) -> str:
        """Format log message with session ID if available.
        
        Args:
            message: Original log message
            
        Returns:
            Formatted message with context
        """
        if self._session_id:
            return f"[{self._session_id}] {message}"
        return message

    def info(self, message: str) -> None:
        """Log info message with context.
        
        Args:
            message: Log message to record
        """
        formatted_message = self._format_message(message)
        self.logger.info(formatted_message)

    def error(self, message: str) -> None:
        """Log error message with context.
        
        Args:
            message: Log message to record
        """
        formatted_message = self._format_message(message)
        self.logger.error(formatted_message)

    def warning(self, message: str) -> None:
        """Log warning message with context.
        
        Args:
            message: Log message to record
        """
        formatted_message = self._format_message(message)
        self.logger.warning(formatted_message)

    def debug(self, message: str) -> None:
        """Log debug message with context.
        
        Args:
            message: Log message to record
        """
        formatted_message = self._format_message(message)
        self.logger.debug(formatted_message)

    def exception(self, message: str) -> None:
        """Log exception with traceback.
        
        Args:
            message: Log message to record
        """
        formatted_message = self._format_message(message)
        self.logger.exception(formatted_message)


@contextmanager
def session_context(logger_instance: AppLogger, session_id: str):
    """Context manager for automatic session ID cleanup.
    
    Args:
        logger_instance: AppLogger instance to use
        session_id: Session ID to set for the context
    """
    logger_instance.set_session_id(session_id)
    try:
        yield
    finally:
        logger_instance.clear_session_id()


def setup_logging(log_level: str = "INFO", log_format: str = "standard") -> None:
    """Configure application-wide logging.
    
    Sets up logging configuration following coding standards:
    - Uses logging.getLogger(__name__) pattern
    - Supports structured logging for production
    - Includes context (session_id) in logs
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format type ("standard" or "structured")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    if log_format.lower() == "structured":
        # Structured format - message already contains all metadata
        # Just add timestamp and level prefix
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    elif log_format.lower() == "json":
        # For JSON format (legacy support)
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        # Standard format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    # Create rotating file handler for persistent debug logs.
    log_file_path = Path(os.getenv("LOG_FILE_PATH", "debug.log"))
    max_bytes = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
    backup_count = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

    if log_file_path.parent != Path("."):
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set level for third-party libraries to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class StructuredLogger:
    """Structured logger with consistent metadata format (key=value pairs, not JSON).
    
    This class provides structured logging with consistent metadata format
    using key=value pairs in text format. All logs include timestamp, session_id,
    component, event_type, and optional data fields.
    
    Example output:
        timestamp=2024-01-01T12:00:00Z session_id=abc123 component=vad event_type=speech_start message="Speech detected" data=speech_prob=0.85 threshold=0.5
    """
    
    def __init__(self, component: str, session_id: Optional[str] = None) -> None:
        """Initialize StructuredLogger.
        
        Args:
            component: Component name (e.g., "vad", "stt", "llm", "tts")
            session_id: Optional session ID for session-scoped logging
        """
        self.component = component
        self.session_id = session_id
        self.logger = logging.getLogger(component)
    
    def set_session_id(self, session_id: Optional[str]) -> None:
        """Set session ID for session-scoped logging.
        
        Args:
            session_id: Session ID to use for logging
        """
        self.session_id = session_id
    
    def _format_data(self, data: Optional[Dict[str, Any]]) -> str:
        """Format data dictionary as key=value pairs.
        
        Args:
            data: Dictionary of data fields
            
        Returns:
            Formatted string with key=value pairs
        """
        if not data:
            return ""
        
        parts = []
        for key, value in data.items():
            # Format value appropriately
            if isinstance(value, str):
                # Escape quotes and wrap in quotes if contains spaces
                if " " in value or "=" in value:
                    escaped_value = value.replace('"', '\\"')
                    value_str = f'"{escaped_value}"'
                else:
                    value_str = value
            elif isinstance(value, (int, float)):
                value_str = str(value)
            elif isinstance(value, bool):
                value_str = "true" if value else "false"
            elif value is None:
                value_str = "null"
            else:
                value_str = str(value)
            
            parts.append(f"{key}={value_str}")
        
        return " ".join(parts)
    
    def _format_message(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Format structured log message with metadata.
        
        Args:
            event_type: Type of event (e.g., "speech_start", "transcript", "llm_response")
            message: Log message text
            data: Optional dictionary of additional data fields
            session_id: Optional session ID (overrides instance session_id)
            
        Returns:
            Formatted structured log message
        """
        # Use provided session_id or instance session_id
        effective_session_id = session_id or self.session_id
        
        # Get current timestamp in ISO format
        timestamp = datetime.now().isoformat()
        
        # Build structured message parts
        parts = [f"timestamp={timestamp}"]
        
        if effective_session_id:
            parts.append(f"session_id={effective_session_id}")
        
        parts.append(f"component={self.component}")
        parts.append(f"event_type={event_type}")
        
        # Format message (wrap in quotes if contains spaces or special chars)
        if " " in message or "=" in message:
            escaped_message = message.replace('"', '\\"')
            message_str = f'message="{escaped_message}"'
        else:
            message_str = f"message={message}"
        parts.append(message_str)
        
        # Add data if provided
        if data:
            data_str = self._format_data(data)
            if data_str:
                parts.append(f"data={data_str}")
        
        return " ".join(parts)
    
    def log_event(
        self,
        level: str,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log structured event with metadata.
        
        Args:
            level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
            event_type: Type of event
            message: Log message
            data: Optional dictionary of additional data
            session_id: Optional session ID (overrides instance session_id)
        """
        formatted_message = self._format_message(event_type, message, data, session_id)
        
        # Get numeric level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Log with appropriate level
        self.logger.log(numeric_level, formatted_message)
    
    def debug(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log DEBUG level event."""
        self.log_event("DEBUG", event_type, message, data, session_id)
    
    def info(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log INFO level event."""
        self.log_event("INFO", event_type, message, data, session_id)
    
    def warning(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log WARNING level event."""
        self.log_event("WARNING", event_type, message, data, session_id)
    
    def error(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        exc_info: bool = False,
    ) -> None:
        """Log ERROR level event.
        
        Args:
            event_type: Type of event
            message: Log message
            data: Optional dictionary of additional data
            session_id: Optional session ID
            exc_info: If True, include exception traceback
        """
        formatted_message = self._format_message(event_type, message, data, session_id)
        self.logger.error(formatted_message, exc_info=exc_info)
    
    def exception(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log exception with traceback."""
        self.error(event_type, message, data, session_id, exc_info=True)


def get_component_logger(component: str, session_id: Optional[str] = None) -> StructuredLogger:
    """Get component-specific structured logger.
    
    Args:
        component: Component name (e.g., "vad", "stt", "llm", "tts", "planner", "fsm", "turn_controller")
        session_id: Optional session ID for session-scoped logging
        
    Returns:
        StructuredLogger instance for the component
    """
    return StructuredLogger(component, session_id)


# Convenience functions for common components
def get_vad_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get VAD component logger."""
    return get_component_logger("vad", session_id)


def get_stt_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get STT component logger."""
    return get_component_logger("stt", session_id)


def get_llm_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get LLM component logger."""
    return get_component_logger("llm", session_id)


def get_tts_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get TTS component logger."""
    return get_component_logger("tts", session_id)


def get_planner_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get Response Planner component logger."""
    return get_component_logger("planner", session_id)


def get_fsm_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get FSM Controller component logger."""
    return get_component_logger("fsm", session_id)


def get_turn_controller_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get Turn Controller component logger."""
    return get_component_logger("turn_controller", session_id)


def get_audio_processor_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get Audio Processor component logger."""
    return get_component_logger("audio_processor", session_id)


def get_websocket_logger(session_id: Optional[str] = None) -> StructuredLogger:
    """Get WebSocket component logger."""
    return get_component_logger("websocket", session_id)

