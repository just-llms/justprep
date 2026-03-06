"""WebSocket message models."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class ControlStartMessage(BaseModel):
    """Control message to start audio capture.
    
    Sent by client to indicate audio capture should begin.
    """

    type: Literal["CONTROL_START"] = Field(
        description="Message type identifier"
    )
    timestamp: datetime = Field(
        description="Timestamp when the message was created"
    )


class ControlStopMessage(BaseModel):
    """Control message to stop audio capture.
    
    Sent by client to indicate audio capture should stop.
    """

    type: Literal["CONTROL_STOP"] = Field(
        description="Message type identifier"
    )
    timestamp: datetime = Field(
        description="Timestamp when the message was created"
    )


class HeartbeatMessage(BaseModel):
    """Heartbeat message for keepalive.
    
    Sent periodically by client to maintain connection.
    Server should respond with HeartbeatAckMessage.
    """

    type: Literal["HEARTBEAT"] = Field(
        description="Message type identifier"
    )
    timestamp: datetime = Field(
        description="Timestamp when the message was created"
    )


class HeartbeatAckMessage(BaseModel):
    """Heartbeat acknowledgment message.
    
    Sent by server in response to HeartbeatMessage.
    """

    type: Literal["HEARTBEAT_ACK"] = Field(
        description="Message type identifier"
    )
    timestamp: datetime = Field(
        description="Timestamp when the acknowledgment was created"
    )


class ErrorMessage(BaseModel):
    """Error message sent to client.
    
    Sent by server when an error occurs that the client should know about.
    """

    type: Literal["ERROR"] = Field(
        description="Message type identifier"
    )
    error_code: str = Field(
        description="Error code identifying the type of error"
    )
    message: str = Field(
        description="Human-readable error message"
    )
    timestamp: datetime = Field(
        description="Timestamp when the error occurred"
    )


# Message type constants for reference
class MessageType:
    """Constants for WebSocket message types."""
    
    AUDIO_CHUNK = "AUDIO_CHUNK"  # Binary frame
    CONTROL_START = "CONTROL_START"  # Text JSON
    CONTROL_STOP = "CONTROL_STOP"  # Text JSON
    HEARTBEAT = "HEARTBEAT"  # Text JSON
    HEARTBEAT_ACK = "HEARTBEAT_ACK"  # Text JSON
    TTS_AUDIO = "TTS_AUDIO"  # Binary frame
    ERROR = "ERROR"  # Text JSON

