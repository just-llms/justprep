"""WebSocket route handlers for the interview system."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.audio import get_audio_processor, remove_audio_processor
from core.connection_manager import ConnectionManager
from core.session_manager import SessionManager
from models.websocket_models import (
    ControlStartMessage,
    ControlStopMessage,
    HeartbeatMessage,
    HeartbeatAckMessage,
    ErrorMessage,
    MessageType,
)
from models.exceptions import SessionNotFoundError
from models.constants import PlannerOutputType
from util.logger import AppLogger, get_websocket_logger

logger = logging.getLogger(__name__)

# Global instances (will be initialized in dependency injection or startup)
connection_manager = ConnectionManager()
session_manager = SessionManager.get_instance()
# #region agent log
import json
import time
with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_sm_init","timestamp":int(time.time()*1000),"location":"websocket_routes.py:29","message":"SessionManager instance retrieved in websocket_routes","data":{"instanceId":str(id(session_manager)),"sessionsCount":len(session_manager.sessions)},"runId":"post-fix","hypothesisId":"A"}) + "\n")
# #endregion

# Router for WebSocket routes
router = APIRouter()

# Heartbeat configuration
HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_TIMEOUT_SECONDS = 60


async def handle_audio_chunk(session_id: str, audio_data: bytes) -> None:
    """Handle incoming audio chunk from client.
    
    Args:
        session_id: Unique identifier for the session
        audio_data: Binary audio data (20ms chunks, 640 bytes at 16kHz)
    """
    # #region agent log
    import json
    import time
    chunk_received_time = time.time()
    with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
        f.write(json.dumps({"id":f"log_{int(chunk_received_time*1000)}_received","timestamp":int(chunk_received_time*1000),"location":"websocket_routes.py:39","message":"Chunk received from WebSocket","data":{"sessionId":session_id,"chunkSize":len(audio_data)},"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + "\n")
    # #endregion
    
    try:
        # Get or create audio processor for session
        audio_processor = await get_audio_processor(session_id)
        
        # #region agent log
        before_process = time.time()
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(before_process*1000)}_before_process","timestamp":int(before_process*1000),"location":"websocket_routes.py:51","message":"Before process_audio_chunk","data":{"sessionId":session_id,"timeSinceReceived":before_process-chunk_received_time},"sessionId":"debug-session","runId":"run1","hypothesisId":"B,E"}) + "\n")
        # #endregion
        
        # Process audio chunk
        await audio_processor.process_audio_chunk(audio_data)
        
        # #region agent log
        after_process = time.time()
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(after_process*1000)}_after_process","timestamp":int(after_process*1000),"location":"websocket_routes.py:51","message":"After process_audio_chunk","data":{"sessionId":session_id,"processDuration":after_process-before_process,"totalTime":after_process-chunk_received_time},"sessionId":"debug-session","runId":"run1","hypothesisId":"B,E"}) + "\n")
        # #endregion
        
        logger.debug(
            f"Processed audio chunk for session {session_id}: {len(audio_data)} bytes"
        )
    except Exception as e:
        logger.error(
            f"Error handling audio chunk for session {session_id}: "
            f"chunk_size={len(audio_data)}, error={type(e).__name__}: {e}",
            exc_info=True
        )
        # Send error to client for critical failures
        # Don't send for every chunk error to avoid spam, but log it
        # Only send if it's a persistent issue (could be tracked separately)
        try:
            # For now, just log - could add error tracking to detect persistent issues
            pass
        except Exception as send_error_exception:
            logger.warning(
                f"Failed to send error message for session {session_id}: {send_error_exception}"
            )


async def handle_control_message(session_id: str, message_text: str) -> None:
    """Parse and handle control messages from client.
    
    Args:
        session_id: Unique identifier for the session
        message_text: JSON string containing the control message
    """
    try:
        message_data = json.loads(message_text)
        message_type = message_data.get("type")
        
        if message_type == MessageType.CONTROL_START:
            message = ControlStartMessage(**message_data)
            logger.info(f"Control START received from session {session_id}")
            # Ensure audio processor is created + started.
            # Note: audio processor is also lazily created on first audio chunk,
            # but CONTROL_START should be sufficient to begin accepting audio.
            try:
                audio_processor = await get_audio_processor(session_id)
                await audio_processor.start()
            except Exception as e:
                logger.error(f"Error starting audio processor for session {session_id}: {e}")
                await send_error(
                    session_id,
                    "AUDIO_START_FAILED",
                    f"Failed to start audio processing: {str(e)}",
                )
            
        elif message_type == MessageType.CONTROL_STOP:
            message = ControlStopMessage(**message_data)
            logger.info(f"Control STOP received from session {session_id}")
            # Stop audio processing (forces STT flush/finalization).
            try:
                audio_processor = await get_audio_processor(session_id)
                await audio_processor.stop()
            except Exception as e:
                logger.error(f"Error stopping audio processor for session {session_id}: {e}")
                await send_error(
                    session_id,
                    "AUDIO_STOP_FAILED",
                    f"Failed to stop audio processing: {str(e)}",
                )
            
        elif message_type == MessageType.HEARTBEAT:
            message = HeartbeatMessage(**message_data)
            await handle_heartbeat(session_id)
            
        else:
            logger.warning(f"Unknown message type '{message_type}' from session {session_id}")
            await send_error(
                session_id,
                "UNKNOWN_MESSAGE_TYPE",
                f"Unknown message type: {message_type}"
            )
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from session {session_id}: {e}")
        await send_error(session_id, "INVALID_JSON", "Message must be valid JSON")
    except Exception as e:
        logger.error(f"Error handling control message from session {session_id}: {e}")
        await send_error(session_id, "PROCESSING_ERROR", f"Error processing message: {str(e)}")


async def handle_heartbeat(session_id: str) -> None:
    """Handle heartbeat message and respond with acknowledgment.
    
    Args:
        session_id: Unique identifier for the session
    """
    ack_message = HeartbeatAckMessage(
        type=MessageType.HEARTBEAT_ACK,
        timestamp=datetime.now()
    )
    await connection_manager.send_control_message(session_id, ack_message.model_dump(mode='json'))


async def send_error(session_id: str, error_code: str, message: str) -> None:
    """Send an error message to the client.
    
    Args:
        session_id: Unique identifier for the session
        error_code: Error code identifying the error type
        message: Human-readable error message
    """
    error_message = ErrorMessage(
        type=MessageType.ERROR,
        error_code=error_code,
        message=message,
        timestamp=datetime.now()
    )
    await connection_manager.send_control_message(session_id, error_message.model_dump(mode='json'))


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for interview sessions.
    
    Handles the complete WebSocket connection lifecycle:
    - Accepts connection
    - Creates or retrieves session
    - Handles incoming messages (binary and text)
    - Manages heartbeat
    - Cleans up on disconnect
    
    Args:
        websocket: The WebSocket connection
        session_id: Unique identifier for the session
    """
    app_logger = AppLogger()
    app_logger.set_session_id(session_id)
    
    # Check rate limit on connection
    try:
        from core.safety import SafetyController
        safety_controller = SafetyController.get_instance()
        # Use client IP or session_id as user identifier
        client_ip = websocket.client.host if websocket.client else None
        user_id = client_ip or session_id
        rate_result = await safety_controller.check_rate_limit(user_id)
        
        if not rate_result.allowed:
            logger.warning(
                f"Rate limit exceeded for {user_id} (session {session_id})"
            )
            websocket_logger.warning(
                "rate_limit_exceeded",
                "Rate limit exceeded on connection",
                {
                    "user_id": user_id,
                    "retry_after": rate_result.retry_after
                },
                session_id=session_id
            )
            # Send error and close connection
            try:
                await websocket.accept()
                await websocket.send_json({
                    "type": "ERROR",
                    "error_type": "rate_limit_exceeded",
                    "message": rate_result.message,
                    "retry_after": rate_result.retry_after,
                    "timestamp": datetime.now().isoformat(),
                })
                await websocket.close()
            except Exception:
                pass
            return
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        # Continue on error to avoid blocking
    
    try:
        # Accept connection
        await connection_manager.connect(websocket, session_id)
        
        # Create or get session
        try:
            # #region agent log
            import json
            import time
            with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_before_get","timestamp":int(time.time()*1000),"location":"websocket_routes.py:199","message":"Before get_session in websocket_routes","data":{"sessionId":session_id,"instanceId":str(id(session_manager)),"sessionsCount":len(session_manager.sessions),"sessionIds":list(session_manager.sessions.keys())},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
            # #endregion
            session = await session_manager.get_session(session_id)
            if session is None:
                # #region agent log
                with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_before_create","timestamp":int(time.time()*1000),"location":"websocket_routes.py:201","message":"Before create_session in websocket_routes","data":{"sessionId":session_id,"instanceId":str(id(session_manager))},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
                # #endregion
                
                # TODO: Replace with actual values from client
                # Hardcoded candidate information for now
                work_experience = "5 years of software engineering experience with expertise in Python, distributed systems, and cloud infrastructure"
                current_role = "Senior Software Engineer"
                target_role = "Staff Software Engineer"
                resume = "Experienced software engineer with 5 years of experience building scalable distributed systems. Proficient in Python, Go, and JavaScript. Led multiple projects involving microservices architecture, containerization with Docker and Kubernetes, and cloud platforms (AWS, GCP). Strong background in system design, database optimization, and API development. Contributed to open-source projects and mentored junior developers."
                
                session = await session_manager.create_session(
                    session_id,
                    work_experience=work_experience,
                    current_role=current_role,
                    target_role=target_role,
                    resume=resume
                )
                # #region agent log
                with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_after_create","timestamp":int(time.time()*1000),"location":"websocket_routes.py:201","message":"After create_session in websocket_routes","data":{"sessionId":session_id,"instanceId":str(id(session_manager)),"sessionsCount":len(session_manager.sessions),"sessionIds":list(session_manager.sessions.keys())},"runId":"post-fix","hypothesisId":"A,C"}) + "\n")
                # #endregion
                logger.info(f"Created new session {session_id}")
                
                # Generate and send initial greeting
                try:
                    from core.ai import get_ai_engine
                    ai_engine = get_ai_engine()
                    planner_output = await ai_engine.start_interview(session_id)
                    
                    # Send greeting to client (as AI_RESPONSE for now, TTS integration later)
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
                    await connection_manager.send_control_message(session_id, ai_response_message)
                    logger.info(f"Initial greeting sent to client for session {session_id}")
                    
                    # If output is SPEAK type, send to TTS
                    if planner_output.type == PlannerOutputType.SPEAK and planner_output.text:
                        try:
                            from core.tts import get_tts_engine
                            from core.audio import get_audio_processor
                            
                            # Get audio processor to access turn controller
                            audio_processor = await get_audio_processor(session_id)
                            
                            # Get TTS engine for session
                            tts_engine = await get_tts_engine(
                                session_id,
                                connection_manager,
                                audio_processor.turn_controller
                            )
                            await tts_engine.speak(planner_output.text)
                            logger.info(
                                f"Initial greeting spoken via TTS for session {session_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error speaking initial greeting for session {session_id}: {e}",
                                exc_info=True
                            )
                            # Don't fail the connection - greeting was sent as control message
                except Exception as e:
                    logger.error(
                        f"Error generating/sending initial greeting for session {session_id}: {e}",
                        exc_info=True
                    )
                    # Send error message but don't fail the connection
                    await send_error(
                        session_id,
                        "GREETING_ERROR",
                        f"Failed to generate initial greeting: {str(e)}"
                    )
            else:
                logger.info(f"Resumed existing session {session_id}")
                
                # Check if initial greeting was already sent
                if not session.initial_greeting_sent:
                    # Generate and send initial greeting for resumed session
                    try:
                        from core.ai import get_ai_engine
                        ai_engine = get_ai_engine()
                        planner_output = await ai_engine.start_interview(session_id)
                        
                        # Send greeting to client
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
                        await connection_manager.send_control_message(session_id, ai_response_message)
                        logger.info(f"Initial greeting sent to resumed session {session_id}")
                        
                        # If output is SPEAK type, send to TTS
                        if planner_output.type == PlannerOutputType.SPEAK and planner_output.text:
                            try:
                                from core.tts import get_tts_engine
                                from core.audio import get_audio_processor
                                
                                audio_processor = await get_audio_processor(session_id)
                                tts_engine = await get_tts_engine(
                                    session_id,
                                    connection_manager,
                                    audio_processor.turn_controller
                                )
                                await tts_engine.speak(planner_output.text)
                                logger.info(f"Initial greeting spoken via TTS for resumed session {session_id}")
                            except Exception as e:
                                logger.error(f"Error speaking greeting for resumed session {session_id}: {e}")
                    except Exception as e:
                        logger.error(f"Error generating greeting for resumed session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Error creating/retrieving session {session_id}: {e}")
            await send_error(session_id, "SESSION_ERROR", "Failed to initialize session")
            await connection_manager.disconnect(session_id, websocket)
            return
        
        # Track last heartbeat time
        last_heartbeat = datetime.now()
        
        # Main message loop
        try:
            while True:
                # Receive message with timeout to check for stale connections
                try:
                    data = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=HEARTBEAT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Check if connection is stale
                    time_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()
                    if time_since_heartbeat >= HEARTBEAT_TIMEOUT_SECONDS:
                        logger.warning(f"Stale connection detected for session {session_id}, closing")
                        break
                    continue
                
                # Handle binary frames (audio chunks)
                if "bytes" in data:
                    audio_data = data["bytes"]
                    await handle_audio_chunk(session_id, audio_data)
                    
                # Handle text frames (control messages)
                elif "text" in data:
                    message_text = data["text"]
                    await handle_control_message(session_id, message_text)
                    
                    # Update heartbeat time if it was a heartbeat message
                    try:
                        message_data = json.loads(message_text)
                        if message_data.get("type") == MessageType.HEARTBEAT:
                            last_heartbeat = datetime.now()
                    except (json.JSONDecodeError, KeyError):
                        pass  # Not a heartbeat, ignore
                        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected normally for session {session_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket message loop for session {session_id}: {e}")
            await send_error(session_id, "INTERNAL_ERROR", "An internal error occurred")
            
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint for session {session_id}: {e}")
    finally:
        # Cleanup only if this handler still owns the active connection.
        is_current_connection = await connection_manager.is_current_connection(session_id, websocket)
        if is_current_connection:
            await connection_manager.disconnect(session_id, websocket)
            await remove_audio_processor(session_id)
            await session_manager.delete_session(session_id)
            logger.info(f"Cleaned up session {session_id}")
        else:
            logger.info(f"Skipped cleanup for stale websocket handler in session {session_id}")

        app_logger.clear_session_id()



