"""Transcript retrieval and download routes."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from core.session_manager import SessionManager
from models.exceptions import SessionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/sessions/{session_id}/transcript")
async def get_transcript(session_id: str) -> dict:
    """Get full interview transcript.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Complete transcript with all turns
    """
    try:
        session_manager = SessionManager.get_instance()
        session = await session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Calculate duration
        duration_seconds = (datetime.now() - session.session_start_time).total_seconds()
        duration_minutes = duration_seconds / 60
        
        # Get phases completed
        phases_completed = list(set(turn.phase.value for turn in session.turn_history))
        
        # Build transcript
        turns = [
            {
                "turn_number": turn.turn_number,
                "question": turn.question,
                "answer": turn.answer,
                "phase": turn.phase.value,
                "timestamp": turn.timestamp.isoformat()
            }
            for turn in session.turn_history
        ]
        
        return {
            "session_id": session_id,
            "turns": turns,
            "duration_minutes": round(duration_minutes, 1),
            "total_turns": session.total_turns,
            "phases_completed": phases_completed
        }
    
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error retrieving transcript for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving transcript: {str(e)}")


@router.get("/api/sessions/{session_id}/transcript/download")
async def download_transcript(
    session_id: str,
    format: str = "pdf"
) -> Response:
    """Download interview transcript as PDF or TXT file.
    
    Args:
        session_id: Session identifier
        format: File format ('pdf' or 'txt'). Default: 'pdf'
        
    Returns:
        File download response
    """
    if format not in ["pdf", "txt"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Allowed: 'pdf' or 'txt'"
        )
    
    try:
        session_manager = SessionManager.get_instance()
        session = await session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Build transcript text
        transcript_lines = []
        transcript_lines.append("Interview Transcript")
        transcript_lines.append(f"Session ID: {session_id}")
        transcript_lines.append(f"Date: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        transcript_lines.append("")
        transcript_lines.append("=" * 80)
        transcript_lines.append("")
        
        for turn in session.turn_history:
            transcript_lines.append(f"Turn {turn.turn_number} - {turn.phase.value.upper()}")
            transcript_lines.append(f"Time: {turn.timestamp.strftime('%H:%M:%S')}")
            transcript_lines.append("")
            if turn.question:
                transcript_lines.append(f"Interviewer: {turn.question}")
            transcript_lines.append(f"You: {turn.answer}")
            transcript_lines.append("")
            transcript_lines.append("-" * 80)
            transcript_lines.append("")
        
        transcript_text = "\n".join(transcript_lines)
        
        if format == "txt":
            # Return as plain text
            return Response(
                content=transcript_text,
                media_type="text/plain",
                headers={
                    "Content-Disposition": f'attachment; filename="transcript_{session_id}.txt"'
                }
            )
        
        else:  # PDF
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.units import inch
                from io import BytesIO
                
                # Create PDF in memory
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Add title
                title = Paragraph("Interview Transcript", styles['Title'])
                story.append(title)
                story.append(Spacer(1, 0.2 * inch))
                
                # Add metadata
                meta = Paragraph(
                    f"Session ID: {session_id}<br/>"
                    f"Date: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                    styles['Normal']
                )
                story.append(meta)
                story.append(Spacer(1, 0.3 * inch))
                
                # Add transcript content
                for turn in session.turn_history:
                    turn_header = Paragraph(
                        f"<b>Turn {turn.turn_number} - {turn.phase.value.upper()}</b>",
                        styles['Heading2']
                    )
                    story.append(turn_header)
                    
                    time_para = Paragraph(
                        f"Time: {turn.timestamp.strftime('%H:%M:%S')}",
                        styles['Normal']
                    )
                    story.append(time_para)
                    story.append(Spacer(1, 0.1 * inch))
                    
                    if turn.question:
                        question_para = Paragraph(
                            f"<b>Interviewer:</b> {turn.question}",
                            styles['Normal']
                        )
                        story.append(question_para)
                        story.append(Spacer(1, 0.1 * inch))
                    
                    answer_para = Paragraph(
                        f"<b>You:</b> {turn.answer}",
                        styles['Normal']
                    )
                    story.append(answer_para)
                    story.append(Spacer(1, 0.2 * inch))
                
                # Build PDF
                doc.build(story)
                buffer.seek(0)
                
                return Response(
                    content=buffer.getvalue(),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f'attachment; filename="transcript_{session_id}.pdf"'
                    }
                )
            
            except ImportError:
                logger.warning("reportlab not installed. Falling back to TXT format.")
                # Fallback to TXT if reportlab not available
                return Response(
                    content=transcript_text,
                    media_type="text/plain",
                    headers={
                        "Content-Disposition": f'attachment; filename="transcript_{session_id}.txt"'
                    }
                )
    
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error generating transcript download for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating transcript: {str(e)}")

