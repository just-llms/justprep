"""Memory manager for short-term, conversation log, and long-term memory.

This module implements the memory system with three distinct components:
1. ShortTermMemory: Sliding window of recent turns for LLM context
2. ConversationLog: Append-only log of all interactions for debugging
3. LongTermMemory: Async structured memory with LLM summarization

Key Principle: All memory updates are non-blocking and never impact response latency.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from models.constants import InterviewPhase
from models.memory_models import LogEntry, LogEntryType
from models.session_models import Turn

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Short-term memory with sliding window of recent turns.
    
    Maintains a sliding window of the most recent turns for fast access
    by the Context Builder. Automatically removes older turns when window
    size is exceeded.
    
    Attributes:
        window_size: Maximum number of turns to keep in sliding window
        short_term_memory: Per-session storage of recent turns
        _locks: Per-session locks for thread-safe operations
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize short-term memory.
        
        Args:
            window_size: Maximum number of turns to keep (default: 5)
        """
        self.window_size = window_size
        self.short_term_memory: Dict[str, List[Turn]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        logger.info(f"ShortTermMemory initialized with window_size={window_size}")

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Async lock for the session
        """
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def add_turn(self, session_id: str, turn: Turn) -> None:
        """Add turn and maintain sliding window.
        
        Adds a turn to the sliding window and automatically removes
        the oldest turn if window size is exceeded.
        
        Args:
            session_id: Session identifier
            turn: Turn to add
        """
        async with self._get_lock(session_id):
            if session_id not in self.short_term_memory:
                self.short_term_memory[session_id] = []
            
            # Add new turn
            self.short_term_memory[session_id].append(turn)
            
            # Maintain sliding window - remove oldest if exceeded
            if len(self.short_term_memory[session_id]) > self.window_size:
                removed_turn = self.short_term_memory[session_id].pop(0)
                logger.debug(
                    f"Removed oldest turn from short-term memory for session {session_id} "
                    f"(turn_number={removed_turn.turn_number})"
                )
            
            logger.debug(
                f"Added turn to short-term memory for session {session_id} "
                f"(turn_number={turn.turn_number}, window_size={len(self.short_term_memory[session_id])})"
            )

    async def get_recent_turns(self, session_id: str, n: int = 5) -> List[Turn]:
        """Get last N turns from sliding window.
        
        Args:
            session_id: Session identifier
            n: Number of recent turns to retrieve (default: 5)
            
        Returns:
            List of Turn objects, most recent last. Returns empty list
            if session not found or no turns available.
        """
        async with self._get_lock(session_id):
            if session_id not in self.short_term_memory:
                return []
            
            turns = self.short_term_memory[session_id]
            # Return last N turns (or all if fewer than N)
            recent_turns = turns[-n:] if len(turns) > n else turns
            
            logger.debug(
                f"Retrieved {len(recent_turns)} recent turns for session {session_id} "
                f"(requested {n}, available {len(turns)})"
            )
            
            return recent_turns.copy()  # Return copy to prevent external modification

    async def clear_old_turns(self, session_id: str) -> None:
        """Clear turns for session.
        
        Args:
            session_id: Session identifier
        """
        async with self._get_lock(session_id):
            if session_id in self.short_term_memory:
                turn_count = len(self.short_term_memory[session_id])
                self.short_term_memory[session_id].clear()
                logger.info(
                    f"Cleared {turn_count} turns from short-term memory for session {session_id}"
                )

    async def get_turn_count(self, session_id: str) -> int:
        """Get current turn count for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of turns in sliding window
        """
        async with self._get_lock(session_id):
            if session_id not in self.short_term_memory:
                return 0
            return len(self.short_term_memory[session_id])


class ConversationLog:
    """Append-only conversation log for debugging and analytics.
    
    Maintains a chronological log of all interactions in the system.
    Entries are append-only and never modified after creation.
    
    Attributes:
        conversation_logs: Per-session storage of log entries
        _locks: Per-session locks for thread-safe operations
        _enable_file_export: Whether to export logs to files
        _export_dir: Directory for exported log files
    """

    def __init__(self) -> None:
        """Initialize conversation log."""
        self.conversation_logs: Dict[str, List[LogEntry]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        
        # Configuration from environment
        self._enable_file_export = os.getenv("MEMORY_ENABLE_FILE_EXPORT", "false").lower() == "true"
        self._export_dir = os.getenv("MEMORY_EXPORT_DIR", "./logs")
        
        logger.info(
            f"ConversationLog initialized "
            f"(file_export={'enabled' if self._enable_file_export else 'disabled'})"
        )

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Async lock for the session
        """
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def append_log(self, session_id: str, entry: LogEntry) -> None:
        """Append log entry.
        
        Adds a log entry to the conversation log. Entries are append-only
        and never modified after creation.
        
        Args:
            session_id: Session identifier
            entry: Log entry to append
        """
        async with self._get_lock(session_id):
            if session_id not in self.conversation_logs:
                self.conversation_logs[session_id] = []
            
            # Append entry (append-only, no modifications)
            self.conversation_logs[session_id].append(entry)
            
            logger.debug(
                f"Appended log entry for session {session_id}: "
                f"type={entry.entry_type.value}, timestamp={entry.timestamp}"
            )
            
            # Optional file export (non-blocking)
            if self._enable_file_export:
                # Fire-and-forget file export
                asyncio.create_task(self._export_to_file_async(session_id, entry))

    async def _export_to_file_async(self, session_id: str, entry: LogEntry) -> None:
        """Export log entry to file asynchronously.
        
        Args:
            session_id: Session identifier
            entry: Log entry to export
        """
        try:
            # Ensure export directory exists
            os.makedirs(self._export_dir, exist_ok=True)
            
            # Create file path: logs/session_id.jsonl
            filepath = os.path.join(self._export_dir, f"{session_id}.jsonl")
            
            # Append entry as JSON line
            with open(filepath, "a", encoding="utf-8") as f:
                json.dump(entry.model_dump(mode="json"), f)
                f.write("\n")
            
            logger.debug(f"Exported log entry to {filepath}")
        except Exception as e:
            # Don't raise - file export is optional
            logger.warning(f"Error exporting log entry to file: {e}")

    async def get_logs(
        self, session_id: str, entry_type: Optional[LogEntryType] = None
    ) -> List[LogEntry]:
        """Get logs for session, optionally filtered by type.
        
        Args:
            session_id: Session identifier
            entry_type: Optional filter by entry type
            
        Returns:
            List of log entries, optionally filtered by type.
            Returns empty list if session not found.
        """
        async with self._get_lock(session_id):
            if session_id not in self.conversation_logs:
                return []
            
            logs = self.conversation_logs[session_id]
            
            # Filter by type if specified
            if entry_type is not None:
                filtered_logs = [log for log in logs if log.entry_type == entry_type]
                return filtered_logs.copy()
            
            return logs.copy()  # Return copy to prevent external modification

    async def get_recent_logs(self, session_id: str, n: int = 100) -> List[LogEntry]:
        """Get last N log entries for session.
        
        Args:
            session_id: Session identifier
            n: Number of recent entries to retrieve (default: 100)
            
        Returns:
            List of last N log entries. Returns empty list if session not found.
        """
        async with self._get_lock(session_id):
            if session_id not in self.conversation_logs:
                return []
            
            logs = self.conversation_logs[session_id]
            recent_logs = logs[-n:] if len(logs) > n else logs
            
            return recent_logs.copy()  # Return copy to prevent external modification

    async def export_to_file(self, session_id: str, filepath: str) -> None:
        """Export all logs for session to JSON file.
        
        Args:
            session_id: Session identifier
            filepath: Path to output file
        """
        async with self._get_lock(session_id):
            if session_id not in self.conversation_logs:
                logger.warning(f"No logs found for session {session_id} to export")
                return
            
            try:
                logs = self.conversation_logs[session_id]
                
                # Export as JSON array
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(
                        [log.model_dump(mode="json") for log in logs],
                        f,
                        indent=2,
                        default=str
                    )
                
                logger.info(
                    f"Exported {len(logs)} log entries for session {session_id} to {filepath}"
                )
            except Exception as e:
                logger.error(f"Error exporting logs to file {filepath}: {e}", exc_info=True)
                raise


class LongTermMemory:
    """Long-term memory with async summarization.
    
    Maintains structured memory with insights, strengths, weaknesses,
    and phase summaries. Updates are made asynchronously and never
    block the main response flow.
    
    Attributes:
        long_term_memory: Per-session storage of structured memory
        _locks: Per-session locks for thread-safe operations
        llm_engine: Optional LLM engine for summarization
    """

    def __init__(self, llm_engine: Optional[Any] = None) -> None:
        """Initialize long-term memory.
        
        Args:
            llm_engine: Optional LLM engine for async summarization
        """
        self.long_term_memory: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self.llm_engine = llm_engine
        self._enabled = os.getenv("MEMORY_LONG_TERM_ENABLED", "true").lower() == "true"
        
        logger.info(
            f"LongTermMemory initialized "
            f"(enabled={self._enabled}, llm_engine={'provided' if llm_engine else 'not provided'})"
        )

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Async lock for the session
        """
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def get_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current memory for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Memory dictionary, or None if session not found
        """
        async with self._get_lock(session_id):
            return self.long_term_memory.get(session_id)

    async def update_memory(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update memory for session (merge, not replace).
        
        Merges updates into existing memory. Never rewrites from scratch.
        This ensures historical accuracy and prevents recursive drift.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates to merge
        """
        async with self._get_lock(session_id):
            if session_id not in self.long_term_memory:
                # Initialize memory structure
                self.long_term_memory[session_id] = {
                    "session_id": session_id,
                    "phases": {},
                    "overall_insights": {},
                    "last_updated": datetime.now().isoformat(),
                }
            
            # Merge updates (deep merge for nested dicts)
            self._deep_merge(self.long_term_memory[session_id], updates)
            self.long_term_memory[session_id]["last_updated"] = datetime.now().isoformat()
            
            logger.info(f"Updated long-term memory for session {session_id}")

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target.
        
        Recursively merges nested dictionaries. Lists are replaced (not merged).
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._deep_merge(target[key], value)
            else:
                # Replace or add value
                target[key] = value

    async def trigger_phase_summary(
        self, session_id: str, phase: InterviewPhase, recent_turns: List[Turn]
    ) -> None:
        """Trigger async phase summary (non-blocking).
        
        Starts a background task to build phase summary. Never blocks.
        
        Args:
            session_id: Session identifier
            phase: Phase that was completed
            recent_turns: Recent turns from the completed phase
        """
        if not self._enabled:
            logger.debug("Long-term memory summarization is disabled")
            return
        
        if not self.llm_engine:
            logger.warning(
                "LLM engine not provided - cannot build phase summary for long-term memory"
            )
            return
        
        # Fire-and-forget background task
        asyncio.create_task(
            self._build_phase_summary_async(session_id, phase, recent_turns)
        )
        logger.info(
            f"Triggered async phase summary for session {session_id}, phase={phase.value}"
        )

    async def _build_phase_summary_async(
        self, session_id: str, phase: InterviewPhase, recent_turns: List[Turn]
    ) -> None:
        """Build phase summary asynchronously.
        
        This runs in a background task and never blocks the main flow.
        If it fails, it logs the error but doesn't raise.
        
        Args:
            session_id: Session identifier
            phase: Phase that was completed
            recent_turns: Recent turns from the completed phase
        """
        try:
            logger.info(
                f"Building phase summary for session {session_id}, phase={phase.value} "
                f"(turns={len(recent_turns)})"
            )
            
            # Format turns for LLM prompt
            turns_text = self._format_turns_for_summary(recent_turns)
            
            # Get current memory state
            current_memory = await self.get_memory(session_id)
            
            # Build LLM prompt for summarization
            prompt = self._build_summary_prompt(phase, turns_text, current_memory)
            
            # Call LLM for summarization (with timeout)
            try:
                # Use LLM engine to generate summary
                # Note: This is a simplified call - actual implementation depends on LLM engine interface
                summary_text = await asyncio.wait_for(
                    self._call_llm_for_summary(prompt),
                    timeout=30.0
                )
                
                # Parse summary (expecting JSON)
                summary_data = json.loads(summary_text)
                
                # Update memory with phase summary (merge, not replace)
                phase_key = phase.value.lower()
                updates = {
                    "phases": {
                        phase_key: {
                            "summary": summary_data.get("summary", ""),
                            "strengths": summary_data.get("strengths", []),
                            "weaknesses": summary_data.get("weaknesses", []),
                            "key_points": summary_data.get("key_points", []),
                            "completed_at": datetime.now().isoformat(),
                        }
                    }
                }
                
                await self.update_memory(session_id, updates)
                
                logger.info(
                    f"Phase summary completed for session {session_id}, phase={phase.value}"
                )
                
            except asyncio.TimeoutError:
                logger.warning(
                    f"LLM summarization timeout for session {session_id}, phase={phase.value}"
                )
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse LLM summary JSON for session {session_id}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Error in LLM summarization for session {session_id}: {e}",
                    exc_info=True
                )
                
        except Exception as e:
            # Never raise - this is a background task
            logger.error(
                f"Error building phase summary for session {session_id}: {e}",
                exc_info=True
            )

    def _format_turns_for_summary(self, turns: List[Turn]) -> str:
        """Format turns as text for LLM summarization.
        
        Args:
            turns: List of turns to format
            
        Returns:
            Formatted text string
        """
        lines = []
        for turn in turns:
            lines.append(f"Q: {turn.question or 'None'}")
            lines.append(f"A: {turn.answer}")
            lines.append("")
        return "\n".join(lines)

    def _build_summary_prompt(
        self, phase: InterviewPhase, turns_text: str, current_memory: Optional[Dict[str, Any]]
    ) -> str:
        """Build LLM prompt for phase summarization.
        
        Args:
            phase: Phase that was completed
            turns_text: Formatted turns text
            current_memory: Current memory state (for context)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Summarize the following interview phase and extract key insights.

Phase: {phase.value.upper()}

Recent conversation:
{turns_text}

Please provide a JSON response with the following structure:
{{
    "summary": "Brief summary of the phase (2-3 sentences)",
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "key_points": ["point1", "point2", ...]
}}

Focus on:
- Technical skills demonstrated (if technical phase)
- Communication clarity
- Problem-solving approach
- Areas that need improvement
"""
        return prompt

    async def _call_llm_for_summary(self, prompt: str) -> str:
        """Call LLM for summarization.
        
        Uses the LLM engine to generate a summary. This is a simplified
        interface that calls the LLM directly with a custom prompt.
        
        Args:
            prompt: Prompt for summarization
            
        Returns:
            LLM response text (expected to be JSON)
            
        Raises:
            NotImplementedError: If LLM engine doesn't support this operation
            RuntimeError: If LLM call fails
        """
        if not self.llm_engine:
            raise NotImplementedError("LLM engine not available")
        
        try:
            # Use OpenAI client directly for summarization
            # Build messages for summarization
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that summarizes interview conversations. "
                               "Always respond with valid JSON only, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call LLM using the engine's client
            # Note: This assumes LLMEngine has a client attribute
            if hasattr(self.llm_engine, 'client'):
                response = await asyncio.to_thread(
                    self.llm_engine.client.chat.completions.create,
                    model=self.llm_engine.model,
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more consistent summaries
                    max_tokens=500,
                    response_format={"type": "json_object"}  # Request JSON format
                )
                
                # Extract response text
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content or "{}"
                else:
                    raise RuntimeError("Empty response from LLM")
            else:
                # Fallback: try to use a generic call method if available
                raise NotImplementedError(
                    "LLM engine doesn't expose client for summarization"
                )
                
        except Exception as e:
            logger.error(f"Error calling LLM for summarization: {e}", exc_info=True)
            raise RuntimeError(f"LLM summarization failed: {e}") from e


# Helper functions for creating log entries
def create_log_entry(
    entry_type: LogEntryType,
    session_id: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
) -> LogEntry:
    """Create a log entry with current timestamp.
    
    Convenience function for creating log entries throughout the system.
    
    Args:
        entry_type: Type of log entry
        session_id: Session identifier
        data: Event-specific data
        metadata: Optional additional metadata
        timestamp: Optional timestamp (defaults to now)
        
    Returns:
        LogEntry instance
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return LogEntry(
        entry_type=entry_type,
        timestamp=timestamp,
        session_id=session_id,
        data=data,
        metadata=metadata,
    )


class MemoryManager:
    """Orchestrates all memory components.
    
    Provides a unified interface for accessing short-term memory,
    conversation log, and long-term memory.
    
    Attributes:
        short_term_memory: ShortTermMemory instance
        conversation_log: ConversationLog instance
        long_term_memory: LongTermMemory instance
    """

    _instance: Optional["MemoryManager"] = None
    _lock = asyncio.Lock()

    def __init__(self, llm_engine: Optional[Any] = None) -> None:
        """Initialize memory manager.
        
        Args:
            llm_engine: Optional LLM engine for long-term memory summarization
        """
        window_size = int(os.getenv("MEMORY_SHORT_TERM_WINDOW_SIZE", "5"))
        self.short_term_memory = ShortTermMemory(window_size=window_size)
        self.conversation_log = ConversationLog()
        self.long_term_memory = LongTermMemory(llm_engine=llm_engine)
        
        logger.info("MemoryManager initialized")

    @classmethod
    async def get_instance(cls, llm_engine: Optional[Any] = None) -> "MemoryManager":
        """Get or create singleton instance.
        
        Args:
            llm_engine: Optional LLM engine (only used on first creation)
            
        Returns:
            MemoryManager singleton instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(llm_engine=llm_engine)
            return cls._instance

    @classmethod
    def get_instance_sync(cls, llm_engine: Optional[Any] = None) -> "MemoryManager":
        """Get or create singleton instance (synchronous version).
        
        For use when async context is not available.
        
        Args:
            llm_engine: Optional LLM engine (only used on first creation)
            
        Returns:
            MemoryManager singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(llm_engine=llm_engine)
        return cls._instance

