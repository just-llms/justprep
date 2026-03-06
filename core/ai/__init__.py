"""AI engine components."""

from core.ai.ai_engine import AIEngine
from core.ai.context_builder import ContextBuilder
from core.ai.fsm_controller import FSMController
from core.ai.llm_engine import LLMEngine
from core.ai.response_planner import ResponsePlanner

__all__ = [
    "AIEngine",
    "ContextBuilder",
    "FSMController",
    "LLMEngine",
    "ResponsePlanner",
]

# Global AI engine instance (singleton pattern)
_ai_engine_instance: AIEngine | None = None


def get_ai_engine() -> AIEngine:
    """Get or create the global AI engine instance.
    
    Creates the AI engine with all its dependencies if it doesn't exist.
    Uses singleton pattern to ensure only one instance exists.
    
    Returns:
        AIEngine instance with all components initialized
    """
    global _ai_engine_instance
    
    if _ai_engine_instance is None:
        from core.session_manager import SessionManager
        from core.memory import MemoryManager
        from core.ai.llm_engine import LLMEngine
        import json
        import time
        
        # Initialize all dependencies - use singleton to share same instance
        session_manager = SessionManager.get_instance()
        # #region agent log
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_sm_init_ai","timestamp":int(time.time()*1000),"location":"core/ai/__init__.py:36","message":"SessionManager instance retrieved in AI engine","data":{"instanceId":str(id(session_manager)),"sessionsCount":len(session_manager.sessions)},"runId":"post-fix","hypothesisId":"A"}) + "\n")
        # #endregion
        
        # Initialize LLM engine first (needed for MemoryManager)
        llm_engine = LLMEngine()
        
        # Initialize MemoryManager (singleton) - use sync version since get_ai_engine is sync
        memory_manager = MemoryManager.get_instance_sync(llm_engine=llm_engine)
        
        fsm_controller = FSMController(session_manager)
        context_builder = ContextBuilder(session_manager, fsm_controller, memory_manager)
        response_planner = ResponsePlanner(session_manager, fsm_controller)
        
        # Create AI engine
        _ai_engine_instance = AIEngine(
            session_manager=session_manager,
            fsm_controller=fsm_controller,
            context_builder=context_builder,
            llm_engine=llm_engine,
            response_planner=response_planner,
        )
    
    return _ai_engine_instance
