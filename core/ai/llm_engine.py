"""LLM engine for OpenRouter API integration.

This module implements the LLM engine that calls OpenRouter API
using OpenAI SDK to get structured JSON responses from the LLM.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai import APITimeoutError, APIError, RateLimitError

from models.constants import ConfidenceLevel, InterviewPhase, LLMAction
from models.response_models import LLMResponse
from util.logger import get_llm_logger

logger = logging.getLogger(__name__)


class LLMEngine:
    """LLM engine for OpenRouter API integration.
    
    This class handles LLM calls to OpenRouter API using OpenAI SDK.
    It formats prompts, calls the LLM, extracts and validates JSON
    responses, and implements retry logic for errors.
    
    Attributes:
        client: OpenAI client configured for OpenRouter
        model: Model identifier for OpenRouter
        api_key: OpenRouter API key
        max_tokens: Maximum tokens for response
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        temperature: Temperature for generation
    """

    # Configuration constants
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/gpt-oss-20b"  # Changed from :free variant to match working config
    DEFAULT_MAX_TOKENS = 800  # Increased to prevent JSON truncation
    DEFAULT_TIMEOUT = 30  # Increased timeout for longer responses
    DEFAULT_MAX_RETRIES = 3  # Increased retries for reliability
    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        """Initialize LLMEngine with OpenAI client for OpenRouter.
        
        Args:
            api_key: OpenRouter API key. If not provided, loads from
                    OPENROUTER_API_KEY environment variable.
            model: Model identifier. Defaults to openai/gpt-oss-20b.
            Can also be set via OPENROUTER_MODEL environment variable.
            max_tokens: Maximum tokens for response. Defaults to 500.
            timeout: Request timeout in seconds. Defaults to 20.
            max_retries: Maximum retry attempts. Defaults to 2.
            temperature: Temperature for generation. Defaults to 0.7.
            
        Raises:
            ValueError: If API key is not provided and not in environment
        """
        # Load API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key must be provided or set in OPENROUTER_API_KEY environment variable"
                )

        self.api_key = api_key
        # Allow model to be overridden via environment variable, otherwise use default
        self.model = model or os.getenv("OPENROUTER_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )

        logger.info(
            f"LLMEngine initialized with model {self.model}, "
            f"max_tokens={max_tokens}, timeout={timeout}s"
        )
        
        # Initialize structured logger
        self.structured_logger = get_llm_logger()
        
        # #region agent log
        with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_llm_init","timestamp":int(time.time()*1000),"location":"llm_engine.py:93","message":"LLMEngine initialized","data":{"model":self.model,"baseUrl":self.OPENROUTER_BASE_URL,"hasApiKey":bool(self.api_key)},"runId":"post-fix","hypothesisId":"model-fix"}) + "\n")
        # #endregion

    def _build_system_prompt(self) -> str:
        """Build system prompt defining interviewer role and JSON format.
        
        Returns:
            System prompt string with instructions for structured JSON output
        """
        prompt = """You are conducting a mock technical interview. Sound like a real human interviewer - warm, curious, and genuinely interested in the candidate.

CRITICAL - RESPOND TO THE CANDIDATE:
- If the candidate asks YOU a question (like "how are you?", "how's your day?"), ANSWER IT FIRST!
- Example: If they say "I'm good, how are you?" → Respond: "I'm doing great, thanks for asking! So..."
- NEVER ignore what they said - acknowledge it before moving on
- This is a two-way conversation, not an interrogation

TONE & STYLE - Be Human, Not Robotic:
- Sound like a curious colleague having a conversation, not a robot reading a script
- Use natural transitions: "So...", "That's interesting...", "I noticed...", "Tell me more about..."
- React to what they say: "Oh that's cool", "That sounds challenging", "Nice!"
- Keep questions concise and conversational (1-2 sentences max)
- Avoid stiff, formal language like "Could you elaborate on..." or "Can you describe..."

QUESTION STYLE - Be Specific, Not Generic:
- ALWAYS reference specific details from their KEY DISCUSSION POINTS or resume
- Tie questions to their actual projects, achievements, or technologies they mentioned
- Ask "how" and "why" questions that show genuine curiosity

Examples of BAD vs GOOD questions:
BAD: "Can you tell me about a challenging project you worked on?"
GOOD: "That migration to microservices you led - what was the trickiest part?"

BAD: "Describe your experience with distributed systems."
GOOD: "You mentioned processing 1M events/day - how did you handle failures in that pipeline?"

BAD: "Tell me about a time when you showed leadership."
GOOD: "When you led that team of 5 on the data platform, how did you handle disagreements on technical decisions?"

You must respond ONLY with valid JSON:
{
  "action": "one of: ask_follow_up, next_question, clarify, repeat_question, end_phase, end_interview, acknowledge",
  "question": "the question text (required for ask_follow_up, next_question, clarify, repeat_question)",
  "confidence": "one of: high, medium, low",
  "reasoning": "brief explanation of your decision (optional)"
}

Actions:
- "ask_follow_up": Dig deeper into current topic
- "next_question": Move to a new main question
- "clarify": Answer was unclear
- "repeat_question": Candidate didn't understand
- "end_phase": Transition to next interview phase
- "end_interview": Conclude the interview
- "acknowledge": Simple acknowledgment ("Got it", "I see")

Special case - Initial Greeting:
- When candidate's answer is "[INITIAL_GREETING]" or "[No answer yet - this is the initial greeting]", you are starting the interview
- Use the candidate's NAME from the CANDIDATE PROFILE section if available
- Generate a brief, casual greeting (1-2 sentences): "Hey [Name]! How's it going?" or "Hi [Name], how are you doing today?"
- Use "next_question" action with your greeting
- Keep it warm and personal by using their name

CRITICAL - Avoid Repetition:
- NEVER ask the same question twice - check "DO NOT REPEAT THESE QUESTIONS" section
- NEVER ask similar questions (e.g., if you asked "how's your day?", don't ask "how's your day been?")
- Each question should BUILD ON their previous answer - reference what they just said
- Reference what the candidate said earlier to show you're listening
- If you see a question in the "Recent Conversation" or "DO NOT REPEAT" section, pick a DIFFERENT topic

FLOW - Build the Conversation:
- Your question should directly relate to what they JUST said
- Example flow: They say "I'm working on ML pipelines" → You ask "What's the trickiest part of those pipelines?"
- DON'T jump to unrelated topics - follow the thread of conversation"""
        
        return prompt

    def _get_phase_guidance(self, phase: InterviewPhase) -> str:
        """Get phase-specific guidance for LLM.
        
        Args:
            phase: Current interview phase
            
        Returns:
            Phase-specific guidance string
        """
        guidance_map = {
            InterviewPhase.GREETING: (
                "GREETING phase - INITIAL greeting ONLY. "
                "Check 'Recent Conversation' - if there are NO turns yet, greet with their name: "
                "'Hey [Name]! How's it going?' "
                "This phase ends after ONE exchange - the system will transition to SMALL_TALK."
            ),
            InterviewPhase.SMALL_TALK: (
                "SMALL_TALK phase - Build rapport through natural conversation. "
                "This is casual chit-chat before diving into interview content. "
                "CRITICAL: If they asked you a question, ANSWER IT FIRST, then ask something NEW. "
                "NEVER repeat variations of the same question (e.g., 'how's your day?' → 'how's your day been?'). "
                "ALWAYS build on what they JUST said - if they mentioned something, ask about THAT. "
                "Example: They say 'pretty busy' → ask 'What's been keeping you so busy?' "
                "NOT: 'How are you feeling about the interview?' (unrelated topic). "
                "Possible topics (pick ONE new topic each turn): weekend plans, recent projects, what brought them here. "
                "Keep it light and friendly - 2-3 exchanges."
            ),
            InterviewPhase.RESUME_DISCUSSION: (
                "RESUME_DISCUSSION phase - High-level overview of their background. "
                "Now transition from small talk to discussing their career at a high level. "
                "Ask about their current role, responsibilities, what they enjoy about their job. "
                "Examples: 'So tell me a bit about what you do at [Company] - what does a typical day look like?', "
                "'I see you're a [Role] - how did you end up in that field?', "
                "'What's been keeping you busy at work lately?' "
                "Keep questions open-ended and high-level - save technical deep-dives for later."
            ),
            InterviewPhase.INTRODUCTION: (
                "INTRODUCTION phase - Get to know them professionally. "
                "Use their name if you have it. Ask what excites them about their current work. "
                "Example: 'So [Name], what's been the most interesting thing you've worked on lately?' "
                "Keep it warm and curious, like meeting someone at a conference."
            ),
            InterviewPhase.WARMUP: (
                "WARMUP phase - Easy questions to get comfortable. "
                "Pick ONE thing from their KEY DISCUSSION POINTS and ask about it casually. "
                "Example: 'I noticed you built that Python library - what made you decide to open source it?' "
                "Goal: Get them talking confidently before harder questions."
            ),
            InterviewPhase.TECHNICAL: (
                "TECHNICAL phase - Dig into their technical work. "
                "Pick a specific project from their KEY DISCUSSION POINTS and ask about technical decisions. "
                "Examples: 'How did you decide between X and Y approach?', 'What would you do differently?', "
                "'Walk me through how that system handles [specific scenario].' "
                "Follow up on interesting details - show you're genuinely curious about their work."
            ),
            InterviewPhase.BEHAVIORAL: (
                "BEHAVIORAL phase - Understand how they work with others. "
                "Reference a specific achievement from their resume and ask about the human side. "
                "Examples: 'When you led that migration, how did you get buy-in from the team?', "
                "'What was the hardest conversation you had during that project?', "
                "'How did you handle it when things didn't go as planned?' "
                "Focus on real situations, not hypotheticals."
            ),
            InterviewPhase.CLOSING: (
                "CLOSING phase - Wrap up warmly. "
                "Ask if they have questions for you. Thank them genuinely. "
                "Example: 'This has been great - any questions you have for me about the role or team?' "
                "Keep it brief and positive."
            ),
            InterviewPhase.ENDED: (
                "Interview ended. Do not ask more questions."
            ),
        }
        return guidance_map.get(phase, "")

    def _build_user_prompt(self, context: str, user_answer: str, phase: Optional[InterviewPhase] = None) -> str:
        """Build user prompt with context and candidate answer.
        
        Args:
            context: Formatted context from ContextBuilder
            user_answer: The candidate's current answer/utterance
            
        Returns:
            User prompt string
        """
        # Get phase-specific guidance if phase is provided
        phase_guidance = ""
        if phase:
            phase_guidance = self._get_phase_guidance(phase)
        
        # Check if this is the initial greeting scenario
        if user_answer == "" or user_answer == "[INITIAL_GREETING]":
            prompt = f"""{context}

This is the START of the interview. The candidate has not spoken yet. Generate a brief, casual, and friendly initial greeting.

IMPORTANT: Use the candidate's NAME from the CANDIDATE PROFILE section above if available!

The greeting should:
- Use their name: "Hey [Name]! How's it going?" or "Hi [Name], how are you doing today?"
- Be brief and casual (1-2 sentences max)
- Be warm and personal
- Use the "next_question" action with your greeting as the question text

Respond with ONLY valid JSON in the required format."""
        else:
            # Build prompt with phase-specific guidance
            if phase_guidance:
                prompt = f"""{context}

{phase_guidance}

Candidate's Current Answer: {user_answer}

CRITICAL INSTRUCTIONS:
1. This is a CONTINUATION of an ongoing conversation - check "Recent Conversation" above
2. NEVER repeat a greeting if you've already greeted them
3. ACKNOWLEDGE what the candidate just said before asking your next question
4. If they asked you a question (like "how are you?"), ANSWER IT FIRST! Example: "I'm doing great, thanks! So..."
5. Reference their KEY DISCUSSION POINTS to ask specific, personalized questions
6. Be conversational - react to what they said with phrases like "Nice!", "That's cool", "Oh interesting"

Respond with ONLY valid JSON in the required format."""
            else:
                prompt = f"""{context}

Candidate's Current Answer: {user_answer}

CRITICAL INSTRUCTIONS:
1. This is a CONTINUATION of an ongoing conversation - check "Recent Conversation" above
2. NEVER repeat a greeting if you've already greeted them
3. ACKNOWLEDGE what the candidate just said before asking your next question
4. If they asked you a question (like "how are you?"), ANSWER IT FIRST! Example: "I'm doing great, thanks! So..."
5. Reference the candidate's background to ask specific questions
6. Be conversational - react to what they said with phrases like "Nice!", "That's cool", "Oh interesting"

Respond with ONLY valid JSON in the required format."""
        
        return prompt

    def _build_messages(
        self, system_prompt: str, user_prompt: str
    ) -> List[Dict[str, str]]:
        """Build messages array for OpenAI API.
        
        Args:
            system_prompt: System prompt string
            user_prompt: User prompt string
            
        Returns:
            List of message dictionaries in OpenAI format
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return messages

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text.
        
        Handles various response formats:
        - Pure JSON
        - Markdown code blocks (```json ... ```)
        - Text with JSON embedded
        - JSON with trailing text
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be extracted
        """
        # Clean the response text
        text = response_text.strip()
        
        # Strategy 1: Try parsing as-is
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code blocks
        # Match ```json ... ``` or ``` ... ```
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON object using regex
        # Match { ... } pattern
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Try to find JSON starting from first {
        start_idx = text.find("{")
        if start_idx >= 0:
            # Try to find matching closing brace
            brace_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start_idx : i + 1])
                        except json.JSONDecodeError:
                            break
        
        # If all strategies fail, raise error
        raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")

    def _validate_response(self, response_dict: Dict[str, Any]) -> LLMResponse:
        """Validate response dictionary and create LLMResponse model.
        
        Args:
            response_dict: Parsed JSON dictionary from LLM
            
        Returns:
            Validated LLMResponse Pydantic model
            
        Raises:
            ValueError: If response doesn't match required schema
        """
        # Validate required fields
        if "action" not in response_dict:
            raise ValueError("Response missing required field: action")
        if "confidence" not in response_dict:
            raise ValueError("Response missing required field: confidence")
        
        # Validate action enum
        try:
            action = LLMAction(response_dict["action"])
        except ValueError:
            valid_actions = [a.value for a in LLMAction]
            raise ValueError(
                f"Invalid action '{response_dict['action']}'. "
                f"Must be one of: {valid_actions}"
            )
        
        # Validate confidence enum
        try:
            confidence = ConfidenceLevel(response_dict["confidence"])
        except ValueError:
            valid_confidences = [c.value for c in ConfidenceLevel]
            raise ValueError(
                f"Invalid confidence '{response_dict['confidence']}'. "
                f"Must be one of: {valid_confidences}"
            )
        
        # Create and validate LLMResponse model
        llm_response = LLMResponse(
            action=action,
            question=response_dict.get("question"),
            confidence=confidence,
            reasoning=response_dict.get("reasoning"),
        )
        
        return llm_response

    def _create_fallback_response(self) -> LLMResponse:
        """Create safe fallback response when LLM fails.
        
        Returns:
            LLMResponse with safe default values
        """
        return LLMResponse(
            action=LLMAction.ACKNOWLEDGE,
            question=None,
            confidence=ConfidenceLevel.LOW,
            reasoning="LLM call failed, using fallback response",
        )

    def _handle_api_error(self, error: Exception, attempt: int) -> None:
        """Handle different types of API errors.
        
        Args:
            error: The exception that occurred
            attempt: Current retry attempt number
        """
        if isinstance(error, APITimeoutError):
            logger.warning(
                f"LLM API timeout on attempt {attempt}: {error}. Will retry if attempts remain."
            )
        elif isinstance(error, RateLimitError):
            logger.warning(
                f"LLM rate limit error on attempt {attempt}: {error}. "
                "Will wait and retry if attempts remain."
            )
        elif isinstance(error, APIError):
            logger.error(
                f"LLM API error on attempt {attempt}: {error}. "
                "Will retry if attempts remain."
            )
        else:
            logger.error(
                f"Unexpected error on attempt {attempt}: {type(error).__name__}: {error}"
            )

    async def call_llm(
        self, 
        context: str, 
        user_answer: str, 
        phase: Optional[InterviewPhase] = None,
        session_id: Optional[str] = None
    ) -> LLMResponse:
        """Call LLM and get structured response.
        
        Main method that builds prompts, calls the LLM, extracts JSON,
        validates response, and implements retry logic.
        
        Args:
            context: Formatted context from ContextBuilder
            user_answer: The candidate's current answer/utterance
            phase: Optional interview phase for phase-specific guidance
            session_id: Optional session identifier for token tracking
            
        Returns:
            LLMResponse with validated structured output, or fallback
            if all retries fail
        """
        start_time = time.time()
        logger.debug(
            f"Calling LLM with context size {len(context)} chars, "
            f"answer size {len(user_answer)} chars"
        )
        
        # Set session_id for structured logger if provided
        if session_id:
            self.structured_logger.set_session_id(session_id)
        
        # Log LLM input
        self.structured_logger.info(
            "llm_input",
            "LLM API call initiated",
            {
                "context_length": len(context),
                "user_answer_length": len(user_answer),
                "phase": phase.value if phase else None,
                "model": self.model
            },
            session_id=session_id
        )
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context, user_answer, phase)
        messages = self._build_messages(system_prompt, user_prompt)
        
        # Retry loop
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Call OpenAI API (non-streaming)
                logger.debug(f"LLM API call attempt {attempt}/{self.max_retries}")
                # #region agent log
                with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_llm_call","timestamp":int(time.time()*1000),"location":"llm_engine.py:354","message":"Before LLM API call","data":{"model":self.model,"attempt":attempt,"maxTokens":self.max_tokens,"temperature":self.temperature,"messagesCount":len(messages)},"runId":"post-fix","hypothesisId":"model-fix"}) + "\n")
                # #endregion
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
                
                # #region agent log
                with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_llm_response","timestamp":int(time.time()*1000),"location":"llm_engine.py:368","message":"LLM API response received","data":{"model":self.model,"hasResponse":bool(response),"hasChoices":bool(response.choices) if response else False},"runId":"post-fix","hypothesisId":"model-fix"}) + "\n")
                # #endregion
                
                # Extract response text
                response_text = response.choices[0].message.content
                if not response_text:
                    logger.warning(f"Empty response from LLM on attempt {attempt}. Will retry.")
                    if attempt < self.max_retries:
                        await asyncio.sleep(1)
                        last_error = ValueError("Empty response from LLM")
                        continue
                    raise ValueError("Empty response from LLM")
                
                logger.debug(f"LLM response received: {len(response_text)} characters")
                logger.info(f"LLM raw response: {response_text}")
                # #region agent log
                with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_llm_text","timestamp":int(time.time()*1000),"location":"llm_engine.py:375","message":"LLM response text extracted","data":{"responseLength":len(response_text),"responseFull":response_text},"runId":"post-fix","hypothesisId":"model-fix"}) + "\n")
                # #endregion
                
                # Extract JSON from response
                try:
                    response_dict = self._extract_json_from_response(response_text)
                except ValueError as e:
                    logger.warning(
                        f"Failed to extract JSON on attempt {attempt}: {e}. "
                        "Will retry with stricter prompt if attempts remain."
                    )
                    # Add stricter instruction for retry
                    if attempt < self.max_retries:
                        user_prompt += "\n\nIMPORTANT: Respond with ONLY valid JSON, no markdown, no extra text. Keep the response SHORT and complete."
                        messages = self._build_messages(system_prompt, user_prompt)
                        # Wait before retry
                        await asyncio.sleep(1)
                    last_error = e
                    continue
                
                # Validate response
                try:
                    llm_response = self._validate_response(response_dict)
                    
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"LLM call successful: action={llm_response.action.value}, "
                        f"confidence={llm_response.confidence.value}, "
                        f"time={elapsed_time:.2f}s"
                    )
                    logger.info(
                        f"LLM response details: action={llm_response.action.value}, "
                        f"confidence={llm_response.confidence.value}, "
                        f"question={llm_response.question}, "
                        f"reasoning={llm_response.reasoning}"
                    )
                    
                    # #region agent log
                    with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_llm_validated","timestamp":int(time.time()*1000),"location":"llm_engine.py:405","message":"LLM response validated","data":{"action":llm_response.action.value,"confidence":llm_response.confidence.value,"question":llm_response.question,"reasoning":llm_response.reasoning,"elapsedTime":elapsed_time},"runId":"post-fix","hypothesisId":"model-fix"}) + "\n")
                    # #endregion
                    
                    # Record token usage for cost tracking
                    if session_id and hasattr(response, 'usage') and response.usage:
                        try:
                            from core.safety import SafetyController
                            safety_controller = SafetyController.get_instance()
                            total_tokens = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                            if total_tokens > 0:
                                await safety_controller.record_llm_tokens(session_id, total_tokens)
                                logger.debug(f"Recorded {total_tokens} tokens for session {session_id}")
                        except Exception as e:
                            logger.warning(f"Failed to record LLM tokens for session {session_id}: {e}")
                    
                    return llm_response
                    
                except ValueError as e:
                    logger.warning(
                        f"Response validation failed on attempt {attempt}: {e}. "
                        "Will retry if attempts remain."
                    )
                    if attempt < self.max_retries:
                        user_prompt += "\n\nIMPORTANT: Ensure your JSON matches the exact format with valid action and confidence values."
                        messages = self._build_messages(system_prompt, user_prompt)
                    last_error = e
                    continue
                    
            except APITimeoutError as e:
                self._handle_api_error(e, attempt)
                if attempt < self.max_retries:
                    # Wait before retry
                    await asyncio.sleep(1)
                last_error = e
                continue
                
            except RateLimitError as e:
                self._handle_api_error(e, attempt)
                if attempt < self.max_retries:
                    # Exponential backoff for rate limits
                    wait_time = 2 ** attempt
                    logger.info(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                last_error = e
                continue
                
            except APIError as e:
                self._handle_api_error(e, attempt)
                # #region agent log
                with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_llm_api_error","timestamp":int(time.time()*1000),"location":"llm_engine.py:436","message":"LLM APIError caught","data":{"errorType":type(e).__name__,"errorMessage":str(e),"attempt":attempt,"model":self.model},"runId":"post-fix","hypothesisId":"model-fix"}) + "\n")
                # #endregion
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                last_error = e
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error in LLM call: {type(e).__name__}: {e}")
                # #region agent log
                with open(r"c:\Users\darsh\Downloads\interview\.cursor\debug.log", "a") as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_llm_exception","timestamp":int(time.time()*1000),"location":"llm_engine.py:443","message":"Unexpected exception in LLM call","data":{"errorType":type(e).__name__,"errorMessage":str(e),"attempt":attempt,"model":self.model},"runId":"post-fix","hypothesisId":"model-fix"}) + "\n")
                # #endregion
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                last_error = e
                continue
        
        # All retries failed, return fallback
        elapsed_time = time.time() - start_time
        logger.warning(
            f"LLM call failed after {self.max_retries} attempts "
            f"(time={elapsed_time:.2f}s). Using fallback response. "
            f"Last error: {last_error}"
        )
        self.structured_logger.error(
            "llm_api_failed",
            "LLM API call failed after all retries",
            {
                "max_retries": self.max_retries,
                "latency_seconds": round(elapsed_time, 3),
                "last_error": str(last_error) if last_error else None,
                "error_type": type(last_error).__name__ if last_error else None
            },
            session_id=session_id
        )
        
        # Raise appropriate exception for error handling
        if isinstance(last_error, asyncio.TimeoutError):
            from models.exceptions import LLMTimeoutError
            raise LLMTimeoutError(
                f"LLM operation timed out after {elapsed_time:.2f}s",
                timeout_seconds=elapsed_time
            ) from last_error
        elif last_error:
            from models.exceptions import LLMError
            raise LLMError(f"LLM operation failed: {last_error}") from last_error
        
        # If no specific error, return fallback
        return self._create_fallback_response()

