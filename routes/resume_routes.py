"""Resume parsing and setup routes."""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI

from core.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

RESUME_EXTRACTION_PROMPT = """You are an expert resume parser. Analyze the following resume text and extract structured information.

Resume Text:
{resume_text}

Extract the following information and return it as a JSON object:

{{
    "name": "Full name of the candidate",
    "current_role": "Current job title/position",
    "current_company": "Current employer/company name",
    "years_of_experience": 5.5,
    "interview_points": [
        "Key point 1 that would be useful to discuss in an interview",
        "Key point 2 - notable achievement or skill",
        "Key point 3 - interesting project or experience",
        "Key point 4 - relevant technical skill or domain expertise",
        "Key point 5 - any unique selling point or differentiator"
    ]
}}

Guidelines:
- For years_of_experience: Calculate total years of professional work experience as a decimal number.
  - Sum the duration of all professional positions listed (exclude internships, education, personal projects)
  - Include months as decimal fractions (e.g., 5 years 6 months = 5.5, 3 years 3 months = 3.25)
  - For overlapping roles, count only once (don't double-count)
  - Round to one decimal place (e.g., 4.33 becomes 4.3)
  - If dates are missing, estimate based on context or use 0 if unable to determine
  - Only count post-graduation professional experience
- For interview_points: Extract 3-7 points that an interviewer would find most relevant
- Focus on achievements, key skills, notable projects, leadership experience
- Include specific technologies, methodologies, or domain expertise
- Highlight any quantifiable achievements (e.g., "Increased revenue by 30%")
- Note career progression or growth indicators
- Include soft skills evidence if mentioned (teamwork, leadership, communication)

Return ONLY the JSON object, no additional text or markdown formatting."""


async def parse_pdf(file_path: Path) -> str:
    """Parse PDF file and extract text.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text from the PDF
    """
    try:
        import pdfplumber
        
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n".join(text_parts)
    
    except ImportError:
        logger.error("pdfplumber not installed. Install with: pip install pdfplumber")
        raise HTTPException(
            status_code=500,
            detail="PDF parsing not available. Please install pdfplumber."
        )
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing PDF: {str(e)}")


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 20
DEFAULT_TEMPERATURE = 0.3


async def extract_resume_info_with_llm(resume_text: str) -> dict[str, Any]:
    """Use OpenRouter LLM to extract structured information from resume.
    
    Args:
        resume_text: Raw text extracted from the resume PDF
        
    Returns:
        Dictionary with extracted information (name, current_role, current_company, interview_points)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not configured"
        )
    
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )
    
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    
    try:
        prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
        
        logger.info(f"Starting LLM request to {model} for resume extraction (prompt length: {len(prompt)} chars)")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            timeout=DEFAULT_TIMEOUT,
        )
        
        finish_reason = response.choices[0].finish_reason if response.choices else "no_choices"
        usage = response.usage
        logger.info(
            f"LLM response received - finish_reason: {finish_reason}, "
            f"tokens: {usage.prompt_tokens}/{usage.completion_tokens}/{usage.total_tokens} (prompt/completion/total)"
            if usage else f"LLM response received - finish_reason: {finish_reason}"
        )
        
        if finish_reason == "length":
            logger.warning("Response was truncated due to max_tokens limit")
        
        # Log full response structure if content is missing
        choice = response.choices[0] if response.choices else None
        if choice:
            response_text = choice.message.content
            if not response_text:
                logger.error(f"Empty content. Message object: {choice.message}")
        else:
            response_text = None
            logger.error(f"No choices in response. Full response: {response}")
        if usage and usage.completion_tokens < 50:
            logger.warning(f"Suspiciously short LLM response ({usage.completion_tokens} tokens). Content: {repr(response_text)}")
        if not response_text or not response_text.strip():
            logger.error(f"Empty or whitespace-only response from LLM. Raw content: {repr(response_text)}")
            raise ValueError("Empty response from LLM")
        
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Extract JSON object using brace matching (handles braces in strings)
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            response_text = response_text[start_idx:end_idx + 1]
        
        logger.debug(f"Cleaned response length: {len(response_text)} chars")
        
        extracted_data = json.loads(response_text)
        
        result = {
            "name": extracted_data.get("name", ""),
            "current_role": extracted_data.get("current_role", ""),
            "current_company": extracted_data.get("current_company", ""),
            "years_of_experience": extracted_data.get("years_of_experience", 0),
            "interview_points": extracted_data.get("interview_points", []),
        }
        
        logger.info(f"Successfully extracted resume info: {result['name']}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e} (response length: {len(response_text) if response_text else 0})")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse resume information from LLM response"
        )
    except Exception as e:
        logger.error(f"Error calling LLM for resume extraction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting resume information: {str(e)}"
        )


@router.post("/api/parse-resume")
async def parse_resume(
    file: UploadFile = File(...),
) -> dict:
    """Parse uploaded resume PDF and extract structured information using LLM.
    
    Args:
        file: Resume file (PDF only)
    
    Returns:
        Extracted information including name, current_role, current_company, 
        interview_points, and session_id
    """
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a PDF resume."
        )
    
    try:
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4()}.pdf"
        
        try:
            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(content)
            
            parsed_text = await parse_pdf(temp_path)
            
            if not parsed_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from PDF. Please ensure the PDF contains readable text."
                )
            
            extracted_info = await extract_resume_info_with_llm(parsed_text)
            
            session_manager = SessionManager.get_instance()
            session_id = str(uuid.uuid4())
            
            await session_manager.create_session(
                session_id=session_id,
                work_experience=parsed_text[:1000],
                current_role=extracted_info.get("current_role", ""),
                resume=parsed_text,
                interview_points=extracted_info.get("interview_points", []),
                years_of_experience=extracted_info.get("years_of_experience"),
                candidate_name=extracted_info.get("name"),
                current_company=extracted_info.get("current_company"),
            )
            
            logger.info(f"Resume parsed and session created: {session_id}")
            
            return {
                "success": True,
                "name": extracted_info.get("name", ""),
                "current_role": extracted_info.get("current_role", ""),
                "current_company": extracted_info.get("current_company", ""),
                "years_of_experience": extracted_info.get("years_of_experience", 0),
                "interview_points": extracted_info.get("interview_points", []),
                "session_id": session_id
            }
        
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing resume: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

