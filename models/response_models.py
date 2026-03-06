"""LLM and planner response models."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, model_validator

from models.constants import LLMAction, ConfidenceLevel, PlannerOutputType


class LLMResponse(BaseModel):
    """LLM response model.
    
    Represents the structured JSON output from the LLM,
    containing the suggested action and optional question text.
    """

    action: LLMAction = Field(
        description="The action the LLM suggests for the interviewer"
    )
    question: Optional[str] = Field(
        default=None,
        description="The question text (required for actions that ask questions)"
    )
    confidence: ConfidenceLevel = Field(
        description="LLM's confidence level in the response"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="LLM's reasoning for the suggested action (optional, for debugging)"
    )

    @model_validator(mode="after")
    def validate_question_for_action(self) -> "LLMResponse":
        """Validate that question is provided for actions that require it."""
        requires_question = self.action in [
            LLMAction.ASK_FOLLOW_UP,
            LLMAction.NEXT_QUESTION,
            LLMAction.CLARIFY,
            LLMAction.REPEAT_QUESTION,
        ]
        if requires_question and (not self.question or not self.question.strip()):
            raise ValueError(
                f"Question is required for action {self.action.value}"
            )
        if self.question:
            self.question = self.question.strip()
        return self


class PlannerOutput(BaseModel):
    """Response planner output model.
    
    Represents the final output from the response planner,
    which validates and potentially overrides the LLM's suggestion.
    """

    type: PlannerOutputType = Field(
        description="Type of output (SPEAK for text to be spoken, SILENT for internal actions)"
    )
    text: Optional[str] = Field(
        default=None,
        description="Speakable text (required if type is SPEAK)"
    )
    action: LLMAction = Field(
        description="The action being taken"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (phase, follow_up_count, etc.)"
    )
    was_overridden: bool = Field(
        default=False,
        description="Whether the planner overrode the LLM's decision"
    )
    override_reason: Optional[str] = Field(
        default=None,
        description="Reason for override if the planner changed the LLM's suggestion"
    )

    @model_validator(mode="after")
    def validate_text_for_speak(self) -> "PlannerOutput":
        """Validate that text is provided when type is SPEAK."""
        if self.type == PlannerOutputType.SPEAK:
            if not self.text or not self.text.strip():
                raise ValueError("Text is required when type is SPEAK")
            self.text = self.text.strip()
        return self

