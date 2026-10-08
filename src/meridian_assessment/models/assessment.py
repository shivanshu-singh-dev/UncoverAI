"""Assessment depth and scope models."""

from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


class AssessmentDepth(StrEnum):
    """Depth levels of assessment derived from criticality tier."""
    COMPREHENSIVE = "COMPREHENSIVE"
    TARGETED = "TARGETED"
    LIGHTWEIGHT = "LIGHTWEIGHT"


class AssessmentScope(BaseModel):
    """Scoping specification defining the depth and boundaries of vendor assessment."""

    model_config = ConfigDict(frozen=True)

    assessment_depth: AssessmentDepth = Field(..., description="Determined assessment depth")
    description: str = Field(..., description="Human-readable description of assessment boundaries")
    focus_areas: list[str] = Field(default_factory=list, description="Targeted scope focus areas for downstream stages")
