"""Criticality models and assessment results."""

from enum import StrEnum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from meridian_assessment.models.assessment import AssessmentDepth
from meridian_assessment.models.vendor import SeverityLevel


class CriticalityTier(StrEnum):
    """Vendor criticality tiers."""
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class CriterionResult(BaseModel):
    """Detailed score breakdown for an individual criticality criterion."""

    model_config = ConfigDict(frozen=True)

    value: SeverityLevel = Field(..., description="Qualitative rating provided for this criterion")
    score: float = Field(..., description="Mapped numerical score for the qualitative rating")
    weight: float = Field(..., description="Configured weighting factor for this criterion")
    weighted_score: float = Field(..., description="Calculated weighted contribution (score * weight)")
    note: Optional[str] = Field(default=None, description="Optional explanatory note")


class CriticalityAssessment(BaseModel):
    """Complete, explainable criticality assessment result for a vendor."""

    model_config = ConfigDict(frozen=True)

    vendor_id: str = Field(..., description="Unique vendor identifier")
    vendor_name: str = Field(..., description="Vendor corporate or trading name")
    domain: str = Field(..., description="Vendor domain")
    criticality_tier: CriticalityTier = Field(..., description="Calculated criticality tier")
    criticality_score: float = Field(..., description="Overall weighted criticality score")
    assessment_depth: AssessmentDepth = Field(..., description="Mapped assessment depth for downstream modules")
    criteria: dict[str, CriterionResult] = Field(..., description="Per-criterion scoring breakdown")
    reasoning: list[str] = Field(default_factory=list, description="Human-readable explainability points")
