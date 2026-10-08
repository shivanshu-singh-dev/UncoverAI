"""Configuration schemas and validation models."""

import math
from typing import Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TierThreshold(BaseModel):
    """Configuration for a specific criticality tier threshold."""

    model_config = ConfigDict(frozen=True)

    min_score: float = Field(..., ge=0.0, description="Minimum score required to trigger this tier")
    label: str = Field(..., description="Human-readable label")
    description: str = Field(..., description="Detailed description of the tier")


class DepthMapping(BaseModel):
    """Configuration for mapping tier to assessment depth."""

    model_config = ConfigDict(frozen=True)

    depth: str = Field(..., description="Depth identifier (COMPREHENSIVE, TARGETED, LIGHTWEIGHT)")
    description: str = Field(..., description="Description of the depth boundaries")


class ReasoningConfig(BaseModel):
    """Configuration for generating explainable reasoning points."""

    model_config = ConfigDict(frozen=True)

    threshold_score_for_mention: float = Field(
        default=2.5,
        description="Minimum score for a criterion to be highlighted in explainability reasoning",
    )
    descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Template descriptions mapped to criteria names",
    )


class CriticalityConfig(BaseModel):
    """Root configuration model for criticality and scoping rules."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(default="1.0", description="Configuration schema version")
    criterion_weights: dict[str, float] = Field(
        ...,
        description="Weights for each of the 5 criteria",
    )
    severity_scores: dict[str, float] = Field(
        ...,
        description="Score mapped to each categorical severity level",
    )
    tier_thresholds: dict[str, TierThreshold] = Field(
        ...,
        description="Tier threshold definitions",
    )
    assessment_depth_mapping: dict[str, DepthMapping] = Field(
        ...,
        description="Assessment depth mapping per tier",
    )
    reasoning_rules: ReasoningConfig = Field(
        default_factory=ReasoningConfig,
        description="Reasoning generation rules",
    )

    @model_validator(mode="after")
    def validate_weights_and_criteria(self) -> Self:
        required_criteria = {
            "data_sensitivity",
            "payment_flows",
            "regulatory_exposure",
            "operational_dependency",
            "customer_data_volume",
        }
        provided_criteria = set(self.criterion_weights.keys())
        missing = required_criteria - provided_criteria
        if missing:
            raise ValueError(f"Missing required criteria weights in configuration: {sorted(missing)}")

        total_weight = sum(self.criterion_weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=1e-3):
            raise ValueError(f"Criterion weights must sum to 1.0 (got {total_weight:.4f})")

        return self
