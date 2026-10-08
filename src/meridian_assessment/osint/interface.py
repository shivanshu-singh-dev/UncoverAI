"""OSINT assessment interface and placeholder."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

from meridian_assessment.models.assessment import AssessmentScope
from meridian_assessment.models.criticality import CriticalityTier
from meridian_assessment.models.vendor import Vendor


class OSINTFindingPlaceholder(BaseModel):
    """Represents pending OSINT investigation output."""

    model_config = ConfigDict(frozen=True)

    vendor_id: str
    status: str = Field(
        default="PENDING",
        description="Status of the OSINT investigation",
    )
    message: str = Field(
        default="OSINT discovery module is not yet available.",
        description="Informational note",
    )
    raw_findings: Dict[str, Any] = Field(default_factory=dict)


class BaseOSINTAssessment(ABC):
    """Abstract base class for OSINT assessment engines."""

    @abstractmethod
    def assess(
        self,
        vendor: Vendor,
        criticality_tier: CriticalityTier,
        assessment_scope: AssessmentScope,
    ) -> OSINTFindingPlaceholder:
        """Execute OSINT discovery and evidence collection for a vendor.

        Args:
            vendor: The vendor entity being investigated.
            criticality_tier: Evaluated criticality tier (TIER_1, TIER_2, TIER_3).
            assessment_scope: Assessment depth and boundaries specification.

        Returns:
            OSINT findings structure.
        """
        raise NotImplementedError("OSINT assessment is not yet implemented.")


class OSINTAssessmentPlaceholder(BaseOSINTAssessment):
    """Placeholder implementation that preserves pipeline compatibility."""

    def assess(
        self,
        vendor: Vendor,
        criticality_tier: CriticalityTier,
        assessment_scope: AssessmentScope,
    ) -> OSINTFindingPlaceholder:
        return OSINTFindingPlaceholder(
            vendor_id=vendor.vendor_id,
            status="PENDING",
            message=f"OSINT investigation pending for {vendor.vendor_name} ({criticality_tier.value} / {assessment_scope.assessment_depth.value}).",
        )
