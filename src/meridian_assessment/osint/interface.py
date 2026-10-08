"""Future-facing OSINT interface placeholder.

NOTE: This interface defines the contract for future OSINT discovery and investigation
modules. Actual search queries, source catalogs, scraping, and LLM triage will be
plugged into this abstraction in Phase 2 without altering the criticality engine.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

from meridian_assessment.models.assessment import AssessmentScope
from meridian_assessment.models.criticality import CriticalityTier
from meridian_assessment.models.vendor import Vendor


class OSINTFindingPlaceholder(BaseModel):
    """Placeholder model representing future OSINT investigation output."""

    model_config = ConfigDict(frozen=True)

    vendor_id: str
    status: str = Field(
        default="NOT_IMPLEMENTED_PHASE_1",
        description="Status indicator indicating OSINT stage is pending implementation",
    )
    message: str = Field(
        default="OSINT discovery module is scheduled for implementation in Phase 2.",
        description="Informational note",
    )
    raw_findings: Dict[str, Any] = Field(default_factory=dict)


class BaseOSINTAssessment(ABC):
    """Abstract base class establishing the contract for OSINT assessment engines."""

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

        Raises:
            NotImplementedError: In Phase 1 placeholder implementations.
        """
        raise NotImplementedError("OSINT assessment will be implemented in Phase 2.")


class OSINTAssessmentPlaceholder(BaseOSINTAssessment):
    """Concrete Phase 1 placeholder implementation that preserves pipeline compatibility."""

    def assess(
        self,
        vendor: Vendor,
        criticality_tier: CriticalityTier,
        assessment_scope: AssessmentScope,
    ) -> OSINTFindingPlaceholder:
        """Return explicit Phase 1 placeholder record without executing actual queries."""
        return OSINTFindingPlaceholder(
            vendor_id=vendor.vendor_id,
            status="NOT_IMPLEMENTED_PHASE_1",
            message=f"OSINT execution deferred to Phase 2 for {vendor.vendor_name} ({criticality_tier.value} / {assessment_scope.assessment_depth.value}).",
        )
