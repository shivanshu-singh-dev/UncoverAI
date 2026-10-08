"""Assessment scoping engine based on criticality tiers."""

from typing import Dict, List
from meridian_assessment.models.assessment import AssessmentDepth, AssessmentScope
from meridian_assessment.models.config import CriticalityConfig
from meridian_assessment.models.criticality import CriticalityTier


class ScopingEngine:
    """Generates structured assessment scope specifications based on criticality tier."""

    FOCUS_AREAS: Dict[AssessmentDepth, List[str]] = {
        AssessmentDepth.COMPREHENSIVE: [
            "Extensive footprint across corporate and technical surface",
            "Executive and developer credential leak monitoring",
            "Cloud infrastructure and public attack surface scan",
            "Regulatory enforcement actions and litigation records",
            "Subprocessor and 4th-party dependency graph analysis",
            "AI/ML pipeline and third-party API exposure audit",
        ],
        AssessmentDepth.TARGETED: [
            "Primary domain and MX/DNS security posture",
            "Core regulatory certifications and SOC2/ISO compliance verification",
            "Public vulnerabilities associated with exposed vendor assets",
            "Key corporate filings and data privacy policy changes",
        ],
        AssessmentDepth.LIGHTWEIGHT: [
            "Basic domain registration and reputation check",
            "Standard TLS/SSL configuration validation",
            "Known high-profile breach search",
        ],
    }

    def __init__(self, config: CriticalityConfig) -> None:
        self.config = config

    def create_scope(self, tier: CriticalityTier) -> AssessmentScope:
        """Create an assessment scope specification for a given criticality tier."""
        tier_key = tier.value
        depth_cfg = self.config.assessment_depth_mapping.get(tier_key)

        if depth_cfg:
            depth = AssessmentDepth(depth_cfg.depth)
            desc = depth_cfg.description
        else:
            depth = AssessmentDepth.LIGHTWEIGHT
            desc = "Lightweight baseline assessment."

        focus_areas = self.FOCUS_AREAS.get(depth, [])

        return AssessmentScope(
            assessment_depth=depth,
            description=desc,
            focus_areas=focus_areas,
        )
