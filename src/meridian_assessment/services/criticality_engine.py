"""Deterministic criticality evaluation engine."""

from typing import Dict, List
from meridian_assessment.models.assessment import AssessmentDepth
from meridian_assessment.models.config import CriticalityConfig
from meridian_assessment.models.criticality import (
    CriticalityAssessment,
    CriticalityTier,
    CriterionResult,
)
from meridian_assessment.models.vendor import Vendor
from meridian_assessment.utils.exceptions import CriticalityEvaluationError
from meridian_assessment.utils.logger import logger


class CriticalityEngine:
    """Calculates deterministic, explainable criticality assessments based on configuration."""

    def __init__(self, config: CriticalityConfig) -> None:
        """Initialize the engine with validated configuration.

        Args:
            config: Validated CriticalityConfig instance.
        """
        self.config = config

    def evaluate(self, vendor: Vendor) -> CriticalityAssessment:
        """Perform a deterministic criticality assessment on a single vendor.

        Args:
            vendor: Validated Vendor instance.

        Returns:
            Structured, explainable CriticalityAssessment.

        Raises:
            CriticalityEvaluationError: If configuration rules are incomplete for vendor data.
        """
        logger.debug("Evaluating criticality for vendor %s (%s)", vendor.vendor_id, vendor.vendor_name)

        criteria_inputs = {
            "data_sensitivity": vendor.data_sensitivity.value,
            "payment_flows": vendor.payment_flows.value,
            "regulatory_exposure": vendor.regulatory_exposure.value,
            "operational_dependency": vendor.operational_dependency.value,
            "customer_data_volume": vendor.customer_data_volume.value,
        }

        criteria_results: Dict[str, CriterionResult] = {}
        total_weighted_score = 0.0

        for crit_name, crit_val in criteria_inputs.items():
            if crit_name not in self.config.criterion_weights:
                raise CriticalityEvaluationError(f"Missing weight configuration for criterion: '{crit_name}'")

            if crit_val not in self.config.severity_scores:
                raise CriticalityEvaluationError(
                    f"Missing severity score mapping for value '{crit_val}' in criterion '{crit_name}'"
                )

            weight = self.config.criterion_weights[crit_name]
            score = self.config.severity_scores[crit_val]
            weighted_score = round(score * weight, 4)
            total_weighted_score += weighted_score

            criteria_results[crit_name] = CriterionResult(
                value=vendor.__getattribute__(crit_name),
                score=score,
                weight=weight,
                weighted_score=weighted_score,
            )

        final_score = round(total_weighted_score, 2)
        tier = self._determine_tier(final_score)
        depth = self._determine_depth(tier)
        reasoning = self._generate_reasoning(criteria_results)

        logger.info(
            "Vendor %s (%s) assessed as %s with score %.2f -> %s",
            vendor.vendor_id,
            vendor.vendor_name,
            tier.value,
            final_score,
            depth.value,
        )

        return CriticalityAssessment(
            vendor_id=vendor.vendor_id,
            vendor_name=vendor.vendor_name,
            domain=vendor.domain,
            criticality_tier=tier,
            criticality_score=final_score,
            assessment_depth=depth,
            criteria=criteria_results,
            reasoning=reasoning,
        )

    def _determine_tier(self, score: float) -> CriticalityTier:
        """Determine criticality tier based on configured thresholds sorted by min_score descending."""
        sorted_thresholds = sorted(
            self.config.tier_thresholds.items(),
            key=lambda item: item[1].min_score,
            reverse=True,
        )

        for tier_name, threshold in sorted_thresholds:
            if score >= threshold.min_score:
                try:
                    return CriticalityTier(tier_name)
                except ValueError:
                    return CriticalityTier[tier_name]

        return CriticalityTier.TIER_3

    def _determine_depth(self, tier: CriticalityTier) -> AssessmentDepth:
        """Map criticality tier to assessment depth from configuration."""
        tier_key = tier.value
        depth_cfg = self.config.assessment_depth_mapping.get(tier_key)
        if depth_cfg:
            return AssessmentDepth(depth_cfg.depth)
        
        # Fallback defaults
        match tier:
            case CriticalityTier.TIER_1:
                return AssessmentDepth.COMPREHENSIVE
            case CriticalityTier.TIER_2:
                return AssessmentDepth.TARGETED
            case _:
                return AssessmentDepth.LIGHTWEIGHT

    def _generate_reasoning(self, criteria_results: Dict[str, CriterionResult]) -> List[str]:
        """Generate human-readable explainability reasoning based on configured threshold."""
        reasoning: List[str] = []
        threshold = self.config.reasoning_rules.threshold_score_for_mention
        descriptions = self.config.reasoning_rules.descriptions

        for crit_name, result in criteria_results.items():
            if result.score >= threshold:
                desc = descriptions.get(crit_name, f"High rating ({result.value.value}) for {crit_name}")
                reasoning.append(f"{desc} (Rating: {result.value.value.upper()}, Score: {result.score})")

        if not reasoning:
            reasoning.append("Standard baseline exposure across all operational and data access dimensions.")

        return reasoning
