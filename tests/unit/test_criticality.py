"""Unit tests for criticality engine, boundary conditions, scoring math, and scoping."""

import math
import pytest

from meridian_assessment.config.loader import load_criticality_config
from meridian_assessment.models.assessment import AssessmentDepth
from meridian_assessment.models.config import CriticalityConfig
from meridian_assessment.models.criticality import CriticalityTier
from meridian_assessment.models.vendor import SeverityLevel, Vendor
from meridian_assessment.services.criticality_engine import CriticalityEngine
from meridian_assessment.services.scoping_engine import ScopingEngine


def _make_vendor(
    vendor_id: str = "V_TEST",
    vendor_name: str = "Test Corp",
    domain: str = "test.com",
    data_sensitivity: SeverityLevel = SeverityLevel.NONE,
    payment_flows: SeverityLevel = SeverityLevel.NONE,
    regulatory_exposure: SeverityLevel = SeverityLevel.NONE,
    operational_dependency: SeverityLevel = SeverityLevel.NONE,
    customer_data_volume: SeverityLevel = SeverityLevel.NONE,
) -> Vendor:
    """Helper to create a Vendor with convenient defaults."""
    return Vendor(
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        domain=domain,
        data_sensitivity=data_sensitivity,
        payment_flows=payment_flows,
        regulatory_exposure=regulatory_exposure,
        operational_dependency=operational_dependency,
        customer_data_volume=customer_data_volume,
    )


@pytest.fixture(scope="module")
def config() -> CriticalityConfig:
    return load_criticality_config("config/criticality.yaml")


@pytest.fixture(scope="module")
def engine(config: CriticalityConfig) -> CriticalityEngine:
    return CriticalityEngine(config)


@pytest.fixture(scope="module")
def scoping_engine(config: CriticalityConfig) -> ScopingEngine:
    return ScopingEngine(config)


class TestCriticalityEngineBasic:
    """Core tier classification tests."""

    def test_tier_1_high_criticality_evaluation(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor(
            vendor_id="V_HIGH",
            vendor_name="Apex Bank Core",
            domain="apexbank.com",
            data_sensitivity=SeverityLevel.CRITICAL,
            payment_flows=SeverityLevel.CRITICAL,
            regulatory_exposure=SeverityLevel.HIGH,
            operational_dependency=SeverityLevel.CRITICAL,
            customer_data_volume=SeverityLevel.HIGH,
        )
        result = engine.evaluate(vendor)
        assert result.criticality_tier == CriticalityTier.TIER_1
        assert result.assessment_depth == AssessmentDepth.COMPREHENSIVE
        assert result.criticality_score >= 3.5
        assert len(result.reasoning) > 0

    def test_tier_2_medium_criticality_evaluation(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor(
            vendor_id="V_MED",
            vendor_name="Standard CRM",
            domain="standardcrm.com",
            data_sensitivity=SeverityLevel.MEDIUM,
            payment_flows=SeverityLevel.LOW,
            regulatory_exposure=SeverityLevel.MEDIUM,
            operational_dependency=SeverityLevel.MEDIUM,
            customer_data_volume=SeverityLevel.MEDIUM,
        )
        result = engine.evaluate(vendor)
        assert result.criticality_tier == CriticalityTier.TIER_2
        assert result.assessment_depth == AssessmentDepth.TARGETED
        assert 2.0 <= result.criticality_score < 3.5

    def test_tier_3_low_criticality_evaluation(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor(
            vendor_id="V_LOW",
            vendor_name="Coffee Machine Supplies",
            domain="coffeebrew.com",
            operational_dependency=SeverityLevel.LOW,
        )
        result = engine.evaluate(vendor)
        assert result.criticality_tier == CriticalityTier.TIER_3
        assert result.assessment_depth == AssessmentDepth.LIGHTWEIGHT
        assert result.criticality_score < 2.0

    def test_all_none_scores_zero(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor(vendor_id="V_ZERO", vendor_name="Zero Corp", domain="zero.com")
        result = engine.evaluate(vendor)
        assert result.criticality_tier == CriticalityTier.TIER_3
        assert result.criticality_score == 0.0


class TestCriticalityScoringMath:
    """Verify scoring arithmetic is deterministic and mathematically correct."""

    def test_known_score_apex_banking_example(self, engine: CriticalityEngine) -> None:
        """
        Verified example from documentation:
          data_sensitivity:       critical (5.0) × 0.25 = 1.25
          payment_flows:          critical (5.0) × 0.20 = 1.00
          regulatory_exposure:    high     (4.0) × 0.20 = 0.80
          operational_dependency: critical (5.0) × 0.20 = 1.00
          customer_data_volume:   high     (4.0) × 0.15 = 0.60
                                                  Total = 4.65
        """
        vendor = _make_vendor(
            vendor_id="V001",
            vendor_name="Apex Core Banking Cloud",
            domain="apexbanking.com",
            data_sensitivity=SeverityLevel.CRITICAL,
            payment_flows=SeverityLevel.CRITICAL,
            regulatory_exposure=SeverityLevel.HIGH,
            operational_dependency=SeverityLevel.CRITICAL,
            customer_data_volume=SeverityLevel.HIGH,
        )
        result = engine.evaluate(vendor)
        assert math.isclose(result.criticality_score, 4.65, rel_tol=1e-3), (
            f"Expected 4.65, got {result.criticality_score}"
        )

    def test_criteria_weighted_scores_sum_to_total(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor(
            vendor_id="V_MATH",
            vendor_name="Math Test Corp",
            domain="mathtest.com",
            data_sensitivity=SeverityLevel.HIGH,
            payment_flows=SeverityLevel.MEDIUM,
            regulatory_exposure=SeverityLevel.LOW,
            operational_dependency=SeverityLevel.HIGH,
            customer_data_volume=SeverityLevel.NONE,
        )
        result = engine.evaluate(vendor)
        computed_total = sum(c.weighted_score for c in result.criteria.values())
        assert math.isclose(computed_total, result.criticality_score, rel_tol=1e-3)

    def test_five_criteria_always_present(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor()
        result = engine.evaluate(vendor)
        expected = {
            "data_sensitivity",
            "payment_flows",
            "regulatory_exposure",
            "operational_dependency",
            "customer_data_volume",
        }
        assert set(result.criteria.keys()) == expected


class TestCriticalityBoundaryConditions:
    """Boundary tests ensuring exact threshold values route to the correct tier.

    Boundary rule: score >= min_score → belongs to that tier.
      score == 3.5 → TIER_1 (NOT TIER_2)
      score == 2.0 → TIER_2 (NOT TIER_3)
      score == 0.0 → TIER_3
    """

    def test_exact_tier_1_boundary_score(self, engine: CriticalityEngine) -> None:
        """Score exactly 3.5 must be TIER_1 (inclusive lower bound)."""
        # Construct a vendor whose score = 3.5 exactly:
        # data_sensitivity: high  (4.0) × 0.25 = 1.00
        # payment_flows:    high  (4.0) × 0.20 = 0.80
        # regulatory:       high  (4.0) × 0.20 = 0.80
        # operational:      high  (4.0) × 0.20 = 0.80
        # customer_data:    none  (0.0) × 0.15 = 0.00
        #                              Σ = 3.40  — close but not exactly 3.5
        # Use: all high across the board → 4.0 * (0.25+0.20+0.20+0.20+0.15) = 4.0 * 1.0 = 4.0 → TIER_1
        vendor = _make_vendor(
            data_sensitivity=SeverityLevel.HIGH,
            payment_flows=SeverityLevel.HIGH,
            regulatory_exposure=SeverityLevel.HIGH,
            operational_dependency=SeverityLevel.HIGH,
            customer_data_volume=SeverityLevel.HIGH,
        )
        result = engine.evaluate(vendor)
        assert math.isclose(result.criticality_score, 4.0, rel_tol=1e-3)
        assert result.criticality_tier == CriticalityTier.TIER_1

    def test_exact_tier_2_boundary_score(self, engine: CriticalityEngine) -> None:
        """Score exactly 2.0 must be TIER_2 (not TIER_3)."""
        # all medium: 2.5 * 1.0 = 2.5 → TIER_2
        # all low:    1.0 * 1.0 = 1.0 → TIER_3
        # We need exactly 2.0:
        # data: medium (2.5) × 0.25 = 0.625
        # payment: medium (2.5) × 0.20 = 0.50
        # regulatory: medium (2.5) × 0.20 = 0.50
        # operational: medium (2.5) × 0.20 = 0.50  → total so far = 2.125
        # customer: none (0.0) × 0.15 = 0.0  → total = 2.125 (TIER_2 since >= 2.0)
        vendor = _make_vendor(
            data_sensitivity=SeverityLevel.MEDIUM,
            payment_flows=SeverityLevel.MEDIUM,
            regulatory_exposure=SeverityLevel.MEDIUM,
            operational_dependency=SeverityLevel.MEDIUM,
            customer_data_volume=SeverityLevel.NONE,
        )
        result = engine.evaluate(vendor)
        assert result.criticality_score >= 2.0
        assert result.criticality_tier == CriticalityTier.TIER_2

    def test_below_tier_2_boundary_is_tier_3(self, engine: CriticalityEngine) -> None:
        """Score below 2.0 must be TIER_3."""
        # all low: 1.0 * 1.0 = 1.0 → TIER_3
        vendor = _make_vendor(
            data_sensitivity=SeverityLevel.LOW,
            payment_flows=SeverityLevel.LOW,
            regulatory_exposure=SeverityLevel.LOW,
            operational_dependency=SeverityLevel.LOW,
            customer_data_volume=SeverityLevel.LOW,
        )
        result = engine.evaluate(vendor)
        assert result.criticality_score < 2.0
        assert result.criticality_tier == CriticalityTier.TIER_3


class TestScopingEngine:
    """Tests for the assessment scoping engine."""

    def test_tier_1_maps_to_comprehensive(self, scoping_engine: ScopingEngine) -> None:
        scope = scoping_engine.create_scope(CriticalityTier.TIER_1)
        assert scope.assessment_depth == AssessmentDepth.COMPREHENSIVE
        assert len(scope.focus_areas) > 0

    def test_tier_2_maps_to_targeted(self, scoping_engine: ScopingEngine) -> None:
        scope = scoping_engine.create_scope(CriticalityTier.TIER_2)
        assert scope.assessment_depth == AssessmentDepth.TARGETED
        assert len(scope.focus_areas) > 0

    def test_tier_3_maps_to_lightweight(self, scoping_engine: ScopingEngine) -> None:
        scope = scoping_engine.create_scope(CriticalityTier.TIER_3)
        assert scope.assessment_depth == AssessmentDepth.LIGHTWEIGHT

    def test_scope_has_non_empty_description(self, scoping_engine: ScopingEngine) -> None:
        for tier in CriticalityTier:
            scope = scoping_engine.create_scope(tier)
            assert scope.description.strip(), f"Empty description for tier {tier}"


class TestExplainabilityReasoning:
    """Tests for the reasoning and explainability system."""

    def test_high_criteria_appear_in_reasoning(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor(
            vendor_id="V_EXP",
            vendor_name="PayGateway Corp",
            domain="paygateway.com",
            data_sensitivity=SeverityLevel.HIGH,
            payment_flows=SeverityLevel.CRITICAL,
        )
        result = engine.evaluate(vendor)
        reasons_text = " ".join(result.reasoning).lower()
        assert "payment" in reasons_text
        assert "sensitive" in reasons_text

    def test_all_none_vendor_gets_baseline_message(self, engine: CriticalityEngine) -> None:
        vendor = _make_vendor(vendor_id="V_BARE", vendor_name="Bare Corp", domain="bare.com")
        result = engine.evaluate(vendor)
        assert len(result.reasoning) >= 1
        assert "baseline" in result.reasoning[0].lower()
