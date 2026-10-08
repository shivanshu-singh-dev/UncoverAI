"""Unit tests for criticality engine and scoping."""

import pytest

from meridian_assessment.config.loader import load_criticality_config
from meridian_assessment.models.assessment import AssessmentDepth
from meridian_assessment.models.criticality import CriticalityTier
from meridian_assessment.models.vendor import SeverityLevel, Vendor
from meridian_assessment.services.criticality_engine import CriticalityEngine
from meridian_assessment.services.scoping_engine import ScopingEngine


class TestCriticalityEngine:
    """Test suite for deterministic criticality evaluation."""

    @pytest.fixture
    def config(self):
        return load_criticality_config("config/criticality.yaml")

    @pytest.fixture
    def engine(self, config):
        return CriticalityEngine(config)

    @pytest.fixture
    def scoping_engine(self, config):
        return ScopingEngine(config)

    def test_tier_1_high_criticality_evaluation(self, engine: CriticalityEngine) -> None:
        high_vendor = Vendor(
            vendor_id="V_HIGH",
            vendor_name="Apex Bank Core",
            domain="apexbank.com",
            data_sensitivity=SeverityLevel.CRITICAL,
            payment_flows=SeverityLevel.CRITICAL,
            regulatory_exposure=SeverityLevel.HIGH,
            operational_dependency=SeverityLevel.CRITICAL,
            customer_data_volume=SeverityLevel.HIGH,
        )
        result = engine.evaluate(high_vendor)
        assert result.criticality_tier == CriticalityTier.TIER_1
        assert result.assessment_depth == AssessmentDepth.COMPREHENSIVE
        assert result.criticality_score >= 3.5
        assert len(result.reasoning) > 0

    def test_tier_2_medium_criticality_evaluation(self, engine: CriticalityEngine) -> None:
        med_vendor = Vendor(
            vendor_id="V_MED",
            vendor_name="Standard CRM",
            domain="standardcrm.com",
            data_sensitivity=SeverityLevel.MEDIUM,
            payment_flows=SeverityLevel.LOW,
            regulatory_exposure=SeverityLevel.MEDIUM,
            operational_dependency=SeverityLevel.MEDIUM,
            customer_data_volume=SeverityLevel.MEDIUM,
        )
        result = engine.evaluate(med_vendor)
        assert result.criticality_tier == CriticalityTier.TIER_2
        assert result.assessment_depth == AssessmentDepth.TARGETED
        assert 2.0 <= result.criticality_score < 3.5

    def test_tier_3_low_criticality_evaluation(self, engine: CriticalityEngine) -> None:
        low_vendor = Vendor(
            vendor_id="V_LOW",
            vendor_name="Coffee Machine Supplies",
            domain="coffeebrew.com",
            data_sensitivity=SeverityLevel.NONE,
            payment_flows=SeverityLevel.NONE,
            regulatory_exposure=SeverityLevel.NONE,
            operational_dependency=SeverityLevel.LOW,
            customer_data_volume=SeverityLevel.NONE,
        )
        result = engine.evaluate(low_vendor)
        assert result.criticality_tier == CriticalityTier.TIER_3
        assert result.assessment_depth == AssessmentDepth.LIGHTWEIGHT
        assert result.criticality_score < 2.0

    def test_explainable_reasoning_generation(self, engine: CriticalityEngine) -> None:
        vendor = Vendor(
            vendor_id="V_EXP",
            vendor_name="PayGateway Corp",
            domain="paygateway.com",
            data_sensitivity=SeverityLevel.HIGH,
            payment_flows=SeverityLevel.CRITICAL,
            regulatory_exposure=SeverityLevel.NONE,
            operational_dependency=SeverityLevel.LOW,
            customer_data_volume=SeverityLevel.NONE,
        )
        result = engine.evaluate(vendor)
        # Should highlight high data sensitivity and critical payment flows
        reasons_text = " ".join(result.reasoning).lower()
        assert "payment" in reasons_text
        assert "sensitive" in reasons_text

    def test_scoping_engine_focus_areas(self, scoping_engine: ScopingEngine) -> None:
        scope_t1 = scoping_engine.create_scope(CriticalityTier.TIER_1)
        assert scope_t1.assessment_depth == AssessmentDepth.COMPREHENSIVE
        assert len(scope_t1.focus_areas) > 0

        scope_t3 = scoping_engine.create_scope(CriticalityTier.TIER_3)
        assert scope_t3.assessment_depth == AssessmentDepth.LIGHTWEIGHT
