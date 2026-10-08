"""Unit tests for Vendor and Assessment models."""

import pytest
from pydantic import ValidationError

from meridian_assessment.models.assessment import AssessmentDepth, AssessmentScope
from meridian_assessment.models.criticality import (
    CriticalityAssessment,
    CriticalityTier,
    CriterionResult,
)
from meridian_assessment.models.vendor import SeverityLevel, Vendor


class TestVendorModel:
    """Test suite for Vendor model validation and normalization."""

    def test_valid_vendor_creation(self) -> None:
        vendor = Vendor(
            vendor_id="V001",
            vendor_name="Example Corp",
            domain="example.com",
            data_sensitivity=SeverityLevel.HIGH,
            payment_involvement=SeverityLevel.MEDIUM,
            regulatory_exposure=SeverityLevel.HIGH,
            operational_dependency=SeverityLevel.HIGH,
            customer_data_volume=SeverityLevel.HIGH,
        )
        assert vendor.vendor_id == "V001"
        assert vendor.vendor_name == "Example Corp"
        assert vendor.domain == "example.com"
        assert vendor.payment_flows == SeverityLevel.MEDIUM

    def test_case_insensitive_severity_parsing(self) -> None:
        vendor = Vendor(
            vendor_id="V002",
            vendor_name="Example Ltd",
            domain="https://example.org/path",
            data_sensitivity="HIGH",  # uppercase string
            payment_flows="NONE",
            regulatory_exposure="Low",
            operational_dependency="critical",
            customer_data_volume="Medium",
        )
        assert vendor.domain == "example.org"
        assert vendor.data_sensitivity == SeverityLevel.HIGH
        assert vendor.payment_flows == SeverityLevel.NONE
        assert vendor.regulatory_exposure == SeverityLevel.LOW
        assert vendor.operational_dependency == SeverityLevel.CRITICAL
        assert vendor.customer_data_volume == SeverityLevel.MEDIUM

    def test_invalid_domain_format(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Vendor(
                vendor_id="V003",
                vendor_name="Bad Domain Corp",
                domain="nodotdomain",
                data_sensitivity=SeverityLevel.LOW,
                payment_flows=SeverityLevel.LOW,
                regulatory_exposure=SeverityLevel.LOW,
                operational_dependency=SeverityLevel.LOW,
                customer_data_volume=SeverityLevel.LOW,
            )
        assert "Invalid domain format" in str(exc_info.value)

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Vendor.model_validate({
                "vendor_id": "V004",
                "vendor_name": "Incomplete Vendor",
                # missing domain and criteria
            })

    def test_blank_vendor_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Vendor(
                vendor_id="   ",
                vendor_name="Blank ID Corp",
                domain="blank.com",
                data_sensitivity=SeverityLevel.LOW,
                payment_flows=SeverityLevel.LOW,
                regulatory_exposure=SeverityLevel.LOW,
                operational_dependency=SeverityLevel.LOW,
                customer_data_volume=SeverityLevel.LOW,
            )


class TestCriticalityModels:
    """Test suite for Criticality and Assessment models."""

    def test_criterion_result_immutability(self) -> None:
        res = CriterionResult(
            value=SeverityLevel.HIGH,
            score=4.0,
            weight=0.25,
            weighted_score=1.0,
        )
        assert res.score == 4.0
        with pytest.raises(ValidationError):
            res.score = 5.0  # type: ignore

    def test_criticality_assessment_serialization(self) -> None:
        assessment = CriticalityAssessment(
            vendor_id="V001",
            vendor_name="Example Corp",
            domain="example.com",
            criticality_tier=CriticalityTier.TIER_1,
            criticality_score=4.2,
            assessment_depth=AssessmentDepth.COMPREHENSIVE,
            criteria={
                "data_sensitivity": CriterionResult(
                    value=SeverityLevel.HIGH,
                    score=4.0,
                    weight=0.25,
                    weighted_score=1.0,
                )
            },
            reasoning=["Processes highly sensitive customer data"],
        )
        data = assessment.model_dump()
        assert data["criticality_tier"] == "TIER_1"
        assert data["assessment_depth"] == "COMPREHENSIVE"
        assert data["criticality_score"] == 4.2
