"""Unit tests for OSINT future interface and placeholder."""

import pytest

from meridian_assessment.models.assessment import AssessmentDepth, AssessmentScope
from meridian_assessment.models.criticality import CriticalityTier
from meridian_assessment.models.vendor import SeverityLevel, Vendor
from meridian_assessment.osint.interface import (
    BaseOSINTAssessment,
    OSINTAssessmentPlaceholder,
)


class TestOSINTInterface:
    """Test suite ensuring OSINT interface adheres to contracts."""

    @pytest.fixture
    def sample_vendor(self) -> Vendor:
        return Vendor(
            vendor_id="V_TEST",
            vendor_name="Test Vendor Corp",
            domain="testvendor.com",
            data_sensitivity=SeverityLevel.HIGH,
            payment_flows=SeverityLevel.HIGH,
            regulatory_exposure=SeverityLevel.HIGH,
            operational_dependency=SeverityLevel.HIGH,
            customer_data_volume=SeverityLevel.HIGH,
        )

    @pytest.fixture
    def sample_scope(self) -> AssessmentScope:
        return AssessmentScope(
            assessment_depth=AssessmentDepth.COMPREHENSIVE,
            description="Comprehensive assessment scope",
            focus_areas=["Technical footprint", "Dark web"],
        )

    def test_abstract_base_class_raises_not_implemented(
        self, sample_vendor: Vendor, sample_scope: AssessmentScope
    ) -> None:
        class UnimplementedOSINT(BaseOSINTAssessment):
            pass

        # Subclass without implementing assess cannot be instantiated
        with pytest.raises(TypeError) as exc_info:
            UnimplementedOSINT()  # type: ignore
        assert "Can't instantiate abstract class" in str(exc_info.value)

        class CallingSuperOSINT(BaseOSINTAssessment):
            def assess(self, vendor, tier, scope):
                return super().assess(vendor, tier, scope)

        instance = CallingSuperOSINT()
        with pytest.raises(NotImplementedError) as exc_info:
            instance.assess(sample_vendor, CriticalityTier.TIER_1, sample_scope)
        assert "not yet implemented" in str(exc_info.value).lower()

    def test_placeholder_returns_structured_pending_record(
        self, sample_vendor: Vendor, sample_scope: AssessmentScope
    ) -> None:
        placeholder = OSINTAssessmentPlaceholder()
        finding = placeholder.assess(sample_vendor, CriticalityTier.TIER_1, sample_scope)
        assert finding.vendor_id == "V_TEST"
        assert finding.status == "PENDING"
        assert "pending" in finding.message.lower()
