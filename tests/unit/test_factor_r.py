"""Unit tests for Factor R (Regulatory Exposure)."""
import pytest
from meridian_assessment.services.factors.r_classifier import classify_regulatory_exposure


class TestFactorR:
    def test_no_regulatory_dependency(self):
        res = classify_regulatory_exposure(
            service_text="Automated batch job scheduling and execution",
            business_process_text="IT operations / batch processing workflow",
        )
        assert res.score == 0

    def test_indirect_compliance_or_reporting_is_r1(self):
        res = classify_regulatory_exposure(
            service_text="Internal performance reporting and analytics dashboard",
            business_process_text="Business intelligence",
        )
        assert res.score == 1

    def test_regulatory_notices_is_r2(self):
        res = classify_regulatory_exposure(
            service_text="Account statement production and regulatory notice generation",
            business_process_text="Customer communications, regulatory notice fulfilment",
        )
        assert res.score == 2
        assert "regulatory_notices" in res.matched_concepts

    def test_regulatory_reporting_is_r2(self):
        res = classify_regulatory_exposure(
            service_text="Required regulatory reporting and compliance filings platform",
            business_process_text="Regulatory compliance",
        )
        assert res.score == 2

    def test_statements_alone_is_not_r2(self):
        # Account statements alone without regulatory label is not R2
        res = classify_regulatory_exposure(
            service_text="Customer account statement printing and mailing",
            business_process_text="Customer communication",
        )
        assert res.score != 2

    def test_critical_financial_infrastructure_is_r3(self):
        res = classify_regulatory_exposure(
            service_text="Payment network infrastructure services — designated financial market utility",
            business_process_text="Critical payment network clearing and settlement",
        )
        assert res.score == 3

    def test_generic_compliance_description_does_not_make_r3(self):
        # Terrapin case: vendor description says "compliance solutions" but service is portfolio data
        res = classify_regulatory_exposure(
            service_text="Portfolio holdings data management, advisor commission calculations",
            business_process_text="Wealth management performance reporting and accounting",
            vendor_description="A financial technology company providing compliance solutions for wealth management firms.",
        )
        # Must NOT inherit generic compliance from vendor description to assign R3 or R2
        assert res.score <= 1
