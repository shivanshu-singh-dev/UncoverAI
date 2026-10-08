"""Unit tests for Factor P (Payment Flow Exposure)."""
import pytest
from meridian_assessment.services.factors.p_classifier import classify_payment_flow


class TestFactorP:
    def test_no_payment_involvement(self):
        res = classify_payment_flow(
            service_text="Automated batch job scheduling and execution",
            business_process_text="IT operations / batch processing workflow",
        )
        assert res.score == 0

    def test_payment_reporting_statements_is_p1(self):
        res = classify_payment_flow(
            service_text="Account statement production and report generation",
            business_process_text="Customer communications",
        )
        assert res.score == 1

    def test_payment_analytics_is_p1(self):
        res = classify_payment_flow(
            service_text="Payment analytics and performance metrics",
            business_process_text="Treasury intelligence",
        )
        assert res.score == 1

    def test_material_transaction_processing_is_p2(self):
        res = classify_payment_flow(
            service_text="Core banking transaction processing and account balance management",
            business_process_text="Core account operations",
        )
        assert res.score == 2

    def test_payment_reconciliation_is_p2(self):
        res = classify_payment_flow(
            service_text="Payment reconciliation and ledger balance matching",
            business_process_text="Finance and settlement reconciliation",
        )
        assert res.score == 2

    def test_rtp_transmission_is_p3(self):
        res = classify_payment_flow(
            service_text="RTP payment file transmission to Federal Reserve and counterparties",
            business_process_text="Payment transmission — RTP real-time payments",
        )
        assert res.score == 3

    def test_clearing_and_settlement_is_p3(self):
        res = classify_payment_flow(
            service_text="ACH and RTP payment clearing and settlement",
            business_process_text="Payment clearing and settlement — ACH and RTP",
        )
        assert res.score == 3

    def test_negation_does_not_process_payments(self):
        res = classify_payment_flow(
            service_text="Cloud infrastructure provider that does not process payments or handle cardholder data",
            business_process_text="General IT hosting",
        )
        # Negation pattern must NOT allow P2/P3
        assert res.score == 0

    def test_category_alone_does_not_dictate_p3(self):
        # Category says Payments, but service is only analytics
        res = classify_payment_flow(
            service_text="Payment reporting and analytics dashboard",
            business_process_text="Management reporting",
            category="Payments",
        )
        assert res.score == 1
        assert res.score != 3
