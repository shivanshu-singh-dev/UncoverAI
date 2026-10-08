"""Unit tests for Factor D (Data Sensitivity)."""
import pytest
from meridian_assessment.services.factors.d_classifier import classify_data_sensitivity
from meridian_assessment.models.factor_result import DeterminationMethod


class TestFactorD:
    def test_no_data(self):
        res = classify_data_sensitivity("No direct customer data.")
        assert res.score == 0
        assert res.determination_method == DeterminationMethod.DETERMINISTIC_RULE

    def test_internal_operational_data(self):
        res = classify_data_sensitivity("Internal operational records and system metrics.")
        assert res.score == 1

    def test_pii(self):
        res = classify_data_sensitivity("Customer names, addresses, and phone numbers.")
        assert res.score == 2

    def test_financial_records(self):
        res = classify_data_sensitivity("Account balances and transaction records.")
        assert res.score == 2

    def test_credentials(self):
        res = classify_data_sensitivity("Production service credentials and system access tokens.")
        assert res.score == 3
        assert "credentials" in res.matched_concepts

    def test_customer_master(self):
        res = classify_data_sensitivity("Entire customer base including names, addresses, account numbers, tax identifiers (SSNs)")
        assert res.score == 3

    def test_mixed_takes_highest(self):
        # Mixed: names (2), account numbers (2), and service credentials (3)
        res = classify_data_sensitivity("Customer names, account numbers, and production service credentials.")
        assert res.score == 3  # Must be 3 (highest), NOT an average

    def test_payment_instructions(self):
        res = classify_data_sensitivity("Payment instructions and account identifiers for RTP transactions.")
        assert res.score == 3

    def test_portfolio_holdings(self):
        res = classify_data_sensitivity("Wealth client portfolio holdings, advisor compensation data, revenue figures.")
        assert res.score == 2
