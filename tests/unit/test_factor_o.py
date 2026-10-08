"""Unit tests for Factor O (Operational Dependency)."""
import pytest
from meridian_assessment.services.factors.o_classifier import classify_operational_dependency
from meridian_assessment.models.factor_result import DeterminationMethod


class TestFactorO:
    def test_low(self):
        res = classify_operational_dependency("Low")
        assert res.score == 0
        assert res.determination_method == DeterminationMethod.DIRECT

    def test_moderate(self):
        res = classify_operational_dependency("Moderate")
        assert res.score == 1
        assert res.determination_method == DeterminationMethod.DIRECT

    def test_high(self):
        res = classify_operational_dependency("High")
        assert res.score == 2
        assert res.determination_method == DeterminationMethod.DIRECT

    def test_critical(self):
        res = classify_operational_dependency("Critical")
        assert res.score == 3
        assert res.determination_method == DeterminationMethod.DIRECT

    def test_case_insensitive_and_whitespace(self):
        assert classify_operational_dependency("moderate").score == 1
        assert classify_operational_dependency("MODERATE").score == 1
        assert classify_operational_dependency("  Moderate  ").score == 1

    def test_unrecognized_returns_unknown(self):
        res = classify_operational_dependency("Extremely High")
        assert res.is_unknown is True
        assert res.score is None
        assert res.determination_method == DeterminationMethod.UNKNOWN
        # Must not silently become 0
        assert res.score != 0
