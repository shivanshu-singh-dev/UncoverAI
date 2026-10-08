"""Unit tests for Factor V (Annual Data Volume)."""
import pytest
from meridian_assessment.services.factors.v_classifier import classify_volume


class TestFactorV:
    def test_na_is_zero(self):
        res = classify_volume("N/A")
        assert res.score == 0
        assert res.volume_type == "NONE"

    def test_small_volume_under_one_million(self):
        res = classify_volume("500,000 messages")
        assert res.score == 1
        assert res.normalized_volume == 500000

    def test_material_volume_one_to_ten_million(self):
        res = classify_volume("2.1 million messages annually")
        assert res.score == 2
        assert res.normalized_volume == 2100000

    def test_enterprise_volume_messages(self):
        res = classify_volume("14 million messages annually")
        assert res.score == 3
        assert res.normalized_volume == 14000000

    def test_enterprise_volume_documents(self):
        res = classify_volume("19 million documents annually")
        assert res.score == 3
        assert res.normalized_volume == 19000000

    def test_enterprise_volume_transactions(self):
        res = classify_volume("240 million transactions annually")
        assert res.score == 3
        assert res.normalized_volume == 240000000

    def test_frequency_indicator_is_unknown(self):
        # "12 monthly cycles" must NOT be treated as count = 12
        res = classify_volume("12 monthly reporting cycles")
        assert res.score is None
        assert res.is_unknown is True
        assert res.volume_type == "FREQUENCY"

    def test_population_scale(self):
        res = classify_volume("38,000 wealth clients")
        assert res.volume_type == "POPULATION"
        assert res.is_unknown is True
        assert res.normalized_volume == 38000
