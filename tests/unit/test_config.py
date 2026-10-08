"""Unit tests for configuration loader and models."""

import pytest
from pathlib import Path
import tempfile
import yaml

from meridian_assessment.config.loader import load_criticality_config
from meridian_assessment.models.config import CriticalityConfig
from meridian_assessment.utils.exceptions import ConfigurationError


class TestConfigLoader:
    """Test suite for configuration loading and validation."""

    def test_load_default_config(self) -> None:
        config = load_criticality_config("config/criticality.yaml")
        assert isinstance(config, CriticalityConfig)
        assert "data_sensitivity" in config.criterion_weights
        assert "TIER_1" in config.tier_thresholds
        assert "TIER_1" in config.assessment_depth_mapping
        assert sum(config.criterion_weights.values()) == pytest.approx(1.0)

    def test_missing_config_file_raises_error(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            load_criticality_config("config/non_existent_config.yaml")
        assert "Configuration file not found" in str(exc_info.value)

    def test_malformed_yaml_raises_error(self, tmp_path: Path) -> None:
        bad_yaml_file = tmp_path / "bad.yaml"
        bad_yaml_file.write_text("criterion_weights: [unbalanced: yaml: {", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            load_criticality_config(bad_yaml_file)
        assert "Failed to parse YAML" in str(exc_info.value)

    def test_invalid_weights_sum_raises_error(self, tmp_path: Path) -> None:
        invalid_data = {
            "version": "1.0",
            "criterion_weights": {
                "data_sensitivity": 0.5,
                "payment_flows": 0.5,
                "regulatory_exposure": 0.5,  # Sum = 1.8 != 1.0
                "operational_dependency": 0.2,
                "customer_data_volume": 0.1,
            },
            "severity_scores": {"high": 4.0, "medium": 2.5, "low": 1.0, "none": 0.0},
            "tier_thresholds": {
                "TIER_1": {"min_score": 3.5, "label": "T1", "description": "T1 desc"}
            },
            "assessment_depth_mapping": {
                "TIER_1": {"depth": "COMPREHENSIVE", "description": "Comp"}
            },
        }
        cfg_file = tmp_path / "invalid_weights.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_data, f)

        with pytest.raises(ConfigurationError) as exc_info:
            load_criticality_config(cfg_file)
        assert "weights must sum to 1.0" in str(exc_info.value)

    def test_missing_required_criterion_weight(self, tmp_path: Path) -> None:
        invalid_data = {
            "version": "1.0",
            "criterion_weights": {
                "data_sensitivity": 0.5,
                "payment_flows": 0.5,
                # Missing regulatory_exposure, operational_dependency, customer_data_volume
            },
            "severity_scores": {"high": 4.0},
            "tier_thresholds": {},
            "assessment_depth_mapping": {},
        }
        cfg_file = tmp_path / "missing_criteria.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(invalid_data, f)

        with pytest.raises(ConfigurationError) as exc_info:
            load_criticality_config(cfg_file)
        assert "Missing required criteria weights" in str(exc_info.value)
