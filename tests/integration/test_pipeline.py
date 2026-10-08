"""Integration tests for the complete vendor assessment pipeline."""

import json
from pathlib import Path
import pytest

from meridian_assessment.services.pipeline import AssessmentPipeline


class TestPipelineIntegration:
    """Test suite for end-to-end pipeline execution."""

    def test_pipeline_run_from_sample_csv(self, tmp_path: Path) -> None:
        csv_file = Path("data/sample/sample_vendors.csv")
        assert csv_file.exists(), "Sample CSV file should exist"

        pipeline = AssessmentPipeline(config_path="config/criticality.yaml")
        results = pipeline.run_from_source(csv_file)

        assert len(results) == 4
        # Check that all results have valid tiers, depths, and criteria breakdown
        for res in results:
            assert res.vendor_id in {"V001", "V002", "V003", "V004"}
            assert res.criticality_tier in {"TIER_1", "TIER_2", "TIER_3"}
            assert res.assessment_depth in {"COMPREHENSIVE", "TARGETED", "LIGHTWEIGHT"}
            assert len(res.criteria) == 5

        # Test JSON export
        output_file = tmp_path / "output_results.json"
        exported_path = pipeline.export_results_json(results, output_file)

        assert exported_path.exists()
        with open(exported_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 4
        assert data[0]["vendor_id"] == "V001"
        assert "criticality_score" in data[0]
        assert "criteria" in data[0]
        assert "reasoning" in data[0]
