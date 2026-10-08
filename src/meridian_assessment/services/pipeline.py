"""Assessment pipeline orchestrating ingestion, criticality calculation, and reporting."""

import json
from pathlib import Path
from typing import List, Optional, Union

from meridian_assessment.config.loader import load_criticality_config
from meridian_assessment.ingestion.base import BaseVendorLoader
from meridian_assessment.ingestion.csv_loader import CSVVendorLoader
from meridian_assessment.models.config import CriticalityConfig
from meridian_assessment.models.criticality import CriticalityAssessment
from meridian_assessment.models.vendor import Vendor
from meridian_assessment.services.criticality_engine import CriticalityEngine
from meridian_assessment.services.scoping_engine import ScopingEngine
from meridian_assessment.utils.logger import logger


class AssessmentPipeline:
    """Orchestrator for the vendor assessment foundation pipeline."""

    def __init__(
        self,
        config: Optional[CriticalityConfig] = None,
        config_path: Union[str, Path] = "config/criticality.yaml",
        loader: Optional[BaseVendorLoader] = None,
    ) -> None:
        """Initialize the assessment pipeline.

        Args:
            config: Optional pre-loaded CriticalityConfig.
            config_path: Path to configuration YAML if config is not provided.
            loader: Optional vendor loader (defaults to CSVVendorLoader).
        """
        self.config = config or load_criticality_config(config_path)
        self.loader = loader or CSVVendorLoader()
        self.criticality_engine = CriticalityEngine(self.config)
        self.scoping_engine = ScopingEngine(self.config)

    def run_from_source(self, source_path: Union[str, Path]) -> List[CriticalityAssessment]:
        """Run the assessment pipeline on a file source.

        Args:
            source_path: Path to input vendor data (e.g. CSV).

        Returns:
            List of CriticalityAssessment results.
        """
        logger.info("Starting vendor assessment pipeline for source: %s", source_path)
        vendors = self.loader.load(source_path)
        return self.run_on_vendors(vendors)

    def run_on_vendors(self, vendors: List[Vendor]) -> List[CriticalityAssessment]:
        """Run assessment directly on a collection of Vendor instances.

        Args:
            vendors: List of validated Vendor objects.

        Returns:
            List of CriticalityAssessment results.
        """
        logger.info("Processing %d vendor(s)...", len(vendors))
        results: List[CriticalityAssessment] = []

        for vendor in vendors:
            assessment = self.criticality_engine.evaluate(vendor)
            results.append(assessment)

        logger.info("Successfully assessed all %d vendor(s).", len(results))
        return results

    def export_results_json(
        self,
        assessments: List[CriticalityAssessment],
        output_path: Union[str, Path] = "data/output/criticality_results.json",
    ) -> Path:
        """Export assessment results to a formatted, machine-readable JSON file.

        Args:
            assessments: List of CriticalityAssessment instances.
            output_path: Target JSON output file path.

        Returns:
            Resolved Path of the written JSON file.
        """
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Serialize using Pydantic model_dump
        payload = [assessment.model_dump(mode="json") for assessment in assessments]

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Exported %d assessment result(s) to %s", len(payload), dest.resolve())
        return dest
