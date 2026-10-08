"""Ingestion module for loading vendor datasets."""

from meridian_assessment.ingestion.base import BaseVendorLoader
from meridian_assessment.ingestion.csv_loader import CSVVendorLoader

__all__ = ["BaseVendorLoader", "CSVVendorLoader"]
