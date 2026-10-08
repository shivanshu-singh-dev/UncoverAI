"""Loader for Meridian vendor dataset CSV format."""
import csv
from pathlib import Path
from typing import Union

from meridian_assessment.ingestion.base import BaseVendorLoader
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.utils.exceptions import IngestionError


class MeridianCSVVendorLoader(BaseVendorLoader):
    """Loads and validates vendor records from the Meridian case study CSV file."""

    REQUIRED_COLUMNS = {"vendor_id", "vendor_name"}

    def load(self, source: Union[str, Path]) -> list[MeridianVendor]:
        """Load, validate, and return all vendors from a Meridian format CSV file."""
        path = Path(source)
        if not path.is_file():
            raise IngestionError(f"Meridian vendor CSV file not found: {path.resolve()}")

        vendors: list[MeridianVendor] = []
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                if reader.fieldnames is None:
                    raise IngestionError(f"Vendor CSV file '{path}' is empty or invalid.")

                # Normalize field headers (lower, strip)
                raw_fields = [f.strip() for f in reader.fieldnames if f]
                norm_map = {f.lower().replace(" ", "_"): f for f in raw_fields}

                missing = [c for c in self.REQUIRED_COLUMNS if c not in norm_map]
                if missing:
                    raise IngestionError(f"CSV '{path}' is missing required columns: {missing}")

                for row_idx, row in enumerate(reader, start=2):
                    cleaned = {}
                    for k, v in row.items():
                        if k:
                            norm_k = k.strip().lower().replace(" ", "_")
                            cleaned[norm_k] = v.strip() if v else None

                    # Filter out completely blank lines
                    if not any(cleaned.values()):
                        continue

                    vendor = MeridianVendor.model_validate(cleaned)
                    vendors.append(vendor)

            return vendors
        except Exception as e:
            if isinstance(e, IngestionError):
                raise
            raise IngestionError(f"Failed to load Meridian vendor CSV '{path}': {e}") from e
