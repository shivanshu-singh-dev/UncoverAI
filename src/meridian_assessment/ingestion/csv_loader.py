"""CSV Vendor Loader with strict schema validation and duplicate checking."""

import csv
from pathlib import Path
from typing import Union
from pydantic import ValidationError

from meridian_assessment.ingestion.base import BaseVendorLoader
from meridian_assessment.models.vendor import Vendor
from meridian_assessment.utils.exceptions import IngestionError, VendorValidationError
from meridian_assessment.utils.logger import logger


class CSVVendorLoader(BaseVendorLoader):
    """Loads and validates vendor records from a CSV file."""

    REQUIRED_COLUMNS = {
        "vendor_id",
        "vendor_name",
        "domain",
        "data_sensitivity",
        "payment_flows",
        "regulatory_exposure",
        "operational_dependency",
        "customer_data_volume",
    }

    def load(self, source: Union[str, Path]) -> list[Vendor]:
        """Load, validate, and return all vendors from a CSV file.

        Args:
            source: Path to the CSV file.

        Returns:
            List of validated Vendor instances.

        Raises:
            IngestionError: If file is missing, empty, or has missing headers.
            VendorValidationError: If row validation fails or duplicate IDs are detected.
        """
        path = Path(source)

        if not path.is_file():
            raise IngestionError(f"Input vendor CSV file not found: {path.resolve()}")

        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                if reader.fieldnames is None:
                    raise IngestionError(f"Vendor CSV file '{path}' is empty or invalid.")

                # Normalize column headers (strip whitespace and lower)
                raw_fieldnames = [f.strip() for f in reader.fieldnames if f]
                normalized_fields = {f.lower(): f for f in raw_fieldnames}

                # Check for required base columns
                missing_cols = [
                    col for col in self.REQUIRED_COLUMNS if col not in normalized_fields
                ]

                if missing_cols:
                    raise IngestionError(
                        f"CSV file '{path}' is missing required columns: {sorted(missing_cols)}"
                    )

                vendors: list[Vendor] = []
                seen_vendor_ids: set[str] = set()
                errors: list[str] = []

                for row_idx, row in enumerate(reader, start=2):  # 1-indexed, header is line 1
                    # Clean and normalize row dict keys and values
                    cleaned_row = {
                        k.strip().lower(): v.strip() for k, v in row.items() if k is not None
                    }

                    # Check for blank row
                    if not any(cleaned_row.values()):
                        logger.warning("Line %d in '%s': Skipping blank row", row_idx, path.name)
                        continue

                    vendor_id = cleaned_row.get("vendor_id", "")
                    if not vendor_id:
                        errors.append(f"Line {row_idx}: 'vendor_id' is missing or blank.")
                        continue

                    # Duplicate vendor ID check
                    if vendor_id in seen_vendor_ids:
                        errors.append(
                            f"Line {row_idx}: Duplicate vendor_id '{vendor_id}' detected."
                        )
                        continue

                    try:
                        vendor = Vendor.model_validate(cleaned_row)
                        vendors.append(vendor)
                        seen_vendor_ids.add(vendor_id)
                    except ValidationError as ve:
                        for err in ve.errors():
                            loc = " -> ".join(str(l) for l in err["loc"])
                            msg = err["msg"]
                            errors.append(f"Line {row_idx} (Vendor '{vendor_id}'): {loc} - {msg}")

                if errors:
                    err_summary = "\n  - ".join(errors)
                    raise VendorValidationError(
                        f"Vendor validation failed with {len(errors)} error(s) in '{path}':\n  - {err_summary}"
                    )

                if not vendors:
                    raise IngestionError(f"No valid vendor records found in '{path}'.")

                logger.info("Successfully loaded and validated %d vendor(s) from %s", len(vendors), path.name)
                return vendors

        except (OSError, csv.Error) as exc:
            if isinstance(exc, (IngestionError, VendorValidationError)):
                raise
            raise IngestionError(f"I/O or CSV parsing error on '{path}': {exc}") from exc
