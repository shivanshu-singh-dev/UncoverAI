"""Unit tests for CSV vendor loader."""

import pytest
from pathlib import Path

from meridian_assessment.ingestion.csv_loader import CSVVendorLoader
from meridian_assessment.models.vendor import SeverityLevel
from meridian_assessment.utils.exceptions import IngestionError, VendorValidationError


class TestCSVVendorLoader:
    """Test suite for CSV vendor data ingestion."""

    @pytest.fixture
    def loader(self) -> CSVVendorLoader:
        return CSVVendorLoader()

    def test_load_valid_csv(self, loader: CSVVendorLoader, tmp_path: Path) -> None:
        csv_content = (
            "vendor_id,vendor_name,domain,data_sensitivity,payment_flows,"
            "regulatory_exposure,operational_dependency,customer_data_volume\n"
            "V001,Example Corp,example.com,high,medium,high,high,high\n"
            "V002,Example Ltd,example.org,low,none,low,medium,low\n"
        )
        csv_file = tmp_path / "valid.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        vendors = loader.load(csv_file)
        assert len(vendors) == 2
        assert vendors[0].vendor_id == "V001"
        assert vendors[0].data_sensitivity == SeverityLevel.HIGH
        assert vendors[0].payment_flows == SeverityLevel.MEDIUM
        assert vendors[1].vendor_id == "V002"
        assert vendors[1].payment_flows == SeverityLevel.NONE

    def test_payment_flows_all_severity_levels(self, loader: CSVVendorLoader, tmp_path: Path) -> None:
        """Verify all SeverityLevel values are accepted for the canonical payment_flows field."""
        for level in ("critical", "high", "medium", "low", "none"):
            csv_content = (
                "vendor_id,vendor_name,domain,data_sensitivity,payment_flows,"
                "regulatory_exposure,operational_dependency,customer_data_volume\n"
                f"V001,Test Corp,test.com,low,{level},low,low,low\n"
            )
            csv_file = tmp_path / f"payment_{level}.csv"
            csv_file.write_text(csv_content, encoding="utf-8")
            vendors = loader.load(csv_file)
            assert vendors[0].payment_flows.value == level

    def test_duplicate_vendor_id_rejected(self, loader: CSVVendorLoader, tmp_path: Path) -> None:
        csv_content = (
            "vendor_id,vendor_name,domain,data_sensitivity,payment_flows,"
            "regulatory_exposure,operational_dependency,customer_data_volume\n"
            "V001,Corp A,corpa.com,low,low,low,low,low\n"
            "V001,Corp B,corpb.com,high,high,high,high,high\n"
        )
        csv_file = tmp_path / "duplicates.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(VendorValidationError) as exc_info:
            loader.load(csv_file)
        assert "Duplicate vendor_id 'V001'" in str(exc_info.value)

    def test_missing_payment_flows_column_raises_error(self, loader: CSVVendorLoader, tmp_path: Path) -> None:
        """payment_flows is now required; missing it must raise IngestionError."""
        csv_content = (
            "vendor_id,vendor_name,domain,data_sensitivity,"
            "regulatory_exposure,operational_dependency,customer_data_volume\n"
            "V001,Incomplete Corp,incomplete.com,high,high,high,high\n"
        )
        csv_file = tmp_path / "missing_payment.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(IngestionError) as exc_info:
            loader.load(csv_file)
        assert "missing required columns" in str(exc_info.value)

    def test_missing_required_column_raises_error(self, loader: CSVVendorLoader, tmp_path: Path) -> None:
        csv_content = (
            "vendor_id,vendor_name,domain,data_sensitivity\n"
            "V001,Incomplete Corp,incomplete.com,high\n"
        )
        csv_file = tmp_path / "missing_col.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(IngestionError) as exc_info:
            loader.load(csv_file)
        assert "missing required columns" in str(exc_info.value)

    def test_invalid_categorical_value_raises_validation_error(
        self, loader: CSVVendorLoader, tmp_path: Path
    ) -> None:
        csv_content = (
            "vendor_id,vendor_name,domain,data_sensitivity,payment_flows,"
            "regulatory_exposure,operational_dependency,customer_data_volume\n"
            "V001,Bad Val Corp,badval.com,super_extreme_danger,none,low,low,low\n"
        )
        csv_file = tmp_path / "invalid_val.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(VendorValidationError) as exc_info:
            loader.load(csv_file)
        assert "Vendor validation failed" in str(exc_info.value)

    def test_nonexistent_file_raises_ingestion_error(self, loader: CSVVendorLoader) -> None:
        with pytest.raises(IngestionError) as exc_info:
            loader.load("non_existent_path.csv")
        assert "not found" in str(exc_info.value)
