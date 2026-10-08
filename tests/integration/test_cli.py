"""Integration tests for the CLI interface."""

import json
import subprocess
import sys
from pathlib import Path


class TestCLIIntegration:
    """Test suite for CLI execution and argument handling."""

    def test_cli_default_run(self, tmp_path: Path) -> None:
        output_file = tmp_path / "cli_output.json"
        cmd = [
            sys.executable,
            "main.py",
            "--input",
            "data/sample/sample_vendors.csv",
            "--output",
            str(output_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "Meridian Vendor Assessment" in result.stdout
        assert "Loaded vendors: 4" in result.stdout
        assert "Assessment completed successfully" in result.stdout
        assert output_file.exists()

        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 4

    def test_cli_missing_input_file_returns_error_code(self) -> None:
        cmd = [
            sys.executable,
            "main.py",
            "--input",
            "data/non_existent_input.csv",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 1
        assert "not found" in result.stderr or "not found" in result.stdout
