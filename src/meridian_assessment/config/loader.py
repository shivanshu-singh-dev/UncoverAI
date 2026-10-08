"""Configuration loader for criticality rules and thresholds."""

from pathlib import Path
from typing import Union
import yaml
from pydantic import ValidationError

from meridian_assessment.models.config import CriticalityConfig
from meridian_assessment.utils.exceptions import ConfigurationError
from meridian_assessment.utils.logger import logger


def load_criticality_config(config_path: Union[str, Path] = "config/criticality.yaml") -> CriticalityConfig:
    """Load and validate the criticality configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated CriticalityConfig instance.

    Raises:
        ConfigurationError: If the file does not exist, cannot be parsed, or fails validation.
    """
    path = Path(config_path)

    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML configuration file '{path}': {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"I/O error reading configuration file '{path}': {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigurationError(f"Configuration root in '{path}' must be a mapping/dictionary, got {type(raw_data).__name__}")

    try:
        config = CriticalityConfig.model_validate(raw_data)
        logger.debug("Successfully loaded criticality configuration from %s (version %s)", path, config.version)
        return config
    except ValidationError as exc:
        raise ConfigurationError(f"Configuration validation failed for '{path}':\n{exc}") from exc
