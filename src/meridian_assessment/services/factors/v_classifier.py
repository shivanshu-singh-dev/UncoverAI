"""V — Annual Data Volume factor classifier."""
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional
import yaml

from meridian_assessment.models.factor_result import DeterminationMethod, FactorResult
from meridian_assessment.services.factors.text_utils import normalize_text

logger = logging.getLogger(__name__)

_VOLUME_CONFIG_PATH = Path("config/volume_thresholds.yaml")
_vol_config: Optional[dict] = None


def _load_volume_config(path: Path) -> dict:
    global _vol_config, _VOLUME_CONFIG_PATH
    if _vol_config is None or _VOLUME_CONFIG_PATH != path:
        _VOLUME_CONFIG_PATH = path
        with open(path) as f:
            _vol_config = yaml.safe_load(f)
    return _vol_config


def parse_annual_volume(raw_text: str, config: dict) -> tuple[Optional[float], str, str]:
    """Parse raw volume text into (numeric_value_or_None, volume_type, rationale).
    
    volume_type: COUNT / POPULATION / FREQUENCY / UNKNOWN
    """
    norm = raw_text.lower().strip()

    # N/A / None
    if norm in ("n/a", "none", "not applicable", "", "-"):
        return None, "NONE", "No annual volume specified."

    freq_indicators = config.get("frequency_indicators", [])
    pop_indicators = config.get("population_indicators", [])
    count_units = config.get("count_units", [])
    scale_multipliers = config.get("scale_multipliers", {})

    # Check if this is a frequency indicator first (e.g., "12 monthly cycles")
    for fi in freq_indicators:
        if fi.lower() in norm:
            return None, "FREQUENCY", f"Text contains frequency indicator '{fi}' — not an annual item count."

    # Try to extract a number + optional scale + unit
    # Examples:
    #   "500,000 messages" -> num: 500000, scale: None, unit: messages
    #   "2.1 million messages" -> num: 2.1, scale: million, unit: messages
    #   "38,000 wealth clients" -> num: 38000, scale: None, unit: wealth clients
    pattern = r'([\d,\.]+)\s*(thousand|million|billion|\bk\b|\bm\b|\bb\b)?\s*([a-zA-Z\s]+)?'
    m = re.search(pattern, norm)

    if m:
        num_str = m.group(1).replace(',', '')
        scale_str = m.group(2)
        unit_str = m.group(3)

        try:
            base_num = float(num_str)
        except ValueError:
            return None, "UNKNOWN", f"Could not parse numeric value from '{raw_text}'."

        scale_key = scale_str.lower().strip() if scale_str else ""
        multiplier = scale_multipliers.get(scale_key, 1)
        annual_volume = base_num * multiplier

        unit_norm = unit_str.lower().strip() if unit_str else ""
        
        # Check if population indicator is present
        is_pop = any(p in unit_norm for p in pop_indicators)
        # Check count units
        is_count = any(u in unit_norm for u in count_units)

        if is_pop and not any(k in unit_norm for k in ["transaction", "message", "document", "file", "event", "payment"]):
            return annual_volume, "POPULATION", f"Parsed {annual_volume:,.0f} as population count ('{unit_norm}'). Not treated as annual transaction count."
        elif is_count:
            return annual_volume, "COUNT", f"Parsed {base_num} × {multiplier} = {annual_volume:,.0f} annual {unit_norm}."
        else:
            return annual_volume, "COUNT", f"Parsed {annual_volume:,.0f} from '{raw_text}'."

    return None, "UNKNOWN", f"Could not extract numeric volume from '{raw_text}'."


def classify_volume(
    raw_text: Optional[str],
    config_path: Path = _VOLUME_CONFIG_PATH,
) -> FactorResult:
    """Classify V — Annual Data Volume."""
    config = _load_volume_config(config_path)
    thresholds = config.get("thresholds", {})
    small_upper = thresholds.get("small_upper", 1_000_000)
    material_upper = thresholds.get("material_upper", 10_000_000)

    if not raw_text or raw_text.strip().lower() in ("n/a", "none", "not applicable", "", "-"):
        return FactorResult(
            factor="V",
            factor_name="Annual Data Volume",
            score=0,
            source_fields=["data_volume_annual"],
            raw_input=raw_text or "",
            normalized_input="",
            matched_concepts=["no_volume"],
            volume_type="NONE",
            normalized_volume=None,
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale="No annual volume specified — V=0.",
            certainty="HIGH",
        )

    norm = normalize_text(raw_text)
    annual_count, volume_type, rationale = parse_annual_volume(raw_text, config)

    if volume_type == "FREQUENCY":
        return FactorResult(
            factor="V",
            factor_name="Annual Data Volume",
            score=None,
            is_unknown=True,
            source_fields=["data_volume_annual"],
            raw_input=raw_text,
            normalized_input=norm,
            volume_type="FREQUENCY",
            normalized_volume=None,
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale=f"Frequency indicator detected: '{raw_text}' is not an annual item count. V=UNKNOWN.",
            certainty="HIGH",
        )

    if volume_type == "POPULATION":
        return FactorResult(
            factor="V",
            factor_name="Annual Data Volume",
            score=None,
            is_unknown=True,
            source_fields=["data_volume_annual"],
            raw_input=raw_text,
            normalized_input=norm,
            volume_type="POPULATION",
            normalized_volume=annual_count,
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale=f"Population count detected: {rationale} Annual transaction volume cannot be determined without explicit data.",
            certainty="MEDIUM",
        )

    if annual_count is None:
        return FactorResult(
            factor="V",
            factor_name="Annual Data Volume",
            score=None,
            is_unknown=True,
            source_fields=["data_volume_annual"],
            raw_input=raw_text,
            normalized_input=norm,
            volume_type="UNKNOWN",
            determination_method=DeterminationMethod.UNKNOWN,
            rationale=f"Could not parse annual volume: {rationale}",
            certainty="LOW",
        )

    # Classify score by thresholds
    if annual_count <= 0:
        score = 0
        score_rationale = "Zero or negative volume — V=0."
    elif annual_count < small_upper:
        score = 1
        score_rationale = f"{annual_count:,.0f} items < {small_upper:,.0f} threshold — limited volume — V=1."
    elif annual_count < material_upper:
        score = 2
        score_rationale = f"{annual_count:,.0f} items in range [{small_upper:,.0f}, {material_upper:,.0f}) — material volume — V=2."
    else:
        score = 3
        score_rationale = f"{annual_count:,.0f} items >= {material_upper:,.0f} threshold — enterprise-scale volume — V=3."

    return FactorResult(
        factor="V",
        factor_name="Annual Data Volume",
        score=score,
        source_fields=["data_volume_annual"],
        raw_input=raw_text,
        normalized_input=norm,
        volume_type=volume_type,
        normalized_volume=annual_count,
        determination_method=DeterminationMethod.DETERMINISTIC_RULE,
        rationale=f"{rationale} {score_rationale}",
        certainty="HIGH",
    )
