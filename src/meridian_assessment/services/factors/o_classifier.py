"""O — Operational Dependency factor classifier."""
from pathlib import Path
from typing import Optional
import yaml

from meridian_assessment.models.factor_result import DeterminationMethod, FactorResult
from meridian_assessment.services.factors.text_utils import normalize_text

_MAPPING_PATH = Path("config/operational_dependency_mapping.yaml")
_mapping: Optional[dict] = None


def _load_mapping(path: Path) -> dict:
    global _mapping, _MAPPING_PATH
    if _mapping is None or _MAPPING_PATH != path:
        _MAPPING_PATH = path
        with open(path) as f:
            _mapping = yaml.safe_load(f)
    return _mapping


def classify_operational_dependency(
    raw_value: Optional[str],
    mapping_path: Path = _MAPPING_PATH,
) -> FactorResult:
    """Classify O — Operational Dependency from a structured field value."""
    config = _load_mapping(mapping_path)
    score_map: dict[str, int] = config.get("mapping", {})

    if not raw_value or raw_value.strip().lower() in ("", "n/a", "none", "not applicable"):
        return FactorResult(
            factor="O",
            factor_name="Operational Dependency",
            score=0,
            source_fields=["operational_dependency"],
            raw_input=raw_value or "",
            normalized_input="",
            matched_concepts=["no_dependency"],
            determination_method=DeterminationMethod.DIRECT,
            rationale="No operational dependency specified — defaulting to O=0.",
            certainty="MEDIUM",
        )

    norm = normalize_text(raw_value)

    # Direct lookup
    score = score_map.get(norm)
    if score is not None:
        return FactorResult(
            factor="O",
            factor_name="Operational Dependency",
            score=score,
            source_fields=["operational_dependency"],
            raw_input=raw_value,
            normalized_input=norm,
            matched_concepts=[norm],
            matched_phrases=[norm],
            determination_method=DeterminationMethod.DIRECT,
            rationale=f"Direct mapping: '{raw_value}' -> O={score}.",
            certainty="HIGH",
        )

    # UNKNOWN — do not silently use 0
    return FactorResult(
        factor="O",
        factor_name="Operational Dependency",
        score=None,
        is_unknown=True,
        source_fields=["operational_dependency"],
        raw_input=raw_value,
        normalized_input=norm,
        determination_method=DeterminationMethod.UNKNOWN,
        rationale=f"Unrecognized operational dependency value: '{raw_value}'. Requires human review.",
        certainty="LOW",
    )
