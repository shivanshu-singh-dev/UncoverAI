"""R — Regulatory Exposure factor classifier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import yaml

from meridian_assessment.models.factor_result import DeterminationMethod, FactorResult, SemanticMatchInfo
from meridian_assessment.services.factors.text_utils import normalize_text, contains_any

logger = logging.getLogger(__name__)

_REGULATORY_TAXONOMY_PATH = Path("config/regulatory_taxonomy.yaml")
_taxonomy: Optional[dict] = None


def _load_taxonomy(path: Path) -> dict:
    global _taxonomy, _REGULATORY_TAXONOMY_PATH
    if _taxonomy is None or _REGULATORY_TAXONOMY_PATH != path:
        _REGULATORY_TAXONOMY_PATH = path
        with open(path, "r", encoding="utf-8") as f:
            _taxonomy = yaml.safe_load(f)
    return _taxonomy


def classify_regulatory_exposure(
    service_text: Optional[str],
    business_process_text: Optional[str],
    vendor_description: Optional[str] = None,
    category: Optional[str] = None,
    taxonomy_path: Path = _REGULATORY_TAXONOMY_PATH,
    semantic_model=None,
    semantic_config: Optional[dict] = None,
) -> FactorResult:
    """Classify R — Regulatory Exposure based on Service / Product and Business Process.
    
    Category and generic vendor description words ("compliance", "financial", "regulated")
    must NEVER be the sole justification for assigning R3 or R2.
    The actual Meridian service must directly support or perform the regulated function.
    """
    taxonomy = _load_taxonomy(taxonomy_path)

    # Primary fields
    parts = [s for s in [service_text, business_process_text] if s]
    combined_primary = " ".join(parts)
    norm = normalize_text(combined_primary)

    source_fields = []
    if service_text:
        source_fields.append("service_product_provided")
    if business_process_text:
        source_fields.append("business_process_supported")

    if not norm:
        return FactorResult(
            factor="R",
            factor_name="Regulatory Exposure",
            score=0,
            source_fields=source_fields,
            raw_input=combined_primary,
            normalized_input=norm,
            matched_concepts=["no_regulatory"],
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale="No service or business process description provided — R=0.",
            certainty="HIGH",
        )

    levels = taxonomy.get("levels", {})

    # Check from highest (R3) to lowest (R0)
    # 1. R3: Critical outsourced or regulated function / critical financial infrastructure
    r3_concepts = levels.get("r3", {}).get("concepts", [])
    for concept in r3_concepts:
        aliases = [a.lower() for a in concept.get("aliases", [])]
        found = contains_any(norm, aliases)
        if found:
            return FactorResult(
                factor="R",
                factor_name="Regulatory Exposure",
                score=3,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=[concept["name"]],
                matched_phrases=found,
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale=concept.get("rationale", "Performs a critical outsourced or regulated function / critical financial infrastructure."),
                certainty="HIGH",
            )

    # 2. R2: Direct regulatory support (regulatory notices, regulatory reporting, regulatory records, material compliance)
    r2_concepts = levels.get("r2", {}).get("concepts", [])
    for concept in r2_concepts:
        aliases = [a.lower() for a in concept.get("aliases", [])]
        found = contains_any(norm, aliases)
        if found:
            return FactorResult(
                factor="R",
                factor_name="Regulatory Exposure",
                score=2,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=[concept["name"]],
                matched_phrases=found,
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale=concept.get("rationale", "Direct material support for a regulatory obligation (notices, filings, recordkeeping)."),
                certainty="HIGH",
            )

    # 3. R1: Indirect compliance support / operational reporting / analytics
    r1_concepts = levels.get("r1", {}).get("concepts", [])
    for concept in r1_concepts:
        aliases = [a.lower() for a in concept.get("aliases", [])]
        found = contains_any(norm, aliases)
        if found:
            return FactorResult(
                factor="R",
                factor_name="Regulatory Exposure",
                score=1,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=[concept["name"]],
                matched_phrases=found,
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale=concept.get("rationale", "Indirect compliance support or operational reporting assistance."),
                certainty="HIGH",
            )

    # 4. R0: Explicit no-regulatory concepts or no regulatory touchpoints
    r0_concepts = levels.get("r0", {}).get("concepts", [])
    for concept in r0_concepts:
        aliases = [a.lower() for a in concept.get("aliases", [])]
        found = contains_any(norm, aliases)
        if found:
            return FactorResult(
                factor="R",
                factor_name="Regulatory Exposure",
                score=0,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=[concept["name"]],
                matched_phrases=found,
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale="Explicit confirmation of no regulatory dependency.",
                certainty="HIGH",
            )

    # Note: Check if vendor description has "compliance" but service does not.
    # The methodology rule: Do NOT inherit generic compliance from vendor description.
    # If the service itself has no regulatory match, it defaults to R0 unless semantic fallback applies.
    if semantic_model is not None and semantic_config:
        return _semantic_fallback_r(combined_primary, norm, taxonomy, source_fields, semantic_model, semantic_config)

    return FactorResult(
        factor="R",
        factor_name="Regulatory Exposure",
        score=0,
        source_fields=source_fields,
        raw_input=combined_primary,
        normalized_input=norm,
        matched_concepts=["no_regulatory_dependency"],
        determination_method=DeterminationMethod.DETERMINISTIC_RULE,
        rationale="No direct or indirect regulatory obligations identified in the actual service description — R=0.",
        certainty="HIGH",
    )


def _semantic_fallback_r(
    raw_text: str, norm: str, taxonomy: dict, source_fields: list,
    semantic_model, semantic_config: dict,
) -> FactorResult:
    """Semantic embedding fallback for R."""
    try:
        refs = taxonomy.get("semantic_references", {})
        reference_texts = [refs.get("r0", ""), refs.get("r1", ""), refs.get("r2", ""), refs.get("r3", "")]
        reference_scores = [0, 1, 2, 3]
        reference_labels = ["R0", "R1", "R2", "R3"]

        top_k = semantic_config.get("top_k", 3)
        high_conf = semantic_config.get("thresholds", {}).get("high_confidence", 0.70)
        model_name = semantic_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        input_emb = semantic_model.encode([norm])
        ref_embs = semantic_model.encode(reference_texts)

        from numpy import dot
        from numpy.linalg import norm as np_norm
        sims = [
            float(dot(input_emb[0], ref_embs[i]) / (np_norm(input_emb[0]) * np_norm(ref_embs[i]) + 1e-9))
            for i in range(len(reference_texts))
        ]

        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        sem_info = SemanticMatchInfo(
            embedding_model=model_name,
            top_k=top_k,
            candidate_concepts=reference_labels,
            similarity_scores=sims,
            selected_candidate=reference_labels[best_idx],
            selected_score_level=reference_scores[best_idx],
            review_status="REVIEW_REQUIRED" if sims[best_idx] < high_conf else "OK",
        )

        if sims[best_idx] >= high_conf:
            return FactorResult(
                factor="R", factor_name="Regulatory Exposure",
                score=reference_scores[best_idx],
                source_fields=source_fields, raw_input=raw_text, normalized_input=norm,
                matched_concepts=[reference_labels[best_idx]],
                semantic_matches=[sem_info],
                determination_method=DeterminationMethod.SEMANTIC_MATCH,
                rationale=f"Semantic match to {reference_labels[best_idx]} (similarity {sims[best_idx]:.2f}).",
                certainty="MEDIUM",
            )
        else:
            return FactorResult(
                factor="R", factor_name="Regulatory Exposure",
                score=None, is_unknown=True,
                source_fields=source_fields, raw_input=raw_text, normalized_input=norm,
                semantic_matches=[sem_info],
                determination_method=DeterminationMethod.SEMANTIC_MATCH,
                rationale=f"Semantic similarity {sims[best_idx]:.2f} below threshold — requires review.",
                certainty="LOW",
            )
    except Exception as e:
        logger.warning("Semantic fallback for R failed: %s", e)
        return FactorResult(
            factor="R", factor_name="Regulatory Exposure",
            score=None, is_unknown=True,
            source_fields=source_fields, raw_input=raw_text, normalized_input=norm,
            determination_method=DeterminationMethod.UNKNOWN,
            rationale=f"Semantic fallback failed: {e}",
            certainty="LOW",
        )
