"""P — Payment Flow Exposure factor classifier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import yaml

from meridian_assessment.models.factor_result import DeterminationMethod, FactorResult, SemanticMatchInfo
from meridian_assessment.services.factors.text_utils import normalize_text, contains_any, has_negation_before

logger = logging.getLogger(__name__)

_PAYMENT_ONTOLOGY_PATH = Path("config/payment_ontology.yaml")
_ontology: Optional[dict] = None


def _load_ontology(path: Path) -> dict:
    global _ontology, _PAYMENT_ONTOLOGY_PATH
    if _ontology is None or _PAYMENT_ONTOLOGY_PATH != path:
        _PAYMENT_ONTOLOGY_PATH = path
        with open(path, "r", encoding="utf-8") as f:
            _ontology = yaml.safe_load(f)
    return _ontology


def detect_negation(text: str, negation_patterns: list[str]) -> bool:
    """Return True if a negation pattern is found in the text."""
    low = text.lower()
    return any(neg.lower() in low for neg in negation_patterns)


def detect_p3_direct_action(norm: str, ontology: dict) -> tuple[bool, list[str]]:
    """Detect direct P3 actions: payment keyword + direct action verb (initiate/transmit/authorize/clear/settle).
    
    Must NOT trigger on 'settlement reconciliation' or 'statement of settlement'.
    """
    if "settlement reconciliation" in norm or "reconciliation" in norm and not any(k in norm for k in ["clearing and settlement", "clear and settle", "ach clearing", "rtp transmission", "transmit payment"]):
        return False, []

    p3_cfg = ontology.get("levels", {}).get("p3", {})
    direct_actions = [a.lower() for a in p3_cfg.get("direct_actions", [])]
    payment_keywords = [k.lower() for k in p3_cfg.get("payment_keywords", [])]

    found_actions = contains_any(norm, direct_actions)
    found_payments = contains_any(norm, payment_keywords)

    if found_actions and found_payments:
        return True, found_actions + found_payments
    return False, []


def classify_payment_flow(
    service_text: Optional[str],
    business_process_text: Optional[str],
    vendor_description: Optional[str] = None,
    category: Optional[str] = None,
    ontology_path: Path = _PAYMENT_ONTOLOGY_PATH,
    semantic_model=None,
    semantic_config: Optional[dict] = None,
) -> FactorResult:
    """Classify P — Payment Flow Exposure based on Service / Product and Business Process.
    
    Category is contextual only and must NEVER be the sole reason for assigning P.
    """
    ontology = _load_ontology(ontology_path)
    negation_patterns = ontology.get("negation_patterns", [])

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
            factor="P",
            factor_name="Payment Flow Exposure",
            score=0,
            source_fields=source_fields,
            raw_input=combined_primary,
            normalized_input=norm,
            matched_concepts=["no_payment"],
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale="No service or business process description provided — P=0.",
            certainty="HIGH",
        )

    # Check for explicit negation
    global_negation = detect_negation(norm, negation_patterns)

    # 1. P3 Direct Action Rule: payment concept + direct action verb (initiate, transmit, authorize, clear, settle)
    if not global_negation:
        is_p3, p3_phrases = detect_p3_direct_action(norm, ontology)
        if is_p3:
            return FactorResult(
                factor="P",
                factor_name="Payment Flow Exposure",
                score=3,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=["direct_payment_flow"],
                matched_phrases=list(set(p3_phrases)),
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale="Direct payment-flow action established: direct handling of payment/funds transfer (initiate/transmit/authorize/clear/settle).",
                certainty="HIGH",
            )

    levels = ontology.get("levels", {})

    # 2. P2 Material processing / control / reconciliation
    p2_concepts = levels.get("p2", {}).get("concepts", [])
    for concept in p2_concepts:
        aliases = [a.lower() for a in concept.get("aliases", [])]
        found = contains_any(norm, aliases)
        if found and not global_negation:
            return FactorResult(
                factor="P",
                factor_name="Payment Flow Exposure",
                score=2,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=[concept["name"]],
                matched_phrases=found,
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale=concept.get("rationale", "Material payment processing/control function established without direct P3 execution."),
                certainty="HIGH",
            )

    # 3. P1 Payment-adjacent support (reporting, statements, analytics, admin)
    p1_concepts = levels.get("p1", {}).get("concepts", [])
    for concept in p1_concepts:
        aliases = [a.lower() for a in concept.get("aliases", [])]
        found = contains_any(norm, aliases)
        if found:
            return FactorResult(
                factor="P",
                factor_name="Payment Flow Exposure",
                score=1,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=[concept["name"]],
                matched_phrases=found,
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale=concept.get("rationale", "Payment-adjacent reporting, analytics, or statement support without processing."),
                certainty="HIGH",
            )

    # 4. Explicit P0 no-payment concepts or negation triggered
    if global_negation:
        return FactorResult(
            factor="P",
            factor_name="Payment Flow Exposure",
            score=0,
            source_fields=source_fields,
            raw_input=combined_primary,
            normalized_input=norm,
            matched_concepts=["no_payment_negation"],
            matched_phrases=["negation_pattern_detected"],
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale="Explicit negation detected ('does not process payments' / 'no payment processing') — P=0.",
            certainty="HIGH",
        )

    p0_concepts = levels.get("p0", {}).get("concepts", [])
    for concept in p0_concepts:
        aliases = [a.lower() for a in concept.get("aliases", [])]
        found = contains_any(norm, aliases)
        if found:
            return FactorResult(
                factor="P",
                factor_name="Payment Flow Exposure",
                score=0,
                source_fields=source_fields,
                raw_input=combined_primary,
                normalized_input=norm,
                matched_concepts=[concept["name"]],
                matched_phrases=found,
                determination_method=DeterminationMethod.DETERMINISTIC_RULE,
                rationale="Explicit non-payment activity confirmed.",
                certainty="HIGH",
            )

    # If no payment keywords at all are present in the text, it's definitively P0
    p3_cfg = levels.get("p3", {})
    payment_keywords = [k.lower() for k in p3_cfg.get("payment_keywords", [])]
    any_payment = contains_any(norm, payment_keywords)
    if not any_payment:
        return FactorResult(
            factor="P",
            factor_name="Payment Flow Exposure",
            score=0,
            source_fields=source_fields,
            raw_input=combined_primary,
            normalized_input=norm,
            matched_concepts=["no_payment_involvement"],
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale="No payment or funds transfer concepts detected in actual service scope — P=0.",
            certainty="HIGH",
        )

    # 5. Semantic fallback for ambiguous cases
    if semantic_model is not None and semantic_config:
        return _semantic_fallback_p(combined_primary, norm, ontology, source_fields, semantic_model, semantic_config)

    # If payment keyword was present but could not determine level:
    return FactorResult(
        factor="P",
        factor_name="Payment Flow Exposure",
        score=None,
        is_unknown=True,
        source_fields=source_fields,
        raw_input=combined_primary,
        normalized_input=norm,
        determination_method=DeterminationMethod.UNKNOWN,
        rationale="Payment keywords detected but specific operational level cannot be definitively established. Requires review.",
        certainty="LOW",
    )


def _semantic_fallback_p(
    raw_text: str, norm: str, ontology: dict, source_fields: list,
    semantic_model, semantic_config: dict,
) -> FactorResult:
    """Semantic embedding fallback for P."""
    try:
        refs = ontology.get("semantic_references", {})
        reference_texts = [refs.get("p0", ""), refs.get("p1", ""), refs.get("p2", ""), refs.get("p3", "")]
        reference_scores = [0, 1, 2, 3]
        reference_labels = ["P0", "P1", "P2", "P3"]

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
                factor="P", factor_name="Payment Flow Exposure",
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
                factor="P", factor_name="Payment Flow Exposure",
                score=None, is_unknown=True,
                source_fields=source_fields, raw_input=raw_text, normalized_input=norm,
                semantic_matches=[sem_info],
                determination_method=DeterminationMethod.SEMANTIC_MATCH,
                rationale=f"Semantic similarity {sims[best_idx]:.2f} below threshold — requires review.",
                certainty="LOW",
            )
    except Exception as e:
        logger.warning("Semantic fallback for P failed: %s", e)
        return FactorResult(
            factor="P", factor_name="Payment Flow Exposure",
            score=None, is_unknown=True,
            source_fields=source_fields, raw_input=raw_text, normalized_input=norm,
            determination_method=DeterminationMethod.UNKNOWN,
            rationale=f"Semantic fallback failed: {e}",
            certainty="LOW",
        )
