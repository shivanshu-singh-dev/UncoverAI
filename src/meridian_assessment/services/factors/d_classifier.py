"""D — Data Sensitivity factor classifier."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from meridian_assessment.models.factor_result import DeterminationMethod, FactorResult, SemanticMatchInfo
from meridian_assessment.services.factors.text_utils import normalize_text, contains_any, has_negation_before

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path("config/data_sensitivity_taxonomy.yaml")
_taxonomy: Optional[dict] = None


def _load_taxonomy() -> dict:
    global _taxonomy
    if _taxonomy is None:
        with open(_TAXONOMY_PATH) as f:
            _taxonomy = yaml.safe_load(f)
    return _taxonomy


def _build_concept_index(taxonomy: dict) -> dict[int, list[dict]]:
    """Build {score: [{name, aliases, rationale}]} index."""
    index: dict[int, list[dict]] = {}
    rubric = taxonomy.get("rubric", {})
    for score_key, score_data in rubric.items():
        score = int(score_key.split("_")[1])  # score_0 → 0
        concepts = score_data.get("concepts", [])
        index[score] = concepts
    return index


def classify_data_sensitivity(
    raw_text: Optional[str],
    taxonomy_path: Path = _TAXONOMY_PATH,
    semantic_model=None,
    semantic_config: Optional[dict] = None,
) -> FactorResult:
    """Classify D — Data Sensitivity from raw data classification text.
    
    Scoring uses HIGHEST applicable level found.
    """
    global _taxonomy, _TAXONOMY_PATH
    _TAXONOMY_PATH = taxonomy_path

    taxonomy = _load_taxonomy()
    concept_index = _build_concept_index(taxonomy)

    if not raw_text or raw_text.strip().lower() in ("n/a", "none", "", "not applicable"):
        return FactorResult(
            factor="D",
            factor_name="Data Sensitivity",
            score=0,
            source_fields=["data_classification_accessed"],
            raw_input=raw_text or "",
            normalized_input="",
            matched_concepts=["no_data"],
            matched_phrases=[],
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale="No data classification provided — D=0.",
            certainty="HIGH",
        )

    norm = normalize_text(raw_text)

    # Check from HIGHEST score downward — take the first (highest) match
    matched_score = None
    matched_concepts_list: list[str] = []
    matched_phrases_list: list[str] = []
    matched_rationale = ""

    for score in sorted(concept_index.keys(), reverse=True):
        concepts = concept_index[score]
        for concept in concepts:
            aliases = [a.lower() for a in concept.get("aliases", [])]
            for alias in aliases:
                # Use regex with word boundaries for short words (<=3 chars) to avoid matching inside words
                if len(alias) <= 3:
                    matched = bool(re.search(r'\b' + re.escape(alias) + r'\b', norm))
                else:
                    matched = alias in norm

                if matched:
                    # If this is a positive concept (score > 0) but preceded by negation, skip it
                    if score > 0 and has_negation_before(norm, alias, window=25):
                        continue
                    if matched_score is None:
                        matched_score = score
                        matched_rationale = concept.get("rationale", "")
                    matched_concepts_list.append(concept["name"])
                    matched_phrases_list.append(alias)

    if matched_score is not None:
        # Recalculate to ensure highest score is used
        # (we already iterate from highest, so first match is highest)
        # But also collect ALL concepts found for the highest matching score
        final_score = matched_score
        return FactorResult(
            factor="D",
            factor_name="Data Sensitivity",
            score=final_score,
            source_fields=["data_classification_accessed"],
            raw_input=raw_text,
            normalized_input=norm,
            matched_concepts=list(set(matched_concepts_list)),
            matched_phrases=list(set(matched_phrases_list)),
            determination_method=DeterminationMethod.DETERMINISTIC_RULE,
            rationale=matched_rationale,
            certainty="HIGH",
        )

    # Semantic fallback
    if semantic_model is not None and semantic_config:
        return _semantic_fallback_d(raw_text, norm, taxonomy, concept_index, semantic_model, semantic_config)

    # Cannot determine
    return FactorResult(
        factor="D",
        factor_name="Data Sensitivity",
        score=None,
        is_unknown=True,
        source_fields=["data_classification_accessed"],
        raw_input=raw_text,
        normalized_input=norm,
        matched_concepts=[],
        matched_phrases=[],
        determination_method=DeterminationMethod.UNKNOWN,
        rationale="Could not match data classification to a known sensitivity concept.",
        certainty="LOW",
    )


def _semantic_fallback_d(
    raw_text: str,
    norm: str,
    taxonomy: dict,
    concept_index: dict,
    semantic_model,
    semantic_config: dict,
) -> FactorResult:
    """Use semantic embeddings to classify D when deterministic matching fails."""
    try:
        rubric = taxonomy.get("rubric", {})
        reference_texts = []
        reference_scores = []
        reference_labels = []

        for score_key, score_data in rubric.items():
            score = int(score_key.split("_")[1])
            label = score_data.get("label", score_key)
            # Use the concept aliases as reference
            for concept in score_data.get("concepts", []):
                for alias in concept.get("aliases", [])[:3]:  # Use first 3 aliases
                    reference_texts.append(alias)
                    reference_scores.append(score)
                    reference_labels.append(f"{label} — {concept['name']}")

        top_k = semantic_config.get("top_k", 3)
        high_conf = semantic_config.get("thresholds", {}).get("high_confidence", 0.70)
        low_conf = semantic_config.get("thresholds", {}).get("low_confidence", 0.45)
        model_name = semantic_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        input_embedding = semantic_model.encode([norm])
        ref_embeddings = semantic_model.encode(reference_texts)

        from numpy import dot
        from numpy.linalg import norm as np_norm

        sims = [
            float(dot(input_embedding[0], ref_embeddings[i]) /
                  (np_norm(input_embedding[0]) * np_norm(ref_embeddings[i]) + 1e-9))
            for i in range(len(reference_texts))
        ]

        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        top_labels = [reference_labels[i] for i in top_indices]
        top_sims = [sims[i] for i in top_indices]
        top_scores = [reference_scores[i] for i in top_indices]

        sem_info = SemanticMatchInfo(
            embedding_model=model_name,
            top_k=top_k,
            candidate_concepts=top_labels,
            similarity_scores=top_sims,
            selected_candidate=top_labels[0] if top_labels else None,
            selected_score_level=top_scores[0] if top_scores else None,
            review_status="REVIEW_REQUIRED" if top_sims[0] < high_conf else "OK",
        )

        if top_sims[0] >= high_conf:
            return FactorResult(
                factor="D",
                factor_name="Data Sensitivity",
                score=top_scores[0],
                source_fields=["data_classification_accessed"],
                raw_input=raw_text,
                normalized_input=norm,
                matched_concepts=[top_labels[0]],
                matched_phrases=[],
                semantic_matches=[sem_info],
                determination_method=DeterminationMethod.SEMANTIC_MATCH,
                rationale=f"Semantic match to '{top_labels[0]}' (similarity {top_sims[0]:.2f}).",
                certainty="MEDIUM",
            )
        else:
            return FactorResult(
                factor="D",
                factor_name="Data Sensitivity",
                score=None,
                is_unknown=True,
                source_fields=["data_classification_accessed"],
                raw_input=raw_text,
                normalized_input=norm,
                matched_concepts=[top_labels[0]] if top_labels else [],
                semantic_matches=[sem_info],
                determination_method=DeterminationMethod.SEMANTIC_MATCH,
                rationale=f"Semantic similarity {top_sims[0]:.2f} below confidence threshold — requires review.",
                certainty="LOW",
            )

    except Exception as e:
        logger.warning("Semantic fallback for D failed: %s", e)
        return FactorResult(
            factor="D",
            factor_name="Data Sensitivity",
            score=None,
            is_unknown=True,
            source_fields=["data_classification_accessed"],
            raw_input=raw_text,
            normalized_input=norm,
            determination_method=DeterminationMethod.UNKNOWN,
            rationale=f"Semantic fallback failed: {e}",
            certainty="LOW",
        )
