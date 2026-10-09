"""Meridian Criticality Engine — implements deterministic D/P/R/O/V methodology with O1-O4 safeguards and O5 human governance."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import yaml

from meridian_assessment.models.factor_result import (
    CriticalityLevel,
    CriticalityResult,
    FactorResult,
    HumanReviewRecord,
    OverrideRecord,
    ScoreStatus,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.services.factors.d_classifier import classify_data_sensitivity
from meridian_assessment.services.factors.o_classifier import classify_operational_dependency
from meridian_assessment.services.factors.v_classifier import classify_volume
from meridian_assessment.services.factors.p_classifier import classify_payment_flow
from meridian_assessment.services.factors.r_classifier import classify_regulatory_exposure

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS_PATH = Path("config/factor_weights.yaml")
_DEFAULT_CRITICALITY_PATH = Path("config/criticality.yaml")
_DEFAULT_SEMANTIC_PATH = Path("config/semantic_config.yaml")

# Hierarchy order for floors
_CRITICALITY_ORDER = {
    CriticalityLevel.LOW: 0,
    CriticalityLevel.MEDIUM: 1,
    CriticalityLevel.HIGH: 2,
    CriticalityLevel.CRITICAL: 3,
}

_DEPTH_MAPPING = {
    CriticalityLevel.CRITICAL: "Comprehensive",
    CriticalityLevel.HIGH: "Comprehensive",
    CriticalityLevel.MEDIUM: "Targeted",
    CriticalityLevel.LOW: "Lightweight",
}


class MeridianCriticalityEngine:
    """Calculates deterministic, explainable criticality assessments based on the D/P/R/O/V rubric.
    
    Formula:
        C = 0.30D + 0.20P + 0.20R + 0.20O + 0.10V
    """

    def __init__(
        self,
        weights_path: Path = _DEFAULT_WEIGHTS_PATH,
        criticality_config_path: Path = _DEFAULT_CRITICALITY_PATH,
        semantic_config_path: Path = _DEFAULT_SEMANTIC_PATH,
        enable_semantic_fallback: bool = False,
    ) -> None:
        self.weights_path = weights_path
        self.criticality_config_path = criticality_config_path
        self.semantic_config_path = semantic_config_path
        self.enable_semantic_fallback = enable_semantic_fallback

        self.weights = self._load_weights()
        self.tier_thresholds, self.depth_mapping = self._load_thresholds()
        self.semantic_config = self._load_semantic_config()
        self._semantic_model = None

    def _load_weights(self) -> dict[str, float]:
        if self.weights_path.is_file():
            with open(self.weights_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("weights", {"D": 0.30, "P": 0.20, "R": 0.20, "O": 0.20, "V": 0.10})
        return {"D": 0.30, "P": 0.20, "R": 0.20, "O": 0.20, "V": 0.10}

    def _load_thresholds(self) -> tuple[dict[str, float], dict[str, str]]:
        thresholds = {
            CriticalityLevel.CRITICAL: 2.50,
            CriticalityLevel.HIGH: 2.00,
            CriticalityLevel.MEDIUM: 1.20,
            CriticalityLevel.LOW: 0.00,
        }
        depths = {
            CriticalityLevel.CRITICAL: "Comprehensive",
            CriticalityLevel.HIGH: "Comprehensive",
            CriticalityLevel.MEDIUM: "Targeted",
            CriticalityLevel.LOW: "Lightweight",
        }
        # First check factor_weights.yaml for provisional_thresholds
        if self.weights_path.is_file():
            with open(self.weights_path, "r", encoding="utf-8") as f:
                w_cfg = yaml.safe_load(f) or {}
                pt = w_cfg.get("provisional_thresholds", {})
                for k, v in pt.items():
                    if isinstance(v, (int, float)):
                        thresholds[k] = float(v)
        # Also allow overrides from criticality.yaml
        if self.criticality_config_path.is_file():
            with open(self.criticality_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                pt = cfg.get("provisional_thresholds", {})
                for k, v in pt.items():
                    if isinstance(v, (int, float)):
                        thresholds[k] = float(v)
                tt = cfg.get("tier_thresholds", {})
                for k, v in tt.items():
                    if isinstance(v, dict) and "min_score" in v:
                        thresholds[k] = float(v["min_score"])
        return thresholds, depths

    def _load_semantic_config(self) -> dict:
        if self.semantic_config_path.is_file():
            with open(self.semantic_config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def semantic_model(self):
        if not self.enable_semantic_fallback:
            return None
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.semantic_config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
                self._semantic_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning("Could not load sentence-transformers model: %s", e)
                self._semantic_model = None
        return self._semantic_model

    def evaluate(
        self,
        vendor: MeridianVendor,
        human_review: Optional[HumanReviewRecord] = None,
    ) -> CriticalityResult:
        """Evaluate a vendor under the D/P/R/O/V methodology with provenance tracking and O1-O5 overrides."""
        # 1. D Factor
        d_res = classify_data_sensitivity(
            vendor.data_classification_accessed,
            semantic_model=self.semantic_model,
            semantic_config=self.semantic_config,
        )

        # 2. P Factor
        p_res = classify_payment_flow(
            service_text=vendor.service_product_provided,
            business_process_text=vendor.business_process_supported,
            vendor_description=vendor.vendor_description,
            category=vendor.category,
            semantic_model=self.semantic_model,
            semantic_config=self.semantic_config,
        )

        # 3. R Factor
        r_res = classify_regulatory_exposure(
            service_text=vendor.service_product_provided,
            business_process_text=vendor.business_process_supported,
            vendor_description=vendor.vendor_description,
            category=vendor.category,
            semantic_model=self.semantic_model,
            semantic_config=self.semantic_config,
        )

        # 4. O Factor
        o_res = classify_operational_dependency(vendor.operational_dependency)

        # 5. V Factor
        v_res = classify_volume(vendor.data_volume_annual)

        # Assign weights & compute contributions
        factors_map = {"D": d_res, "P": p_res, "R": r_res, "O": o_res, "V": v_res}
        updated_factors: dict[str, FactorResult] = {}
        for factor_code, f_res in factors_map.items():
            w = self.weights.get(factor_code, 0.0)
            contrib = round(f_res.score * w, 4) if f_res.score is not None else None
            f_dict = f_res.model_dump()
            f_dict["weight"] = w
            f_dict["weighted_contribution"] = contrib
            updated_factors[factor_code] = FactorResult(**f_dict)

        d_final = updated_factors["D"]
        p_final = updated_factors["P"]
        r_final = updated_factors["R"]
        o_final = updated_factors["O"]
        v_final = updated_factors["V"]

        # Check for UNKNOWN factors
        unknown_factors = [f for f in [d_final, p_final, r_final, o_final, v_final] if f.is_unknown or f.score is None]
        requires_review_factors = [
            f for f in [d_final, p_final, r_final, o_final, v_final]
            if f.certainty == "LOW" or any(sm.review_status == "REVIEW_REQUIRED" for sm in f.semantic_matches)
        ]

        review_notes: list[str] = []
        if unknown_factors:
            review_notes.append(f"Factors with UNKNOWN values: {[f.factor for f in unknown_factors]}")
        if requires_review_factors:
            review_notes.append(f"Factors requiring review: {[f.factor for f in requires_review_factors]}")

        # Determine Score Status & Base Score
        if len(unknown_factors) == 0:
            score_status = ScoreStatus.COMPLETE
            base_score = (
                d_final.weighted_contribution +
                p_final.weighted_contribution +
                r_final.weighted_contribution +
                o_final.weighted_contribution +
                v_final.weighted_contribution
            )
            base_score = round(base_score, 2)
        elif len(unknown_factors) <= 2:
            score_status = ScoreStatus.PARTIAL
            known_contribs = [f.weighted_contribution for f in [d_final, p_final, r_final, o_final, v_final] if f.weighted_contribution is not None]
            base_score = round(sum(known_contribs), 2)
        else:
            score_status = ScoreStatus.UNKNOWN
            base_score = None

        if requires_review_factors and score_status == ScoreStatus.COMPLETE:
            score_status = ScoreStatus.REQUIRES_REVIEW

        # Step 3: Determine provisional criticality from threshold configuration
        provisional_criticality = self._determine_provisional_criticality(base_score)

        # Step 4: Apply automatic overrides O1-O4
        overrides_evaluated, proposed_criticality, automatic_override_applied = self._evaluate_automatic_overrides(
            vendor=vendor,
            D=d_final,
            P=p_final,
            R=r_final,
            O=o_final,
            V=v_final,
            provisional=provisional_criticality,
        )

        # Step 5: O5 User Review / Manual Governance
        final_criticality = proposed_criticality
        final_depth = _DEPTH_MAPPING.get(final_criticality, "Lightweight")

        if human_review is not None and human_review.user_decision == "CHANGE_CRITICALITY":
            final_criticality = human_review.final_criticality
            final_depth = _DEPTH_MAPPING.get(final_criticality, "Lightweight")

        return CriticalityResult(
            vendor_id=vendor.vendor_id,
            vendor_name=vendor.vendor_name,
            D=d_final,
            P=p_final,
            R=r_final,
            O=o_final,
            V=v_final,
            base_score=base_score,
            score_status=score_status,
            provisional_criticality=provisional_criticality,
            overrides_evaluated=overrides_evaluated,
            automatic_override_applied=automatic_override_applied,
            proposed_criticality=proposed_criticality,
            human_review=human_review,
            final_criticality=final_criticality,
            criticality_tier=final_criticality,  # backwards compatibility alias
            assessment_depth=final_depth,
            requires_review=bool(review_notes or score_status in (ScoreStatus.PARTIAL, ScoreStatus.REQUIRES_REVIEW, ScoreStatus.UNKNOWN)),
            review_notes=review_notes,
        )

    def _determine_provisional_criticality(self, score: Optional[float]) -> str:
        """Map base score on the 0-3 scale to provisional criticality using configured thresholds.

        Inclusive lower-bound thresholds (evaluated highest-first):
          score >= Critical (2.50) -> Critical
          score >= High     (2.00) -> High
          score >= Medium   (1.20) -> Medium
          otherwise                -> Low
        """
        if score is None:
            return CriticalityLevel.LOW

        crit_thresh = self.tier_thresholds.get(CriticalityLevel.CRITICAL, self.tier_thresholds.get("Critical", 2.50))
        high_thresh = self.tier_thresholds.get(CriticalityLevel.HIGH, self.tier_thresholds.get("High", 2.00))
        med_thresh = self.tier_thresholds.get(CriticalityLevel.MEDIUM, self.tier_thresholds.get("Medium", 1.20))

        if score >= crit_thresh:
            return CriticalityLevel.CRITICAL
        elif score >= high_thresh:
            return CriticalityLevel.HIGH
        elif score >= med_thresh:
            return CriticalityLevel.MEDIUM
        return CriticalityLevel.LOW

    def _evaluate_automatic_overrides(
        self,
        vendor: MeridianVendor,
        D: FactorResult,
        P: FactorResult,
        R: FactorResult,
        O: FactorResult,
        V: FactorResult,
        provisional: str,
    ) -> tuple[list[OverrideRecord], str, bool]:
        """Evaluate automatic safeguards O1-O4 and compute the maximum floor."""
        overrides: list[OverrideRecord] = []
        floors: list[str] = [provisional]

        # Combine text fields to inspect for explicit wording
        text_corpus = " ".join([
            vendor.vendor_description or "",
            vendor.service_product_provided or "",
            vendor.business_process_supported or "",
            vendor.data_classification_accessed or "",
        ]).lower()

        # ----------------------------------------------------
        # O1: Privileged Access -> Minimum HIGH floor
        # Explicit evidence of privileged/elevated/administrative access to production
        # ----------------------------------------------------
        privileged_patterns = [
            "privileged access",
            "administrative access",
            "production administrator access",
            "production administrator",
            "elevated privileges",
            "privileged production access",
            "privileged user access",
        ]
        o1_evidence = None
        for pat in privileged_patterns:
            if pat in text_corpus:
                o1_evidence = pat
                break

        o1_triggered = bool(o1_evidence)
        overrides.append(
            OverrideRecord(
                override_id="O1",
                description="Privileged / administrative access to production systems",
                condition_evaluated="Explicit evidence of privileged/elevated/administrative access",
                triggered=o1_triggered,
                resulting_floor=CriticalityLevel.HIGH if o1_triggered else None,
                evidence=f"Matched phrase '{o1_evidence}'" if o1_evidence else "None identified",
                rationale="Vendor holds confirmed privileged/administrative access to production systems; imposes minimum High floor."
                if o1_triggered else "Condition not met.",
            )
        )
        if o1_triggered:
            floors.append(CriticalityLevel.HIGH)

        # ----------------------------------------------------
        # O2: Sole-Source + Critical/Total Dependency (O=3) -> Critical floor
        # Requires BOTH: explicit sole-source confirmation AND O == 3
        # ----------------------------------------------------
        sole_source_patterns = [
            "sole-source",
            "sole source",
            "only provider",
            "single supplier",
            "no alternative provider",
            "no available substitute",
            "sole supplier",
        ]
        o2_evidence = None
        for pat in sole_source_patterns:
            if pat in text_corpus:
                o2_evidence = pat
                break

        o2_triggered = bool(o2_evidence and O.score == 3)
        overrides.append(
            OverrideRecord(
                override_id="O2",
                description="Sole-source supplier with Critical operational dependency (O=3)",
                condition_evaluated="SoleSourceConfirmed == TRUE and O == 3",
                triggered=o2_triggered,
                resulting_floor=CriticalityLevel.CRITICAL if o2_triggered else None,
                evidence=f"Sole-source phrase '{o2_evidence}', O={O.score}" if o2_evidence else f"O={O.score}, no sole-source evidence",
                rationale="Confirmed sole-source dependency combined with Critical operational path (O=3); imposes Critical floor."
                if o2_triggered else "Condition not met.",
            )
        )
        if o2_triggered:
            floors.append(CriticalityLevel.CRITICAL)

        # ----------------------------------------------------
        # O3: D3 + V3 -> Minimum HIGH floor
        # Maximum sensitivity + enterprise-scale volume
        # ----------------------------------------------------
        o3_triggered = bool(D.score == 3 and V.score == 3)
        overrides.append(
            OverrideRecord(
                override_id="O3",
                description="Maximum data sensitivity (D=3) with enterprise-scale volume (V=3)",
                condition_evaluated="D == 3 and V == 3",
                triggered=o3_triggered,
                resulting_floor=CriticalityLevel.HIGH if o3_triggered else None,
                evidence=f"D={D.score}, V={V.score}",
                rationale="Simultaneous maximum data sensitivity (D3) and enterprise-scale volume (V3); imposes minimum High floor."
                if o3_triggered else "Condition not met.",
            )
        )
        if o3_triggered:
            floors.append(CriticalityLevel.HIGH)

        # ----------------------------------------------------
        # O4: P3 + High-or-Greater Operational Dependency (O >= 2) -> Minimum HIGH floor
        # ----------------------------------------------------
        o4_triggered = bool(P.score == 3 and O.score is not None and O.score >= 2)
        overrides.append(
            OverrideRecord(
                override_id="O4",
                description="Direct payment-flow execution (P=3) with High+ operational dependency (O >= 2)",
                condition_evaluated="P == 3 and O >= 2",
                triggered=o4_triggered,
                resulting_floor=CriticalityLevel.HIGH if o4_triggered else None,
                evidence=f"P={P.score}, O={O.score}",
                rationale="Direct payment-flow activity (P3) combined with operational dependency O >= 2; imposes minimum High floor."
                if o4_triggered else "Condition not met.",
            )
        )
        if o4_triggered:
            floors.append(CriticalityLevel.HIGH)

        # Compute proposed criticality as maximum floor
        highest_order = max(_CRITICALITY_ORDER.get(f, 0) for f in floors)
        proposed_criticality = next(lvl for lvl, order in _CRITICALITY_ORDER.items() if order == highest_order)
        automatic_override_applied = proposed_criticality != provisional

        return overrides, proposed_criticality, automatic_override_applied

    def apply_human_override(
        self,
        current_result: CriticalityResult,
        decision: str,  # KEEP_PROPOSED or CHANGE_CRITICALITY
        new_criticality: Optional[str] = None,
        rationale: str = "",
    ) -> CriticalityResult:
        """Apply O5 human governance step to an existing result."""
        if decision == "CHANGE_CRITICALITY":
            if new_criticality not in _CRITICALITY_ORDER:
                raise ValueError(f"Invalid criticality selection: '{new_criticality}'. Must be one of {list(_CRITICALITY_ORDER.keys())}")
            if not rationale.strip():
                raise ValueError("A written rationale is required when changing the proposed criticality.")
            final_crit = new_criticality
        else:
            final_crit = current_result.proposed_criticality or current_result.provisional_criticality or CriticalityLevel.LOW

        review_rec = HumanReviewRecord(
            user_decision=decision,
            original_criticality=current_result.proposed_criticality or current_result.provisional_criticality or CriticalityLevel.LOW,
            final_criticality=final_crit,
            rationale=rationale.strip(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        final_depth = _DEPTH_MAPPING.get(final_crit, "Lightweight")

        res_dict = current_result.model_dump()
        res_dict["human_review"] = review_rec
        res_dict["final_criticality"] = final_crit
        res_dict["criticality_tier"] = final_crit
        res_dict["assessment_depth"] = final_depth
        return CriticalityResult(**res_dict)

    def generate_audit_trail(self, result: CriticalityResult) -> dict[str, Any]:
        """Generate full factor-level audit trail dictionary for a vendor."""
        return {
            "vendor_id": result.vendor_id,
            "vendor_name": result.vendor_name,
            "base_score": result.base_score,
            "score_status": result.score_status.value,
            "provisional_criticality": result.provisional_criticality,
            "automatic_override_applied": result.automatic_override_applied,
            "proposed_criticality": result.proposed_criticality,
            "final_criticality": result.final_criticality,
            "assessment_depth": result.assessment_depth,
            "human_review": result.human_review.model_dump() if result.human_review else None,
            "requires_review": result.requires_review,
            "review_notes": result.review_notes,
            "factors": {
                "D": result.D.debug_dict(),
                "P": result.P.debug_dict(),
                "R": result.R.debug_dict(),
                "O": result.O.debug_dict(),
                "V": result.V.debug_dict(),
            },
            "overrides_evaluated": [o.model_dump() for o in result.overrides_evaluated],
        }
