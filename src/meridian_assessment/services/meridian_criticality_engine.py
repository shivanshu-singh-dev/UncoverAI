"""Meridian Criticality Engine — implements deterministic D/P/R/O/V methodology."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import yaml

from meridian_assessment.models.factor_result import (
    CriticalityResult,
    FactorResult,
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
        thresholds = {"TIER_1": 3.5, "TIER_2": 2.0, "TIER_3": 0.0}
        depths = {"TIER_1": "COMPREHENSIVE", "TIER_2": "TARGETED", "TIER_3": "LIGHTWEIGHT"}
        if self.criticality_config_path.is_file():
            with open(self.criticality_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                tt = cfg.get("tier_thresholds", {})
                for k, v in tt.items():
                    if isinstance(v, dict) and "min_score" in v:
                        thresholds[k] = float(v["min_score"])
                dm = cfg.get("assessment_depth_mapping", {})
                for k, v in dm.items():
                    if isinstance(v, dict) and "depth" in v:
                        depths[k] = str(v["depth"])
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

    def evaluate(self, vendor: MeridianVendor) -> CriticalityResult:
        """Evaluate a vendor under the D/P/R/O/V methodology with provenance tracking."""
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
            # Recreate immutable model with weight information
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

        # Determine Score Status
        if len(unknown_factors) == 0:
            score_status = ScoreStatus.COMPLETE
            # Calculate base score
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
            # Calculate available partial score
            known_contribs = [f.weighted_contribution for f in [d_final, p_final, r_final, o_final, v_final] if f.weighted_contribution is not None]
            base_score = round(sum(known_contribs), 2)
        else:
            score_status = ScoreStatus.UNKNOWN
            base_score = None

        if requires_review_factors and score_status == ScoreStatus.COMPLETE:
            score_status = ScoreStatus.REQUIRES_REVIEW

        # Determine base tier from provisional thresholds
        # Note: In the new 0-3 scale, max theoretical score is 3.0.
        # Provisional thresholds are kept configurable and evaluated dynamically.
        tier, depth = self._determine_tier_and_depth(base_score)

        # Overrides evaluation (O1-O5)
        overrides, final_tier, final_depth, override_applied = self._evaluate_overrides(
            d_final, p_final, r_final, o_final, v_final, tier, depth
        )

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
            overrides_evaluated=overrides,
            override_applied=override_applied,
            override_tier=final_tier if override_applied else None,
            criticality_tier=final_tier,
            assessment_depth=final_depth,
            requires_review=bool(review_notes or score_status in (ScoreStatus.PARTIAL, ScoreStatus.REQUIRES_REVIEW, ScoreStatus.UNKNOWN)),
            review_notes=review_notes,
        )

    def _determine_tier_and_depth(self, score: Optional[float]) -> tuple[Optional[str], Optional[str]]:
        if score is None:
            return None, None
        
        # In the 0-3 scale:
        # Tier 1: >= 2.25 (or configured threshold)
        # For compatibility with provisional threshold checking:
        # We sort descending:
        sorted_tiers = sorted(self.tier_thresholds.items(), key=lambda x: x[1], reverse=True)
        for t_name, min_s in sorted_tiers:
            # Scale adjustment if thresholds are on 0-5 scale vs 0-3 scale:
            # If threshold is >= 3.0 on 0-3 scale, adjust provisionally:
            # 3.5 / 5.0 = 70% of max -> in 0-3 scale 70% is 2.10
            adj_min = min_s if min_s <= 3.0 else (min_s / 5.0) * 3.0
            if score >= adj_min:
                return t_name, self.depth_mapping.get(t_name, "LIGHTWEIGHT")
        return "TIER_3", self.depth_mapping.get("TIER_3", "LIGHTWEIGHT")

    def _evaluate_overrides(
        self,
        D: FactorResult,
        P: FactorResult,
        R: FactorResult,
        O: FactorResult,
        V: FactorResult,
        current_tier: Optional[str],
        current_depth: Optional[str],
    ) -> tuple[list[OverrideRecord], Optional[str], Optional[str], bool]:
        """Evaluate O1-O5 overrides framework."""
        overrides: list[OverrideRecord] = []
        tier = current_tier
        depth = current_depth
        override_applied = False

        # O1: Critical Operational Dependency (O=3) with significant volume/data
        o1_triggered = (O.score == 3 and D.score is not None and D.score >= 2)
        overrides.append(
            OverrideRecord(
                override_id="O1",
                description="Critical Operational Dependency with high data sensitivity",
                condition_evaluated="O == 3 and D >= 2",
                triggered=o1_triggered,
                resulting_tier="TIER_1" if o1_triggered else None,
                rationale="Critical infrastructure dependency directly handling sensitive core assets." if o1_triggered else "Condition not met.",
            )
        )
        if o1_triggered and tier != "TIER_1":
            tier = "TIER_1"
            depth = "COMPREHENSIVE"
            override_applied = True

        # O2: Direct Payment Flow Execution (P=3)
        o2_triggered = (P.score == 3)
        overrides.append(
            OverrideRecord(
                override_id="O2",
                description="Direct Payment Flow execution (clearing/settlement/transmission)",
                condition_evaluated="P == 3",
                triggered=o2_triggered,
                resulting_tier="TIER_1" if o2_triggered else None,
                rationale="Direct transmission/clearing/settlement of payment transactions." if o2_triggered else "Condition not met.",
            )
        )
        if o2_triggered and tier != "TIER_1":
            tier = "TIER_1"
            depth = "COMPREHENSIVE"
            override_applied = True

        # O3: Critical Regulated Function (R=3)
        o3_triggered = (R.score == 3)
        overrides.append(
            OverrideRecord(
                override_id="O3",
                description="Critical outsourced or regulated infrastructure",
                condition_evaluated="R == 3",
                triggered=o3_triggered,
                resulting_tier="TIER_1" if o3_triggered else None,
                rationale="Critical designated financial market utility or regulated function." if o3_triggered else "Condition not met.",
            )
        )
        if o3_triggered and tier != "TIER_1":
            tier = "TIER_1"
            depth = "COMPREHENSIVE"
            override_applied = True

        # O4: Privileged Access / Credentials (D=3 via credentials)
        o4_triggered = (D.score == 3 and "credentials" in D.matched_concepts and O.score is not None and O.score >= 2)
        overrides.append(
            OverrideRecord(
                override_id="O4",
                description="Privileged production access / credentials with High+ operational dependency",
                condition_evaluated="D == 3 (credentials) and O >= 2",
                triggered=o4_triggered,
                resulting_tier="TIER_1" if o4_triggered else None,
                rationale="Direct access to production credentials across core execution environments." if o4_triggered else "Condition not met.",
            )
        )
        if o4_triggered and tier != "TIER_1":
            tier = "TIER_1"
            depth = "COMPREHENSIVE"
            override_applied = True

        # O5: Enterprise Scale Customer Population (V=3 and D >= 2)
        o5_triggered = (V.score == 3 and D.score is not None and D.score >= 2)
        overrides.append(
            OverrideRecord(
                override_id="O5",
                description="Enterprise-scale volume with sensitive customer data",
                condition_evaluated="V == 3 and D >= 2",
                triggered=o5_triggered,
                resulting_tier="TIER_1" if o5_triggered else None,
                rationale="Full-population scale customer data access." if o5_triggered else "Condition not met.",
            )
        )
        if o5_triggered and tier != "TIER_1":
            tier = "TIER_1"
            depth = "COMPREHENSIVE"
            override_applied = True

        return overrides, tier, depth, override_applied

    def generate_audit_trail(self, result: CriticalityResult) -> dict[str, Any]:
        """Generate full factor-level audit trail dictionary for a vendor."""
        return {
            "vendor_id": result.vendor_id,
            "vendor_name": result.vendor_name,
            "base_score": result.base_score,
            "score_status": result.score_status.value,
            "criticality_tier": result.criticality_tier,
            "assessment_depth": result.assessment_depth,
            "override_applied": result.override_applied,
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
