"""Level 9 — Deterministic AI Security Risk Rating Engine.

Implements the case-study brief equation:
    Risk Score = Criticality Multiplier (M_c) * sum(Exposure Severity (S_i) * Evidentiary Weight (W_i))

Enforces strict governance boundaries:
  - In production (`config/ai_risk_scoring.yaml`), undecided numerical parameters remain
    explicitly `"UNCONFIGURED"`, returning `SCORING_POLICY_UNCONFIGURED — findings available;
    final numerical rating not activated.` without inventing defaults or returning 0 / Low.
  - When a complete approved or test-only fixture policy is supplied, validates all ranges and
    monotone rating thresholds, deduplicates repeated exposure findings, resolves multi-source
    evidentiary weights, executes `M_c * sum(S_i * W_i)`, evaluates optional escalation rules,
    and returns the full calculation breakdown.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
import yaml

from meridian_assessment.models.ai_risk import (
    AIRiskFinding,
    DeterministicRiskScoreResult,
    ExposureDetermination,
    FindingScoreBreakdown,
    FindingSeverity,
    ScoringPolicyStatus,
    WeightedEvidenceItem,
)

_DEFAULT_SCORING_CONFIG_PATH = Path("config/ai_risk_scoring.yaml")

_RATING_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_unconfigured_val(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().upper() in ("UNCONFIGURED", "NONE", "TBD", ""):
        return True
    return False


class DeterministicRiskScoringPolicyError(ValueError):
    """Raised when a configured scoring policy contains contradictory or illegal numerical values."""


class DeterministicRiskScoringEngine:
    """Validates scoring policy configuration and executes `Risk Score = M_c * sum(S_i * W_i)`."""

    def __init__(
        self,
        policy_path: Path = _DEFAULT_SCORING_CONFIG_PATH,
        *,
        policy_override: Optional[dict[str, Any]] = None,
    ) -> None:
        self.policy_path = policy_path
        self.policy_data: dict[str, Any] = (
            policy_override if policy_override is not None else self._load_yaml(policy_path)
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def validate_policy(self) -> tuple[ScoringPolicyStatus, str, list[str]]:
        """Inspect policy metadata and parameters.

        Returns `(policy_status, policy_version, unconfigured_parameter_names)`.
        Raises `DeterministicRiskScoringPolicyError` if numeric parameters are provided but invalid/contradictory.
        """
        meta = self.policy_data.get("policy_metadata", {})
        version = str(meta.get("policy_version", "UNKNOWN"))
        raw_status = str(meta.get("policy_status", "SCORING_POLICY_UNCONFIGURED"))

        ev_cfg = self.policy_data.get("evidentiary_weighting", {})
        det_cfg = self.policy_data.get("deterministic_scoring", {})

        unconfigured: list[str] = []

        # 1. Criticality multipliers
        mults = det_cfg.get("criticality_multipliers")
        if not isinstance(mults, dict) or any(
            _is_unconfigured_val(mults.get(k)) for k in ("Critical", "High", "Medium", "Low")
        ):
            unconfigured.append("deterministic_scoring.criticality_multipliers")
        else:
            for k in ("Critical", "High", "Medium", "Low"):
                v = float(mults[k])
                if v <= 0:
                    raise DeterministicRiskScoringPolicyError(
                        f"Criticality multiplier for '{k}' must be > 0, got {v}."
                    )

        # 2. Exposure severity mapping
        sev_map = det_cfg.get("exposure_severity_mapping")
        if not isinstance(sev_map, dict) or any(
            _is_unconfigured_val(sev_map.get(k)) for k in ("Low", "Medium", "High", "Critical")
        ):
            unconfigured.append("deterministic_scoring.exposure_severity_mapping")
        else:
            vals = [float(sev_map[k]) for k in ("Low", "Medium", "High", "Critical")]
            if any(v <= 0 for v in vals) or not (vals[0] < vals[1] < vals[2] < vals[3]):
                raise DeterministicRiskScoringPolicyError(
                    f"Exposure severity mapping must be strictly positive and increasing (Low < Medium < High < Critical), got {sev_map}."
                )

        # 3. Exact scalar evidentiary weights
        scalar_weights = ev_cfg.get("exact_scalar_weights")
        if not isinstance(scalar_weights, dict) or _is_unconfigured_val(scalar_weights):
            unconfigured.append("evidentiary_weighting.exact_scalar_weights")
        else:
            for k, v in scalar_weights.items():
                if _is_unconfigured_val(v):
                    unconfigured.append(f"evidentiary_weighting.exact_scalar_weights.{k}")
                else:
                    fv = float(v)
                    if not (0.0 <= fv <= 1.0):
                        raise DeterministicRiskScoringPolicyError(
                            f"Evidentiary scalar weight '{k}' must be within [0.0, 1.0], got {fv}."
                        )

        # 4. Multi-source weight resolution method
        resolution = ev_cfg.get("multi_source_weight_resolution")
        if _is_unconfigured_val(resolution):
            unconfigured.append("evidentiary_weighting.multi_source_weight_resolution")
        elif str(resolution).upper() not in ("MAX", "MEAN", "PRIMARY"):
            raise DeterministicRiskScoringPolicyError(
                f"Unsupported multi_source_weight_resolution '{resolution}'; expected MAX, MEAN, or PRIMARY."
            )

        # 5. Rating thresholds
        thresh = det_cfg.get("rating_thresholds")
        if not isinstance(thresh, dict) or any(
            _is_unconfigured_val(thresh.get(k)) for k in ("Critical", "High", "Medium", "Low")
        ):
            unconfigured.append("deterministic_scoring.rating_thresholds")
        else:
            c_t = float(thresh["Critical"])
            h_t = float(thresh["High"])
            m_t = float(thresh["Medium"])
            l_t = float(thresh["Low"])
            if not (c_t > h_t > m_t >= l_t >= 0.0):
                raise DeterministicRiskScoringPolicyError(
                    f"Contradictory rating_thresholds: require Critical > High > Medium >= Low >= 0, got {thresh}."
                )

        # 6. Escalation rules & normalization/cap (optional or explicitly configured)
        esc_rules = det_cfg.get("minimum_rating_escalation_rules")
        if _is_unconfigured_val(esc_rules):
            unconfigured.append("deterministic_scoring.minimum_rating_escalation_rules")

        norm_cap = det_cfg.get("aggregation_normalization_or_cap")
        if _is_unconfigured_val(norm_cap):
            unconfigured.append("deterministic_scoring.aggregation_normalization_or_cap")

        # Required core calculation fields: 1 through 5
        core_unconfigured = [
            u
            for u in unconfigured
            if not u.endswith("minimum_rating_escalation_rules")
            and not u.endswith("aggregation_normalization_or_cap")
        ]
        if core_unconfigured:
            return ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED, version, unconfigured

        try:
            status_enum = ScoringPolicyStatus(raw_status)
        except ValueError:
            status_enum = ScoringPolicyStatus.PROVISIONAL_HUMAN_SELECTED
        return status_enum, version, unconfigured

    def calculate_vendor_risk_score(
        self,
        *,
        vendor_id: str,
        criticality_tier: str,
        findings: Sequence[AIRiskFinding],
        weighted_evidence: Sequence[WeightedEvidenceItem],
        assessment_completeness_status: str = "INCOMPLETE",
    ) -> DeterministicRiskScoreResult:
        """Calculate the deterministic AI security risk score or report `SCORING_POLICY_UNCONFIGURED`."""
        now_iso = _utc_now_iso()
        policy_status, policy_version, unconfigured_params = self.validate_policy()
        det_cfg = self.policy_data.get("deterministic_scoring", {})
        ev_cfg = self.policy_data.get("evidentiary_weighting", {})
        calc_method = str(
            self.policy_data.get("policy_metadata", {}).get("calculation_method", "M_c * sum(S_i * W_i)")
        )

        # Deduplicate findings by `dedup_exposure_key`
        seen_keys: dict[str, FindingScoreBreakdown] = {}
        breakdowns: list[FindingScoreBreakdown] = []
        duplicates_suppressed = 0

        # If policy is unconfigured, still build transparent finding breakdowns without inventing numbers
        if policy_status == ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED:
            for f in findings:
                is_dup = f.dedup_exposure_key in seen_keys
                if is_dup:
                    duplicates_suppressed += 1
                bd = FindingScoreBreakdown(
                    finding_id=f.finding_id,
                    risk_rule_id=f.risk_rule_id,
                    dedup_exposure_key=f.dedup_exposure_key,
                    severity_label=f.severity.value,
                    severity_numeric=None,
                    evidentiary_weight=None,
                    weighted_contribution=None,
                    source_evidence_ids=list(f.evidence_ids),
                    is_deduplicated_primary=not is_dup,
                    notes="Numerical scoring parameters are UNCONFIGURED; finding retained qualitatively.",
                )
                if not is_dup:
                    seen_keys[f.dedup_exposure_key] = bd
                breakdowns.append(bd)

            return DeterministicRiskScoreResult(
                vendor_id=vendor_id,
                policy_status=ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED,
                policy_version=policy_version,
                calculation_method=calc_method,
                status_message="SCORING_POLICY_UNCONFIGURED — findings available; final numerical rating not activated.",
                criticality_tier=criticality_tier,
                criticality_multiplier=None,
                finding_breakdowns=breakdowns,
                distinct_findings_scored=len(seen_keys),
                duplicates_suppressed=duplicates_suppressed,
                sum_weighted_exposures=None,
                raw_risk_score=None,
                final_risk_score=None,
                overall_risk_rating=None,
                escalation_rules_evaluated=[],
                escalation_applied=False,
                unconfigured_parameters=unconfigured_params,
                assessment_completeness_status=assessment_completeness_status,
                calculated_at=now_iso,
            )

        # Policy IS configured (e.g. test-only fixture or future approved production policy)
        mults: dict[str, float] = {k: float(v) for k, v in det_cfg["criticality_multipliers"].items()}
        sev_map: dict[str, float] = {k: float(v) for k, v in det_cfg["exposure_severity_mapping"].items()}
        scalar_weights: dict[str, float] = {
            k: float(v) for k, v in ev_cfg["exact_scalar_weights"].items() if not _is_unconfigured_val(v)
        }
        resolution_mode = str(ev_cfg["multi_source_weight_resolution"]).upper()
        thresh: dict[str, float] = {k: float(v) for k, v in det_cfg["rating_thresholds"].items()}

        m_c = mults.get(criticality_tier, mults.get("Low", 1.0))
        weight_by_ev_id = {w.evidence_id: w for w in weighted_evidence}

        # Group findings by dedup_exposure_key so duplicate representations merge evidence without double-counting
        grouped_findings: dict[str, list[AIRiskFinding]] = {}
        for f in findings:
            grouped_findings.setdefault(f.dedup_exposure_key, []).append(f)

        sum_s_w = 0.0
        distinct_scored = 0

        for dedup_key, group in grouped_findings.items():
            # Select highest-severity finding in the group as primary
            group_sorted = sorted(
                group,
                key=lambda item: sev_map.get(item.severity.value, 0.0),
                reverse=True,
            )
            primary = group_sorted[0]
            all_ev_ids: list[str] = []
            for gf in group:
                for eid in gf.evidence_ids:
                    if eid not in all_ev_ids:
                        all_ev_ids.append(eid)

            for idx, gf in enumerate(group_sorted):
                if idx > 0:
                    duplicates_suppressed += 1
                    breakdowns.append(
                        FindingScoreBreakdown(
                            finding_id=gf.finding_id,
                            risk_rule_id=gf.risk_rule_id,
                            dedup_exposure_key=dedup_key,
                            severity_label=gf.severity.value,
                            severity_numeric=0.0,
                            evidentiary_weight=0.0,
                            weighted_contribution=0.0,
                            source_evidence_ids=list(gf.evidence_ids),
                            is_deduplicated_primary=False,
                            notes=f"Duplicate exposure merged into primary finding {primary.finding_id}.",
                        )
                    )
                    continue

                if primary.exposure_determination == ExposureDetermination.MITIGATED_BY_EVIDENCE:
                    breakdowns.append(
                        FindingScoreBreakdown(
                            finding_id=primary.finding_id,
                            risk_rule_id=primary.risk_rule_id,
                            dedup_exposure_key=dedup_key,
                            severity_label=primary.severity.value,
                            severity_numeric=0.0,
                            evidentiary_weight=0.0,
                            weighted_contribution=0.0,
                            source_evidence_ids=all_ev_ids,
                            is_deduplicated_primary=True,
                            notes="Exposure mitigated by verified contractual/technical evidence (0 risk points).",
                        )
                    )
                    continue

                s_i = sev_map.get(primary.severity.value, 0.0)
                w_i = self._resolve_evidentiary_weight(
                    all_ev_ids,
                    weight_by_ev_id=weight_by_ev_id,
                    scalar_weights=scalar_weights,
                    resolution_mode=resolution_mode,
                )
                contrib = round(s_i * w_i, 6)
                sum_s_w += contrib
                distinct_scored += 1

                breakdowns.append(
                    FindingScoreBreakdown(
                        finding_id=primary.finding_id,
                        risk_rule_id=primary.risk_rule_id,
                        dedup_exposure_key=dedup_key,
                        severity_label=primary.severity.value,
                        severity_numeric=s_i,
                        evidentiary_weight=round(w_i, 4),
                        weighted_contribution=round(contrib, 4),
                        source_evidence_ids=all_ev_ids,
                        is_deduplicated_primary=True,
                        notes=f"S_i={s_i} * W_i={w_i:.4f} (resolution={resolution_mode}).",
                    )
                )

        raw_score = round(m_c * sum_s_w, 4)
        final_score = raw_score

        norm_cap = det_cfg.get("aggregation_normalization_or_cap")
        if isinstance(norm_cap, dict) and norm_cap.get("max_score") is not None:
            final_score = min(final_score, float(norm_cap["max_score"]))

        overall_rating = self._map_score_to_rating(final_score, thresh)

        # Evaluate optional configured minimum-rating escalation rules
        esc_evaluated: list[dict[str, Any]] = []
        esc_applied = False
        esc_rules = det_cfg.get("minimum_rating_escalation_rules")
        if isinstance(esc_rules, list):
            for rule in esc_rules:
                if not isinstance(rule, dict):
                    continue
                r_id = str(rule.get("rule_id", "ESC_RULE"))
                target_risk = str(rule.get("risk_rule_id", ""))
                req_det = str(rule.get("exposure_determination", "CONFIRMED_EXPOSURE"))
                min_rating = str(rule.get("minimum_rating", "High"))

                matched = any(
                    f.risk_rule_id == target_risk and f.exposure_determination.value == req_det
                    for f in findings
                )
                esc_evaluated.append({
                    "rule_id": r_id,
                    "risk_rule_id": target_risk,
                    "exposure_determination": req_det,
                    "minimum_rating": min_rating,
                    "triggered": matched,
                })
                if matched and _RATING_ORDER.get(min_rating, 0) > _RATING_ORDER.get(overall_rating, 0):
                    overall_rating = min_rating
                    esc_applied = True

        return DeterministicRiskScoreResult(
            vendor_id=vendor_id,
            policy_status=policy_status,
            policy_version=policy_version,
            calculation_method=calc_method,
            status_message=f"Deterministic score calculated under policy '{policy_version}' ({policy_status.value}).",
            criticality_tier=criticality_tier,
            criticality_multiplier=m_c,
            finding_breakdowns=breakdowns,
            distinct_findings_scored=distinct_scored,
            duplicates_suppressed=duplicates_suppressed,
            sum_weighted_exposures=round(sum_s_w, 4),
            raw_risk_score=raw_score,
            final_risk_score=final_score,
            overall_risk_rating=overall_rating,
            escalation_rules_evaluated=esc_evaluated,
            escalation_applied=esc_applied,
            unconfigured_parameters=unconfigured_params,
            assessment_completeness_status=assessment_completeness_status,
            calculated_at=now_iso,
        )

    @staticmethod
    def _resolve_evidentiary_weight(
        evidence_ids: Sequence[str],
        *,
        weight_by_ev_id: dict[str, WeightedEvidenceItem],
        scalar_weights: dict[str, float],
        resolution_mode: str,
    ) -> float:
        weights: list[float] = []
        for eid in evidence_ids:
            w_item = weight_by_ev_id.get(eid)
            if w_item is None:
                continue
            if w_item.configured_scalar_weight is not None:
                weights.append(float(w_item.configured_scalar_weight))
                continue
            key_specific = f"{w_item.source_reliability_grade}_{w_item.evidence_strength.value}"
            val = (
                scalar_weights.get(key_specific)
                or scalar_weights.get(w_item.evidence_strength.value)
                or scalar_weights.get(w_item.weight_band.value)
            )
            if val is not None:
                weights.append(float(val))

        if not weights:
            return float(scalar_weights.get("DEFAULT", 0.5))
        if resolution_mode == "MAX":
            return max(weights)
        if resolution_mode == "MEAN":
            return sum(weights) / len(weights)
        return weights[0]

    @staticmethod
    def _map_score_to_rating(score: float, thresholds: dict[str, float]) -> str:
        if score >= thresholds["Critical"]:
            return "Critical"
        if score >= thresholds["High"]:
            return "High"
        if score >= thresholds["Medium"]:
            return "Medium"
        return "Low"
