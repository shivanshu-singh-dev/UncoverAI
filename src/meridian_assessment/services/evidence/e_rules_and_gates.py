"""Level 7 — Complete E1–E14 Evidentiary Rule Engine, Evidentiary Weighting, G1–G4 Gates, and AI-Use Verdict.

Preserves strict separation between:
  1. Source reliability (`source_reliability_grade`: A, B, C, D from S1–S12 hierarchy)
  2. Evidence strength (`evidence_strength`: CONFIRMED, STRONG, INDICATIVE, MARKETING,
     DISCARDED_HOSTILE, NONE, REQUIRES_REVIEW from rules E1–E14)
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Optional, Sequence

from meridian_assessment.models.ai_risk import (
    AIType,
    AIUseVerdictResult,
    AIUseVerdictValue,
    ERuleApplication,
    EvaluatedGateResult,
    EvidenceRecord,
    EvidenceScope,
    EvidenceStrength,
    EvidentiaryWeightBand,
    SignalCategory,
    SignalClassificationResult,
    WeightedEvidenceItem,
)
from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CoverageState,
    GateStatus,
    OSINTInvestigationPlan,
)
from meridian_assessment.services.classification.signal_classifier import detect_prompt_injection


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_claim_text(text: str) -> str:
    """Normalize excerpt text to detect verbatim duplicate claims copied across sources (Rule E9)."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_older_than_24_months(
    published_date_str: Optional[str],
    reference_dt: Optional[datetime] = None,
) -> bool:
    """Return True if `published_date_str` is more than 24 months (730 days) before `reference_dt`."""
    if not published_date_str:
        return False
    ref = reference_dt or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(published_date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (ref - dt).days > 730
    except Exception:
        return False


def _downgrade_one_level(strength: EvidenceStrength) -> EvidenceStrength:
    """Downgrade an evidence strength by one tier under Rule E10 (historical / removed source)."""
    mapping = {
        EvidenceStrength.CONFIRMED: EvidenceStrength.STRONG,
        EvidenceStrength.STRONG: EvidenceStrength.INDICATIVE,
        EvidenceStrength.INDICATIVE: EvidenceStrength.MARKETING,
        EvidenceStrength.MARKETING: EvidenceStrength.MARKETING,
        EvidenceStrength.NONE: EvidenceStrength.NONE,
        EvidenceStrength.DISCARDED_HOSTILE: EvidenceStrength.DISCARDED_HOSTILE,
        EvidenceStrength.REQUIRES_REVIEW: EvidenceStrength.REQUIRES_REVIEW,
    }
    return mapping.get(strength, strength)


# ─── E1–E14 Evidentiary Rule Engine & Weighting Service ───────────────────────


class ERuleEngineAndWeighter:
    """Applies all 14 rulebook evidentiary rules (E1–E14) and assigns evidentiary weight bands."""

    def __init__(
        self,
        scalar_weight_policy: Optional[dict[str, float]] = None,
    ) -> None:
        # In production, scalar_weight_policy is None ("UNCONFIGURED_BAND_ONLY") unless explicitly supplied
        self.scalar_weight_policy = scalar_weight_policy

    def evaluate_evidence_batch(
        self,
        evidence_records: Sequence[EvidenceRecord],
        classifications: Optional[Sequence[SignalClassificationResult]] = None,
        *,
        reference_dt: Optional[datetime] = None,
    ) -> list[WeightedEvidenceItem]:
        """Apply single-item rules (E1–E8, E10–E14) followed by cross-source corroboration (E9)."""
        cls_by_ev_id: dict[str, SignalClassificationResult] = {}
        if classifications:
            for c in classifications:
                cls_by_ev_id[c.evidence_id] = c

        items: list[WeightedEvidenceItem] = []
        for ev in evidence_records:
            cls_res = cls_by_ev_id.get(ev.evidence_id)
            item = self._evaluate_single_item(ev, cls_res, reference_dt=reference_dt)
            items.append(item)

        # Apply E9 (Independent corroboration across distinct source classes with non-duplicate text)
        self._apply_e9_corroboration(evidence_records, items, reference_dt=reference_dt)

        # Finalize weight bands and synchronize back onto EvidenceRecord objects
        for ev, item in zip(evidence_records, items):
            self._assign_weight_band(item)
            ev.evidence_strength = item.evidence_strength
            ev.prior_strength_before_downgrade = item.prior_strength_before_downgrade
            ev.e_rule_ids = list(item.fired_e_rule_ids)

        return items

    def _evaluate_single_item(
        self,
        ev: EvidenceRecord,
        cls_res: Optional[SignalClassificationResult],
        *,
        reference_dt: Optional[datetime] = None,
    ) -> WeightedEvidenceItem:
        text = ev.verbatim_excerpt or ""
        low = text.lower()
        src = (ev.source_class or "S2").upper()
        applied: list[ERuleApplication] = []
        fired_ids: list[str] = []

        def record_rule(rule_id: str, name: str, triggered: bool, strength: Optional[EvidenceStrength], reason: str) -> None:
            applied.append(
                ERuleApplication(
                    rule_id=rule_id,
                    rule_name=name,
                    triggered=triggered,
                    resulting_strength=strength if triggered else None,
                    rationale=reason,
                )
            )
            if triggered and rule_id not in fired_ids:
                fired_ids.append(rule_id)

        # ── E14: Hostile webpage instructions (highest priority) ──
        injected, sec_events = detect_prompt_injection(text)
        if ev.hostile_instruction_detected or injected:
            ev.hostile_instruction_detected = True
            for se in sec_events:
                if se not in ev.security_events:
                    ev.security_events.append(se)
            record_rule(
                "E14",
                "Hostile webpage instructions",
                True,
                EvidenceStrength.DISCARDED_HOSTILE,
                "Untrusted source text attempted prompt injection or rule override; isolated and discarded.",
            )
            return WeightedEvidenceItem(
                evidence_id=ev.evidence_id,
                vendor_id=ev.vendor_id,
                source_class=src,
                source_reliability_grade=ev.source_reliability_grade,
                source_reliability_rating=ev.source_reliability_rating,
                evidence_strength=EvidenceStrength.DISCARDED_HOSTILE,
                applied_e_rules=applied,
                fired_e_rule_ids=fired_ids,
                disposition_explanation="Discarded under Rule E14 due to hostile prompt-injection instructions.",
            )
        record_rule("E14", "Hostile webpage instructions", False, None, "No hostile instructions detected.")

        # ── E6: Automation distinction ──
        is_non_ai_automation = (
            (cls_res and cls_res.signal_category == SignalCategory.NON_AI_AUTOMATION)
            or ev.ai_type == AIType.NONE_AUTOMATION_ONLY
            or (
                any(
                    k in low
                    for k in ("batch scheduling", "batch job", "job orchestration", "rules engine", "rpa", "robotic process automation")
                )
                and not any(k in low for k in ("llm", "machine learning", "neural", "openai", "bedrock", "anthropic", "generative ai"))
            )
        )
        if is_non_ai_automation:
            ev.ai_type = AIType.NONE_AUTOMATION_ONLY
            record_rule(
                "E6",
                "Automation distinction",
                True,
                EvidenceStrength.NONE,
                "Ordinary workflow, batch scheduler, rules engine, or RPA without ML/LLM evidence is classified as non-AI.",
            )
            return WeightedEvidenceItem(
                evidence_id=ev.evidence_id,
                vendor_id=ev.vendor_id,
                source_class=src,
                source_reliability_grade=ev.source_reliability_grade,
                source_reliability_rating=ev.source_reliability_rating,
                evidence_strength=EvidenceStrength.NONE,
                applied_e_rules=applied,
                fired_e_rule_ids=fired_ids,
                disposition_explanation="Non-AI automation under Rule E6 (AI type = NONE_AUTOMATION_ONLY).",
            )
        record_rule("E6", "Automation distinction", False, None, "Not ordinary non-AI automation.")

        # Starting baseline strength
        current_strength: EvidenceStrength = EvidenceStrength.REQUIRES_REVIEW
        explanations: list[str] = []

        # ── E8: Patent / research paper (S10) ──
        is_patent_or_research = src == "S10" or any(
            k in low for k in ("patent application", "u.s. patent", "arxiv preprint", "research paper", "r&d prototype")
        )
        if is_patent_or_research:
            current_strength = EvidenceStrength.INDICATIVE
            explanations.append("Patent or R&D publication is INDICATIVE only (Rule E8), not proof of production deployment.")
            record_rule("E8", "Patent/research capability", True, EvidenceStrength.INDICATIVE, explanations[-1])
        else:
            record_rule("E8", "Patent/research capability", False, None, "Not a patent or research paper.")

        # ── E7: Partnership announcement without technical integration details ──
        is_partnership_only = (
            ("partner" in low or "partnership" in low or "alliance" in low or "announces collaboration" in low)
            and not any(
                k in low
                for k in ("api endpoint", "subprocessor register", "model input", "inference pipeline", "integrated into")
            )
        )
        if is_partnership_only:
            current_strength = EvidenceStrength.INDICATIVE
            explanations.append(
                "Partnership announcement without concrete technical integration details is capped at INDICATIVE and cannot establish G4 alone (Rule E7)."
            )
            record_rule("E7", "Partnership announcement", True, EvidenceStrength.INDICATIVE, explanations[-1])
        else:
            record_rule("E7", "Partnership announcement", False, None, "Not an unsubstantiated partnership announcement.")

        # ── E11: Third-party reporting (S11) ──
        if src == "S11":
            quotes_primary = any(
                k in low
                for k in (
                    "according to the vendor's dpa",
                    "quoted in official documentation",
                    "sec filing states",
                    "official subprocessor register lists",
                    "verified primary disclosure",
                )
            )
            if quotes_primary:
                current_strength = EvidenceStrength.STRONG
                explanations.append("Third-party report quotes verifiable vendor primary source -> STRONG (Rule E11).")
                record_rule("E11", "Third-party reporting", True, EvidenceStrength.STRONG, explanations[-1])
            else:
                current_strength = EvidenceStrength.INDICATIVE
                explanations.append("Third-party reporting without quoted primary vendor verification is INDICATIVE (Rule E11).")
                record_rule("E11", "Third-party reporting", True, EvidenceStrength.INDICATIVE, explanations[-1])
        else:
            record_rule("E11", "Third-party reporting", False, None, "Source is not S11 third-party reporting.")

        # ── E3: Technical hiring (S6) ──
        if src == "S6":
            has_prod_stack = any(
                k in low
                for k in (
                    "openai",
                    "anthropic",
                    "bedrock",
                    "vertex",
                    "langchain",
                    "rag pipeline",
                    "production llm",
                    "inference service",
                    "pytorch",
                    "fine-tuning pipeline",
                )
            ) and ev.scope in (EvidenceScope.MERIDIAN_SERVICE, EvidenceScope.VENDOR_PRODUCT)
            if has_prod_stack:
                current_strength = EvidenceStrength.STRONG
                explanations.append("Production technical job posting naming AI stack/provider tied to product -> STRONG (Rule E3).")
                record_rule("E3", "Technical hiring", True, EvidenceStrength.STRONG, explanations[-1])
            else:
                current_strength = EvidenceStrength.INDICATIVE
                explanations.append("Generic or research AI job posting without specific product stack -> INDICATIVE (Rule E3).")
                record_rule("E3", "Technical hiring", True, EvidenceStrength.INDICATIVE, explanations[-1])
        else:
            record_rule("E3", "Technical hiring", False, None, "Source is not S6 technical hiring.")

        # ── E4: Executive / IR statement (S7) ──
        if src == "S7":
            has_specific_product_metric = any(
                k in low
                for k in ("deployed in", "processes", "million", "workflow", "production", "module", "platform")
            ) and not (cls_res and cls_res.signal_category == SignalCategory.MARKETING_COMMENTARY)
            if has_specific_product_metric and ev.ai_type in (AIType.GENAI_LLM, AIType.CLASSICAL_ML, AIType.AGENTIC_AI):
                current_strength = EvidenceStrength.STRONG
                explanations.append("Executive/IR statement naming specific product, workflow, or scale -> STRONG (Rule E4).")
                record_rule("E4", "Executive/IR statement", True, EvidenceStrength.STRONG, explanations[-1])
            else:
                current_strength = EvidenceStrength.MARKETING
                explanations.append("Vague executive/IR promotional statement without specific workflow/outcome -> MARKETING (Rule E4).")
                record_rule("E4", "Executive/IR statement", True, EvidenceStrength.MARKETING, explanations[-1])
        else:
            record_rule("E4", "Executive/IR statement", False, None, "Source is not S7 executive/IR statement.")

        # ── E1: Named model / provider / API ──
        has_named_model_or_provider = bool(ev.named_model or ev.named_provider or "api endpoint" in low)
        if (
            has_named_model_or_provider
            and src in ("S1", "S2", "S3", "S4", "S5", "S8")
            and not is_partnership_only
            and not is_patent_or_research
        ):
            if ev.scope == EvidenceScope.MERIDIAN_SERVICE and src in ("S1", "S2", "S3", "S5"):
                current_strength = EvidenceStrength.CONFIRMED
                explanations.append(
                    f"Named AI model/provider ({ev.named_model or ev.named_provider or 'API'}) in authoritative source {src} "
                    "within Meridian service scope -> CONFIRMED (Rule E1)."
                )
            else:
                # Confirms vendor AI use & records provider, but scope is broader than Meridian service
                current_strength = (
                    EvidenceStrength.CONFIRMED
                    if src in ("S1", "S2", "S3")
                    else EvidenceStrength.STRONG
                )
                explanations.append(
                    f"Named AI model/provider ({ev.named_model or ev.named_provider or 'API'}) in {src} "
                    f"(scope={ev.scope.value}); records provider under E1 without auto-confirming G2 service scope."
                )
            record_rule("E1", "Named model/provider/API", True, current_strength, explanations[-1])
        else:
            record_rule("E1", "Named model/provider/API", False, None, "No explicit model/provider/API disclosure in authoritative doc.")

        # ── E2: Specific AI functionality (inputs, outputs, or settings) ──
        has_concrete_io = any(
            k in low
            for k in (
                "input",
                "output",
                "inference",
                "embedding",
                "rag",
                "classification",
                "anomaly detection",
                "summarization",
                "prompt",
                "completion",
            )
        ) and ev.ai_type in (AIType.GENAI_LLM, AIType.CLASSICAL_ML, AIType.AGENTIC_AI)
        if has_concrete_io and not is_patent_or_research and not is_partnership_only and src not in ("S6", "S7", "S11"):
            if ev.scope == EvidenceScope.MERIDIAN_SERVICE and src in ("S1", "S2", "S3", "S4", "S5"):
                current_strength = EvidenceStrength.CONFIRMED
                explanations.append("Concrete AI functionality with inputs/outputs in Meridian service scope -> CONFIRMED (Rule E2).")
                record_rule("E2", "Specific AI functionality", True, EvidenceStrength.CONFIRMED, explanations[-1])
            elif current_strength not in (EvidenceStrength.CONFIRMED, EvidenceStrength.STRONG):
                current_strength = EvidenceStrength.INDICATIVE
                explanations.append("Concrete AI functionality described, but product/Meridian scope is unclear -> INDICATIVE (Rule E2).")
                record_rule("E2", "Specific AI functionality", True, EvidenceStrength.INDICATIVE, explanations[-1])
            else:
                record_rule("E2", "Specific AI functionality", True, current_strength, "Concrete AI functionality present.")
        else:
            record_rule("E2", "Specific AI functionality", False, None, "No concrete operational I/O functionality matched under E2.")

        # ── E5: Marketing-only claim ──
        is_marketing_only = (
            (cls_res and cls_res.signal_category == SignalCategory.MARKETING_COMMENTARY)
            or (src == "S12" and current_strength == EvidenceStrength.REQUIRES_REVIEW)
        )
        if is_marketing_only and current_strength not in (EvidenceStrength.CONFIRMED, EvidenceStrength.STRONG, EvidenceStrength.INDICATIVE):
            current_strength = EvidenceStrength.MARKETING
            explanations.append("Broad promotional AI claim without technical mechanism -> MARKETING (Rule E5).")
            record_rule("E5", "Marketing-only claim", True, EvidenceStrength.MARKETING, explanations[-1])
        else:
            record_rule("E5", "Marketing-only claim", False, None, "Not a marketing-only claim.")

        # ── E12: Opt-in / off-by-default feature ──
        if ev.is_opt_in_only or any(
            k in low for k in ("opt-in", "opt in", "disabled by default", "off by default", "optional add-on")
        ):
            ev.is_opt_in_only = True
            ev.activation_caveats = ev.activation_caveats or "AI feature is opt-in / disabled by default; availability does not prove activation for Meridian (Rule E12)."
            explanations.append(ev.activation_caveats)
            record_rule("E12", "Opt-in/off-by-default feature", True, current_strength, ev.activation_caveats)
        else:
            record_rule("E12", "Opt-in/off-by-default feature", False, None, "No opt-in/off-by-default caveat detected.")

        # ── E13: No-training / ZDR assertion ──
        if ev.no_training_commitment or ev.zdr_commitment:
            if src in ("S1", "S3", "S5"):
                if current_strength in (EvidenceStrength.REQUIRES_REVIEW, EvidenceStrength.INDICATIVE, EvidenceStrength.MARKETING):
                    current_strength = EvidenceStrength.STRONG
                e13_msg = f"Contract-grade ({src}) no-training / ZDR commitment is STRONG (Rule E13)."
                explanations.append(e13_msg)
                record_rule("E13", "No-training/ZDR assertion", True, EvidenceStrength.STRONG, e13_msg)
            else:
                if current_strength == EvidenceStrength.REQUIRES_REVIEW:
                    current_strength = EvidenceStrength.INDICATIVE
                e13_msg = f"Non-contractual / marketing ({src}) no-training claim is INDICATIVE only (Rule E13)."
                explanations.append(e13_msg)
                record_rule("E13", "No-training/ZDR assertion", True, EvidenceStrength.INDICATIVE, e13_msg)
        else:
            record_rule("E13", "No-training/ZDR assertion", False, None, "No explicit no-training or ZDR commitment.")

        # Handle non-AI informational excerpts cleanly
        if current_strength == EvidenceStrength.REQUIRES_REVIEW and ev.ai_type == AIType.NONE:
            current_strength = EvidenceStrength.NONE
            explanations.append("No AI signal present in excerpt.")

        # ── E10: Old (>24 months) or removed/archived evidence downgrade ──
        is_archived_or_old = (
            ev.is_historical_or_removed
            or "web.archive.org" in (ev.original_url or "").lower()
            or _is_older_than_24_months(ev.published_or_observed_date, reference_dt=reference_dt)
            or any(k in low for k in ("deprecated in 2022", "removed from product", "archived page"))
        )
        prior_strength: Optional[EvidenceStrength] = None
        if is_archived_or_old and current_strength in (
            EvidenceStrength.CONFIRMED,
            EvidenceStrength.STRONG,
            EvidenceStrength.INDICATIVE,
        ):
            ev.is_historical_or_removed = True
            prior_strength = current_strength
            current_strength = _downgrade_one_level(current_strength)
            e10_msg = (
                f"Evidence is >24 months old or from a removed/archived source; "
                f"downgraded one level from {prior_strength.value} to {current_strength.value} (Rule E10)."
            )
            explanations.append(e10_msg)
            record_rule("E10", "Old or removed evidence", True, current_strength, e10_msg)
        else:
            record_rule("E10", "Old or removed evidence", False, None, "Evidence is current (<24 months) and active.")

        return WeightedEvidenceItem(
            evidence_id=ev.evidence_id,
            vendor_id=ev.vendor_id,
            source_class=src,
            source_reliability_grade=ev.source_reliability_grade,
            source_reliability_rating=ev.source_reliability_rating,
            evidence_strength=current_strength,
            prior_strength_before_downgrade=prior_strength,
            applied_e_rules=applied,
            fired_e_rule_ids=fired_ids,
            disposition_explanation=" ".join(explanations) or f"Classified as {current_strength.value}.",
        )

    @staticmethod
    def _apply_e9_corroboration(
        evidence_records: Sequence[EvidenceRecord],
        items: list[WeightedEvidenceItem],
        *,
        reference_dt: Optional[datetime] = None,
    ) -> None:
        """Apply Rule E9: Two independent source classes supporting the same indicative operational claim upgrade it to STRONG.

        Duplicate claims copied verbatim across sources do NOT count as independent corroboration.
        Also respects E10 if an item was already downgraded for age/removal.
        """
        # Identify operational INDICATIVE items that were NOT downgraded from STRONG by E10
        indicative_indices: list[int] = []
        for idx, (ev, item) in enumerate(zip(evidence_records, items)):
            if (
                item.evidence_strength == EvidenceStrength.INDICATIVE
                and ev.ai_type in (AIType.GENAI_LLM, AIType.CLASSICAL_ML, AIType.AGENTIC_AI)
                and not ev.is_historical_or_removed
            ):
                indicative_indices.append(idx)

        for i in indicative_indices:
            ev_i = evidence_records[i]
            item_i = items[i]
            norm_i = _normalize_claim_text(ev_i.verbatim_excerpt)
            corroborating_ids: list[str] = []
            corroborating_sources: set[str] = {item_i.source_class}

            for j in indicative_indices:
                if i == j:
                    continue
                ev_j = evidence_records[j]
                item_j = items[j]
                # Must be a distinct source class
                if item_j.source_class == item_i.source_class:
                    continue
                # Must not be a duplicate copy of the exact same text/hash
                norm_j = _normalize_claim_text(ev_j.verbatim_excerpt)
                if ev_i.content_sha256 == ev_j.content_sha256 or norm_i == norm_j:
                    continue
                # Must support the same AI type or provider/scope
                if ev_i.ai_type == ev_j.ai_type:
                    corroborating_ids.append(ev_j.evidence_id)
                    corroborating_sources.add(item_j.source_class)

            if len(corroborating_sources) >= 2:
                item_i.evidence_strength = EvidenceStrength.STRONG
                item_i.corroborated_by_evidence_ids = corroborating_ids
                item_i.corroborating_source_classes = sorted(list(corroborating_sources))
                e9_msg = (
                    f"Upgraded from INDICATIVE to STRONG via independent corroboration across "
                    f"source classes {', '.join(item_i.corroborating_source_classes)} (Rule E9)."
                )
                item_i.applied_e_rules.append(
                    ERuleApplication(
                        rule_id="E9",
                        rule_name="Independent corroboration",
                        triggered=True,
                        resulting_strength=EvidenceStrength.STRONG,
                        rationale=e9_msg,
                    )
                )
                if "E9" not in item_i.fired_e_rule_ids:
                    item_i.fired_e_rule_ids.append("E9")
                item_i.disposition_explanation = f"{item_i.disposition_explanation} {e9_msg}".strip()
            else:
                item_i.applied_e_rules.append(
                    ERuleApplication(
                        rule_id="E9",
                        rule_name="Independent corroboration",
                        triggered=False,
                        resulting_strength=None,
                        rationale="No independent second source class with non-duplicate text corroborated this claim.",
                    )
                )

    def _assign_weight_band(self, item: WeightedEvidenceItem) -> None:
        """Assign broad weight band (`HIGH`, `MEDIUM`, `LOW`, `NONE`) while keeping production scalar weight unconfigured unless supplied."""
        strength = item.evidence_strength
        grade = (item.source_reliability_grade or "C").upper()

        if strength in (EvidenceStrength.DISCARDED_HOSTILE, EvidenceStrength.NONE, EvidenceStrength.MARKETING):
            item.weight_band = EvidentiaryWeightBand.NONE
            item.weight_band_range = (0.0, 0.0)
        elif strength == EvidenceStrength.CONFIRMED or (strength == EvidenceStrength.STRONG and grade in ("A", "B")):
            item.weight_band = EvidentiaryWeightBand.HIGH
            item.weight_band_range = (0.80, 1.00)
        elif strength == EvidenceStrength.STRONG or (strength == EvidenceStrength.INDICATIVE and grade in ("A", "B")):
            item.weight_band = EvidentiaryWeightBand.MEDIUM
            item.weight_band_range = (0.50, 0.70)
        else:
            item.weight_band = EvidentiaryWeightBand.LOW
            item.weight_band_range = (0.10, 0.40)

        if self.scalar_weight_policy is not None:
            key_specific = f"{grade}_{strength.value}"
            scalar = self.scalar_weight_policy.get(key_specific)
            if scalar is None:
                scalar = self.scalar_weight_policy.get(strength.value)
            if scalar is None:
                scalar = self.scalar_weight_policy.get(item.weight_band.value)
            if scalar is not None:
                item.configured_scalar_weight = float(scalar)
                item.weight_status = "CONFIGURED_BY_POLICY"
                return

        item.configured_scalar_weight = None
        item.weight_status = "UNCONFIGURED_BAND_ONLY"


# ─── G1–G4 Gate Evaluator & AI-Use Verdict Engine ─────────────────────────────


class GateAndVerdictEvaluator:
    """Evaluates G1–G4 gates (with mandatory E1/G2 & E7/G4 safeguards) and computes the AI-Use Verdict."""

    @staticmethod
    def evaluate_gates_and_verdict(
        plan: OSINTInvestigationPlan,
        evidence_records: Sequence[EvidenceRecord],
        weighted_items: Sequence[WeightedEvidenceItem],
        *,
        analyst_gate_overrides: Optional[dict[str, tuple[str, str]]] = None,
    ) -> tuple[list[EvaluatedGateResult], AIUseVerdictResult]:
        """Evaluate G1–G4 and produce the rulebook-compliant AI-use verdict."""
        now_iso = _utc_now_iso()
        weight_by_id = {w.evidence_id: w for w in weighted_items}

        # Filter out hostile discarded evidence (E14)
        valid_evidence = [
            ev
            for ev in evidence_records
            if weight_by_id.get(ev.evidence_id, None) is None
            or weight_by_id[ev.evidence_id].evidence_strength != EvidenceStrength.DISCARDED_HOSTILE
        ]

        # Check whether the defined assessment depth has genuinely been completed
        depth_completed, incomplete_reasons = GateAndVerdictEvaluator._check_depth_completeness(
            plan, valid_evidence
        )

        # ── G1: AI Operational Use (by vendor) ──
        op_confirmed = [
            ev
            for ev in valid_evidence
            if ev.ai_type in (AIType.GENAI_LLM, AIType.CLASSICAL_ML, AIType.AGENTIC_AI)
            and ev.evidence_strength == EvidenceStrength.CONFIRMED
        ]
        op_strong = [
            ev
            for ev in valid_evidence
            if ev.ai_type in (AIType.GENAI_LLM, AIType.CLASSICAL_ML, AIType.AGENTIC_AI)
            and ev.evidence_strength == EvidenceStrength.STRONG
        ]
        op_indicative = [
            ev
            for ev in valid_evidence
            if ev.ai_type in (AIType.GENAI_LLM, AIType.CLASSICAL_ML, AIType.AGENTIC_AI)
            and ev.evidence_strength == EvidenceStrength.INDICATIVE
        ]
        strong_sources_all = {ev.source_class for ev in op_strong}

        if op_confirmed or len(strong_sources_all) >= 2:
            g1_status = GateStatus.QUALIFYING_POSITIVE_EVIDENCE.value
            g1_ev_ids = [ev.evidence_id for ev in (op_confirmed + op_strong)]
            g1_rat = (
                f"Confirmed operational AI use by vendor ({len(op_confirmed)} CONFIRMED, "
                f"{len(op_strong)} STRONG across source classes {sorted(list(strong_sources_all))})."
            )
            g1_open: list[str] = []
        elif op_strong or op_indicative:
            g1_status = GateStatus.UNRESOLVED.value
            g1_ev_ids = [ev.evidence_id for ev in (op_strong + op_indicative)]
            g1_rat = "Indicative or uncorroborated operational AI signals found; insufficient to conclusively resolve G1."
            g1_open = ["Corroborate operational AI use across a second independent authoritative source class."]
        elif depth_completed:
            g1_status = GateStatus.QUALIFYING_NEGATIVE_EVIDENCE.value
            g1_ev_ids = [ev.evidence_id for ev in valid_evidence]
            g1_rat = "Defined OSINT depth completed with only non-AI automation or marketing claims; no operational AI identified."
            g1_open = []
        else:
            g1_status = GateStatus.NOT_INVESTIGATED.value if not valid_evidence else GateStatus.UNRESOLVED.value
            g1_ev_ids = [ev.evidence_id for ev in valid_evidence]
            g1_rat = "Required OSINT depth not yet completed; cannot conclude absence of operational AI."
            g1_open = list(incomplete_reasons)

        # ── G2: Service Scope Relevance (MANDATORY E1/G2 SAFEGUARD) ──
        # Only evidence explicitly in `MERIDIAN_SERVICE` scope can confirm G2!
        in_scope_confirmed = [
            ev for ev in op_confirmed if ev.scope == EvidenceScope.MERIDIAN_SERVICE
        ]
        in_scope_strong = [
            ev for ev in op_strong if ev.scope == EvidenceScope.MERIDIAN_SERVICE
        ]
        in_scope_strong_sources = {ev.source_class for ev in in_scope_strong}
        out_of_scope_or_broad_op = [
            ev
            for ev in (op_confirmed + op_strong + op_indicative)
            if ev.scope != EvidenceScope.MERIDIAN_SERVICE
        ]

        if in_scope_confirmed or len(in_scope_strong_sources) >= 2:
            g2_status = GateStatus.QUALIFYING_POSITIVE_EVIDENCE.value
            g2_ev_ids = [ev.evidence_id for ev in (in_scope_confirmed + in_scope_strong)]
            g2_rat = (
                f"Operational AI is directly linked to Meridian's contracted service "
                f"('{plan.service_product_provided or 'purchased service'}')."
            )
            g2_open: list[str] = []
        elif g1_status == GateStatus.QUALIFYING_NEGATIVE_EVIDENCE.value:
            g2_status = GateStatus.QUALIFYING_NEGATIVE_EVIDENCE.value
            g2_ev_ids = g1_ev_ids
            g2_rat = "No operational AI identified for vendor at the assessed depth."
            g2_open = []
        elif in_scope_strong or out_of_scope_or_broad_op:
            g2_status = GateStatus.UNRESOLVED.value
            g2_ev_ids = [ev.evidence_id for ev in (in_scope_strong + out_of_scope_or_broad_op)]
            g2_rat = (
                "E1/G2 Safeguard: Vendor AI activity was identified, but evidence is either firm-wide/other-product "
                "or lacks required corroboration for Meridian's specific contracted service."
            )
            g2_open = [
                f"Verify whether AI features are active inside Meridian's contracted service: '{plan.service_product_provided}'."
            ]
        else:
            g2_status = GateStatus.NOT_INVESTIGATED.value if not valid_evidence else GateStatus.UNRESOLVED.value
            g2_ev_ids = []
            g2_rat = "Service-scope AI involvement has not yet been established."
            g2_open = [f"Inspect S1/S2 product documentation for '{plan.service_product_provided}'."]

        # ── G3: Customer Data Exposure ──
        data_touch_ev = [
            ev
            for ev in valid_evidence
            if (
                "R1" in ev.risk_tags
                or "R4" in ev.risk_tags
                or any(
                    k in (ev.verbatim_excerpt or "").lower()
                    for k in ("customer data", "pii", "account balances", "transaction records", "payment instructions", "prompts")
                )
            )
            and ev.ai_type in (AIType.GENAI_LLM, AIType.CLASSICAL_ML, AIType.AGENTIC_AI)
        ]
        if data_touch_ev and g2_status == GateStatus.QUALIFYING_POSITIVE_EVIDENCE.value:
            g3_status = GateStatus.QUALIFYING_POSITIVE_EVIDENCE.value
            g3_ev_ids = [ev.evidence_id for ev in data_touch_ev]
            g3_rat = "Evidence indicates customer/financial data can enter the in-scope AI processing or inference path."
            g3_open: list[str] = []
        elif g1_status == GateStatus.QUALIFYING_NEGATIVE_EVIDENCE.value:
            g3_status = GateStatus.QUALIFYING_NEGATIVE_EVIDENCE.value
            g3_ev_ids = g1_ev_ids
            g3_rat = "No operational AI processing path identified at the assessed depth."
            g3_open = []
        elif data_touch_ev:
            g3_status = GateStatus.UNRESOLVED.value
            g3_ev_ids = [ev.evidence_id for ev in data_touch_ev]
            g3_rat = "Potential customer data handling in AI features noted, but Meridian service scope or data boundary remains unresolved."
            g3_open = ["Confirm data ingestion boundaries and DPA AI clauses (S1/S3)."]
        else:
            g3_status = GateStatus.NOT_INVESTIGATED.value if not valid_evidence else GateStatus.UNRESOLVED.value
            g3_ev_ids = []
            g3_rat = "Customer data exposure to AI processing paths is unresolved from public disclosures."
            g3_open = ["Review S1 DPA and S3 Privacy Policy for AI data ingestion and retention terms."]

        # ── G4: Supply Chain, Provider, Model & Hosting Identity (with E7 Safeguard) ──
        # E7 partnership announcements alone cannot resolve G4!
        qualifying_g4_ev = [
            ev
            for ev in valid_evidence
            if (ev.named_provider or ev.named_model or ev.hosting_infrastructure)
            and "E7" not in ev.e_rule_ids
            and ev.evidence_strength in (EvidenceStrength.CONFIRMED, EvidenceStrength.STRONG)
        ]
        e7_only_g4_ev = [
            ev
            for ev in valid_evidence
            if (ev.named_provider or ev.named_model) and "E7" in ev.e_rule_ids
        ]

        if qualifying_g4_ev:
            g4_status = GateStatus.QUALIFYING_POSITIVE_EVIDENCE.value
            g4_ev_ids = [ev.evidence_id for ev in qualifying_g4_ev]
            providers = sorted({ev.named_provider for ev in qualifying_g4_ev if ev.named_provider})
            models = sorted({ev.named_model for ev in qualifying_g4_ev if ev.named_model})
            hosts = sorted({ev.hosting_infrastructure for ev in qualifying_g4_ev if ev.hosting_infrastructure})
            g4_rat = (
                f"Identified AI supply chain from authoritative evidence — "
                f"Providers: {providers or ['Unspecified']}, Models: {models or ['Unspecified']}, Hosting: {hosts or ['Unspecified']}."
            )
            g4_open: list[str] = []
        elif e7_only_g4_ev:
            g4_status = GateStatus.UNRESOLVED.value
            g4_ev_ids = [ev.evidence_id for ev in e7_only_g4_ev]
            g4_rat = "Rule E7 Safeguard: Partnership announcement alone is insufficient to confirm G4 model/hosting integration."
            g4_open = ["Obtain S1 subprocessor register or S2 technical documentation confirming model/hosting integration."]
        elif g1_status == GateStatus.QUALIFYING_NEGATIVE_EVIDENCE.value:
            g4_status = GateStatus.QUALIFYING_NEGATIVE_EVIDENCE.value
            g4_ev_ids = g1_ev_ids
            g4_rat = "No third-party AI model or subprocessor applicable (no operational AI identified at assessed depth)."
            g4_open = []
        else:
            g4_status = GateStatus.NOT_INVESTIGATED.value if not valid_evidence else GateStatus.UNRESOLVED.value
            g4_ev_ids = []
            g4_rat = "Underlying AI model provider, foundation model, or hosting infrastructure is not publicly confirmed."
            g4_open = ["Inspect S1 subprocessor register for AI hyperscalers (OpenAI, Anthropic, Bedrock, Vertex AI)."]

        gates = [
            EvaluatedGateResult(
                gate_id="G1",
                title="AI Operational Use",
                question="Is AI genuinely used by the vendor?",
                status=g1_status,
                rationale=g1_rat,
                supporting_evidence_ids=g1_ev_ids,
                evaluated_at=now_iso,
                open_questions=g1_open,
            ),
            EvaluatedGateResult(
                gate_id="G2",
                title="Service Scope Relevance",
                question="Is AI used inside the specific service or product Meridian purchases?",
                status=g2_status,
                rationale=g2_rat,
                supporting_evidence_ids=g2_ev_ids,
                evaluated_at=now_iso,
                open_questions=g2_open,
            ),
            EvaluatedGateResult(
                gate_id="G3",
                title="Customer Data Exposure",
                question="Can Meridian/customer/financial/proprietary data touch the AI path?",
                status=g3_status,
                rationale=g3_rat,
                supporting_evidence_ids=g3_ev_ids,
                evaluated_at=now_iso,
                open_questions=g3_open,
            ),
            EvaluatedGateResult(
                gate_id="G4",
                title="Supply Chain & Hyperscaler Identity",
                question="Which model, AI API, subprocessor, and hosting infrastructure are involved?",
                status=g4_status,
                rationale=g4_rat,
                supporting_evidence_ids=g4_ev_ids,
                evaluated_at=now_iso,
                open_questions=g4_open,
            ),
        ]

        # Apply optional analyst overrides with recorded rationale
        if analyst_gate_overrides:
            for g in gates:
                if g.gate_id in analyst_gate_overrides:
                    new_st, ov_rat = analyst_gate_overrides[g.gate_id]
                    g.status = new_st
                    g.review_status = "ANALYST_OVERRIDDEN"
                    g.analyst_override_rationale = ov_rat
                    g.rationale = f"{g.rationale} [Analyst Override: {ov_rat}]"

        # Synchronize gate statuses back onto plan.gate_checklist
        gate_map = {g.gate_id: g for g in gates}
        for plan_gate in plan.gate_checklist:
            if plan_gate.gate_id in gate_map:
                ev_g = gate_map[plan_gate.gate_id]
                try:
                    plan_gate.status = GateStatus(ev_g.status)
                except ValueError:
                    pass
                plan_gate.evidence_notes = ev_g.rationale

        # ── Compute Rulebook AI-Use Verdict (Section 17) ──
        verdict_res = GateAndVerdictEvaluator._compute_verdict(
            valid_evidence=valid_evidence,
            in_scope_confirmed=in_scope_confirmed,
            in_scope_strong=in_scope_strong,
            in_scope_strong_sources=in_scope_strong_sources,
            op_confirmed=op_confirmed,
            op_strong=op_strong,
            op_indicative=op_indicative,
            depth_completed=depth_completed,
            incomplete_reasons=incomplete_reasons,
        )

        return gates, verdict_res

    @staticmethod
    def _check_depth_completeness(
        plan: OSINTInvestigationPlan,
        valid_evidence: Sequence[EvidenceRecord],
    ) -> tuple[bool, list[str]]:
        """Verify whether the required OSINT investigation depth was genuinely completed."""
        reasons: list[str] = []

        if plan.stop_status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE:
            reasons.append("POC timebox was exhausted before required source classes and gates were resolved.")

        unreviewed_required = [
            s_id
            for s_id, rec in plan.coverage_records.items()
            if rec.is_required
            and rec.status not in (CoverageState.REVIEWED_EVIDENCE_RECORDED, CoverageState.REVIEWED_NO_EVIDENCE)
        ]
        if unreviewed_required:
            reasons.append(
                f"Required source classes not yet reviewed to completion: {', '.join(sorted(unreviewed_required))}."
            )

        unavailable_queries = [
            q.query_id for q in plan.planned_queries if q.execution_status.value == "UNAVAILABLE"
        ]
        if unavailable_queries and unreviewed_required:
            reasons.append(
                f"{len(unavailable_queries)} planned search queries could not execute (search provider unconfigured/unavailable)."
            )

        return (len(reasons) == 0), reasons

    @staticmethod
    def _compute_verdict(
        *,
        valid_evidence: Sequence[EvidenceRecord],
        in_scope_confirmed: Sequence[EvidenceRecord],
        in_scope_strong: Sequence[EvidenceRecord],
        in_scope_strong_sources: set[str],
        op_confirmed: Sequence[EvidenceRecord],
        op_strong: Sequence[EvidenceRecord],
        op_indicative: Sequence[EvidenceRecord],
        depth_completed: bool,
        incomplete_reasons: list[str],
    ) -> AIUseVerdictResult:
        # Check for material conflict (e.g. an authoritative S1/S3 document explicitly states "No AI is used in the service"
        # while another operational source claims AI is active in the same service)
        explicit_no_ai_in_service = [
            ev
            for ev in valid_evidence
            if any(
                p in (ev.verbatim_excerpt or "").lower()
                for p in ("does not use artificial intelligence", "no ai or machine learning is used in")
            )
            and ev.source_class in ("S1", "S2", "S3")
        ]
        if explicit_no_ai_in_service and (in_scope_confirmed or in_scope_strong):
            return AIUseVerdictResult(
                verdict=AIUseVerdictValue.INCONCLUSIVE,
                verdict_statement="INCONCLUSIVE — Materially conflicting primary evidence regarding in-scope AI use.",
                basis_summary=(
                    "Authoritative disclosures contain conflicting statements on whether AI is active "
                    "in Meridian's contracted service; escalated for analyst validation."
                ),
                supporting_evidence_ids=[
                    ev.evidence_id for ev in (*in_scope_confirmed, *in_scope_strong, *explicit_no_ai_in_service)
                ],
                corroborating_source_classes=sorted(list(in_scope_strong_sources)),
                depth_completed=depth_completed,
                incomplete_reasons=incomplete_reasons,
                requires_human_review=True,
            )

        # 1. YES: >= 1 CONFIRMED in-scope operational AI signal OR >= 2 independent STRONG source classes in scope
        if in_scope_confirmed or len(in_scope_strong_sources) >= 2:
            supp_ids = [ev.evidence_id for ev in (*in_scope_confirmed, *in_scope_strong)]
            src_classes = sorted({ev.source_class for ev in (*in_scope_confirmed, *in_scope_strong)})
            return AIUseVerdictResult(
                verdict=AIUseVerdictValue.YES,
                verdict_statement="YES — Operational AI use confirmed within Meridian's contracted service scope.",
                basis_summary=(
                    f"Supported by {len(in_scope_confirmed)} CONFIRMED and {len(in_scope_strong)} STRONG "
                    f"in-scope operational evidence item(s) across source classes {', '.join(src_classes)}."
                ),
                supporting_evidence_ids=supp_ids,
                corroborating_source_classes=src_classes,
                depth_completed=depth_completed,
                incomplete_reasons=incomplete_reasons,
                requires_human_review=False,
            )

        # 2. INCONCLUSIVE: 1 STRONG signal without corroboration, INDICATIVE-only evidence, or unresolved service scope
        if op_confirmed or op_strong or op_indicative:
            supp_ids = [ev.evidence_id for ev in (*op_confirmed, *op_strong, *op_indicative)]
            src_classes = sorted({ev.source_class for ev in (*op_confirmed, *op_strong, *op_indicative)})
            if op_confirmed or len({ev.source_class for ev in op_strong}) >= 2:
                reason_detail = (
                    "Operational AI is confirmed at the vendor/product level, but Meridian service scope (G2) "
                    "remains unresolved."
                )
            elif op_strong:
                reason_detail = (
                    "Single STRONG operational AI signal found without independent second-source-class corroboration."
                )
            else:
                reason_detail = "Only INDICATIVE operational AI signals identified without authoritative confirmation."

            return AIUseVerdictResult(
                verdict=AIUseVerdictValue.INCONCLUSIVE,
                verdict_statement=f"INCONCLUSIVE — {reason_detail}",
                basis_summary=reason_detail,
                supporting_evidence_ids=supp_ids,
                corroborating_source_classes=src_classes,
                depth_completed=depth_completed,
                incomplete_reasons=incomplete_reasons,
                requires_human_review=True,
            )

        # 3. NO: Only MARKETING / NONE evidence OR no qualifying evidence AFTER defined depth is genuinely completed
        if depth_completed:
            supp_ids = [ev.evidence_id for ev in valid_evidence]
            return AIUseVerdictResult(
                verdict=AIUseVerdictValue.NO,
                verdict_statement="NO — None identified at the assessed depth.",
                basis_summary=(
                    "Defined OSINT assessment depth was completed across all required source classes. "
                    "Only marketing claims or ordinary non-AI automation were found; none identified at the assessed depth."
                ),
                supporting_evidence_ids=supp_ids,
                corroborating_source_classes=[],
                depth_completed=True,
                incomplete_reasons=[],
                requires_human_review=False,
            )

        # 4. INCOMPLETE_ASSESSMENT: Never claim NO when searches/extractions/sources are unconfigured or incomplete
        return AIUseVerdictResult(
            verdict=AIUseVerdictValue.INCOMPLETE_ASSESSMENT,
            verdict_statement=(
                "INCOMPLETE_ASSESSMENT — Required OSINT depth has not been completed; AI-use verdict cannot be finalized."
            ),
            basis_summary=(
                "Investigation is incomplete due to unexecuted searches, unreviewed required source classes, "
                "or unavailable external adapters. Refusing to issue a false-negative 'NO' verdict."
            ),
            supporting_evidence_ids=[ev.evidence_id for ev in valid_evidence],
            corroborating_source_classes=[],
            depth_completed=False,
            incomplete_reasons=incomplete_reasons,
            requires_human_review=True,
        )
