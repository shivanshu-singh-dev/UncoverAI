"""End-to-End 9-Level Orchestration Pipeline, Human Validation Service, and Report Generator.

Connects all nine original Meridian project levels into one auditable workflow:
  Level 1: Criticality Tiering (`MeridianCriticalityEngine`, D/P/R/O/V + O1–O5)
  Level 2: Assessment Depth Scoping (`OSINTInvestigationPlanner`, timeboxes & S1–S12 profiles)
  Level 3: Source Identification & Reconnaissance (`OSINTQueryPlanner`, `OSINTSearchCollector`, `OSINTURLTriageService`)
  Level 4: Data Scraping & Point-of-Capture Traceability (`OSINTExtractionService`, `EvidenceRepository`)
  Level 5: Semantic Filtering & RAG Chunking (`RAGChunkerAndRetriever`, `EmbeddingProvider`)
  Level 6: Signal Classification (`SignalClassificationService`, prompt-injection defence & optional Ollama adapter)
  Level 7: Evidentiary Weighting (`ERuleEngineAndWeighter` E1–E14, `GateAndVerdictEvaluator` G1–G4 & AI-Use Verdict)
  Level 8: Risk Surfacing & Summarization (`AIRiskAnalyzer` R1–R8)
  Level 9: Deterministic Risk Rating (`DeterministicRiskScoringEngine` M_c * sum(S_i * W_i))
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from meridian_assessment.models.ai_risk import (
    AIRiskFinding,
    AIUseVerdictResult,
    AIUseVerdictValue,
    DeterministicRiskScoreResult,
    EvaluatedGateResult,
    EvidenceRecord,
    EvidenceScope,
    ExtractedDocument,
    ExtractionStatus,
    FindingSeverity,
    HumanValidationRecord,
    ScoringPolicyStatus,
    SignalClassificationResult,
    StageExecutionRecord,
    URLTriageAction,
    URLTriageDecision,
    VendorEndToEndAssessmentReport,
    WeightedEvidenceItem,
)
from meridian_assessment.models.factor_result import (
    CriticalityResult,
    HumanReviewRecord,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.models.osint_plan import (
    CoverageState,
    OSINTInvestigationPlan,
)
from meridian_assessment.services.classification import (
    LocalLLMClassificationAdapter,
    OllamaClassificationAdapter,
    SignalClassificationService,
    UnconfiguredOllamaAdapter,
)
from meridian_assessment.services.evidence import (
    ERuleEngineAndWeighter,
    GateAndVerdictEvaluator,
)
from meridian_assessment.services.extraction import (
    ContentFetchAdapter,
    DisabledOfflineFetchAdapter,
    EvidenceRepository,
    OSINTExtractionService,
)
from meridian_assessment.services.llm.model_router import (
    LLMTaskType,
    ModelRoutingConfig,
    OllamaModelRouter,
)
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine
from meridian_assessment.services.osint import (
    OSINTCoverageTracker,
    OSINTInvestigationPlanner,
    OSINTSearchCollector,
    SearchBackend,
    UnconfiguredSearchBackend,
    build_search_backend_from_env,
)
from meridian_assessment.services.osint.url_triage import (
    OSINTURLTriageService,
    OllamaURLTriageProvider,
    URLTriageLLMProvider,
)
from meridian_assessment.services.rag import (
    EmbeddingProvider,
    RAGChunkerAndRetriever,
    UnconfiguredEmbeddingProvider,
)
from meridian_assessment.services.risk import (
    AIRiskAnalyzer,
    DeterministicRiskScoringEngine,
    OllamaRiskSummaryAdapter,
    RiskSummaryLLMProvider,
)
import os



def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HumanValidationService:
    """Evaluates human-review requirements (Section 20) and records auditable analyst validation decisions."""

    @staticmethod
    def evaluate_review_triggers(
        *,
        effective_criticality: str,
        gates: Sequence[EvaluatedGateResult],
        verdict: AIUseVerdictResult,
        findings: Sequence[AIRiskFinding],
        score_result: DeterministicRiskScoreResult,
        evidence_records: Sequence[EvidenceRecord],
    ) -> list[str]:
        """Return all triggered human-validation reasons for the vendor assessment."""
        triggers: list[str] = []
        gate_map = {g.gate_id: g for g in gates}

        # 1. Conflicting evidence or inconclusive verdict
        if verdict.verdict == AIUseVerdictValue.INCONCLUSIVE:
            triggers.append(f"AI-use verdict is INCONCLUSIVE: {verdict.basis_summary}")
        if verdict.verdict == AIUseVerdictValue.INCOMPLETE_ASSESSMENT:
            triggers.append("OSINT investigation depth is incomplete; analyst review required before sign-off.")

        # 2. Ambiguous G2 service scope when G1 operational AI is indicated/confirmed
        g1 = gate_map.get("G1")
        g2 = gate_map.get("G2")
        if g1 and g2 and g1.status == "QUALIFYING_POSITIVE_EVIDENCE" and g2.status != "QUALIFYING_POSITIVE_EVIDENCE":
            triggers.append("Gate G2 (Meridian Service Scope) is unresolved despite confirmed vendor AI activity (G1).")

        # 3. Unresolved G3 data exposure in an important context
        g3 = gate_map.get("G3")
        if g3 and g3.status == "UNRESOLVED" and effective_criticality in ("Critical", "High"):
            triggers.append(f"Gate G3 (Customer Data Exposure) is unresolved for {effective_criticality} vendor.")

        # 4. Incomplete G4 provider identity for a material finding
        g4 = gate_map.get("G4")
        if g4 and g4.status == "UNRESOLVED" and findings:
            triggers.append("Gate G4 (AI Provider/Model/Hosting Identity) is incomplete while risk findings exist.")

        # 5. High/Critical risk findings
        high_crit_findings = [
            f.risk_rule_id for f in findings if f.severity in (FindingSeverity.HIGH, FindingSeverity.CRITICAL)
        ]
        if high_crit_findings:
            triggers.append(
                f"High/Critical AI security risk findings surfaced: {', '.join(sorted(set(high_crit_findings)))}."
            )

        # 6. High/Critical vendor with any unresolved material risks
        if effective_criticality in ("Critical", "High") and any(f.requires_human_review for f in findings):
            triggers.append(f"{effective_criticality} criticality vendor has risk findings requiring analyst validation.")

        # 7. Incomplete scoring policy
        if score_result.policy_status == ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED:
            triggers.append(
                "Deterministic AI risk scoring policy is unconfigured; final numerical rating requires policy calibration."
            )

        # 8. Hostile prompt-injection security event
        if any(e.hostile_instruction_detected for e in evidence_records):
            triggers.append("Hostile webpage/PDF prompt-injection instructions detected and isolated under Rule E14.")

        return triggers

    @staticmethod
    def create_validation_record(
        *,
        vendor_id: str,
        target_type: str,
        target_id: str,
        proposed_automated_finding: str,
        human_decision: str,
        rationale: str,
        original_status: str,
        updated_status: str,
        evidence_references: Optional[list[str]] = None,
        reviewer_id: str = "Primary Analyst",
        outstanding_questions: Optional[list[str]] = None,
    ) -> HumanValidationRecord:
        if not rationale or not rationale.strip():
            raise ValueError("A written rationale is required for all human validation decisions.")
        ts = _utc_now_iso()
        val_id = f"HVAL_{vendor_id}_{target_type}_{target_id}"
        return HumanValidationRecord(
            validation_id=val_id,
            vendor_id=vendor_id,
            target_type=target_type,
            target_id=target_id,
            proposed_automated_finding=proposed_automated_finding,
            evidence_references=list(evidence_references or []),
            human_decision=human_decision,
            rationale=rationale.strip(),
            reviewer_id=reviewer_id,
            reviewed_at=ts,
            original_status=original_status,
            updated_status=updated_status,
            outstanding_questions=list(outstanding_questions or []),
        )


class MeridianEndToEndPipeline:
    """Orchestrates all 9 project levels for a single vendor or a full vendor inventory."""

    def __init__(
        self,
        *,
        criticality_engine: Optional[MeridianCriticalityEngine] = None,
        investigation_planner: Optional[OSINTInvestigationPlanner] = None,
        search_backend: Optional[SearchBackend] = None,
        model_router: Optional[OllamaModelRouter] = None,
        enable_local_llm_from_env: Optional[bool] = None,
        triage_llm_provider: Optional[URLTriageLLMProvider] = None,
        fetch_adapter: Optional[ContentFetchAdapter] = None,
        evidence_repository: Optional[EvidenceRepository] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        classification_llm_adapter: Optional[LocalLLMClassificationAdapter] = None,
        risk_summary_llm_adapter: Optional[RiskSummaryLLMProvider] = None,
        scoring_engine: Optional[DeterministicRiskScoringEngine] = None,
    ) -> None:
        self.criticality_engine = criticality_engine or MeridianCriticalityEngine()
        self.investigation_planner = investigation_planner or OSINTInvestigationPlanner()
        self.search_backend = search_backend or build_search_backend_from_env()

        if model_router is not None:
            self.model_router = model_router
        else:
            # In normal CLI/UI execution, load .env and enable dual-model routing if MERIDIAN_OLLAMA_ENABLED=true.
            # When running inside pytest without an explicit model_router or enable_local_llm_from_env=True,
            # default to offline deterministic mode so unit tests remain fast and network-free.
            in_pytest = "PYTEST_CURRENT_TEST" in os.environ
            cfg = ModelRoutingConfig.from_env_and_config()
            if enable_local_llm_from_env is True:
                cfg.enabled = True
            elif enable_local_llm_from_env is False or (enable_local_llm_from_env is None and in_pytest):
                cfg.enabled = False
            self.model_router = OllamaModelRouter(config=cfg)

        if triage_llm_provider is not None:
            resolved_triage_llm: Optional[URLTriageLLMProvider] = triage_llm_provider
        elif self.model_router.config.enabled:
            resolved_triage_llm = OllamaURLTriageProvider(router=self.model_router)
        else:
            resolved_triage_llm = None

        if classification_llm_adapter is not None:
            resolved_cls_llm: Optional[LocalLLMClassificationAdapter] = classification_llm_adapter
        elif self.model_router.config.enabled:
            resolved_cls_llm = OllamaClassificationAdapter(router=self.model_router)
        else:
            resolved_cls_llm = None

        if risk_summary_llm_adapter is not None:
            resolved_sum_llm: Optional[RiskSummaryLLMProvider] = risk_summary_llm_adapter
        elif self.model_router.config.enabled:
            resolved_sum_llm = OllamaRiskSummaryAdapter(router=self.model_router)
        else:
            resolved_sum_llm = None

        self.triage_service = OSINTURLTriageService(llm_provider=resolved_triage_llm)
        self.fetch_adapter = fetch_adapter or DisabledOfflineFetchAdapter()
        self.evidence_repository = evidence_repository
        self.extraction_service = OSINTExtractionService(
            fetch_adapter=self.fetch_adapter,
            source_registry=self.investigation_planner.source_registry,
            artifact_dir=self.evidence_repository.artifact_dir if self.evidence_repository else None,
        )
        self.rag_service = RAGChunkerAndRetriever(embedding_provider=embedding_provider)
        self.classification_service = SignalClassificationService(llm_adapter=resolved_cls_llm)
        self.scoring_engine = scoring_engine or DeterministicRiskScoringEngine()
        self.risk_analyzer = AIRiskAnalyzer(summary_llm_adapter=resolved_sum_llm)

    def run_vendor_assessment(
        self,
        vendor: MeridianVendor,
        *,
        criticality_result: Optional[CriticalityResult] = None,
        human_criticality_override: Optional[HumanReviewRecord] = None,
        existing_plan: Optional[OSINTInvestigationPlan] = None,
        execute_searches: bool = False,
        execute_extraction: bool = False,
        pre_supplied_evidence: Optional[Sequence[EvidenceRecord]] = None,
        mark_extracted_sources_reviewed: bool = False,
        reviewed_no_evidence_sources: Optional[Sequence[str]] = None,
        analyst_gate_overrides: Optional[dict[str, tuple[str, str]]] = None,
        human_validations: Optional[Sequence[HumanValidationRecord]] = None,
        persist_to_repository: bool = True,
    ) -> VendorEndToEndAssessmentReport:
        """Execute all 9 project levels in order and produce a complete `VendorEndToEndAssessmentReport`."""
        now_iso = _utc_now_iso()
        stage_ledger: list[StageExecutionRecord] = []
        unavailable_integrations: list[str] = []
        limitations: list[str] = []

        # Reset per-run call telemetry on router so each vendor report captures its exact calls
        self.model_router.call_history.clear()
        self.model_router.vram_unload_events.clear()

        # ── LEVEL 1: Criticality Tiering (D/P/R/O/V + O1–O5) ──
        crit_res = criticality_result or self.criticality_engine.evaluate(
            vendor, human_review=human_criticality_override
        )
        if criticality_result is not None and human_criticality_override is not None:
            crit_res = self.criticality_engine.evaluate(
                vendor, human_review=human_criticality_override
            )
        effective_crit, approval_status = self.investigation_planner.resolve_effective_criticality(
            crit_res, human_criticality_override
        )
        crit_audit = self.criticality_engine.generate_audit_trail(crit_res)
        stage_ledger.append(
            StageExecutionRecord(
                level_number=1,
                level_name="Criticality Tiering",
                status="COMPLETED" if not crit_res.requires_review else "REQUIRES_REVIEW",
                summary=(
                    f"Base score={crit_res.base_score}, Effective Criticality='{effective_crit}' "
                    f"({approval_status.value})."
                ),
                limitations=list(crit_res.review_notes),
                selected_model="DETERMINISTIC_RULEBOOK",
                model_task="CRITICALITY_D_P_R_O_V",
                model_execution_status="DETERMINISTIC_ONLY",
            )
        )

        # ── LEVEL 2: Assessment Depth Scoping ──
        plan = self.investigation_planner.plan_investigation(
            vendor=vendor,
            criticality_input=crit_res,
            human_review=human_criticality_override,
            existing_plan=existing_plan,
        )
        stage_ledger.append(
            StageExecutionRecord(
                level_number=2,
                level_name="Assessment Depth Scoping",
                status="COMPLETED",
                summary=(
                    f"Selected '{plan.selected_depth_profile}' profile "
                    f"(POC Timebox: {plan.timebox.planning_target}; "
                    f"{len(plan.required_source_classes)} required sources; "
                    f"{len(plan.planned_queries)} deduplicated queries)."
                ),
                limitations=list(plan.unresolved_planning_issues),
                selected_model="DETERMINISTIC_RULEBOOK",
                model_task="OSINT_DEPTH_SCOPING",
                model_execution_status="DETERMINISTIC_ONLY",
            )
        )

        # ── LEVEL 3: Source Identification & Reconnaissance (Search + URL Triage via Model A llama3.2:latest) ──
        search_ok, search_msg = self.search_backend.is_configured()
        if execute_searches:
            collector = OSINTSearchCollector(backend=self.search_backend, retry_backoff_seconds=0.0)
            collector.execute_planned_queries(plan, persist=False)
        if not search_ok:
            unavailable_integrations.append(f"Search Backend ({self.search_backend.provider_name}): {search_msg}")

        triage_llm_ok, triage_llm_msg = self.triage_service.llm_provider.is_configured()
        triage_model_name = (
            getattr(self.triage_service.llm_provider, "model_name", None)
            or self.model_router.resolve_model_for_task(LLMTaskType.URL_TRIAGE)
        )
        triage_decisions: list[URLTriageDecision] = self.triage_service.triage_plan_candidates(plan)
        fetch_decisions = [d for d in triage_decisions if d.action == URLTriageAction.FETCH]

        triage_llm_succeeded = sum(1 for d in triage_decisions if d.decision_method.value == "LOCAL_LLM")
        triage_task_tel = self.model_router.get_task_telemetry_summary(LLMTaskType.URL_TRIAGE)
        if not triage_llm_ok:
            l3_model_status = "SKIPPED_DETERMINISTIC_FALLBACK"
        elif not triage_decisions:
            l3_model_status = "SKIPPED_NO_INPUT"
        elif triage_llm_succeeded > 0:
            l3_model_status = "SUCCEEDED" if triage_llm_succeeded == len(triage_decisions) else "PARTIAL_SUCCEEDED"
        else:
            l3_model_status = triage_task_tel["overall_status"]

        l3_status = (
            "COMPLETED"
            if (plan.candidate_urls or (execute_searches and search_ok))
            else "NOT_CONFIGURED"
            if not search_ok
            else "PARTIAL"
        )
        l3_limits: list[str] = [] if search_ok else [search_msg]
        if triage_task_tel.get("errors"):
            l3_limits.extend(triage_task_tel["errors"])
        stage_ledger.append(
            StageExecutionRecord(
                level_number=3,
                level_name="Source Identification & Reconnaissance",
                status=l3_status,
                summary=(
                    f"{len(plan.planned_queries)} planned queries; "
                    f"{len(plan.candidate_urls)} candidate URLs collected; "
                    f"{len(fetch_decisions)} URLs triaged as FETCH "
                    f"(Triage Model: {triage_model_name} [{l3_model_status}])."
                ),
                limitations=l3_limits,
                selected_model=triage_model_name,
                model_task=LLMTaskType.URL_TRIAGE.value,
                model_execution_status=l3_model_status,
            )
        )

        # ── LEVEL 4: Data Scraping & Point-of-Capture Traceability ──
        extracted_docs: list[ExtractedDocument] = []
        evidence_records: list[EvidenceRecord] = (
            [ev.model_copy(deep=True) for ev in pre_supplied_evidence]
            if pre_supplied_evidence
            else []
        )

        fetch_ok, fetch_msg = self.fetch_adapter.is_configured()
        if execute_extraction and fetch_decisions:
            seq_ev = len(evidence_records) + 1
            for doc_idx, t_dec in enumerate(fetch_decisions, start=1):
                cand_obj = next((c for c in plan.candidate_urls if c.candidate_id == t_dec.candidate_id), None)
                q_ids = cand_obj.query_ids if cand_obj else []
                doc = self.extraction_service.extract_from_url(
                    vendor_id=vendor.vendor_id,
                    url=t_dec.url,
                    source_class=t_dec.candidate_source_class,
                    query_ids=q_ids,
                    triage_action=t_dec.action.value,
                    document_index=doc_idx,
                )
                extracted_docs.append(doc)
                if doc.extraction_status == ExtractionStatus.EXTRACTED:
                    new_evs = self.extraction_service.build_evidence_records_from_document(
                        doc,
                        legal_entity=vendor.vendor_name,
                        domain=vendor.domain,
                        starting_seq=seq_ev,
                    )
                    seq_ev += len(new_evs)
                    evidence_records.extend(new_evs)
                else:
                    limitations.extend(doc.extraction_limitations)

        if not fetch_ok and not evidence_records:
            unavailable_integrations.append(f"Web/PDF Fetch Adapter: {fetch_msg}")

        # Update source coverage records if analyst-reviewed / completed mode is specified
        tracker = OSINTCoverageTracker(plan.coverage_records)
        if mark_extracted_sources_reviewed:
            by_source: dict[str, list[EvidenceRecord]] = {}
            for ev in evidence_records:
                ev.is_reviewed_and_accepted = True
                by_source.setdefault(ev.source_class, []).append(ev)
            for s_id, ev_list in by_source.items():
                if s_id in plan.coverage_records:
                    tracker.record_evidence_found(
                        s_id,
                        evidence_references=[e.evidence_id for e in ev_list],
                        candidate_urls=len(ev_list),
                        reviewed_urls=len(ev_list),
                        produced_new_evidence=True,
                    )
        if reviewed_no_evidence_sources:
            for s_id in reviewed_no_evidence_sources:
                if s_id in plan.coverage_records:
                    tracker.record_reviewed_no_evidence(s_id, candidate_urls=1, reviewed_urls=1)
        tracker.sync_plan_stop_status(plan)

        is_synthetic_run = any(e.is_synthetic_test_fixture for e in evidence_records) or any(
            d.is_synthetic_fixture for d in extracted_docs
        )

        l4_status = (
            "COMPLETED"
            if evidence_records or reviewed_no_evidence_sources
            else "NOT_CONFIGURED"
            if not fetch_ok
            else "SKIPPED_NO_INPUT"
        )
        stage_ledger.append(
            StageExecutionRecord(
                level_number=4,
                level_name="Data Scraping & Point-of-Capture Traceability",
                status=l4_status,
                summary=(
                    f"{len(extracted_docs)} documents processed; "
                    f"{len(evidence_records)} traceable EvidenceRecord items created."
                ),
                limitations=[] if evidence_records else ["No extracted evidence records present."],
                selected_model="N/A",
                model_task="DOCUMENT_EXTRACTION",
                model_execution_status="DETERMINISTIC_ONLY",
            )
        )

        # ── LEVEL 5: Semantic Filtering & RAG Chunking (Separate Embedding Model) ──
        _, rag_summary = self.rag_service.retrieve_relevant_chunks(evidence_records)
        emb_ok, emb_msg = self.rag_service.embedding_provider.is_available()
        if not emb_ok:
            unavailable_integrations.append(f"RAG Embedding Model ({self.rag_service.model_name}): {emb_msg}")

        l5_model_status = (
            "SUCCEEDED"
            if (rag_summary.model_available and rag_summary.chunks_embedded > 0)
            else "SKIPPED_NO_INPUT"
            if rag_summary.model_available
            else "UNAVAILABLE"
        )
        stage_ledger.append(
            StageExecutionRecord(
                level_number=5,
                level_name="Semantic Filtering & RAG Chunking",
                status="COMPLETED" if rag_summary.model_available else "NOT_CONFIGURED",
                summary=(
                    f"{rag_summary.chunks_created} chunks created; "
                    f"{rag_summary.chunks_embedded} embedded; "
                    f"{len(rag_summary.matches)} concept matches retrieved ({rag_summary.status})."
                ),
                limitations=list(rag_summary.limitations),
                selected_model=self.rag_service.model_name,
                model_task=LLMTaskType.RAG_EMBEDDING.value,
                model_execution_status=l5_model_status,
            )
        )

        # ── LEVEL 6: Signal Classification (Model B mistral:latest + Prompt-Injection Defence) ──
        llm_ok, llm_msg = self.classification_service.llm_adapter.is_configured()
        cls_model_name = (
            getattr(self.classification_service.llm_adapter, "model_name", None)
            or self.model_router.resolve_model_for_task(LLMTaskType.EVIDENCE_CLASSIFICATION)
        )
        if not llm_ok:
            unavailable_integrations.append(f"Local LLM Classifier ({cls_model_name}): {llm_msg}")

        classifications = self.classification_service.classify_evidence_batch(
            evidence_records,
            meridian_service_scope=vendor.service_product_provided,
            meridian_business_process=vendor.business_process_supported,
        )
        cls_llm_succeeded = sum(
            1 for c in classifications if c.classification_method.value == "LOCAL_LLM_VALIDATED"
        )
        cls_task_tel = self.model_router.get_task_telemetry_summary(LLMTaskType.EVIDENCE_CLASSIFICATION)
        if not llm_ok:
            l6_model_status = "SKIPPED_DETERMINISTIC_FALLBACK"
        elif not classifications:
            l6_model_status = "SKIPPED_NO_INPUT"
        elif cls_llm_succeeded > 0:
            l6_model_status = "SUCCEEDED" if cls_llm_succeeded == len(classifications) else "PARTIAL_SUCCEEDED"
        else:
            l6_model_status = cls_task_tel["overall_status"]

        l6_limits: list[str] = [] if llm_ok else [llm_msg]
        if cls_task_tel.get("errors"):
            l6_limits.extend(cls_task_tel["errors"])
        stage_ledger.append(
            StageExecutionRecord(
                level_number=6,
                level_name="Signal Classification",
                status="COMPLETED" if classifications else "SKIPPED_NO_INPUT",
                summary=(
                    f"{len(classifications)} evidence items classified "
                    f"(Analysis Model: {cls_model_name} [{l6_model_status}])."
                ),
                limitations=l6_limits,
                selected_model=cls_model_name,
                model_task=LLMTaskType.EVIDENCE_CLASSIFICATION.value,
                model_execution_status=l6_model_status,
            )
        )

        # ── LEVEL 7: Evidentiary Weighting (E1–E14), G1–G4 Gates & AI-Use Verdict ──
        scalar_weights_cfg = self.scoring_engine.policy_data.get("evidentiary_weighting", {}).get("exact_scalar_weights")
        active_scalars = (
            {k: float(v) for k, v in scalar_weights_cfg.items() if isinstance(v, (int, float))}
            if isinstance(scalar_weights_cfg, dict)
            else None
        )
        e_engine = ERuleEngineAndWeighter(scalar_weight_policy=active_scalars)
        weighted_items = e_engine.evaluate_evidence_batch(evidence_records, classifications)

        gates, ai_verdict = GateAndVerdictEvaluator.evaluate_gates_and_verdict(
            plan,
            evidence_records,
            weighted_items,
            analyst_gate_overrides=analyst_gate_overrides,
        )
        tracker.sync_plan_stop_status(plan)

        stage_ledger.append(
            StageExecutionRecord(
                level_number=7,
                level_name="Evidentiary Weighting, Gates G1–G4 & AI-Use Verdict",
                status="COMPLETED" if weighted_items or ai_verdict.depth_completed else "PARTIAL",
                summary=(
                    f"Evaluated E1–E14 across {len(weighted_items)} items; "
                    f"AI-Use Verdict: {ai_verdict.verdict.value}."
                ),
                limitations=list(ai_verdict.incomplete_reasons),
                selected_model="DETERMINISTIC_RULEBOOK",
                model_task="E1_E14_AND_G1_G4_VERDICT",
                model_execution_status="DETERMINISTIC_ONLY",
            )
        )

        # ── LEVEL 8: Risk Surfacing & Summarization (R1–R8 + Model B mistral:latest Summaries) ──
        sum_llm_ok, sum_llm_msg = self.risk_analyzer.summary_llm_adapter.is_configured()
        sum_model_name = (
            getattr(self.risk_analyzer.summary_llm_adapter, "model_name", None)
            or self.model_router.resolve_model_for_task(LLMTaskType.RISK_SUMMARIZATION)
        )
        risk_findings = self.risk_analyzer.analyze_vendor_risks(
            vendor_id=vendor.vendor_id,
            evidence_records=evidence_records,
            weighted_items=weighted_items,
            gates=gates,
            verdict=ai_verdict,
            data_classification_accessed=vendor.data_classification_accessed,
            service_product_provided=vendor.service_product_provided,
        )
        sum_llm_succeeded = sum(1 for rf in risk_findings if rf.summary_method == "LOCAL_LLM_VALIDATED")
        sum_task_tel = self.model_router.get_task_telemetry_summary(LLMTaskType.RISK_SUMMARIZATION)
        if not sum_llm_ok:
            l8_model_status = "SKIPPED_DETERMINISTIC_FALLBACK"
        elif not risk_findings:
            l8_model_status = "SKIPPED_NO_INPUT"
        elif sum_llm_succeeded > 0:
            l8_model_status = "SUCCEEDED" if sum_llm_succeeded == len(risk_findings) else "PARTIAL_SUCCEEDED"
        else:
            l8_model_status = sum_task_tel["overall_status"]

        l8_limits: list[str] = []
        if sum_task_tel.get("errors"):
            l8_limits.extend(sum_task_tel["errors"])
        stage_ledger.append(
            StageExecutionRecord(
                level_number=8,
                level_name="Risk Surfacing & Summarization (R1–R8)",
                status="COMPLETED" if risk_findings or ai_verdict.verdict == AIUseVerdictValue.NO else "PARTIAL",
                summary=(
                    f"Surfaced {len(risk_findings)} evidence-backed R1–R8 risk finding(s) "
                    f"(Summary Model: {sum_model_name} [{l8_model_status}])."
                ),
                limitations=l8_limits,
                selected_model=sum_model_name,
                model_task=LLMTaskType.RISK_SUMMARIZATION.value,
                model_execution_status=l8_model_status,
            )
        )

        # Unload active generative model at end of Level 8 if configured
        if self.model_router.config.unload_on_model_switch and self.model_router.active_generative_model:
            self.model_router.unload_active_model(
                reason="Level 8 risk summarization completed; releasing VRAM"
            )

        # ── LEVEL 9: Deterministic Risk Rating (`M_c * sum(S_i * W_i)`) ──
        completeness_str = "COMPLETED" if ai_verdict.depth_completed else "INCOMPLETE"
        risk_score_res = self.scoring_engine.calculate_vendor_risk_score(
            vendor_id=vendor.vendor_id,
            criticality_tier=effective_crit,
            findings=risk_findings,
            weighted_evidence=weighted_items,
            assessment_completeness_status=completeness_str,
        )
        l9_status = (
            "NOT_CONFIGURED"
            if risk_score_res.policy_status == ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED
            else "COMPLETED"
        )
        stage_ledger.append(
            StageExecutionRecord(
                level_number=9,
                level_name="Deterministic Risk Rating",
                status=l9_status,
                summary=risk_score_res.status_message,
                limitations=[
                    f"Unconfigured scoring parameter: {p}" for p in risk_score_res.unconfigured_parameters
                ],
                selected_model="DETERMINISTIC_RULEBOOK",
                model_task="DETERMINISTIC_RISK_SCORING",
                model_execution_status="DETERMINISTIC_ONLY",
            )
        )


        # ── Human Validation & Final Report Assembly (Sections 20 & 21) ──
        review_triggers = HumanValidationService.evaluate_review_triggers(
            effective_criticality=effective_crit,
            gates=gates,
            verdict=ai_verdict,
            findings=risk_findings,
            score_result=risk_score_res,
            evidence_records=evidence_records,
        )

        all_validations: list[HumanValidationRecord] = list(human_validations or [])
        if self.evidence_repository is not None:
            stored_vals = self.evidence_repository.list_human_validations(vendor.vendor_id)
            seen_val_ids = {v.validation_id for v in all_validations}
            for sv in stored_vals:
                if sv.validation_id not in seen_val_ids:
                    all_validations.append(sv)

        valid_non_hostile = [e for e in evidence_records if not e.hostile_instruction_detected]
        ai_types_found = sorted({
            e.ai_type.value
            for e in valid_non_hostile
            if e.ai_type.value not in ("NONE", "UNRESOLVED")
        })
        named_models = sorted({e.named_model for e in valid_non_hostile if e.named_model})
        named_providers = sorted({e.named_provider for e in valid_non_hostile if e.named_provider})
        hosting_infras = sorted({e.hosting_infrastructure for e in valid_non_hostile if e.hosting_infrastructure})

        if any(e.scope == EvidenceScope.MERIDIAN_SERVICE for e in valid_non_hostile):
            det_scope = EvidenceScope.MERIDIAN_SERVICE.value
        elif any(e.scope == EvidenceScope.VENDOR_PRODUCT for e in valid_non_hostile):
            det_scope = EvidenceScope.VENDOR_PRODUCT.value
        elif any(e.scope == EvidenceScope.VENDOR_FIRM_WIDE for e in valid_non_hostile):
            det_scope = EvidenceScope.VENDOR_FIRM_WIDE.value
        else:
            det_scope = EvidenceScope.UNRESOLVED.value

        g4_gate = next((g for g in gates if g.gate_id == "G4"), None)
        if g4_gate and g4_gate.status == "QUALIFYING_POSITIVE_EVIDENCE":
            supply_chain_status = "CONFIRMED"
        elif named_providers or named_models:
            supply_chain_status = "INFERRED_INDICATIVE"
        else:
            supply_chain_status = "UNKNOWN"

        known_controls: list[str] = []
        remediations: list[dict[str, str]] = []
        for rf in risk_findings:
            for ctrl in rf.known_controls_and_mitigations:
                if ctrl and ctrl not in known_controls:
                    known_controls.append(ctrl)
            remediations.append({
                "finding_id": rf.finding_id,
                "risk_rule_id": rf.risk_rule_id,
                "risk_title": rf.risk_title,
                "severity": rf.severity.value,
                "recommended_remediation": rf.recommended_remediation,
            })

        unresolved_questions: list[str] = []
        for g in gates:
            for oq in g.open_questions:
                if oq not in unresolved_questions:
                    unresolved_questions.append(oq)
        for cls_r in classifications:
            for oq in cls_r.unresolved_questions:
                if oq not in unresolved_questions:
                    unresolved_questions.append(oq)

        cov_summary = tracker.get_summary()
        osint_summary = {
            "plan_id": plan.plan_id,
            "selected_depth_profile": plan.selected_depth_profile,
            "selection_reason": plan.selection_reason,
            "required_source_classes": plan.required_source_classes,
            "conditional_source_classes": plan.conditional_source_classes,
            "corroborative_source_classes": plan.corroborative_source_classes,
            "planned_queries_count": len(plan.planned_queries),
            "executed_queries_count": sum(1 for q in plan.planned_queries if q.execution_status.value == "EXECUTED"),
            "unavailable_queries_count": sum(1 for q in plan.planned_queries if q.execution_status.value == "UNAVAILABLE"),
            "candidate_urls_count": len(plan.candidate_urls),
            "coverage_summary": cov_summary,
            "timebox": plan.timebox.model_dump(mode="json"),
            "stopping_rule_id": plan.stopping_rule_id,
            "stop_status": plan.stop_status.value,
            "stop_progress_notes": plan.stop_progress_notes,
            "depth_completed": ai_verdict.depth_completed,
        }

        report = VendorEndToEndAssessmentReport(
            report_id=f"RPT_{vendor.vendor_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            generated_at=now_iso,
            is_synthetic_evidence_run=is_synthetic_run,
            vendor_id=vendor.vendor_id,
            legal_entity=vendor.vendor_name,
            domain=vendor.domain,
            category=vendor.category,
            vendor_description=vendor.vendor_description,
            service_product_provided=vendor.service_product_provided,
            business_process_supported=vendor.business_process_supported,
            criticality_summary=crit_audit,
            osint_plan_summary=osint_summary,
            url_triage_decisions=triage_decisions,
            extracted_documents_count=len(extracted_docs),
            evidence_records=evidence_records,
            semantic_retrieval=rag_summary,
            signal_classifications=classifications,
            weighted_evidence=weighted_items,
            gates=gates,
            ai_use_verdict=ai_verdict,
            determined_ai_types=ai_types_found,
            determined_scope=det_scope,
            named_models=named_models,
            named_providers=named_providers,
            hosting_infrastructures=hosting_infras,
            supply_chain_identity_status=supply_chain_status,
            risk_findings=risk_findings,
            known_controls_summary=known_controls,
            remediation_recommendations=remediations,
            deterministic_risk_score=risk_score_res,
            human_Review_required=bool(review_triggers),
            human_review_triggers=review_triggers,
            human_validations=all_validations,
            unresolved_questions=unresolved_questions,
            stage_ledger=stage_ledger,
            model_routing_telemetry=self.model_router.get_full_telemetry_summary(),
            unavailable_integrations=unavailable_integrations,
            assessment_limitations=limitations + list(ai_verdict.incomplete_reasons),
        )


        if persist_to_repository and self.evidence_repository is not None:
            for doc in extracted_docs:
                self.evidence_repository.save_document(doc)
            self.evidence_repository.save_evidence_batch(evidence_records)
            for hv in all_validations:
                self.evidence_repository.save_human_validation(hv)
            self.evidence_repository.save_vendor_report(report)

        return report
