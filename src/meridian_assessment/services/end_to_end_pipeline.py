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
import time


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
        enable_triage_llm: Optional[bool] = None,
        enable_rag_embeddings: Optional[bool] = None,
        enable_classification_llm: Optional[bool] = None,
        enable_summary_llm: Optional[bool] = None,
        triage_llm_provider: Optional[URLTriageLLMProvider] = None,
        fetch_adapter: Optional[ContentFetchAdapter] = None,
        evidence_repository: Optional[EvidenceRepository] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        classification_llm_adapter: Optional[LocalLLMClassificationAdapter] = None,
        risk_summary_llm_adapter: Optional[RiskSummaryLLMProvider] = None,
        scoring_engine: Optional[DeterministicRiskScoringEngine] = None,
        readiness: Optional[Any] = None,
    ) -> None:
        self.criticality_engine = criticality_engine or MeridianCriticalityEngine()
        self.investigation_planner = investigation_planner or OSINTInvestigationPlanner()
        self.search_backend = search_backend or build_search_backend_from_env()
        self._local_llm_disable_reason: Optional[str] = None

        any_stage_llm_requested = any(
            flag is True for flag in (enable_triage_llm, enable_classification_llm, enable_summary_llm)
        )
        effective_master_llm = (
            True
            if any_stage_llm_requested
            else enable_local_llm_from_env
        )

        if model_router is not None:
            self.model_router = model_router
        else:
            # In normal CLI/UI execution, load .env and enable dual-model routing if MERIDIAN_OLLAMA_ENABLED=true
            # AND the local Ollama server and required models are reachable/installed.
            # When running inside pytest without an explicit model_router or enable_local_llm_from_env=True,
            # default to offline deterministic mode so unit tests remain fast and network-free.
            in_pytest = "PYTEST_CURRENT_TEST" in os.environ
            cfg = ModelRoutingConfig.from_env_and_config()
            if enable_rag_embeddings is not None:
                cfg.enable_rag_embeddings = bool(enable_rag_embeddings)
            if effective_master_llm is False:
                cfg.enabled = False
                self._local_llm_disable_reason = (
                    "Local AI analysis disabled for deterministic run; deterministic rules applied."
                )
            elif effective_master_llm is None and in_pytest:
                cfg.enabled = False
                self._local_llm_disable_reason = (
                    "Local AI analysis bypassed in offline test mode; deterministic rules applied."
                )
            else:
                if effective_master_llm is True:
                    cfg.enabled = True
                if cfg.enabled:
                    if readiness is None and not in_pytest:
                        from meridian_assessment.services.readiness_and_exports import (
                            SystemReadinessInspector,
                        )

                        readiness = SystemReadinessInspector.check_system_readiness(
                            local_analysis_override=effective_master_llm
                        )
                    if readiness is not None and not getattr(readiness, "ollama_reachable", False):
                        cfg.enabled = False
                        self._local_llm_disable_reason = (
                            "Local analysis unavailable: local model service is unreachable."
                        )
                else:
                    self._local_llm_disable_reason = (
                        "Local analysis unavailable: disabled in configuration."
                    )
            self.model_router = OllamaModelRouter(config=cfg)

        llama_ok = True if readiness is None else bool(getattr(readiness, "llama3_2_installed", False))
        mistral_ok = True if readiness is None else bool(getattr(readiness, "mistral_installed", False))

        triage_enabled = (
            enable_triage_llm
            if enable_triage_llm is not None
            else self.model_router.config.enabled
        )
        cls_enabled = (
            enable_classification_llm
            if enable_classification_llm is not None
            else self.model_router.config.enabled
        )
        sum_enabled = (
            enable_summary_llm
            if enable_summary_llm is not None
            else self.model_router.config.enabled
        )

        if triage_llm_provider is not None:
            resolved_triage_llm: Optional[URLTriageLLMProvider] = triage_llm_provider
        elif self.model_router.config.enabled and triage_enabled and llama_ok:
            resolved_triage_llm = OllamaURLTriageProvider(router=self.model_router)
        else:
            resolved_triage_llm = None

        if classification_llm_adapter is not None:
            resolved_cls_llm: Optional[LocalLLMClassificationAdapter] = classification_llm_adapter
        elif self.model_router.config.enabled and cls_enabled and mistral_ok:
            resolved_cls_llm = OllamaClassificationAdapter(router=self.model_router)
        else:
            if enable_classification_llm is False:
                cls_reason = "Level 6 Local AI classification disabled by stage toggle; deterministic rules applied."
            elif self.model_router.config.enabled and not mistral_ok:
                cls_reason = (
                    f"Local analysis unavailable: required analysis model "
                    f"'{self.model_router.config.analysis_model}' is not installed."
                )
            elif self._local_llm_disable_reason:
                cls_reason = self._local_llm_disable_reason
            else:
                cls_reason = "Local analysis unavailable: local model service is not configured."
            resolved_cls_llm = UnconfiguredOllamaAdapter(reason=cls_reason)

        if risk_summary_llm_adapter is not None:
            resolved_sum_llm: Optional[RiskSummaryLLMProvider] = risk_summary_llm_adapter
        elif self.model_router.config.enabled and sum_enabled and mistral_ok:
            resolved_sum_llm = OllamaRiskSummaryAdapter(router=self.model_router)
        else:
            resolved_sum_llm = None

        from meridian_assessment.services.rag.chunker_retriever import SentenceTransformerEmbeddingProvider
        from meridian_assessment.services.readiness_and_exports import _check_huggingface_model_cached_locally
        import importlib.util

        in_pytest_env = "PYTEST_CURRENT_TEST" in os.environ
        if enable_rag_embeddings is not None:
            rag_enabled_flag = bool(enable_rag_embeddings)
        elif in_pytest_env and model_router is None and effective_master_llm is None:
            rag_enabled_flag = False
        else:
            rag_enabled_flag = bool(self.model_router.config.enable_rag_embeddings)

        if embedding_provider is not None:
            resolved_emb_provider: EmbeddingProvider = embedding_provider
        elif (
            rag_enabled_flag
            and importlib.util.find_spec("sentence_transformers") is not None
            and _check_huggingface_model_cached_locally(self.model_router.config.embedding_model)
        ):
            resolved_emb_provider = SentenceTransformerEmbeddingProvider(
                model_name=self.model_router.config.embedding_model,
                auto_load=True,
            )
        else:
            emb_reason = (
                f"Embedding model '{self.model_router.config.embedding_model}' is disabled "
                "(Level 5 RAG toggle off); Level 5 runs deterministic 500-token chunking "
                "and passes all extracted evidence directly to Level 6."
                if not rag_enabled_flag
                else f"Embedding model '{self.model_router.config.embedding_model}' or 'sentence-transformers' is not installed/cached locally; "
                "Level 5 runs deterministic 500-token chunking and passes all extracted evidence directly to Level 6."
            )
            resolved_emb_provider = UnconfiguredEmbeddingProvider(
                model_name=self.model_router.config.embedding_model,
                reason=emb_reason,
            )

        self.triage_service = OSINTURLTriageService(llm_provider=resolved_triage_llm)
        self.fetch_adapter = fetch_adapter or DisabledOfflineFetchAdapter()
        self.evidence_repository = evidence_repository
        self.extraction_service = OSINTExtractionService(
            fetch_adapter=self.fetch_adapter,
            source_registry=self.investigation_planner.source_registry,
            artifact_dir=self.evidence_repository.artifact_dir if self.evidence_repository else None,
        )
        self.rag_service = RAGChunkerAndRetriever(
            embedding_provider=resolved_emb_provider,
            evidence_repository=self.evidence_repository,
        )
        self.classification_service = SignalClassificationService(
            llm_adapter=resolved_cls_llm,
            evidence_repository=self.evidence_repository,
        )
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
        reuse_cached_artifacts: bool = True,
        use_verified_local_artifacts_only: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> VendorEndToEndAssessmentReport:
        """Execute all 9 project levels in order and produce a complete `VendorEndToEndAssessmentReport`."""
        t_pipeline_start = time.perf_counter()
        stage_timings_ms: dict[str, float] = {}
        now_iso = _utc_now_iso()
        run_id = f"RUN_{vendor.vendor_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        stage_ledger: list[StageExecutionRecord] = []
        unavailable_integrations: list[str] = []
        limitations: list[str] = []

        _emitted_3arg_levels: set[int] = set()

        def _notify_progress(
            level_num: int,
            stage_title: str,
            detail: str,
            completed: int = 0,
            total: int = 1,
        ) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(level_num, stage_title, detail, completed, total)
                except TypeError:
                    if level_num not in _emitted_3arg_levels:
                        _emitted_3arg_levels.add(level_num)
                        try:
                            progress_callback(level_num, stage_title, detail)
                        except Exception:
                            pass
                except Exception:
                    pass

        # Reset per-run call telemetry on router so each vendor report captures its exact calls
        self.model_router.call_history.clear()
        self.model_router.vram_unload_events.clear()

        # ── LEVEL 1: Criticality Tiering (D/P/R/O/V + O1–O5) ──
        t_l1 = time.perf_counter()
        _notify_progress(
            1,
            "Level 1: Criticality Tiering",
            f"Evaluating D/P/R/O/V factors and O1–O5 overrides for {vendor.vendor_name}...",
            completed=0,
            total=1,
        )
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
        stage_timings_ms["level_1_ms"] = round((time.perf_counter() - t_l1) * 1000.0, 2)
        _notify_progress(
            1,
            "Level 1: Criticality Tiering",
            f"Completed criticality tiering: Effective Criticality='{effective_crit}' (Base={crit_res.base_score}).",
            completed=1,
            total=1,
        )
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
        t_l2 = time.perf_counter()
        _notify_progress(
            2,
            "Level 2: Assessment Depth Scoping",
            f"Mapping '{effective_crit}' criticality to S1–S12 depth profile and query plan...",
            completed=0,
            total=1,
        )
        plan = self.investigation_planner.plan_investigation(
            vendor=vendor,
            criticality_input=crit_res,
            human_review=human_criticality_override,
            existing_plan=existing_plan,
        )
        stage_timings_ms["level_2_ms"] = round((time.perf_counter() - t_l2) * 1000.0, 2)
        _notify_progress(
            2,
            "Level 2: Assessment Depth Scoping",
            f"Scoped '{plan.selected_depth_profile}' profile ({len(plan.planned_queries)} queries across {len(plan.required_source_classes)} required sources).",
            completed=1,
            total=1,
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
        t_l3 = time.perf_counter()
        search_ok, search_msg = self.search_backend.is_configured()
        triage_llm_ok, triage_llm_msg = self.triage_service.llm_provider.is_configured()
        triage_model_name = (
            getattr(self.triage_service.llm_provider, "model_name", None)
            or self.model_router.resolve_model_for_task(LLMTaskType.URL_TRIAGE)
        )
        total_planned_q = max(1, len(plan.planned_queries))
        _notify_progress(
            3,
            "Level 3: Source Identification & Reconnaissance",
            f"Starting Level 3A OSINT search ({self.search_backend.provider_name}) across {len(plan.planned_queries)} queries...",
            completed=0,
            total=total_planned_q,
        )
        if execute_searches and not use_verified_local_artifacts_only:
            collector = OSINTSearchCollector(backend=self.search_backend, retry_backoff_seconds=0.0)
            collector.execute_planned_queries(
                plan,
                persist=False,
                progress_callback=lambda q_done, q_tot, q_obj: _notify_progress(
                    3,
                    "Level 3: Source Identification & Reconnaissance (3A Search)",
                    f"Executing search query {q_done}/{q_tot} ({max(0, q_tot - q_done)} left): {q_obj.query_id}",
                    completed=q_done,
                    total=q_tot,
                ),
            )
        if not search_ok and not use_verified_local_artifacts_only:
            unavailable_integrations.append(f"Search Backend ({self.search_backend.provider_name}): {search_msg}")

        def _on_triage_progress(c_done: int, c_tot: int, c_obj: Any, is_llm: bool = False) -> None:
            method_str = triage_model_name if (is_llm and triage_llm_ok) else "Deterministic"
            _notify_progress(
                3,
                "Level 3: Source Identification & Reconnaissance (3B URL Triage)",
                f"Triaging candidate URL {c_done}/{c_tot} ({max(0, c_tot - c_done)} left) [{method_str}]: {c_obj.url[:60]}",
                completed=c_done,
                total=c_tot,
            )

        triage_decisions: list[URLTriageDecision] = self.triage_service.triage_plan_candidates(
            plan,
            progress_callback=_on_triage_progress,
        )
        fetch_decisions = [d for d in triage_decisions if d.action == URLTriageAction.FETCH]
        stage_timings_ms["level_3_ms"] = round((time.perf_counter() - t_l3) * 1000.0, 2)

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
            if (plan.candidate_urls or (execute_searches and search_ok) or use_verified_local_artifacts_only)
            else "NOT_CONFIGURED"
            if not search_ok
            else "PARTIAL"
        )
        l3_limits: list[str] = [] if (search_ok or use_verified_local_artifacts_only) else [search_msg]
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
                    f"(Level 3B Triage Model: {triage_model_name} [{l3_model_status}])."
                ),
                limitations=l3_limits,
                selected_model=triage_model_name,
                model_task=LLMTaskType.URL_TRIAGE.value,
                model_execution_status=l3_model_status,
            )
        )

        # ── LEVEL 4: Data Scraping & Point-of-Capture Traceability ──
        t_l4 = time.perf_counter()
        total_fetches = max(1, len(fetch_decisions))
        _notify_progress(
            4,
            "Level 4: Data Scraping & Point-of-Capture Traceability",
            f"Extracting content and computing SHA-256 provenance from {len(fetch_decisions)} triaged URL(s)...",
            completed=0,
            total=total_fetches,
        )
        extracted_docs: list[ExtractedDocument] = []
        evidence_records: list[EvidenceRecord] = (
            [ev.model_copy(deep=True) for ev in pre_supplied_evidence]
            if pre_supplied_evidence
            else []
        )

        from meridian_assessment.services.extraction.extractor import StubFetchAdapter
        is_stub_fetch = isinstance(self.fetch_adapter, StubFetchAdapter)

        seen_normalized_hashes: dict[str, EvidenceRecord] = {}
        seen_shingle_sets: list[tuple[frozenset[str], EvidenceRecord]] = []
        seen_doc_hashes: dict[str, list[EvidenceRecord]] = {}
        raw_passages_total = 0
        deterministic_retained_total = 0
        deterministic_screened_out_total = 0
        artifact_cache_hits = 0
        duplicate_documents_skipped = 0

        fetch_ok, fetch_msg = self.fetch_adapter.is_configured()

        if use_verified_local_artifacts_only and self.evidence_repository is not None:
            local_artifacts = self.evidence_repository.discover_verified_vendor_artifacts(
                vendor.vendor_id,
                include_synthetic=False,
            )
            seq_ev = len(evidence_records) + 1
            for art_idx, (art_path, raw_bytes, _full_sha, seq_num, _ext) in enumerate(local_artifacts, start=1):
                doc_id = f"DOC_{vendor.vendor_id}_{seq_num:03d}"
                _notify_progress(
                    4,
                    "Level 4: Data Scraping & Point-of-Capture Traceability",
                    f"Reusing verified local artifact {art_idx}/{len(local_artifacts)}: {art_path.name}",
                    completed=art_idx,
                    total=max(1, len(local_artifacts)),
                )
                cached_doc = (
                    self.evidence_repository.find_cached_document(
                        vendor_id=vendor.vendor_id,
                        document_id=doc_id,
                    )
                    if reuse_cached_artifacts
                    else None
                )
                synthetic_url = (
                    cached_doc.url
                    if cached_doc is not None
                    else f"https://{vendor.domain}/local-artifact/{art_path.name}"
                )
                source_class = cached_doc.source_class if cached_doc is not None else "S1"
                doc = self.extraction_service.extract_from_artifact_bytes(
                    vendor_id=vendor.vendor_id,
                    document_id=doc_id,
                    content_bytes=raw_bytes,
                    url=synthetic_url,
                    source_class=source_class,
                    artifact_path=str(art_path),
                    is_synthetic_fixture=False,
                )
                artifact_cache_hits += 1
                extracted_docs.append(doc)
                if doc.extraction_status == ExtractionStatus.EXTRACTED:
                    if doc.content_sha256 in seen_doc_hashes:
                        duplicate_documents_skipped += 1
                        for existing_ev in seen_doc_hashes[doc.content_sha256]:
                            if doc.document_id not in existing_ev.corroborating_document_ids:
                                existing_ev.corroborating_document_ids.append(doc.document_id)
                            if doc.canonical_url and doc.canonical_url not in existing_ev.corroborating_urls:
                                existing_ev.corroborating_urls.append(doc.canonical_url)
                        if persist_to_repository and self.evidence_repository is not None:
                            self.evidence_repository.save_document(doc)
                        continue
                    new_evs = self.extraction_service.build_evidence_records_from_document(
                        doc,
                        legal_entity=vendor.vendor_name,
                        domain=vendor.domain,
                        starting_seq=seq_ev,
                        seen_normalized_hashes=seen_normalized_hashes,
                        seen_shingle_sets=seen_shingle_sets,
                    )
                    seen_doc_hashes[doc.content_sha256] = new_evs
                    raw_passages_total += doc.raw_blocks_count
                    deterministic_retained_total += doc.unique_passages_count
                    deterministic_screened_out_total += (
                        doc.boilerplate_filtered_count
                        + doc.exact_duplicates_filtered_count
                        + doc.near_duplicates_filtered_count
                    )
                    seq_ev += len(new_evs)
                    evidence_records.extend(new_evs)
                    if persist_to_repository and self.evidence_repository is not None:
                        self.evidence_repository.save_document(doc)
                        if new_evs:
                            self.evidence_repository.save_evidence_batch(new_evs)
                else:
                    if persist_to_repository and self.evidence_repository is not None:
                        self.evidence_repository.save_document(doc)
                    limitations.extend(doc.extraction_limitations)

        elif execute_extraction and fetch_decisions:
            seq_ev = len(evidence_records) + 1
            for doc_idx, t_dec in enumerate(fetch_decisions, start=1):
                _notify_progress(
                    4,
                    "Level 4: Data Scraping & Point-of-Capture Traceability",
                    f"Scraping & verifying URL {doc_idx}/{len(fetch_decisions)} ({max(0, len(fetch_decisions) - doc_idx)} left): {t_dec.url[:65]}",
                    completed=doc_idx,
                    total=len(fetch_decisions),
                )
                cand_obj = next((c for c in plan.candidate_urls if c.candidate_id == t_dec.candidate_id), None)
                q_ids = cand_obj.query_ids if cand_obj else []
                doc_id_candidate = f"DOC_{vendor.vendor_id}_{doc_idx:03d}"

                doc: Optional[ExtractedDocument] = None
                if reuse_cached_artifacts and self.evidence_repository is not None and not is_stub_fetch:
                    t_url = getattr(t_dec, "url", getattr(t_dec, "canonical_url", None))
                    cached_doc = self.evidence_repository.find_cached_document(
                        vendor_id=vendor.vendor_id,
                        canonical_url=t_url,
                        document_id=doc_id_candidate,
                    )
                    if cached_doc is not None and cached_doc.artifact_path:
                        art_path = Path(cached_doc.artifact_path)
                        if art_path.exists():
                            raw_bytes = art_path.read_bytes()
                            doc = self.extraction_service.extract_from_artifact_bytes(
                                vendor_id=vendor.vendor_id,
                                document_id=cached_doc.document_id,
                                content_bytes=raw_bytes,
                                url=t_dec.url,
                                source_class=t_dec.candidate_source_class,
                                artifact_path=str(art_path),
                                query_ids=q_ids,
                                triage_action=t_dec.action.value,
                                is_synthetic_fixture=False,
                            )
                            artifact_cache_hits += 1

                if doc is None:
                    try:
                        doc = self.extraction_service.extract_from_url(
                            vendor_id=vendor.vendor_id,
                            url=t_dec.url,
                            source_class=t_dec.candidate_source_class,
                            query_ids=q_ids,
                            triage_action=t_dec.action.value,
                            document_index=doc_idx,
                            is_synthetic_fixture=is_stub_fetch,
                        )
                    except Exception as exc:
                        doc = ExtractedDocument(
                            document_id=doc_id_candidate,
                            vendor_id=vendor.vendor_id,
                            url=t_dec.url,
                            canonical_url=t_dec.url,
                            requested_url=t_dec.url,
                            final_response_url="UNKNOWN",
                            retrieval_method="FETCH_FAILED",
                            source_integrity_status="FETCH_ERROR",
                            source_class=t_dec.candidate_source_class,
                            content_type="unknown",
                            retrieved_at=now_iso,
                            extraction_status=ExtractionStatus.FETCH_FAILED,
                            extraction_limitations=[f"Extraction failed for candidate URL: {exc}"],
                            query_ids=q_ids,
                            triage_action=t_dec.action.value,
                            is_synthetic_fixture=is_stub_fetch,
                        )

                extracted_docs.append(doc)
                if doc.extraction_status == ExtractionStatus.EXTRACTED:
                    if doc.content_sha256 in seen_doc_hashes:
                        duplicate_documents_skipped += 1
                        for existing_ev in seen_doc_hashes[doc.content_sha256]:
                            if doc.document_id not in existing_ev.corroborating_document_ids:
                                existing_ev.corroborating_document_ids.append(doc.document_id)
                            if doc.canonical_url and doc.canonical_url not in existing_ev.corroborating_urls:
                                existing_ev.corroborating_urls.append(doc.canonical_url)
                        if persist_to_repository and self.evidence_repository is not None:
                            self.evidence_repository.save_document(doc)
                        continue
                    new_evs = self.extraction_service.build_evidence_records_from_document(
                        doc,
                        legal_entity=vendor.vendor_name,
                        domain=vendor.domain,
                        starting_seq=seq_ev,
                        seen_normalized_hashes=seen_normalized_hashes,
                        seen_shingle_sets=seen_shingle_sets,
                    )
                    seen_doc_hashes[doc.content_sha256] = new_evs
                    raw_passages_total += doc.raw_blocks_count
                    deterministic_retained_total += doc.unique_passages_count
                    deterministic_screened_out_total += (
                        doc.boilerplate_filtered_count
                        + doc.exact_duplicates_filtered_count
                        + doc.near_duplicates_filtered_count
                    )
                    seq_ev += len(new_evs)
                    evidence_records.extend(new_evs)
                    if persist_to_repository and self.evidence_repository is not None:
                        self.evidence_repository.save_document(doc)
                        if new_evs:
                            self.evidence_repository.save_evidence_batch(new_evs)
                else:
                    if persist_to_repository and self.evidence_repository is not None:
                        self.evidence_repository.save_document(doc)
                    limitations.extend(doc.extraction_limitations)

        if not fetch_ok and not evidence_records and not use_verified_local_artifacts_only:
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

        mismatched_docs = [
            d for d in extracted_docs if d.extraction_status == ExtractionStatus.SOURCE_IDENTITY_MISMATCH
        ]
        stage_timings_ms["level_4_ms"] = round((time.perf_counter() - t_l4) * 1000.0, 2)

        if persist_to_repository and self.evidence_repository is not None:
            self.evidence_repository.save_checkpoint(
                vendor_id=vendor.vendor_id,
                run_id=run_id,
                completed_level=4,
                checkpoint_state={
                    "extracted_documents_count": len(extracted_docs),
                    "evidence_records_count": len(evidence_records),
                    "raw_passages_total": raw_passages_total,
                    "deterministic_retained_total": deterministic_retained_total,
                    "deterministic_screened_out_total": deterministic_screened_out_total,
                    "artifact_cache_hits": artifact_cache_hits,
                    "duplicate_documents_skipped": duplicate_documents_skipped,
                },
            )

        l4_status = (
            "COMPLETED"
            if evidence_records or reviewed_no_evidence_sources
            else "NOT_CONFIGURED"
            if not fetch_ok and not use_verified_local_artifacts_only
            else "PARTIAL"
            if mismatched_docs
            else "SKIPPED_NO_INPUT"
        )
        l4_limits: list[str] = []
        for m_doc in mismatched_docs:
            l4_limits.extend(m_doc.extraction_limitations)
        if not evidence_records and not l4_limits:
            l4_limits.append("No extracted evidence records present.")
        mismatch_note = (
            f" ({len(mismatched_docs)} rejected due to SOURCE_IDENTITY_MISMATCH)"
            if mismatched_docs
            else ""
        )
        stage_ledger.append(
            StageExecutionRecord(
                level_number=4,
                level_name="Data Scraping & Point-of-Capture Traceability",
                status=l4_status,
                summary=(
                    f"{len(extracted_docs)} documents processed{mismatch_note}; "
                    f"{len(evidence_records)} deduplicated EvidenceRecord items created "
                    f"(from {raw_passages_total or len(evidence_records)} raw blocks; "
                    f"{deterministic_screened_out_total} boilerplate/duplicate blocks screened out)."
                ),
                limitations=l4_limits,
                selected_model="N/A",
                model_task="DOCUMENT_EXTRACTION",
                model_execution_status="DETERMINISTIC_ONLY",
            )
        )

        # ── LEVEL 5: Semantic Filtering & RAG Chunking (Separate Embedding Model) ──
        t_l5 = time.perf_counter()
        total_ev_l5 = max(1, len(evidence_records))
        _notify_progress(
            5,
            "Level 5: Semantic Filtering & RAG Chunking",
            f"Chunking & ranking {len(evidence_records)} evidence item(s) ({self.rag_service.model_name})...",
            completed=0,
            total=total_ev_l5,
        )
        _, rag_summary = self.rag_service.retrieve_relevant_chunks(
            evidence_records,
            legal_entity=vendor.vendor_name,
        )
        stage_timings_ms["level_5_ms"] = round((time.perf_counter() - t_l5) * 1000.0, 2)
        _notify_progress(
            5,
            "Level 5: Semantic Filtering & RAG Chunking",
            (
                f"Created {rag_summary.chunks_created} chunks; selected "
                f"{rag_summary.candidates_selected_for_llm} high-signal candidate(s) for Level 6 "
                f"({rag_summary.candidates_screened_out_semantic} screened out)."
            ),
            completed=total_ev_l5,
            total=total_ev_l5,
        )
        selected_for_llm_ids = {
            ev.evidence_id
            for ev in evidence_records
            if ev.screening_status == "SELECTED_FOR_LLM"
        }
        if selected_for_llm_ids:
            rag_candidate_ids = selected_for_llm_ids
        else:
            rag_candidate_ids = {
                m.chunk.evidence_id
                for m in rag_summary.matches
                if m.chunk is not None and m.chunk.evidence_id
            }

        if persist_to_repository and self.evidence_repository is not None and evidence_records:
            self.evidence_repository.save_evidence_batch(evidence_records)
            self.evidence_repository.save_checkpoint(
                vendor_id=vendor.vendor_id,
                run_id=run_id,
                completed_level=5,
                checkpoint_state={
                    "chunks_created": rag_summary.chunks_created,
                    "chunks_embedded": rag_summary.chunks_embedded,
                    "embedding_cache_hits": rag_summary.embedding_cache_hits,
                    "candidates_selected_for_llm": rag_summary.candidates_selected_for_llm,
                    "candidates_screened_out_semantic": rag_summary.candidates_screened_out_semantic,
                    "selected_evidence_ids": sorted(rag_candidate_ids),
                },
            )

        emb_ok, emb_msg = self.rag_service.embedding_provider.is_available()
        if not emb_ok:
            unavailable_integrations.append(f"RAG Embedding Model ({self.rag_service.model_name}): {emb_msg}")

        l5_model_status = (
            "SUCCEEDED"
            if (rag_summary.model_available and (rag_summary.chunks_embedded > 0 or rag_summary.embedding_cache_hits > 0))
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
                    f"{rag_summary.chunks_embedded} embedded ({rag_summary.embedding_cache_hits} cached); "
                    f"{rag_summary.candidates_selected_for_llm} candidates selected for Level 6 "
                    f"({rag_summary.status})."
                ),
                limitations=list(rag_summary.limitations),
                selected_model=self.rag_service.model_name,
                model_task=LLMTaskType.RAG_EMBEDDING.value,
                model_execution_status=l5_model_status,
            )
        )

        # ── LEVEL 6: Signal Classification (Model B mistral:latest + Prompt-Injection Defence) ──
        t_l6 = time.perf_counter()
        llm_ok, llm_msg = self.classification_service.llm_adapter.is_configured()
        cls_model_name = (
            getattr(self.classification_service.llm_adapter, "model_name", None)
            or self.model_router.resolve_model_for_task(LLMTaskType.EVIDENCE_CLASSIFICATION)
        )
        total_ev_l6 = max(1, len(evidence_records))
        _notify_progress(
            6,
            "Level 6: Signal Classification",
            f"Classifying {len(evidence_records)} evidence item(s) ({len(rag_candidate_ids)} routed to batched {cls_model_name if llm_ok else 'Deterministic'}, E14 guard active)...",
            completed=0,
            total=total_ev_l6,
        )
        if not llm_ok:
            unavailable_integrations.append(f"Local LLM Classifier ({cls_model_name}): {llm_msg}")

        def _l6_checkpoint_cb(completed_items: int, total_items: int, partial_cls: Sequence[SignalClassificationResult]) -> None:
            if persist_to_repository and self.evidence_repository is not None:
                self.evidence_repository.save_checkpoint(
                    vendor_id=vendor.vendor_id,
                    run_id=run_id,
                    completed_level=6,
                    checkpoint_state={
                        "status": "IN_PROGRESS",
                        "completed_items": completed_items,
                        "total_items": total_items,
                        "classified_ids": [c.evidence_id for c in partial_cls],
                    },
                )

        classifications = self.classification_service.classify_evidence_batch(
            evidence_records,
            meridian_service_scope=vendor.service_product_provided,
            meridian_business_process=vendor.business_process_supported,
            rag_candidate_evidence_ids=rag_candidate_ids,
            max_llm_calls=self.rag_service.max_total_candidates,
            progress_callback=lambda ev_done, ev_tot, ev_obj: _notify_progress(
                6,
                "Level 6: Signal Classification",
                f"Classifying evidence item {ev_done}/{ev_tot} ({max(0, ev_tot - ev_done)} left) [{cls_model_name if llm_ok else 'Deterministic'}]: {ev_obj.evidence_id}",
                completed=ev_done,
                total=ev_tot,
            ),
            checkpoint_callback=_l6_checkpoint_cb,
        )
        stage_timings_ms["level_6_ms"] = round((time.perf_counter() - t_l6) * 1000.0, 2)

        if persist_to_repository and self.evidence_repository is not None:
            if evidence_records:
                self.evidence_repository.save_evidence_batch(evidence_records)
            self.evidence_repository.save_checkpoint(
                vendor_id=vendor.vendor_id,
                run_id=run_id,
                completed_level=6,
                checkpoint_state={
                    "status": "COMPLETED",
                    "classified_count": len(classifications),
                    "batch_telemetry": dict(self.classification_service.last_batch_telemetry),
                },
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
        l6_batch_tel = self.classification_service.last_batch_telemetry
        stage_ledger.append(
            StageExecutionRecord(
                level_number=6,
                level_name="Signal Classification",
                status="COMPLETED" if classifications else "SKIPPED_NO_INPUT",
                summary=(
                    f"{len(classifications)} evidence items classified "
                    f"({l6_batch_tel.get('llm_batches_invoked', 0)} LLM batches / "
                    f"{l6_batch_tel.get('llm_calls_invoked', 0)} calls; "
                    f"{l6_batch_tel.get('cache_hits', 0)} cache hits; "
                    f"Analysis Model: {cls_model_name} [{l6_model_status}])."
                ),
                limitations=l6_limits,
                selected_model=cls_model_name,
                model_task=LLMTaskType.EVIDENCE_CLASSIFICATION.value,
                model_execution_status=l6_model_status,
            )
        )

        # ── LEVEL 7: Evidentiary Weighting (E1–E14), G1–G4 Gates & AI-Use Verdict ──
        t_l7 = time.perf_counter()
        total_ev_l7 = max(1, len(evidence_records))
        _notify_progress(
            7,
            "Level 7: Evidentiary Weighting",
            "Applying deterministic E1–E14 rules, G1–G4 gates, and AI-use verdict logic...",
            completed=0,
            total=total_ev_l7,
        )
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
        stage_timings_ms["level_7_ms"] = round((time.perf_counter() - t_l7) * 1000.0, 2)
        _notify_progress(
            7,
            "Level 7: Evidentiary Weighting",
            f"Completed E1–E14 weighting across {len(weighted_items)} items; AI-Use Verdict: {ai_verdict.verdict.value}.",
            completed=total_ev_l7,
            total=total_ev_l7,
        )

        stage_ledger.append(
            StageExecutionRecord(
                level_number=7,
                level_name="Evidentiary Weighting",
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
        t_l8 = time.perf_counter()
        sum_llm_ok, sum_llm_msg = self.risk_analyzer.summary_llm_adapter.is_configured()
        sum_model_name = (
            getattr(self.risk_analyzer.summary_llm_adapter, "model_name", None)
            or self.model_router.resolve_model_for_task(LLMTaskType.RISK_SUMMARIZATION)
        )
        _notify_progress(
            8,
            "Level 8: Risk Surfacing & Summarization",
            f"Evaluating R1–R8 risk rules and drafting evidence-grounded explanations ({sum_model_name if sum_llm_ok else 'Deterministic'})...",
            completed=0,
            total=8,
        )
        risk_findings = self.risk_analyzer.analyze_vendor_risks(
            vendor_id=vendor.vendor_id,
            evidence_records=evidence_records,
            weighted_items=weighted_items,
            gates=gates,
            verdict=ai_verdict,
            data_classification_accessed=vendor.data_classification_accessed,
            service_product_provided=vendor.service_product_provided,
            progress_callback=lambda rf_done, rf_tot, rf_obj: _notify_progress(
                8,
                "Level 8: Risk Surfacing & Summarization",
                f"Summarizing risk finding {rf_done}/{rf_tot} ({max(0, rf_tot - rf_done)} left) [{sum_model_name if sum_llm_ok else 'Deterministic'}]: {rf_obj.risk_rule_id}",
                completed=rf_done,
                total=rf_tot,
            ),
        )
        stage_timings_ms["level_8_ms"] = round((time.perf_counter() - t_l8) * 1000.0, 2)
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
                level_name="Risk Surfacing & Summarization",
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
        t_l9 = time.perf_counter()
        _notify_progress(
            9,
            "Level 9: Deterministic Risk Rating",
            "Evaluating deterministic risk formula (M_c * sum(S_i * W_i)) and policy readiness...",
            completed=1,
            total=1,
        )
        completeness_str = "COMPLETED" if ai_verdict.depth_completed else "INCOMPLETE"
        risk_score_res = self.scoring_engine.calculate_vendor_risk_score(
            vendor_id=vendor.vendor_id,
            criticality_tier=effective_crit,
            findings=risk_findings,
            weighted_evidence=weighted_items,
            assessment_completeness_status=completeness_str,
        )
        stage_timings_ms["level_9_ms"] = round((time.perf_counter() - t_l9) * 1000.0, 2)
        stage_timings_ms["total_pipeline_ms"] = round((time.perf_counter() - t_pipeline_start) * 1000.0, 2)
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
        valid_ai_evidence = [
            e for e in valid_non_hostile
            if e.ai_type.value in ("GENAI_LLM", "CLASSICAL_ML", "AGENTIC_AI")
        ]
        ai_types_found = sorted({e.ai_type.value for e in valid_ai_evidence})
        named_models = sorted({e.named_model for e in valid_ai_evidence if e.named_model})
        named_providers = sorted({e.named_provider for e in valid_ai_evidence if e.named_provider})
        hosting_infras = sorted({e.hosting_infrastructure for e in valid_ai_evidence if e.hosting_infrastructure})

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

        if mismatched_docs:
            review_triggers.append(
                f"Source-identity mismatch detected on {len(mismatched_docs)} fetched document(s); rejected from evidence register."
            )

        pipeline_perf_telemetry = {
            "run_id": run_id,
            "stage_timings_ms": stage_timings_ms,
            "level_4_extraction": {
                "documents_processed": len(extracted_docs),
                "artifact_cache_hits": artifact_cache_hits,
                "duplicate_documents_skipped": duplicate_documents_skipped,
                "raw_passages_total": raw_passages_total or len(evidence_records),
                "deterministic_retained_total": len(evidence_records),
                "deterministic_screened_out_total": deterministic_screened_out_total,
            },
            "level_5_retrieval": {
                "status": rag_summary.status,
                "model_available": rag_summary.model_available,
                "chunks_created": rag_summary.chunks_created,
                "chunks_embedded": rag_summary.chunks_embedded,
                "embedding_cache_hits": rag_summary.embedding_cache_hits,
                "candidates_selected_for_llm": rag_summary.candidates_selected_for_llm,
                "candidates_screened_out_semantic": rag_summary.candidates_screened_out_semantic,
                "retrieval_duration_ms": rag_summary.retrieval_duration_ms,
            },
            "level_6_classification": dict(l6_batch_tel),
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
            data_classification_accessed=vendor.data_classification_accessed,
            operational_dependency=vendor.operational_dependency,
            data_volume_annual=vendor.data_volume_annual,
            criticality_summary=crit_audit,
            osint_plan_summary=osint_summary,
            url_triage_decisions=triage_decisions,
            extracted_documents_count=len(extracted_docs),
            extracted_documents=extracted_docs,
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
            pipeline_performance_telemetry=pipeline_perf_telemetry,
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
            self.evidence_repository.save_checkpoint(
                vendor_id=vendor.vendor_id,
                run_id=run_id,
                completed_level=9,
                checkpoint_state={
                    "status": "COMPLETED",
                    "report_id": report.report_id,
                    "pipeline_performance_telemetry": pipeline_perf_telemetry,
                },
            )

        return report
