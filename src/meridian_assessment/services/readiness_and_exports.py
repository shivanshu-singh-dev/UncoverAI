"""System Readiness Inspector, Canonical 9-Level Stage Tracker Builder, Plain-Language Analyst Formatters, Vendor Input Validator, and Clean Report Exporter.

Provides:
  1. `SystemReadinessInspector`: Single reliable source of truth for pre-assessment and runtime readiness:
       - Analyst-facing capability readiness (`external_research_ready`, `local_analysis_ready`,
         `semantic_retrieval_ready`, `risk_score_calibration_ready`) and plain-language labels.
       - Technical diagnostics (`ollama_endpoint`, `ollama_reachable`, `llama3_2_installed`, `mistral_installed`,
         `unload_on_model_switch`, `search_provider`, `embedding_model_name`, `scoring_policy_status`)
         for `Settings -> Technical Diagnostics` and unit tests.
  2. `format_plain_status` & `format_analyst_unavailable_integrations`: Translates internal enums and
     technical status strings into concise product-level analyst wording.
  3. `validate_vendor_assessment_input`: Validates single-vendor assessment inputs while preserving `UNKNOWN`.
  4. `build_nine_level_stage_tracker` & `format_analyst_stage_tracker`: Produces the canonical 9-level
     stage tracker rows preserving the exact 9 project levels (where Level 7 is `Evidentiary Weighting`).
  5. `AssessmentReportExporter`: Clean, secret-free exports in JSON, Evidence CSV, Markdown, and HTML formats.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request

from pydantic import BaseModel, Field

from meridian_assessment.config.env_loader import load_dotenv_if_present
from meridian_assessment.models.ai_risk import (
    ScoringPolicyStatus,
    StageExecutionRecord,
    VendorEndToEndAssessmentReport,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.services.llm.model_router import ModelRoutingConfig
from meridian_assessment.services.osint.search_collector import build_search_backend_from_env
from meridian_assessment.services.risk.deterministic_scorer import DeterministicRiskScoringEngine


CANONICAL_NINE_LEVEL_NAMES: dict[int, str] = {
    1: "Criticality Tiering",
    2: "Assessment Depth Scoping",
    3: "Source Identification & Reconnaissance",
    4: "Data Scraping & Point-of-Capture Traceability",
    5: "Semantic Filtering & RAG Chunking",
    6: "Signal Classification",
    7: "Evidentiary Weighting",
    8: "Risk Surfacing & Summarization",
    9: "Deterministic Risk Rating",
}


PLAIN_STATUS_LABELS: dict[str, str] = {
    # AI-Use Determinations
    "CONFIRMED": "Confirmed AI Use",
    "INDICATIVE_ONLY": "Indicative AI Signal Only",
    "NO": "No AI Identified",
    "INCOMPLETE_ASSESSMENT": "Incomplete Assessment",
    # Evidentiary Gate Statuses
    "QUALIFYING_POSITIVE_EVIDENCE": "Confirmed Positive Evidence",
    "QUALIFYING_NEGATIVE_EVIDENCE": "Qualified Negative Finding",
    "UNRESOLVED": "Unresolved",
    "PENDING_INVESTIGATION": "Pending Investigation",
    # Product & Organizational Scopes
    "MERIDIAN_SERVICE": "Purchased Meridian Service",
    "VENDOR_PRODUCT": "Vendor Product",
    "VENDOR_FIRM_WIDE": "Vendor Firm-Wide",
    # AI Types
    "GENAI_LLM": "Generative AI / Large Language Model",
    "CLASSICAL_ML": "Classical Machine Learning",
    "AGENTIC_AI": "Autonomous / Agentic AI",
    "NONE_AUTOMATION_ONLY": "Deterministic Automation Only (Non-AI)",
    "NONE": "None Identified",
    # Signal Categories
    "OPERATIONAL_AI": "Operational AI Capability",
    "MARKETING_COMMENTARY": "Marketing / Promotional Claim",
    "NON_AI_AUTOMATION": "Standard Non-AI Automation",
    "NO_AI_SIGNAL": "No AI Signal",
    # Evidence Strength & Weighting
    "STRONG": "Strong",
    "MEDIUM": "Medium",
    "WEAK": "Weak",
    "CONFLICTING": "Conflicting",
    "HIGH": "High Weight",
    "MODERATE": "Moderate Weight",
    "LOW": "Low Weight",
    "ZERO_OR_NEGATIVE": "Exclusionary / Negative",
    # Criticality & Governance
    "DIRECT": "Direct Mapping",
    "DETERMINISTIC_RULE": "Rule Match",
    "SEMANTIC_MATCH": "Semantic Match",
    "HUMAN_REVIEW": "Analyst Review",
    "UNKNOWN": "Unknown (Requires Review)",
    "COMPLETE": "Complete",
    "PARTIAL": "Partial (Incomplete Inputs)",
    "REQUIRES_REVIEW": "Requires Analyst Review",
    "APPROVED_BY_ANALYST": "Approved by Analyst",
    "PROPOSED_UNREVIEWED": "Pending Analyst Review",
    # Source Coverage & Recon
    "EVIDENCE_FOUND": "Evidence Captured",
    "REVIEWED_NO_EVIDENCE": "Reviewed — No AI Evidence",
    "SEARCH_ATTEMPTED_ONLY": "Search Attempted (Unreviewed)",
    "IN_PROGRESS": "In Progress",
    "NOT_STARTED": "Not Started",
    "NOT_APPLICABLE": "Not Applicable",
    "UNAVAILABLE": "External Research Unavailable",
    "EXECUTED": "Executed",
    "PLANNED": "Planned",
    "FAILED": "Failed",
    "FETCH": "Selected for Extraction",
    "HOLD": "Held for Review",
    "SKIP": "Skipped (Low Relevance)",
    "REJECT": "Rejected (Out of Scope)",
    # Stopping Conditions
    "STOP_CONDITION_MET": "Stopping Condition Met",
    "INCOMPLETE_TIMEBOX_EXHAUSTED": "Incomplete (Timebox Reached)",
    "INCOMPLETE_GATES_UNRESOLVED": "Incomplete (Open Gates)",
    # Supply Chain
    "INFERRED_INDICATIVE": "Indicative (Unconfirmed)",
    # Risk Exposure & Scoring
    "EXPOSED": "Confirmed Exposure",
    "POTENTIALLY_EXPOSED": "Potential Exposure",
    "MITIGATED": "Control Identified",
    "SCORING_POLICY_UNCONFIGURED": "Calibrated numerical risk score unavailable",
    "CONFIGURED": "Calibrated numerical risk score available",
    # Stage & Model Execution
    "COMPLETED": "Completed",
    "NOT_CONFIGURED": "External Research Unavailable",
    "SKIPPED_NO_INPUT": "Not Executed (No Upstream Evidence)",
    "DETERMINISTIC_TEMPLATE": "Rule-Based Analysis",
    "FALLBACK_DETERMINISTIC": "Rule-Based Analysis",
    "DETERMINISTIC_RULES": "Rule-Based Analysis",
    "LOCAL_LLM_VALIDATED": "Local AI Analysis (Validated)",
    "MODEL_UNAVAILABLE_FALLBACK": "Semantic evidence retrieval unavailable",
    "SEMANTIC_RETRIEVAL_ENABLED": "Semantic retrieval enabled",
    "HYBRID_SEMANTIC_MINILM": "Semantic retrieval enabled",
    "HYBRID_BROAD_RECALL_FALLBACK": "Semantic retrieval enabled",
    "DETERMINISTIC_FALLBACK": "Deterministic retrieval fallback",
    "MODEL_UNAVAILABLE_NOT_CONFIGURED": "Deterministic retrieval fallback",
    "NO_DOCUMENTS": "Retrieval incomplete",
    "RETRIEVAL_INCOMPLETE": "Retrieval incomplete",
    # Retrieval & Source Integrity
    "SOURCE_IDENTITY_MISMATCH": "Source Identity Mismatch (Rejected)",
    "VERIFIED": "Verified Source Identity",
    "WARNING_SAME_HOST_PATH_DIFF": "Verified (Same-Host Path Variant)",
    "DIRECT_HTTP": "Direct HTTP",
    "DIRECT_PDF": "Direct PDF",
    "SCRAPERAPI": "ScraperAPI Fallback",
    "SCRAPERAPI_PDF": "ScraperAPI PDF Fallback",
    "SYNTHETIC_SAMPLE": "Sample Fixture",
}


def format_plain_status(raw_value: Any) -> str:
    """Convert an internal enum or status code into a clean analyst-facing label."""
    if raw_value is None:
        return "—"
    val_str = getattr(raw_value, "value", str(raw_value)).strip()
    if not val_str:
        return "—"
    upper = val_str.upper()
    if upper in PLAIN_STATUS_LABELS:
        return PLAIN_STATUS_LABELS[upper]
    if "_" in val_str and val_str == upper:
        return val_str.replace("_", " ").title()
    return val_str


def format_analyst_unavailable_integrations(
    report: VendorEndToEndAssessmentReport,
    readiness: Optional["SystemReadinessReport"] = None,
) -> list[str]:
    """Return deduplicated, plain-language capability notices for the analyst interface."""
    notices: list[str] = []

    def _add_once(msg: str) -> None:
        if msg not in notices:
            notices.append(msg)

    raw_items = list(report.unavailable_integrations or [])
    for item in raw_items:
        low = item.lower()
        if "search" in low or "serper" in low or "serpapi" in low or "scraperapi" in low or "fetch adapter" in low:
            _add_once("External research unavailable")
        elif "disabled for deterministic run" in low or "bypassed in offline test mode" in low:
            _add_once("Rule-based analysis active (local AI analysis bypassed for this run)")
        elif "llm" in low or "ollama" in low or "mistral" in low or "llama" in low:
            _add_once("Local analysis unavailable")
        elif "embedding" in low or "rag" in low or "sentence-transformers" in low:
            _add_once("Semantic evidence retrieval unavailable")

    if (
        report.deterministic_risk_score
        and report.deterministic_risk_score.policy_status == ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED
    ):
        _add_once("Calibrated numerical risk score unavailable")

    if readiness is not None and not notices:
        if not readiness.external_research_ready and not report.is_synthetic_evidence_run:
            _add_once("External research unavailable")
        if not readiness.risk_score_calibration_ready:
            _add_once("Calibrated numerical risk score unavailable")

    return notices


class SystemReadinessReport(BaseModel):
    """Single reliable source of truth for application readiness and technical diagnostics."""

    # ── Analyst-Facing Capability Statuses (Plain Product Language) ──
    external_research_ready: bool = False
    external_research_label: str = "External research unavailable"
    external_research_consequence: str = (
        "Live public web reconnaissance is unavailable. Running a live assessment without external research "
        "will complete criticality and scoping and mark the AI-use determination as Incomplete Assessment. "
        "Use a Sample Assessment to evaluate pre-collected vendor evidence."
    )

    local_analysis_enabled: bool = False
    local_analysis_ready: bool = False
    local_triage_ready: bool = False
    local_classifier_ready: bool = False
    local_summary_ready: bool = False
    local_analysis_state: str = "DISABLED_IN_CONFIG"
    local_analysis_label: str = "Local analysis unavailable"

    semantic_retrieval_ready: bool = False
    semantic_retrieval_label: str = "Semantic evidence retrieval unavailable"
    retrieval_status_label: str = "Deterministic retrieval fallback"

    risk_score_calibration_ready: bool = False
    risk_score_calibration_label: str = "Calibrated numerical risk score unavailable"

    # ── Technical Diagnostics (Settings -> Integrations & Technical Diagnostics) ──
    # 1. Ollama Server & Installed Models
    ollama_endpoint: str
    ollama_reachable: bool
    ollama_status_message: str
    installed_models: list[str] = Field(default_factory=list)
    llama3_2_installed: bool = False
    mistral_installed: bool = False

    # 2. Dual-Model Routing Configuration
    llm_routing_enabled_in_env: bool
    triage_model: str
    analysis_model: str
    summary_model: str
    unload_on_model_switch: bool
    routing_summary: str

    # 3. Search Provider & Credential Readiness
    search_provider: str
    search_configured: bool
    search_status_message: str
    scraperapi_configured: bool = False
    scraperapi_status_message: str = ""
    web_retrieval_label: str = "Direct HTTP retrieval active"

    # 4. Level 5 RAG & Embedding Readiness
    rag_embeddings_enabled_in_env: bool
    embedding_model_name: str
    sentence_transformers_installed: bool
    embedding_model_cached_locally: bool
    rag_active_mode: str
    rag_status_message: str

    # 5. Level 9 Production Risk-Scoring Policy Readiness
    scoring_policy_status: str
    scoring_policy_ready: bool
    scoring_unconfigured_parameters: list[str] = Field(default_factory=list)
    scoring_status_message: str


def _default_http_get_json(url: str, timeout: float = 1.5) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}


def _check_huggingface_model_cached_locally(model_name: str) -> bool:
    """Check whether a HuggingFace / sentence-transformers model is present in the local disk cache without network calls."""
    cleaned = model_name.strip().replace("/", "--")
    if not cleaned.startswith("models--"):
        repo_dir_name = f"models--{cleaned}"
    else:
        repo_dir_name = cleaned

    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    candidates = [
        Path(hf_home) / "hub" / repo_dir_name,
        Path(os.path.expanduser("~")) / ".cache" / "torch" / "sentence_transformers" / model_name.replace("/", "_"),
    ]
    for cand in candidates:
        if (cand.exists() and any(cand.rglob("*.safetensors"))) or (cand.exists() and any(cand.rglob("*.bin"))):
            return True
    return False


class SystemReadinessInspector:
    """Inspects local environment, Ollama server, search credentials, embeddings, and scoring policy."""

    @staticmethod
    def check_system_readiness(
        *,
        env: Optional[dict[str, str]] = None,
        load_dotenv: bool = True,
        http_get_fn: Optional[Callable[[str, float], dict[str, Any]]] = None,
        scoring_policy_path: Path = Path("config/ai_risk_scoring.yaml"),
        ollama_timeout_seconds: float = 1.5,
        local_analysis_override: Optional[bool] = None,
    ) -> SystemReadinessReport:
        if env is None:
            if load_dotenv:
                load_dotenv_if_present()
            environ = dict(os.environ)
        else:
            environ = dict(env)

        if "MERIDIAN_OLLAMA_ENABLED" not in environ and "MERIDIAN_ENABLE_LOCAL_LLM" in environ:
            environ["MERIDIAN_OLLAMA_ENABLED"] = environ["MERIDIAN_ENABLE_LOCAL_LLM"]

        routing_cfg = ModelRoutingConfig.from_env_and_config(env=environ, load_dotenv=False)
        getter = http_get_fn or _default_http_get_json

        # 1. Check Ollama server & models via GET /api/tags
        tags_url = f"{routing_cfg.endpoint.rstrip('/')}/api/tags"
        installed_models: list[str] = []
        ollama_reachable = False
        try:
            payload = getter(tags_url, ollama_timeout_seconds)
            ollama_reachable = True
            raw_models = payload.get("models", [])
            if isinstance(raw_models, list):
                for m in raw_models:
                    if isinstance(m, dict) and m.get("name"):
                        installed_models.append(str(m["name"]))
        except Exception as exc:
            ollama_status_msg = (
                f"Ollama server unreachable at {routing_cfg.endpoint} ({type(exc).__name__}). "
                "Pipeline will automatically use deterministic rules."
            )
        else:
            ollama_status_msg = (
                f"Ollama server online at {routing_cfg.endpoint} "
                f"({len(installed_models)} model(s) installed)."
            )

        llama_installed = any(
            m == routing_cfg.triage_model or m.startswith("llama3.2:")
            for m in installed_models
        )
        mistral_installed = any(
            m in (routing_cfg.analysis_model, routing_cfg.summary_model) or m.startswith("mistral:")
            for m in installed_models
        )

        # Unified local analysis readiness calculation (eliminates contradictory status claims)
        effective_llm_enabled = (
            routing_cfg.enabled if local_analysis_override is None else bool(local_analysis_override)
        )
        local_triage_ready = bool(effective_llm_enabled and ollama_reachable and llama_installed)
        local_classifier_ready = bool(effective_llm_enabled and ollama_reachable and mistral_installed)
        local_summary_ready = bool(effective_llm_enabled and ollama_reachable and mistral_installed)
        local_analysis_ready = bool(local_triage_ready and local_classifier_ready and local_summary_ready)

        if local_analysis_ready:
            local_analysis_state = "READY"
            local_analysis_label = "Local AI analysis ready"
        elif local_analysis_override is False:
            local_analysis_state = "DISABLED_BY_USER"
            local_analysis_label = "Rule-based analysis only"
        elif not routing_cfg.enabled:
            local_analysis_state = "DISABLED_IN_CONFIG"
            local_analysis_label = "Local analysis unavailable"
        elif not ollama_reachable:
            local_analysis_state = "SERVER_UNREACHABLE"
            local_analysis_label = "Local analysis unavailable"
        else:
            local_analysis_state = "MODEL_MISSING"
            local_analysis_label = "Local analysis unavailable"

        routing_summary = (
            f"Level 3B URL Triage -> {routing_cfg.triage_model} | "
            f"Level 6 Signal Classification -> {routing_cfg.analysis_model} | "
            f"Level 8 Risk Summarization -> {routing_cfg.summary_model} "
            f"(Sequential VRAM Unload: {'Enabled' if routing_cfg.unload_on_model_switch else 'Disabled'})"
        )

        # 2. Check Search Provider & Credential Status (never expose secret values)
        search_backend = build_search_backend_from_env(env=environ, load_dotenv=False)
        search_ok, raw_search_msg = search_backend.is_configured()
        selected_provider = (
            environ.get("MERIDIAN_SEARCH_PROVIDER", "").strip().lower()
            or getattr(search_backend, "provider_name", "unconfigured")
        )
        if not search_ok:
            if selected_provider in ("serper", "serper_dev"):
                search_status_message = (
                    "Search is not configured: add MERIDIAN_SERPER_API_KEY to the local .env file."
                )
            elif selected_provider == "serpapi":
                search_status_message = (
                    "Search is not configured: add MERIDIAN_SERPAPI_KEY to the local .env file."
                )
            elif selected_provider in ("google_cse", "google"):
                search_status_message = (
                    "Search is not configured: add MERIDIAN_GOOGLE_API_KEY and MERIDIAN_GOOGLE_CSE_ID to the local .env file."
                )
            elif selected_provider == "brave":
                search_status_message = (
                    "Search is not configured: add MERIDIAN_BRAVE_API_KEY to the local .env file."
                )
            elif selected_provider == "bing":
                search_status_message = (
                    "Search is not configured: add MERIDIAN_BING_API_KEY to the local .env file."
                )
            else:
                search_status_message = (
                    f"Search is not configured ({raw_search_msg}). Add search provider credentials to the local .env file."
                )
        else:
            search_status_message = f"Search provider '{selected_provider}' is configured and ready."

        scraper_key = (environ.get("MERIDIAN_SCRAPERAPI_KEY") or "").strip()
        scraperapi_configured = bool(scraper_key)
        if scraperapi_configured:
            scraperapi_status_message = "ScraperAPI retrieval fallback is configured and ready."
            web_retrieval_label = "Direct HTTP + ScraperAPI fallback active"
        else:
            scraperapi_status_message = (
                "ScraperAPI fallback is not configured (add MERIDIAN_SCRAPERAPI_KEY to .env for "
                "blocked-page fallback; direct HTTP retrieval remains active)."
            )
            web_retrieval_label = "Direct HTTP retrieval active (ScraperAPI fallback not configured)"

        external_research_label = (
            "External research available" if search_ok else "External research unavailable"
        )
        external_research_consequence = (
            "Live public source reconnaissance and evidence extraction are active."
            if search_ok
            else (
                "External research is currently unavailable. Starting a live assessment without external research "
                "will complete criticality and scoping and mark the AI-use determination as Incomplete Assessment. "
                "Select Sample Assessment to evaluate pre-collected vendor evidence."
            )
        )

        # 3. Check Level 5 RAG & Embedding Readiness
        st_installed = importlib.util.find_spec("sentence_transformers") is not None
        emb_model_name = routing_cfg.embedding_model
        emb_cached = _check_huggingface_model_cached_locally(emb_model_name)

        if routing_cfg.enable_rag_embeddings and st_installed and emb_cached:
            rag_active_mode = "SEMANTIC_VECTOR_RETRIEVAL"
            rag_status_msg = (
                f"Semantic vector retrieval active ({emb_model_name} cached locally)."
            )
            semantic_retrieval_ready = True
            semantic_retrieval_label = "Semantic evidence retrieval available"
            retrieval_status_label = "Semantic retrieval enabled"
        elif routing_cfg.enable_rag_embeddings and not st_installed:
            rag_active_mode = "DETERMINISTIC_CHUNK_PASSTHROUGH_TO_L6"
            rag_status_msg = (
                f"MERIDIAN_ENABLE_RAG_EMBEDDINGS=true, but 'sentence-transformers' is not installed in .venv. "
                "Level 5 runs deterministic 500-token chunking with SHA-256 provenance and passes all extracted evidence directly to Level 6."
            )
            semantic_retrieval_ready = False
            semantic_retrieval_label = "Semantic evidence retrieval unavailable"
            retrieval_status_label = "Deterministic retrieval fallback"
        elif routing_cfg.enable_rag_embeddings and not emb_cached:
            rag_active_mode = "DETERMINISTIC_CHUNK_PASSTHROUGH_TO_L6"
            rag_status_msg = (
                f"MERIDIAN_ENABLE_RAG_EMBEDDINGS=true, but '{emb_model_name}' is not cached locally. "
                "Level 5 runs deterministic 500-token chunking and passes all extracted evidence directly to Level 6."
            )
            semantic_retrieval_ready = False
            semantic_retrieval_label = "Semantic evidence retrieval unavailable"
            retrieval_status_label = "Deterministic retrieval fallback"
        else:
            rag_active_mode = "DETERMINISTIC_CHUNK_PASSTHROUGH_TO_L6"
            rag_status_msg = (
                "Semantic embeddings disabled (MERIDIAN_ENABLE_RAG_EMBEDDINGS=false). "
                "Level 5 executes deterministic 500-token chunking with SHA-256 provenance and passes all extracted evidence directly to Level 6."
            )
            semantic_retrieval_ready = False
            semantic_retrieval_label = "Semantic evidence retrieval unavailable"
            retrieval_status_label = "Deterministic retrieval fallback"

        # 4. Check Level 9 Production Risk-Scoring Policy Readiness
        scorer = DeterministicRiskScoringEngine(policy_path=scoring_policy_path)
        policy_status, _, unconfigured_params = scorer.validate_policy()
        policy_ready = policy_status != ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED
        if not policy_ready:
            scoring_msg = (
                "Production risk-scoring policy is unconfigured (SCORING_POLICY_UNCONFIGURED): "
                "a final calibrated Level 9 numerical risk score cannot be produced until Meridian's "
                f"approved parameters ({', '.join(unconfigured_params)}) are supplied."
            )
            risk_score_calibration_label = "Calibrated numerical risk score unavailable"
        else:
            scoring_msg = f"Level 9 risk-scoring policy configured ({policy_status.value})."
            risk_score_calibration_label = "Calibrated numerical risk score available"

        return SystemReadinessReport(
            external_research_ready=search_ok,
            external_research_label=external_research_label,
            external_research_consequence=external_research_consequence,
            local_analysis_enabled=effective_llm_enabled,
            local_analysis_ready=local_analysis_ready,
            local_triage_ready=local_triage_ready,
            local_classifier_ready=local_classifier_ready,
            local_summary_ready=local_summary_ready,
            local_analysis_state=local_analysis_state,
            local_analysis_label=local_analysis_label,
            semantic_retrieval_ready=semantic_retrieval_ready,
            semantic_retrieval_label=semantic_retrieval_label,
            retrieval_status_label=retrieval_status_label,
            risk_score_calibration_ready=policy_ready,
            risk_score_calibration_label=risk_score_calibration_label,
            ollama_endpoint=routing_cfg.endpoint,
            ollama_reachable=ollama_reachable,
            ollama_status_message=ollama_status_msg,
            installed_models=installed_models,
            llama3_2_installed=llama_installed,
            mistral_installed=mistral_installed,
            llm_routing_enabled_in_env=routing_cfg.enabled,
            triage_model=routing_cfg.triage_model,
            analysis_model=routing_cfg.analysis_model,
            summary_model=routing_cfg.summary_model,
            unload_on_model_switch=routing_cfg.unload_on_model_switch,
            routing_summary=routing_summary,
            search_provider=selected_provider,
            search_configured=search_ok,
            search_status_message=search_status_message,
            scraperapi_configured=scraperapi_configured,
            scraperapi_status_message=scraperapi_status_message,
            web_retrieval_label=web_retrieval_label,
            rag_embeddings_enabled_in_env=routing_cfg.enable_rag_embeddings,
            embedding_model_name=emb_model_name,
            sentence_transformers_installed=st_installed,
            embedding_model_cached_locally=emb_cached,
            rag_active_mode=rag_active_mode,
            rag_status_message=rag_status_msg,
            scoring_policy_status=policy_status.value,
            scoring_policy_ready=policy_ready,
            scoring_unconfigured_parameters=unconfigured_params,
            scoring_status_message=scoring_msg,
        )


def validate_vendor_assessment_input(
    *,
    vendor_id: str,
    vendor_name: str,
    domain: str = "",
    category: Optional[str] = None,
    vendor_description: Optional[str] = None,
    service_product_provided: Optional[str] = None,
    business_process_supported: Optional[str] = None,
    data_classification_accessed: Optional[str] = None,
    operational_dependency: Optional[str] = None,
    data_volume_annual: Optional[str] = None,
    is_sole_source: bool = False,
) -> tuple[Optional[MeridianVendor], list[str], list[str]]:
    """Validate single-vendor assessment inputs and return `(vendor_or_none, errors, warnings)`.

    Preserves `UNKNOWN` when inputs are unknown rather than coercing unknown values to zero.
    """
    errors: list[str] = []
    warnings: list[str] = []

    vid = (vendor_id or "").strip()
    vname = (vendor_name or "").strip()
    dom = (domain or "").strip().lower()

    if not vid:
        errors.append("Vendor ID is required (for example, 'V-007').")
    if not vname:
        errors.append("Vendor legal/display name is required.")

    if dom:
        if dom.startswith(("http://", "https://")):
            parsed = urllib.parse.urlsplit(dom)
            dom = (parsed.hostname or dom).lower()
        if "." not in dom or " " in dom:
            errors.append(f"Primary domain '{domain}' is not a valid hostname (expected e.g. 'vendor.com').")
    else:
        warnings.append(
            "Primary domain is blank: domain-scoped reconnaissance queries will have reduced specificity."
        )

    def _norm_optional(val: Optional[str], field_label: str) -> Optional[str]:
        if val is None or not str(val).strip():
            warnings.append(
                f"{field_label} is not specified; corresponding factor will evaluate as Unknown and require analyst review."
            )
            return "UNKNOWN"
        cleaned = str(val).strip()
        if cleaned.upper() == "UNKNOWN":
            warnings.append(f"{field_label} is marked Unknown; corresponding factor will require analyst review.")
            return "UNKNOWN"
        return cleaned

    srv = _norm_optional(service_product_provided, "Service / Product Purchased by Meridian")
    bp = _norm_optional(business_process_supported, "Business Process Supported")
    dc = _norm_optional(data_classification_accessed, "Data Classification Accessed (D)")
    od = _norm_optional(operational_dependency, "Operational Dependency (O)")
    dv = _norm_optional(data_volume_annual, "Annual Data Volume (V)")

    if errors:
        return None, errors, warnings

    vendor = MeridianVendor(
        vendor_id=vid,
        vendor_name=vname,
        domain=dom,
        category=(category or "").strip() or None,
        vendor_description=(vendor_description or "").strip() or None,
        service_product_provided=srv,
        business_process_supported=bp,
        data_classification_accessed=dc,
        operational_dependency=od,
        data_volume_annual=dv,
        is_sole_source=bool(is_sole_source),
    )
    return vendor, errors, warnings


class StageTrackerRow(BaseModel):
    """Structured presentation row for one of the canonical nine project levels."""

    level_number: int
    level_label: str
    level_name: str
    status: str
    badge_label: str
    concise_output: str
    relevant_counts: str
    execution_method: str
    selected_model: str
    blocker: Optional[str] = None
    limitations: list[str] = Field(default_factory=list)
    substeps: list[dict[str, str]] = Field(default_factory=list)

    # Plain-language analyst presentation fields (free of raw env vars, model tags, and provider IDs)
    analyst_status: str = "Completed"
    analyst_method: str = "Rule-Based Evaluation"
    analyst_counts: str = ""
    analyst_summary: str = ""


def _format_stage_badge(status: str, model_execution_status: Optional[str]) -> str:
    st = (status or "").upper()
    ms = (model_execution_status or "").upper()
    if st == "NOT_CONFIGURED":
        return "BLOCKED BY CONFIG"
    if st == "REQUIRES_REVIEW":
        return "AWAITING HUMAN REVIEW"
    if st == "FAILED":
        return "FAILED"
    if st == "SKIPPED_NO_INPUT":
        return "SKIPPED (NO INPUT)"
    if st in ("INCOMPLETE", "PARTIAL"):
        if "FALLBACK" in ms:
            return "INCOMPLETE (DETERMINISTIC FALLBACK)"
        return "INCOMPLETE"
    if st == "COMPLETED":
        if "FALLBACK" in ms:
            return "DETERMINISTIC FALLBACK"
        return "COMPLETED"
    return st or "UNKNOWN"


def build_nine_level_stage_tracker(
    report: VendorEndToEndAssessmentReport,
) -> list[StageTrackerRow]:
    """Build the 9-level stage tracker rows preserving the exact 9 project levels and showing Level 3B as a substep of Level 3."""
    ledger_by_num: dict[int, StageExecutionRecord] = {
        rec.level_number: rec for rec in report.stage_ledger
    }
    crit = report.criticality_summary or {}
    osint = report.osint_plan_summary or {}
    rag = report.semantic_retrieval

    rows: list[StageTrackerRow] = []

    for lvl in range(1, 10):
        canonical_name = CANONICAL_NINE_LEVEL_NAMES[lvl]
        rec = ledger_by_num.get(lvl)
        status = rec.status if rec else "INCOMPLETE"
        summary = rec.summary if rec else ""
        model_name = (rec.selected_model if rec and rec.selected_model else "DETERMINISTIC")
        model_status = (rec.model_execution_status if rec and rec.model_execution_status else "DETERMINISTIC_ONLY")
        limits = list(rec.limitations) if rec else []

        counts_str = ""
        exec_method = "Deterministic Rulebook"
        blocker: Optional[str] = None
        substeps: list[dict[str, str]] = []

        analyst_status = "Completed"
        analyst_method = "Rule-Based Evaluation"
        analyst_counts = ""
        analyst_summary = ""

        if lvl == 1:
            factors = crit.get("factors", {})
            known_cnt = sum(
                1 for f in ("D", "P", "R", "O", "V")
                if isinstance(factors.get(f), dict) and factors[f].get("score") != "UNKNOWN"
            )
            ov_trig = len(crit.get("overrides_triggered", []))
            counts_str = f"{known_cnt}/5 factors scored | {ov_trig} override(s)"
            exec_method = "Deterministic Formula (0.30D+0.20P+0.20R+0.20O+0.10V) + O1–O5"
            if status == "REQUIRES_REVIEW":
                blocker = "; ".join(limits) if limits else "Factor ambiguity or O5 escalation requires analyst review."
                analyst_status = "Requires Analyst Review"
            else:
                analyst_status = "Completed"
            analyst_method = "Deterministic Formula & Safeguards"
            analyst_counts = f"{known_cnt}/5 factors scored · {ov_trig} safeguard override(s)"
            base_sc = crit.get("base_score")
            base_sc_txt = f"{base_sc:.2f}" if isinstance(base_sc, (int, float)) else "Unknown"
            analyst_summary = (
                f"Base score {base_sc_txt} ({crit.get('provisional_criticality', 'Unknown')}); "
                f"final tier assigned as {crit.get('final_criticality', 'Unknown')}."
            )

        elif lvl == 2:
            req_cnt = len(osint.get("required_source_classes", []))
            cond_cnt = len(osint.get("conditional_source_classes", []))
            q_cnt = int(osint.get("planned_queries_count", 0))
            counts_str = f"{req_cnt} required, {cond_cnt} conditional sources | {q_cnt} queries"
            exec_method = "Deterministic S1–S12 Depth & Timebox Planner"
            analyst_status = "Completed"
            analyst_method = "Policy Depth Scoping"
            analyst_counts = f"{req_cnt} required sources · {q_cnt} planned queries"
            analyst_summary = (
                f"Assigned {osint.get('selected_depth_profile', 'Targeted')} investigation depth "
                f"across {req_cnt} mandatory public source classes."
            )

        elif lvl == 3:
            q_plan = int(osint.get("planned_queries_count", 0))
            q_exec = int(osint.get("executed_queries_count", 0))
            q_unavail = int(osint.get("unavailable_queries_count", 0))
            cand_cnt = int(osint.get("candidate_urls_count", 0))
            triage_cnt = len(report.url_triage_decisions)
            fetch_cnt = sum(1 for d in report.url_triage_decisions if d.action.value == "FETCH")
            counts_str = (
                f"{q_exec}/{q_plan} queries executed | {cand_cnt} URLs | {fetch_cnt}/{triage_cnt} triaged FETCH"
            )
            if "llama" in model_name.lower() and model_status in ("SUCCEEDED", "PARTIAL_SUCCEEDED"):
                exec_method = f"Search Collector + Level 3B Model A ({model_name})"
                analyst_method = "Public Source Discovery & Local AI Triage"
            elif "FALLBACK" in model_status:
                exec_method = f"Search Collector + Deterministic URL Triage Fallback ({model_name} [{model_status}])"
                analyst_method = "Public Source Discovery & Rule-Based Triage"
            else:
                exec_method = "Search Collector + Deterministic URL Triage"
                analyst_method = "Public Source Discovery & Rule-Based Triage"

            if status in ("NOT_CONFIGURED", "INCOMPLETE"):
                blocker = limits[0] if limits else "External search provider credential not configured."
                analyst_status = "External research unavailable" if status == "NOT_CONFIGURED" else "Incomplete"
                analyst_summary = (
                    f"{q_plan} reconnaissance queries planned; external research unavailable."
                    if status == "NOT_CONFIGURED"
                    else f"{q_exec}/{q_plan} queries executed; {cand_cnt} candidate source URLs identified."
                )
            else:
                analyst_status = "Completed"
                analyst_summary = (
                    f"{q_exec}/{q_plan} queries executed; {cand_cnt} candidate URLs evaluated and "
                    f"{fetch_cnt} prioritized for evidence extraction."
                )
            analyst_counts = f"{q_exec}/{q_plan} queries · {cand_cnt} candidate URLs"

            search_sub_status = "COMPLETED" if q_exec > 0 else ("BLOCKED BY CONFIG" if status == "NOT_CONFIGURED" else "INCOMPLETE")
            triage_sub_status = (
                f"COMPLETED ({model_name} [{model_status}])"
                if triage_cnt > 0
                else ("SKIPPED (0 candidate URLs)" if cand_cnt == 0 else model_status)
            )
            substeps = [
                {
                    "substep_id": "Level 3A",
                    "substep_name": "Query Execution & Candidate-URL Collection",
                    "status": search_sub_status,
                    "detail": (
                        f"{q_exec}/{q_plan} planned queries executed ({q_unavail} unavailable); "
                        f"{cand_cnt} deduplicated candidate URL(s) collected."
                    ),
                },
                {
                    "substep_id": "Level 3B",
                    "substep_name": "Candidate-URL Triage (Substep of Level 3)",
                    "status": triage_sub_status,
                    "detail": (
                        f"Model A ({model_name}): {triage_cnt} URL(s) triaged -> {fetch_cnt} FETCH, "
                        f"{sum(1 for d in report.url_triage_decisions if d.action.value == 'SKIP')} SKIP."
                    ),
                },
            ]

        elif lvl == 4:
            doc_cnt = report.extracted_documents_count
            ev_cnt = len(report.evidence_records)
            counts_str = f"{doc_cnt} document(s) fetched | {ev_cnt} evidence excerpt(s)"
            exec_method = "Point-of-Capture Extractor + SHA-256 Provenance Store"
            analyst_method = "Point-of-Capture Document Extraction"
            analyst_counts = f"{doc_cnt} document(s) · {ev_cnt} verbatim excerpt(s)"
            if status in ("NOT_CONFIGURED", "SKIPPED_NO_INPUT"):
                blocker = limits[0] if limits else "No triaged candidate URLs available for extraction."
                analyst_status = "Not Executed (No Candidate Sources)"
                analyst_summary = "No candidate URLs were available for document extraction."
            else:
                analyst_status = "Completed"
                analyst_summary = (
                    f"Extracted {ev_cnt} traceable verbatim excerpt(s) from {doc_cnt} source document(s) "
                    "with cryptographic SHA-256 verification."
                )

        elif lvl == 5:
            counts_str = (
                f"{rag.chunks_created} chunk(s) created | {rag.chunks_embedded} embedded | {len(rag.matches)} concept match(es)"
            )
            analyst_counts = f"{rag.chunks_created} passage chunk(s)"
            if rag.status == "NO_DOCUMENTS" or rag.chunks_created == 0:
                exec_method = "Token Chunker (No Upstream Documents)"
                if rag.limitations:
                    blocker = rag.limitations[0]
                analyst_status = "Retrieval incomplete"
                analyst_method = "Semantic Evidence Retrieval"
                analyst_summary = "No upstream extracted evidence passages were available for retrieval."
            elif rag.model_available:
                exec_method = f"Token Chunker + Semantic Embeddings ({rag.model_name})"
                analyst_status = "Semantic retrieval enabled"
                analyst_method = "Semantic Evidence Retrieval"
                analyst_summary = (
                    f"Segmented evidence into {rag.chunks_created} passage chunk(s), matched "
                    f"{len(rag.matches)} AI risk concept(s), and selected {rag.candidates_selected_for_llm} candidate(s)."
                )
            else:
                exec_method = "Deterministic 500-Token Chunker (Deterministic Candidate Selection)"
                if rag.limitations:
                    blocker = rag.limitations[0]
                analyst_status = "Deterministic retrieval fallback"
                analyst_method = "Deterministic Candidate Retrieval"
                analyst_summary = (
                    f"Segmented {rag.chunks_created} passage chunk(s) and selected "
                    f"{rag.candidates_selected_for_llm or rag.chunks_created} candidate(s) via deterministic retrieval fallback."
                )

        elif lvl == 6:
            cls_cnt = len(report.signal_classifications)
            llm_cnt = sum(
                1 for c in report.signal_classifications if c.classification_method.value == "LOCAL_LLM_VALIDATED"
            )
            counts_str = f"{cls_cnt} classified ({llm_cnt} via {model_name})"
            analyst_counts = f"{cls_cnt} excerpt(s) classified"
            if llm_cnt > 0:
                exec_method = f"Model B ({model_name}) + Deterministic Schema & E14 Guardrails"
                analyst_method = "Local AI Analysis & Rule Verification"
            elif "FALLBACK" in model_status:
                exec_method = f"Deterministic Classifier Fallback ({model_name} [{model_status}])"
                analyst_method = "Rule-Based Signal Classification"
            else:
                exec_method = "Deterministic Signal & Scope Classifier"
                analyst_method = "Rule-Based Signal Classification"
            if not report.signal_classifications:
                if limits:
                    blocker = limits[0]
                analyst_status = "Not Executed (No Evidence)"
                analyst_summary = "No extracted evidence excerpts available to classify."
            else:
                analyst_status = "Completed"
                hostile_cnt = sum(1 for e in report.evidence_records if e.hostile_instruction_detected)
                hostile_note = f" ({hostile_cnt} untrusted instruction isolated)" if hostile_cnt else ""
                analyst_summary = (
                    f"Classified {cls_cnt} evidence excerpt(s) by AI capability type, product scope, "
                    f"and fourth-party provider identity{hostile_note}."
                )

        elif lvl == 7:
            w_cnt = len(report.weighted_evidence)
            strong_cnt = sum(1 for w in report.weighted_evidence if w.evidence_strength.value == "STRONG")
            med_cnt = sum(1 for w in report.weighted_evidence if w.evidence_strength.value == "MEDIUM")
            qual_gates = sum(1 for g in report.gates if g.status == "QUALIFYING_POSITIVE_EVIDENCE")
            neg_gates = sum(1 for g in report.gates if g.status == "QUALIFYING_NEGATIVE_EVIDENCE")
            counts_str = (
                f"{w_cnt} weighted item(s) | {qual_gates}/4 positive gates | Verdict: {report.ai_use_verdict.verdict.value}"
            )
            exec_method = "Deterministic E1–E14 Weighting & G1–G4 Verdict Engine"
            analyst_method = "Evidentiary Rule Engine (E1–E14 & Gates G1–G4)"
            analyst_counts = (
                f"{w_cnt} weighted item(s) ({strong_cnt} Strong, {med_cnt} Medium) · "
                f"{qual_gates + neg_gates}/4 gates resolved"
            )
            if not report.ai_use_verdict.depth_completed:
                blocker = (
                    report.ai_use_verdict.incomplete_reasons[0]
                    if report.ai_use_verdict.incomplete_reasons
                    else "Required OSINT source depth is incomplete."
                )
                analyst_status = "Incomplete Coverage"
            else:
                analyst_status = "Completed"
            analyst_summary = (
                f"Weighted {w_cnt} evidence item(s) under rules E1–E14 and evaluated gates G1–G4, "
                f"resulting in determination: {format_plain_status(report.ai_use_verdict.verdict)}."
            )

        elif lvl == 8:
            rf_cnt = len(report.risk_findings)
            llm_sum_cnt = sum(1 for f in report.risk_findings if f.summary_method == "LOCAL_LLM_VALIDATED")
            counts_str = f"{rf_cnt} R1–R8 finding(s) ({llm_sum_cnt} LLM-summarized)"
            analyst_counts = f"{rf_cnt} risk finding(s)"
            if llm_sum_cnt > 0:
                exec_method = f"Deterministic R1–R8 Rules + Model B ({model_name}) Grounded Summary"
                analyst_method = "Risk Domain Rules (R1–R8) & Local AI Narrative"
            elif "FALLBACK" in model_status:
                exec_method = f"Deterministic R1–R8 Rules + Deterministic Summary Fallback ({model_status})"
                analyst_method = "Risk Domain Rules (R1–R8)"
            else:
                exec_method = "Deterministic R1–R8 Risk Rule Engine"
                analyst_method = "Risk Domain Rules (R1–R8)"
            analyst_status = "Completed"
            if rf_cnt > 0:
                analyst_summary = (
                    f"Surfaced {rf_cnt} evidence-backed security and governance risk finding(s) "
                    "with recommended vendor remediation actions."
                )
            else:
                analyst_summary = "No material AI security risk findings surfaced from current evidence."

        elif lvl == 9:
            ds = report.deterministic_risk_score
            counts_str = (
                f"Policy: {ds.policy_status.value} | Score: "
                f"{f'{ds.final_risk_score:.2f}' if ds.final_risk_score is not None else 'UNCONFIGURED'}"
            )
            exec_method = "Deterministic Formula: Risk Score = M_c * sum(S_i * W_i)"
            analyst_method = "Deterministic Risk Scoring Policy"
            if ds.policy_status == ScoringPolicyStatus.SCORING_POLICY_UNCONFIGURED:
                blocker = (
                    "Production scoring policy unconfigured: M_c, S_i, W_i, and rating thresholds require formal calibration."
                )
                analyst_status = "Calibrated numerical risk score unavailable"
                analyst_counts = "Qualitative severity findings active"
                analyst_summary = (
                    "Calibrated numerical risk score unavailable until institutional scoring weights "
                    "and thresholds are approved; qualitative finding severities govern review."
                )
            else:
                analyst_status = "Completed"
                analyst_counts = f"Composite score: {ds.final_risk_score:.2f}" if ds.final_risk_score is not None else "Scored"
                analyst_summary = f"Calculated composite risk score {ds.final_risk_score:.2f} ({ds.risk_rating or 'Rated'})."

        rows.append(
            StageTrackerRow(
                level_number=lvl,
                level_label=f"Level {lvl}",
                level_name=canonical_name,
                status=status,
                badge_label=_format_stage_badge(status, model_status),
                concise_output=summary,
                relevant_counts=counts_str,
                execution_method=exec_method,
                selected_model=model_name,
                blocker=blocker,
                limitations=limits,
                substeps=substeps,
                analyst_status=analyst_status,
                analyst_method=analyst_method,
                analyst_counts=analyst_counts,
                analyst_summary=analyst_summary,
            )
        )

    return rows


def format_analyst_stage_tracker(report: VendorEndToEndAssessmentReport) -> list[dict[str, str]]:
    """Return the canonical 9-level stage rows formatted in clean, non-technical analyst language."""
    tracker = build_nine_level_stage_tracker(report)
    table_rows: list[dict[str, str]] = []
    for row in tracker:
        table_rows.append({
            "Stage": f"{row.level_number}",
            "Assessment Stage": row.level_name,
            "Status": row.analyst_status,
            "Method": row.analyst_method,
            "Scope & Volume": row.analyst_counts,
            "Outcome Summary": row.analyst_summary,
        })
    return table_rows


def gates_for_evidence(report: VendorEndToEndAssessmentReport, evidence_id: str) -> list[str]:
    """Return all G1–G4 gate IDs supported by a given evidence_id."""
    supported: list[str] = []
    for g in report.gates:
        if evidence_id in g.supporting_evidence_ids and g.gate_id not in supported:
            supported.append(g.gate_id)
    return supported


CANONICAL_VENDOR_INVENTORY_COLUMNS: list[str] = [
    # 10 Supplied Vendor Input Fields (Verbatim preservation)
    "Vendor ID",
    "Vendor Name",
    "Vendor Description",
    "Service / Product Provided to Meridian",
    "Category",
    "Public Website",
    "Business Process Supported",
    "Operational Dependency",
    "Data Classification Accessed",
    "Data Volume (annual)",
    # 11 Student-Derived Fields (Larkspur worked example analytical rigor)
    "Criticality Tier",
    "Criticality Rationale",
    "Assessment Depth Applied",
    "AI Usage Detected",
    "Evidence Description along with OSINT Source(s) / Evidence Link",
    "How Vendor May Be Using AI",
    "Named AI Sub-processors / Model Providers",
    "AI Security Risk Classification",
    "Risk Rationale",
    "Recommended Action",
    "Assessed By / Date",
]


def build_canonical_vendor_inventory_record(
    report: VendorEndToEndAssessmentReport,
    vendor: Optional[MeridianVendor] = None,
) -> dict[str, str]:
    """Synthesize the canonical 21-field Meridian Vendor Inventory assessment record.

    Strictly preserves the 10 supplied vendor input fields without mutation and derives
    the 11 governance fields matching the analytical rigor, evidence traceability,
    and structural completeness of the Larkspur worked example.
    """
    # ── 1. 10 Supplied Vendor Input Fields (Verbatim Preservation) ──
    v_id = (vendor.vendor_id if vendor else None) or report.vendor_id or ""
    v_name = (vendor.vendor_name if vendor else None) or report.legal_entity or ""
    v_desc = (vendor.vendor_description if vendor else None) or report.vendor_description or ""
    v_svc = (vendor.service_product_provided if vendor else None) or report.service_product_provided or ""
    v_cat = (vendor.category if vendor else None) or report.category or ""
    v_web = (vendor.domain if vendor else None) or report.domain or ""
    v_proc = (vendor.business_process_supported if vendor else None) or report.business_process_supported or ""
    v_op = (vendor.operational_dependency if vendor else None) or report.operational_dependency or ""
    v_data = (vendor.data_classification_accessed if vendor else None) or report.data_classification_accessed or ""
    v_vol = (vendor.data_volume_annual if vendor else None) or report.data_volume_annual or ""

    # ── 2. Field 1: Criticality Tier ──
    crit = report.criticality_summary or {}
    crit_tier = crit.get("final_criticality") or crit.get("criticality_tier") or "Unrated"

    # ── 3. Field 2: Criticality Rationale ──
    base_score = crit.get("base_score")
    prov_tier = crit.get("provisional_criticality")
    factors = crit.get("factors") or {}
    d_f = factors.get("D") or {}
    o_f = factors.get("O") or {}
    p_f = factors.get("P") or {}
    r_f = factors.get("R") or {}
    v_f = factors.get("V") or {}

    rat_parts: list[str] = [f"Assigned {crit_tier} Criticality Tier."]
    if base_score is not None and prov_tier:
        rat_parts.append(
            f"Base scoring calculation yielded {base_score:.1f} (Provisional Tier: {prov_tier})."
        )

    overrides = crit.get("overrides_evaluated") or []
    triggered = [o for o in overrides if o.get("triggered")]
    if triggered:
        ov_notes = []
        for o in triggered:
            ov_id = o.get("override_id", "Override")
            ov_rat = o.get("rationale") or o.get("description")
            ov_fl = o.get("resulting_floor")
            ov_notes.append(f"Safeguard {ov_id} triggered ({ov_rat}), enforcing a mandatory floor of {ov_fl}")
        rat_parts.append(f"Automated Governance Safeguards: {'; '.join(ov_notes)}.")
    elif crit.get("automatic_override_applied"):
        rat_parts.append("Automated governance safeguards evaluated and applied.")

    factor_items = []
    if d_f:
        factor_items.append(
            f"Data Sensitivity D={d_f.get('score', 'N/A')} ({d_f.get('rationale', 'Assessed data access')})"
        )
    if o_f:
        factor_items.append(
            f"Operational Dependency O={o_f.get('score', 'N/A')} ({o_f.get('rationale', 'Direct dependency mapping')})"
        )
    if p_f:
        factor_items.append(
            f"Payment Flow Exposure P={p_f.get('score', 'N/A')} ({p_f.get('rationale', 'Evaluated funds transfer')})"
        )
    if r_f:
        factor_items.append(
            f"Regulatory Exposure R={r_f.get('score', 'N/A')} ({r_f.get('rationale', 'Evaluated regulatory obligations')})"
        )
    if v_f:
        factor_items.append(
            f"Annual Volume V={v_f.get('score', 'N/A')} ({v_f.get('rationale', 'Evaluated data volume')})"
        )

    if factor_items:
        rat_parts.append("Methodology Breakdown: " + " | ".join(factor_items))
    crit_rationale = " ".join(rat_parts)

    # ── 4. Field 3: Assessment Depth Applied ──
    osint = report.osint_plan_summary or {}
    depth_profile = osint.get("selected_depth_profile") or crit.get("assessment_depth") or "Standard"
    sel_reason = osint.get("selection_reason") or f"{crit_tier} criticality tier mandates proportional investigation depth"

    depth_exec: list[str] = [
        f"Executed {depth_profile} OSINT investigation ({sel_reason}).",
        f"Reconnaissance triaged {len(report.url_triage_decisions)} candidate URLs across prioritized source classes.",
        f"Ingested and verified {len(report.extracted_documents)} primary documents yielding {len(report.evidence_records)} point-of-capture evidence items.",
    ]
    if report.semantic_retrieval and report.semantic_retrieval.chunks_created > 0:
        depth_exec.append(
            f"Level 5 semantic retrieval evaluated {report.semantic_retrieval.chunks_created} text chunks."
        )
    depth_exec.append(
        f"Level 6 signal classification evaluated {len(report.signal_classifications)} signals under prompt-injection defenses."
    )
    depth_exec.append(
        f"Level 7 evidentiary weighting evaluated E1–E14 rules across {len(report.weighted_evidence)} weighted items and evaluated G1–G4 gates."
    )
    depth_applied = " ".join(depth_exec)

    # ── 5. Field 4: AI Usage Detected ──
    verdict_raw = (
        report.ai_use_verdict.verdict.value
        if hasattr(report.ai_use_verdict.verdict, "value")
        else str(report.ai_use_verdict.verdict)
    )
    if verdict_raw == "YES":
        ai_detected = "Yes"
    elif verdict_raw == "NO":
        ai_detected = "No"
    else:
        ai_detected = "Inconclusive"

    # ── 6. Field 5: Evidence Description along with OSINT Source(s) / Evidence Link ──
    weight_map = {w.evidence_id: w for w in report.weighted_evidence}

    def _ev_priority_key(ev: EvidenceRecord) -> tuple[int, int, str]:
        we = weight_map.get(ev.evidence_id)
        strength = we.evidence_strength if we else ev.evidence_strength
        gates = gates_for_evidence(report, ev.evidence_id)
        is_op_ai = ev.ai_type.value in ("GENAI_LLM", "CLASSICAL_ML", "AGENTIC_AI")
        has_gate = bool(gates)
        if is_op_ai and has_gate:
            p = 0
        elif is_op_ai:
            p = 1
        elif strength.value in ("CONFIRMED", "STRONG"):
            p = 2
        else:
            p = 3
        s_rank = {"CONFIRMED": 0, "STRONG": 1, "INDICATIVE": 2, "MARKETING": 3, "NONE": 4}.get(strength.value, 5)
        return (p, s_rank, ev.evidence_id)

    sorted_records = sorted(report.evidence_records, key=_ev_priority_key)
    max_detailed_ev = 15
    selected_ev = sorted_records[:max_detailed_ev]

    ev_entries: list[str] = []
    for ev in selected_ev:
        we = weight_map.get(ev.evidence_id)
        gates = gates_for_evidence(report, ev.evidence_id)
        erules = we.fired_e_rule_ids if we else ev.e_rule_ids
        strength = we.evidence_strength.value if we else ev.evidence_strength.value

        entry_parts = [
            f"[{ev.evidence_id} | Class {ev.source_class} (Reliability {ev.source_reliability_grade})]",
            f"Direct URL: {ev.canonical_url}",
            f"Retrieval Date: {ev.retrieved_at}" + (f" (Published/Observed: {ev.published_or_observed_date})" if ev.published_or_observed_date else ""),
            f"SHA-256: {ev.content_sha256}",
            f"Verbatim Excerpt: \"{ev.verbatim_excerpt}\"",
            f"Evidentiary Analysis: Strength={strength}, Scope={ev.scope.value}, AI-Type={ev.ai_type.value}, E-Rules={'; '.join(erules) or 'None'}, Supported Gates={'; '.join(gates) or 'None'}",
        ]
        if ev.corroborating_urls:
            entry_parts.append(f"Corroborating Sources: {'; '.join(ev.corroborating_urls)}")
        if ev.extraction_limitations:
            entry_parts.append(f"Source Limitations: {'; '.join(ev.extraction_limitations)}")
        ev_entries.append(" | ".join(entry_parts))

    if ev_entries:
        evidence_desc = "\n\n".join(ev_entries)
        if len(report.evidence_records) > len(selected_ev):
            overflow = len(report.evidence_records) - len(selected_ev)
            evidence_desc += (
                f"\n\n[Evidentiary Register Note: {overflow} additional point-of-capture evidence items "
                f"verified and retained in complete evidentiary ledger.]"
            )
    else:
        lim_str = "; ".join(report.assessment_limitations) or "Proprietary vendor platform with no public developer disclosures"
        evidence_desc = (
            f"No qualifying public AI evidence detected during OSINT reconnaissance. "
            f"Assessment Limitations: {lim_str}. "
            f"Notice: Absence of public documentation does not confirm absence of internal or sub-processor AI usage; direct vendor diligence required."
        )

    # ── 7. Field 6: How Vendor May Be Using AI ──
    confirmed_signals = [
        c for c in report.signal_classifications
        if c.signal_category.value == "OPERATIONAL_AI"
        and c.ai_type.value in ("GENAI_LLM", "CLASSICAL_ML", "AGENTIC_AI")
        and c.confidence in ("HIGH", "CONFIRMED", "MEDIUM")
    ]
    confirmed_caps = []
    seen_caps: set[str] = set()
    for c in confirmed_signals:
        cap_desc = c.capability_summary or f"{c.ai_type.value} capability"
        norm_cap = re.sub(r"\s+", " ", cap_desc.strip().lower())
        if norm_cap in seen_caps:
            continue
        seen_caps.add(norm_cap)

        c_inputs = f" Inputs: {', '.join(c.known_inputs)}." if c.known_inputs else ""
        c_outputs = f" Outputs: {', '.join(c.known_outputs)}." if c.known_outputs else ""
        cust_status = (
            f" Customer data involved: {'Yes' if c.customer_data_involved else 'No / Unverified'}."
            if c.customer_data_involved is not None
            else ""
        )
        confirmed_caps.append(
            f"- Confirmed Production Capability: {cap_desc} ({c.ai_type.value}, Scope: {c.scope.value}).{c_inputs}{c_outputs}{cust_status}"
        )
        if len(confirmed_caps) >= 8:
            break

    how_sections = []
    if confirmed_caps:
        how_sections.append("CONFIRMED AI USE CASES (EVIDENCED):\n" + "\n".join(confirmed_caps))
    else:
        how_sections.append(
            f"CONFIRMED AI USE CASES:\nNone evidenced in public technical documentation for contracted service '{v_svc}'."
        )

    hypothesized = (
        f"POTENTIAL / HYPOTHESIZED AI INTEGRATION VECTORS:\n"
        f"Given vendor's operational role providing '{v_svc}' supporting '{v_proc}', "
        f"potential unconfirmed AI integration vectors include: "
        f"(1) Automated telemetry anomaly detection and predictive failure analytics in production workloads; "
        f"(2) Internal developer code assistance and automated batch script maintenance; "
        f"(3) Automated customer support and ticketing classification. "
        f"These represent potential operational exposures pending bilateral questionnaire verification."
    )
    how_sections.append(hypothesized)
    how_ai_used = "\n\n".join(how_sections)

    # ── 8. Field 7: Named AI Sub-processors / Model Providers ──
    is_operational_ai = (ai_detected == "Yes") or bool(confirmed_caps)
    conf_prov = [p for p in report.named_providers if p and p != "UNSPECIFIED"] if is_operational_ai else []
    conf_mod = [m for m in report.named_models if m and m != "UNSPECIFIED"] if is_operational_ai else []
    conf_infra = [h for h in report.hosting_infrastructures if h and h != "UNSPECIFIED"]

    sp_parts = []
    if conf_prov or conf_mod:
        sp_parts.append(f"Confirmed AI Sub-processors: {', '.join(conf_prov) if conf_prov else 'None explicitly confirmed'}")
        sp_parts.append(f"Confirmed Foundation Models: {', '.join(conf_mod) if conf_mod else 'None explicitly confirmed'}")
    else:
        sp_parts.append("Confirmed AI Sub-processors: None evidenced in public documentation.")
        sp_parts.append("Confirmed Foundation Models: None evidenced in public documentation.")

    sp_parts.append(f"Hosting Infrastructure: {', '.join(conf_infra) if conf_infra else 'Not disclosed in public documentation'}")
    sp_parts.append(f"Supply Chain Identity Status: {report.supply_chain_identity_status if is_operational_ai else 'NOT_APPLICABLE'}")
    sp_parts.append(
        "Unidentified / Unverified Providers: Upstream foundation model providers, API inference routing, and third-party data sub-processors remain unverified due to lack of public sub-processor transparency; requires contractual sub-processor list submission."
    )
    named_subprocessors = "\n".join(sp_parts)

    # ── 9. Field 8: AI Security Risk Classification ──
    ds = report.deterministic_risk_score
    if ds.policy_status == ScoringPolicyStatus.APPROVED_ACTIVE and ds.final_risk_score is not None:
        risk_class = f"{ds.overall_risk_rating} (Calibrated Deterministic Score: {ds.final_risk_score:.2f})"
    elif report.risk_findings:
        severities = [rf.severity.value for rf in report.risk_findings]
        order = ["Critical", "High", "Medium", "Low", "UNRESOLVED"]
        top_sev = next((s for s in order if s in severities), "Medium")
        rule_summary = ", ".join(f"{rf.risk_rule_id} ({rf.severity.value})" for rf in report.risk_findings)
        risk_class = f"{top_sev} (Qualitative Risk Rules: {rule_summary} — Policy Unconfigured)"
    else:
        risk_class = "Low (No AI-specific security risk findings identified — Policy Unconfigured)"

    # ── 10. Field 9: Risk Rationale ──
    rr_parts = [
        f"Risk classification synthesizes {crit_tier} criticality tier, {v_op} operational dependency, "
        f"and access to data classification: '{v_data}'."
    ]
    if report.risk_findings:
        rf_items = []
        for rf in report.risk_findings:
            rf_items.append(
                f"- {rf.risk_rule_id} ({rf.risk_title}): Severity {rf.severity.value}. Exposure: {rf.exposure_determination.value}. {rf.finding_description}"
            )
        rr_parts.append("Evaluated AI Risk Findings:\n" + "\n".join(rf_items))
    else:
        rr_parts.append("Evaluated AI Risk Findings: No evidence-backed R1–R8 risk findings surfaced in public documentation.")

    if report.known_controls_summary:
        rr_parts.append("Observed Mitigations & Controls: " + "; ".join(report.known_controls_summary))
    else:
        rr_parts.append("Observed Mitigations: No contractual AI controls or Zero-Data-Retention commitments publicly verified.")
    risk_rationale = "\n\n".join(rr_parts)

    # ── 11. Field 10: Recommended Action ──
    is_high_or_crit = crit_tier in ("Critical", "High")
    deadline = "Prior to contract renewal or within 30 days" if is_high_or_crit else "Within 60 days of assessment"
    action_items = [
        f"1. Governance & Diligence Review (Owner: Meridian Third-Party Risk Management / Deadline: {deadline}):",
        "   - Issue Meridian Supplementary AI Diligence Questionnaire to vendor security team to resolve identified disclosure gaps.",
        "2. Contractual Safeguards (Owner: Legal & Procurement / Deadline: Next Contract Renewal):",
        "   - Execute formal AI Data Processing Addendum (DPA) incorporating: (a) explicit prohibition against training or fine-tuning models on Meridian data; (b) enforceable Zero Data Retention (ZDR) commitment for inference prompts and outputs; (c) mandatory 60-day prior written notice for new AI sub-processors.",
        "3. Technical & Security Controls (Owner: Information Security Architecture / Deadline: Ongoing):",
        "   - Validate vendor's SOC 2 Type II with AI Trust Services Criteria or ISO/IEC 42001 certification.",
        "   - Ensure continuous API telemetry logging and strict tenant vector/RAG isolation.",
    ]
    if report.remediation_recommendations:
        action_items.append("4. Finding-Specific Remediation Requirements:")
        for r in report.remediation_recommendations:
            action_items.append(f"   - [{r.get('risk_rule_id')}] {r.get('risk_title')} ({r.get('severity')}): {r.get('recommended_remediation')}")
    recommended_action = "\n".join(action_items)

    # ── 12. Field 11: Assessed By / Date ──
    date_str = report.generated_at[:10] if report.generated_at else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assessed_by = f"Meridian Third-Party AI Risk Governance Team / {date_str}"

    return {
        "Vendor ID": v_id,
        "Vendor Name": v_name,
        "Vendor Description": v_desc,
        "Service / Product Provided to Meridian": v_svc,
        "Category": v_cat,
        "Public Website": v_web,
        "Business Process Supported": v_proc,
        "Operational Dependency": v_op,
        "Data Classification Accessed": v_data,
        "Data Volume (annual)": v_vol,
        "Criticality Tier": crit_tier,
        "Criticality Rationale": crit_rationale,
        "Assessment Depth Applied": depth_applied,
        "AI Usage Detected": ai_detected,
        "Evidence Description along with OSINT Source(s) / Evidence Link": evidence_desc,
        "How Vendor May Be Using AI": how_ai_used,
        "Named AI Sub-processors / Model Providers": named_subprocessors,
        "AI Security Risk Classification": risk_class,
        "Risk Rationale": risk_rationale,
        "Recommended Action": recommended_action,
        "Assessed By / Date": assessed_by,
    }


class AssessmentReportExporter:
    """Generates clean, secret-free JSON, Evidence CSV, Markdown, HTML, and Canonical Inventory exports."""

    @staticmethod
    def to_inventory_dict(
        report: VendorEndToEndAssessmentReport,
        vendor: Optional[MeridianVendor] = None,
    ) -> dict[str, str]:
        """Produce the canonical 21-column Meridian Vendor Inventory assessment record as a dictionary."""
        return build_canonical_vendor_inventory_record(report, vendor)

    @staticmethod
    def to_inventory_csv(
        report: VendorEndToEndAssessmentReport,
        vendor: Optional[MeridianVendor] = None,
    ) -> str:
        """Export the single vendor's canonical 21-column inventory record as RFC 4180 CSV (Excel-compatible)."""
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=CANONICAL_VENDOR_INVENTORY_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\r\n",
        )
        writer.writeheader()
        rec = build_canonical_vendor_inventory_record(report, vendor)
        writer.writerow(rec)
        return buf.getvalue()

    @staticmethod
    def to_batch_inventory_csv(
        reports: Sequence[VendorEndToEndAssessmentReport],
        vendors: Optional[Sequence[MeridianVendor]] = None,
    ) -> str:
        """Export a portfolio batch of canonical 21-column inventory records as RFC 4180 CSV (Excel-compatible)."""
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=CANONICAL_VENDOR_INVENTORY_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\r\n",
        )
        writer.writeheader()
        vendor_map = {v.vendor_id: v for v in (vendors or [])}
        for r in reports:
            v_match = vendor_map.get(r.vendor_id)
            rec = build_canonical_vendor_inventory_record(r, v_match)
            writer.writerow(rec)
        return buf.getvalue()

    @staticmethod
    def to_json(report: VendorEndToEndAssessmentReport) -> str:
        payload = report.model_dump(mode="json")
        return json.dumps(payload, indent=2)

    @staticmethod
    def to_evidence_csv(report: VendorEndToEndAssessmentReport) -> str:
        buf = io.StringIO()
        fieldnames = [
            "report_id",
            "evidence_id",
            "vendor_id",
            "legal_entity",
            "source_class",
            "source_reliability_grade",
            "canonical_url",
            "original_url",
            "final_url",
            "retrieval_method",
            "source_integrity_status",
            "retrieved_at",
            "published_or_observed_date",
            "content_sha256",
            "scope",
            "ai_type",
            "evidence_strength",
            "weight_band",
            "e_rule_ids",
            "supports_gates",
            "risk_tags",
            "named_provider",
            "named_model",
            "hosting_infrastructure",
            "hostile_instruction_detected",
            "is_synthetic_test_fixture",
            "verbatim_excerpt",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        weight_map = {w.evidence_id: w for w in report.weighted_evidence}
        for ev in report.evidence_records:
            we = weight_map.get(ev.evidence_id)
            sup_gates = gates_for_evidence(report, ev.evidence_id)
            writer.writerow({
                "report_id": report.report_id,
                "evidence_id": ev.evidence_id,
                "vendor_id": ev.vendor_id,
                "legal_entity": ev.legal_entity,
                "source_class": ev.source_class,
                "source_reliability_grade": ev.source_reliability_grade,
                "canonical_url": ev.canonical_url,
                "original_url": ev.original_url,
                "final_url": ev.final_url or "",
                "retrieval_method": ev.retrieval_method or "DIRECT_HTTP",
                "source_integrity_status": ev.source_integrity_status or "VERIFIED",
                "retrieved_at": ev.retrieved_at,
                "published_or_observed_date": ev.published_or_observed_date or "",
                "content_sha256": ev.content_sha256,
                "scope": ev.scope.value,
                "ai_type": ev.ai_type.value,
                "evidence_strength": we.evidence_strength.value if we else ev.evidence_strength.value,
                "weight_band": we.weight_band.value if we else "",
                "e_rule_ids": ";".join(we.fired_e_rule_ids if we else ev.e_rule_ids),
                "supports_gates": ";".join(sup_gates),
                "risk_tags": ";".join(ev.risk_tags),
                "named_provider": ev.named_provider or "",
                "named_model": ev.named_model or "",
                "hosting_infrastructure": ev.hosting_infrastructure or "",
                "hostile_instruction_detected": str(ev.hostile_instruction_detected),
                "is_synthetic_test_fixture": str(ev.is_synthetic_test_fixture),
                "verbatim_excerpt": ev.verbatim_excerpt,
            })
        return buf.getvalue()

    @staticmethod
    def to_markdown(report: VendorEndToEndAssessmentReport) -> str:
        crit = report.criticality_summary or {}
        osint = report.osint_plan_summary or {}
        ds = report.deterministic_risk_score
        tracker = build_nine_level_stage_tracker(report)
        weight_map = {w.evidence_id: w for w in report.weighted_evidence}

        mode_label = (
            "Sample Assessment (SYNTHETIC FIXTURE / DEMO WALKTHROUGH — NOT REAL VENDOR EVIDENCE)"
            if report.is_synthetic_evidence_run
            else "Live OSINT Assessment"
        )

        lines: list[str] = [
            f"# Meridian Financial — Third-Party AI Risk Assessment Dossier",
            f"",
            f"- **Report Run ID:** `{report.report_id}`",
            f"- **Generated At (UTC):** `{report.generated_at}`",
            f"- **Assessment Type:** **{mode_label}**",
            f"- **Vendor:** {report.legal_entity} (`{report.vendor_id}`)",
            f"- **Primary Domain:** `{report.domain or 'N/A'}`",
            f"- **Service Purchased by Meridian:** {report.service_product_provided or 'N/A'}",
            f"- **Business Process Supported:** {report.business_process_supported or 'N/A'}",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary & Governance Determination",
            f"",
            f"| Dimension | Determination | Summary Details |",
            f"| :--- | :--- | :--- |",
            f"| **Effective Criticality Tier** | **{crit.get('final_criticality', 'UNKNOWN')}** | Base Score: `{crit.get('base_score', 'N/A')}` (Provisional Tier: `{crit.get('provisional_criticality', 'N/A')}`) |",
            f"| **Investigation Depth** | **{osint.get('selected_depth_profile', 'N/A')}** | Depth Completed: `{'Yes' if report.ai_use_verdict.depth_completed else 'No'}` |",
            f"| **AI-Use Determination** | **{format_plain_status(report.ai_use_verdict.verdict)}** (`{report.ai_use_verdict.verdict.value}`) | {report.ai_use_verdict.basis_summary} |",
            f"| **Determined Service Scope** | {format_plain_status(report.determined_scope)} | AI Types: `{', '.join(format_plain_status(t) for t in report.determined_ai_types) if report.determined_ai_types else 'None Identified'}` |",
            f"| **Security Risk Findings (`R1–R8`)** | **{len(report.risk_findings)} finding(s)** | Fourth-Party Supply-Chain Status: `{format_plain_status(report.supply_chain_identity_status)}` |",
            f"| **Numerical Risk Rating** | **{format_plain_status(ds.policy_status)}** (`{ds.policy_status.value}`) | Final Score: `{ds.final_risk_score if ds.final_risk_score is not None else 'Calibrated numerical risk score unavailable'}` |",
            f"| **Analyst Review Status** | **{'Review Required' if report.human_Review_required else 'Complete'}** | {len(report.human_validations)} analyst sign-off record(s) logged |",
            f"",
            f"## 2. Nine-Level Assessment Workflow Summary",
            f"",
            f"| Stage | Framework Level | Status | Method | Scope & Volume | Outcome Summary |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for row in tracker:
            lines.append(
                f"| **{row.level_label}** | {row.level_name} | {row.analyst_status} | {row.analyst_method} | {row.analyst_counts} | {row.analyst_summary} |"
            )
            for sub in row.substeps:
                lines.append(
                    f"| ↳ *{sub['substep_id']}* | *{sub['substep_name']}* | `{sub['status']}` | — | — | {sub['detail']} |"
                )

        lines.extend([
            f"",
            f"## 3. Criticality Factor Breakdown (`D, P, R, O, V`) & `O1–O5` Safeguards",
            f"",
            f"| Factor | Score | Weight | Weighted | Method | Rationale |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        factors = crit.get("factors", {})
        for fk in ("D", "P", "R", "O", "V"):
            fd = factors.get(fk, {})
            lines.append(
                f"| **{fk}** | `{fd.get('score', 'UNKNOWN')}` | `{fd.get('weight', '')}` | "
                f"`{fd.get('weighted_contribution', '')}` | {format_plain_status(fd.get('determination_method', ''))} | {fd.get('rationale', '')} |"
            )

        overrides = crit.get("overrides_evaluated", [])
        if overrides:
            lines.extend([
                f"",
                f"### Evaluated Governance Overrides (`O1–O5`)",
                f"",
                f"| Override ID | Triggered | Resulting Tier | Condition & Rationale |",
                f"| :--- | :--- | :--- | :--- |",
            ])
            for ov in overrides:
                lines.append(
                    f"| `{ov.get('override_id')}` | `{'Yes' if ov.get('triggered') else 'No'}` | "
                    f"`{ov.get('resulting_tier') or '—'}` | {ov.get('rationale', '')} |"
                )

        cond_raw = osint.get("conditional_source_classes", [])
        cond_strs = [
            f"{c.get('source_id', '')} ({c.get('condition', '')})" if isinstance(c, dict) else str(c)
            for c in cond_raw
        ]
        lines.extend([
            f"",
            f"## 4. Investigation Scope & Source Coverage (`S1–S12`)",
            f"",
            f"- **Selected Depth Profile:** `{osint.get('selected_depth_profile', 'N/A')}` ({osint.get('selection_reason', '')})",
            f"- **Required Source Classes:** `{', '.join(osint.get('required_source_classes', []))}`",
            f"- **Conditional Source Classes:** `{', '.join(cond_strs) or 'None'}`",
            f"- **Planned Queries:** `{osint.get('planned_queries_count', 0)}` | **Executed:** `{osint.get('executed_queries_count', 0)}` | **Unavailable:** `{osint.get('unavailable_queries_count', 0)}`",
            f"- **Stopping Condition (`{osint.get('stopping_rule_id', '')}`):** {format_plain_status(osint.get('stop_status', ''))}",
            f"",
            f"## 5. Evidentiary Weighting (`E1–E14`), Gates (`G1–G4`) & AI-Use Determination",
            f"",
            f"| Gate ID | Gate Name | Status | Supporting Evidence IDs | Rationale |",
            f"| :--- | :--- | :--- | :--- | :--- |",
        ])
        for g in report.gates:
            lines.append(
                f"| `{g.gate_id}` | {g.title} | {format_plain_status(g.status)} | "
                f"`{', '.join(g.supporting_evidence_ids) or 'None'}` | {g.rationale} |"
            )

        lines.extend([
            f"",
            f"## 6. Point-of-Capture Evidence Register (`E1–E14`)",
            f"",
        ])
        if not report.evidence_records:
            lines.append("*No external or sample evidence records were captured during this run.*")
        else:
            lines.extend([
                f"| Evidence ID | Source | Retrieved At | Scope | Strength | E-Rules | Gates | Source URL & Verbatim Excerpt |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for ev in report.evidence_records:
                we = weight_map.get(ev.evidence_id)
                strength = we.evidence_strength.value if we else ev.evidence_strength.value
                erules = ", ".join(we.fired_e_rule_ids if we else ev.e_rule_ids)
                gates_str = ", ".join(gates_for_evidence(report, ev.evidence_id))
                excerpt_clean = ev.verbatim_excerpt.replace("\n", " ").replace("|", "\\|")
                lines.append(
                    f"| `{ev.evidence_id}` | `{ev.source_class}` (Grade {ev.source_reliability_grade}) | "
                    f"`{ev.retrieved_at}` | {format_plain_status(ev.scope)} | {format_plain_status(strength)} | `{erules}` | `{gates_str}` | "
                    f"[{ev.canonical_url}]({ev.canonical_url}) — \"{excerpt_clean}\" (`SHA-256: {ev.content_sha256[:12]}...`) |"
                )

        lines.extend([
            f"",
            f"## 7. Security & Governance Risk Findings (`R1–R8`) & Recommended Follow-Up",
            f"",
        ])
        if not report.risk_findings:
            lines.append("*No R1–R8 risk findings surfaced for this run.*")
        else:
            for rf in report.risk_findings:
                lines.extend([
                    f"### {rf.risk_rule_id}: {rf.risk_title} (`{rf.finding_id}`)",
                    f"- **Severity:** `{rf.severity.value}` | **Exposure Determination:** {format_plain_status(rf.exposure_determination)}",
                    f"- **Finding Description:** {rf.finding_description}",
                    f"- **Supporting Evidence IDs:** `{', '.join(rf.evidence_ids)}`",
                    f"- **Supporting Verbatim Quote(s):** {'; '.join(f'\"{q}\"' for q in rf.quotations)}",
                    f"- **Analytical Summary:** {rf.analyst_explanation or rf.finding_description}",
                    f"- **Recommended Follow-Up Action:** {rf.recommended_remediation}",
                    f"",
                ])

        lines.extend([
            f"## 8. Capability Notices, Limitations & Analyst Sign-Off",
            f"",
            f"### Capability Notices",
        ])
        analyst_notices = format_analyst_unavailable_integrations(report)
        if analyst_notices:
            for notice in analyst_notices:
                lines.append(f"- {notice}")
        else:
            lines.append("- All assessment capabilities active")

        lines.extend([f"", f"### Assessment Limitations"])
        if report.assessment_limitations:
            for lim in report.assessment_limitations:
                lines.append(f"- {lim}")
        else:
            lines.append("- None")

        lines.extend([f"", f"### Analyst Review Triggers"])
        if report.human_review_triggers:
            for trig in report.human_review_triggers:
                lines.append(f"- {trig}")
        else:
            lines.append("- None")

        if report.human_validations:
            lines.extend([
                f"",
                f"### Logged Analyst Sign-Off Decisions",
                f"",
                f"| Validation ID | Target | Decision | Reviewer | Timestamp | Rationale |",
                f"| :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for hv in report.human_validations:
                lines.append(
                    f"| `{hv.validation_id}` | `{hv.target_type}:{hv.target_id}` | `{hv.human_decision}` | "
                    f"{hv.reviewer_id} | `{hv.reviewed_at}` | {hv.rationale} |"
                )

        return "\n".join(lines) + "\n"

    @staticmethod
    def to_html(report: VendorEndToEndAssessmentReport) -> str:
        """Render a clean, self-contained enterprise HTML assessment report suitable for archival or printing."""
        crit = report.criticality_summary or {}
        osint = report.osint_plan_summary or {}
        ds = report.deterministic_risk_score
        tracker = build_nine_level_stage_tracker(report)
        weight_map = {w.evidence_id: w for w in report.weighted_evidence}

        mode_banner = (
            '<div class="banner banner-synth">Sample Assessment (Pre-Collected Demonstration Evidence — Not Live Vendor Evidence)</div>'
            if report.is_synthetic_evidence_run
            else '<div class="banner banner-real">Third-Party AI Risk Assessment Dossier</div>'
        )

        stage_rows_html = []
        for row in tracker:
            stage_rows_html.append(
                f"<tr>"
                f"<td><strong>{html.escape(row.level_label)}</strong></td>"
                f"<td>{html.escape(row.level_name)}</td>"
                f"<td><code>{html.escape(row.analyst_status)}</code></td>"
                f"<td>{html.escape(row.analyst_method)}</td>"
                f"<td>{html.escape(row.analyst_counts)}</td>"
                f"<td>{html.escape(row.analyst_summary)}</td>"
                f"</tr>"
            )

        ev_rows_html = []
        for ev in report.evidence_records:
            we = weight_map.get(ev.evidence_id)
            strength = we.evidence_strength.value if we else ev.evidence_strength.value
            erules = ", ".join(we.fired_e_rule_ids if we else ev.e_rule_ids)
            sup_gates = gates_for_evidence(report, ev.evidence_id)
            safe_url = html.escape(ev.canonical_url)
            ev_rows_html.append(
                f"<tr>"
                f"<td><code>{html.escape(ev.evidence_id)}</code></td>"
                f"<td><code>{html.escape(ev.source_class)}</code> (Grade {html.escape(ev.source_reliability_grade)})</td>"
                f"<td><code>{html.escape(ev.retrieved_at)}</code></td>"
                f"<td>{html.escape(format_plain_status(ev.scope))}</td>"
                f"<td><strong>{html.escape(format_plain_status(strength))}</strong></td>"
                f"<td><code>{html.escape(erules)}</code></td>"
                f"<td><code>{html.escape(', '.join(sup_gates))}</code></td>"
                f"<td><a href='{safe_url}' target='_blank' rel='noopener noreferrer'>{safe_url}</a><br/>"
                f"<em>\"{html.escape(ev.verbatim_excerpt)}\"</em><br/>"
                f"<small>SHA-256: <code>{html.escape(ev.content_sha256)}</code></small></td>"
                f"</tr>"
            )

        rf_rows_html = []
        for rf in report.risk_findings:
            rf_rows_html.append(
                f"<tr>"
                f"<td><strong>{html.escape(rf.risk_rule_id)}</strong><br/><small>{html.escape(rf.risk_title)}</small></td>"
                f"<td><strong>{html.escape(rf.severity.value)}</strong><br/><small>{html.escape(format_plain_status(rf.exposure_determination))}</small></td>"
                f"<td>{html.escape(rf.finding_description)}<br/>"
                f"<small>Evidence: <code>{html.escape(', '.join(rf.evidence_ids))}</code></small></td>"
                f"<td>{html.escape(rf.analyst_explanation or rf.finding_description)}</td>"
                f"<td>{html.escape(rf.recommended_remediation)}</td>"
                f"</tr>"
            )

        analyst_notices = format_analyst_unavailable_integrations(report)
        lim_items = analyst_notices + list(report.assessment_limitations)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Meridian Third-Party AI Risk Assessment — {html.escape(report.legal_entity)} ({html.escape(report.vendor_id)})</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #0f172a; margin: 28px; line-height: 1.45; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; color: #0f172a; }}
  h2 {{ font-size: 15px; margin-top: 22px; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; color: #1e293b; text-transform: uppercase; letter-spacing: 0.04em; }}
  .meta {{ font-size: 12.5px; color: #475569; margin-bottom: 14px; }}
  .banner {{ padding: 8px 12px; border-radius: 4px; font-weight: 600; font-size: 12px; margin-bottom: 16px; }}
  .banner-synth {{ background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }}
  .banner-real {{ background: #e0f2fe; color: #075985; border: 1px solid #0284c7; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12.5px; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f8fafc; font-weight: 600; color: #334155; }}
  .substep-row td {{ background: #f8fafc; color: #475569; font-size: 12px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
  {mode_banner}
  <h1>Meridian Financial — Third-Party AI Risk Assessment Dossier</h1>
  <div class="meta">
    Run ID: <code>{html.escape(report.report_id)}</code> &nbsp;|&nbsp;
    Generated (UTC): <code>{html.escape(report.generated_at)}</code> &nbsp;|&nbsp;
    Vendor: <strong>{html.escape(report.legal_entity)}</strong> (<code>{html.escape(report.vendor_id)}</code>, Domain: <code>{html.escape(report.domain or 'N/A')}</code>)<br/>
    Service Purchased: {html.escape(report.service_product_provided or 'N/A')} &nbsp;|&nbsp;
    Business Process: {html.escape(report.business_process_supported or 'N/A')}
  </div>

  <h2>1. Executive Summary</h2>
  <table>
    <tr>
      <th>Effective Criticality</th>
      <th>Investigation Depth</th>
      <th>AI-Use Determination</th>
      <th>Risk Findings (R1–R8)</th>
      <th>Numerical Risk Score</th>
      <th>Analyst Review</th>
    </tr>
    <tr>
      <td><strong>{html.escape(str(crit.get('final_criticality', 'UNKNOWN')))}</strong> (Base: <code>{html.escape(str(crit.get('base_score', 'N/A')))}</code>)</td>
      <td><code>{html.escape(str(osint.get('selected_depth_profile', 'N/A')))}</code> (Completed: <code>{'Yes' if report.ai_use_verdict.depth_completed else 'No'}</code>)</td>
      <td><strong>{html.escape(format_plain_status(report.ai_use_verdict.verdict))}</strong><br/><small>{html.escape(report.ai_use_verdict.basis_summary)}</small></td>
      <td><strong>{len(report.risk_findings)}</strong> finding(s)</td>
      <td>{html.escape(format_plain_status(ds.policy_status))}<br/><small>Score: <code>{ds.final_risk_score if ds.final_risk_score is not None else 'Not calibrated'}</code></small></td>
      <td><code>{'Review Required' if report.human_Review_required else 'Complete'}</code> ({len(report.human_validations)} logged)</td>
    </tr>
  </table>

  <h2>2. Nine-Level Assessment Workflow Summary</h2>
  <table>
    <thead>
      <tr><th>Stage</th><th>Framework Level</th><th>Status</th><th>Method</th><th>Scope &amp; Volume</th><th>Outcome Summary</th></tr>
    </thead>
    <tbody>
      {''.join(stage_rows_html)}
    </tbody>
  </table>

  <h2>3. Point-of-Capture Evidence Register (E1–E14)</h2>
  <table>
    <thead>
      <tr><th>Evidence ID</th><th>Source</th><th>Retrieved At</th><th>Scope</th><th>Strength</th><th>E-Rules</th><th>Gates</th><th>URL &amp; Verbatim Excerpt</th></tr>
    </thead>
    <tbody>
      {''.join(ev_rows_html) if ev_rows_html else '<tr><td colspan="8">No evidence records captured in this run.</td></tr>'}
    </tbody>
  </table>

  <h2>4. Security &amp; Governance Risk Findings (R1–R8) &amp; Follow-Up Actions</h2>
  <table>
    <thead>
      <tr><th>Rule</th><th>Severity</th><th>Finding &amp; Supporting Evidence</th><th>Analytical Explanation</th><th>Recommended Follow-Up</th></tr>
    </thead>
    <tbody>
      {''.join(rf_rows_html) if rf_rows_html else '<tr><td colspan="5">No R1–R8 risk findings surfaced in this run.</td></tr>'}
    </tbody>
  </table>

  <h2>5. Capability Notices, Limitations &amp; Review Triggers</h2>
  <ul>
    {''.join(f'<li>{html.escape(lim)}</li>' for lim in lim_items) or '<li>None</li>'}
  </ul>
</body>
</html>
"""
