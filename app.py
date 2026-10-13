"""Meridian Financial Services — UncoverAI Enterprise Third-Party AI Risk Governance Platform.

Provides a cohesive enterprise analyst workspace across five top-level views:
  - Portfolio: Risk dashboard, 5-KPI summary, filterable vendor inventory table with full text wrapping,
    and structured selected-vendor dossier preview.
  - Assessments: Three-stage assessment workspace (A. Setup -> B. Execution -> C. Results) supporting
    both Live OSINT Assessments (Serper + Direct HTTP / ScraperAPI fallback) and explicitly labeled
    Sample Assessments.
  - Evidence: Point-of-capture evidence workbench with filters across Vendor, Source Class (S1–S12),
    Evidence Strength, Scope, Gate (G1–G4), and Retrieval Method, plus source-integrity diagnostics.
  - Reports: Executive-ready vendor assessment dossier and multi-format export center (HTML, Markdown,
    Evidence CSV, JSON).
  - Settings: Consolidated technical diagnostics for Serper, ScraperAPI, Source-Integrity Guard,
    Ollama dual-model routing (llama3.2:latest + mistral:latest), Level 5 RAG embeddings, and Level 9
    scoring policy calibration.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional

import streamlit as st

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.models.ai_risk import (
    ExtractionStatus,
    HumanValidationRecord,
    ScoringPolicyStatus,
    VendorEndToEndAssessmentReport,
)
from meridian_assessment.models.factor_result import (
    CriticalityResult,
    HumanReviewRecord,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.services.end_to_end_pipeline import (
    HumanValidationService,
    MeridianEndToEndPipeline,
)
from meridian_assessment.services.extraction import (
    EvidenceRepository,
    build_fetch_adapter_from_env,
)
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine
from meridian_assessment.services.offline_demo_fixtures import (
    build_synthetic_demo_adapters_for_vendor,
)
from meridian_assessment.services.osint import (
    OSINTInvestigationPlanner,
    OSINTSourceRegistry,
    build_search_backend_from_env,
)
from meridian_assessment.services.readiness_and_exports import (
    AssessmentReportExporter,
    CANONICAL_NINE_LEVEL_NAMES,
    SystemReadinessInspector,
    SystemReadinessReport,
    build_nine_level_stage_tracker,
    format_analyst_unavailable_integrations,
    format_plain_status,
    gates_for_evidence,
    validate_vendor_assessment_input,
)

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="UncoverAI — Meridian Third-Party AI Risk Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_DEFAULT_VENDORS_CSV = Path("data/input/meridian_vendors.csv")
_DEFAULT_DB_PATH = Path("data/state/evidence_repository.sqlite3")
_DEFAULT_ARTIFACT_DIR = Path("data/state/artifacts")
_DEFAULT_ACTIVE_STATUS_PATH = Path("data/state/active_assessment_status.json")

_NAV_PAGES = ("Portfolio", "Assessments", "Evidence", "Reports", "Settings")
_TIER_SORT_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

_GLOBAL_JOB_LOCK = threading.Lock()
_GLOBAL_JOB_STATE: dict[str, Any] = {
    "status": "IDLE",
    "vendor_id": None,
    "vendor_name": None,
    "mode": None,
    "level_num": 0,
    "stage_title": "",
    "detail": "",
    "completed": 0,
    "total": 1,
    "remaining": 0,
    "level_pct": 0,
    "overall_pct": 0,
    "report_id": None,
    "error": None,
    "cancel_requested": False,
    "updated_at": None,
    "latest_report": None,
    "latest_vendor": None,
}


# ─── Enterprise Visual Design System (CSS) ────────────────────────────────────


def _inject_enterprise_styles() -> None:
    st.markdown(
        """
        <style>
        /* Hide Streamlit default chrome and sidebar completely */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        div[data-testid="stToolbar"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        div[data-testid="collapsedControl"] {
            display: none !important;
        }
        #MainMenu, footer {
            visibility: hidden;
        }

        /* Application canvas & typography */
        .stApp {
            background-color: #F4F6F9;
            color: #0F172A;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
        }
        .block-container {
            max-width: 1440px !important;
            padding-top: 1.1rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Enterprise Shell Header */
        .uc-shell-header {
            background: #0F172A;
            color: #F8FAFC;
            border-radius: 8px;
            padding: 14px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
            border: 1px solid #1E293B;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
            margin-bottom: 10px;
        }
        .uc-brand-group {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .uc-brand-mark {
            background: #1D4ED8;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            padding: 6px 10px;
            border-radius: 5px;
            text-transform: uppercase;
        }
        .uc-brand-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.01em;
            line-height: 1.2;
        }
        .uc-brand-subtitle {
            font-size: 0.78rem;
            color: #94A3B8;
            margin-top: 2px;
        }
        .uc-header-pills {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .uc-header-pill {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 999px;
            background: #1E293B;
            color: #E2E8F0;
            border: 1px solid #334155;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .uc-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }
        .uc-dot-green { background: #10B981; }
        .uc-dot-amber { background: #F59E0B; }
        .uc-dot-blue { background: #3B82F6; }

        /* Page Header Banner */
        .uc-page-header {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 14px 20px;
            margin-top: 6px;
            margin-bottom: 14px;
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        h1.uc-page-title, .uc-page-title {
            font-size: 1.32rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
            margin: 0 0 3px 0 !important;
            padding: 0 !important;
            line-height: 1.25 !important;
            letter-spacing: -0.015em !important;
        }
        p.uc-page-desc, .uc-page-desc {
            font-size: 0.85rem !important;
            color: #475569 !important;
            margin: 0 !important;
            line-height: 1.45 !important;
        }

        /* KPI Metric Cards */
        .uc-kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }
        @media (max-width: 1100px) {
            .uc-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        .uc-kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: 3px solid #CBD5E1;
            border-radius: 8px;
            padding: 13px 16px;
            min-height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        .uc-kpi-blue { border-top-color: #1D4ED8; }
        .uc-kpi-green { border-top-color: #059669; }
        .uc-kpi-red { border-top-color: #DC2626; }
        .uc-kpi-amber { border-top-color: #D97706; }
        .uc-kpi-slate { border-top-color: #475569; }
        .uc-kpi-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #475569;
        }
        .uc-kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.15;
            margin: 5px 0 3px 0;
        }
        .uc-kpi-sub {
            font-size: 0.75rem;
            color: #64748B;
        }

        /* Content Panels & Section Headings */
        .uc-panel {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        .uc-section-header {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #1D4ED8;
            border-radius: 6px;
            padding: 11px 16px;
            margin-top: 8px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
        }
        .uc-panel .uc-section-header {
            background: transparent;
            border: none;
            border-bottom: 1px solid #E2E8F0;
            border-radius: 0;
            padding: 0 0 10px 0;
            margin-top: 0;
            margin-bottom: 12px;
            box-shadow: none;
        }
        h2.uc-section-title, h3.uc-section-title, .uc-section-title {
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.3 !important;
        }
        p.uc-section-subtitle, .uc-section-subtitle {
            font-size: 0.79rem !important;
            color: #64748B !important;
            margin: 2px 0 0 0 !important;
            line-height: 1.35 !important;
        }

        /* Semantic Badges */
        .uc-badge {
            display: inline-block;
            padding: 3px 9px;
            border-radius: 4px;
            font-size: 0.73rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            line-height: 1.3;
            white-space: nowrap;
        }
        .uc-badge-critical { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
        .uc-badge-high { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
        .uc-badge-medium { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
        .uc-badge-low { background: #F8FAFC; color: #334155; border: 1px solid #CBD5E1; }
        .uc-badge-success { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
        .uc-badge-neutral { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }
        .uc-badge-purple { background: #F5F3FF; color: #5B21B6; border: 1px solid #DDD6FE; }

        /* Enterprise Data Tables (Wrapped, Non-Clipped) */
        .uc-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            background: #FFFFFF;
            margin-bottom: 14px;
        }
        table.uc-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.84rem;
            color: #0F172A;
        }
        table.uc-table th {
            background: #F8FAFC;
            color: #334155;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 10px 12px;
            border-bottom: 1px solid #E2E8F0;
            text-align: left;
            vertical-align: middle;
        }
        table.uc-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #F1F5F9;
            vertical-align: top;
            line-height: 1.45;
            white-space: normal;
            word-break: break-word;
        }
        table.uc-table tr:last-child td {
            border-bottom: none;
        }
        table.uc-table tr:hover td {
            background: #F8FAFC;
        }
        table.uc-table tr.uc-row-selected td {
            background: #EFF6FF;
        }
        .uc-mono {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.78rem;
            color: #1E293B;
            background: #F1F5F9;
            padding: 2px 5px;
            border-radius: 4px;
        }

        /* Callout Banners */
        .uc-callout {
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 0.84rem;
            line-height: 1.45;
            margin-bottom: 14px;
            border-left: 4px solid #CBD5E1;
        }
        .uc-callout-info {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-left: 4px solid #1D4ED8;
            color: #1E3A8A;
        }
        .uc-callout-warn {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 4px solid #D97706;
            color: #78350F;
        }
        .uc-callout-danger {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-left: 4px solid #DC2626;
            color: #7F1D1D;
        }
        .uc-callout-success {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-left: 4px solid #059669;
            color: #064E3B;
        }

        /* Key-Value Metadata Grid */
        .uc-kv-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 12px;
        }
        .uc-kv-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 10px 12px;
        }
        .uc-kv-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748B;
            margin-bottom: 3px;
        }
        .uc-kv-val {
            font-size: 0.88rem;
            font-weight: 600;
            color: #0F172A;
            word-break: break-word;
        }

        /* Empty State */
        .uc-empty-state {
            background: #FFFFFF;
            border: 1px dashed #CBD5E1;
            border-radius: 8px;
            padding: 28px 24px;
            text-align: center;
            color: #475569;
            margin: 12px 0;
        }
        .uc-empty-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 6px;
        }
        .uc-empty-desc {
            font-size: 0.84rem;
            color: #64748B;
            max-width: 560px;
            margin: 0 auto;
        }

        /* Button styling refinements */
        div.stButton > button {
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.83rem !important;
            padding: 0.42rem 0.9rem !important;
            transition: all 0.12s ease-in-out !important;
        }
        div.stButton > button[kind="primary"] {
            background-color: #1D4ED8 !important;
            border-color: #1D4ED8 !important;
            color: #FFFFFF !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1E40AF !important;
            border-color: #1E40AF !important;
        }
        div.stButton > button[kind="secondary"] {
            background-color: #FFFFFF !important;
            border-color: #CBD5E1 !important;
            color: #1E293B !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            background-color: #F8FAFC !important;
            border-color: #94A3B8 !important;
            color: #0F172A !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Badge & Formatting Helpers ───────────────────────────────────────────────


def _tier_badge(tier: Optional[str]) -> str:
    t = (tier or "Unknown").strip()
    cls_map = {
        "Critical": "uc-badge-critical",
        "High": "uc-badge-high",
        "Medium": "uc-badge-medium",
        "Low": "uc-badge-low",
    }
    cls = cls_map.get(t, "uc-badge-neutral")
    return f'<span class="uc-badge {cls}">{html.escape(t)}</span>'


def _status_badge(label: str, variant: str = "neutral") -> str:
    cls_map = {
        "critical": "uc-badge-critical",
        "high": "uc-badge-high",
        "medium": "uc-badge-medium",
        "low": "uc-badge-low",
        "success": "uc-badge-success",
        "neutral": "uc-badge-neutral",
        "purple": "uc-badge-purple",
    }
    cls = cls_map.get(variant, "uc-badge-neutral")
    return f'<span class="uc-badge {cls}">{html.escape(label)}</span>'


def _verdict_badge(verdict_val: str) -> str:
    v = (verdict_val or "").upper()
    if v == "YES":
        return _status_badge("Confirmed AI Use (YES)", "critical")
    if v == "INCONCLUSIVE":
        return _status_badge("Indicative / Inconclusive", "high")
    if v == "NO":
        return _status_badge("No AI Identified (NO)", "success")
    if v == "INCOMPLETE_ASSESSMENT":
        return _status_badge("Incomplete Assessment", "high")
    return _status_badge(format_plain_status(v), "neutral")


def _severity_badge(sev: str) -> str:
    s = (sev or "").capitalize()
    if s == "Critical":
        return _status_badge("Critical", "critical")
    if s == "High":
        return _status_badge("High", "high")
    if s == "Medium":
        return _status_badge("Medium", "medium")
    if s == "Low":
        return _status_badge("Low", "low")
    return _status_badge(s or "Unresolved", "neutral")


def _format_short_date(iso_str: Optional[str]) -> str:
    if not iso_str:
        return "—"
    s = str(iso_str).strip()
    if len(s) >= 16:
        return s[:16].replace("T", " ") + " UTC"
    return s


# ─── Session State & Data Initialization ──────────────────────────────────────


class _AssessmentCancelledError(RuntimeError):
    """Raised when the analyst cancels an active background assessment run."""


def _save_job_state_to_disk(snapshot: dict[str, Any]) -> None:
    try:
        _DEFAULT_ACTIVE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: v for k, v in snapshot.items() if k not in ("latest_report", "latest_vendor")
        }
        _DEFAULT_ACTIVE_STATUS_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    except Exception:
        pass


def _update_global_job_progress(
    *,
    status: str,
    vendor_id: Optional[str] = None,
    vendor_name: Optional[str] = None,
    mode: Optional[str] = None,
    level_num: int = 0,
    stage_title: str = "",
    detail: str = "",
    completed: int = 0,
    total: int = 1,
    report_id: Optional[str] = None,
    error: Optional[str] = None,
    latest_report: Optional[VendorEndToEndAssessmentReport] = None,
    latest_vendor: Optional[MeridianVendor] = None,
) -> dict[str, Any]:
    safe_total = max(1, int(total))
    safe_comp = max(0, min(int(completed), safe_total))
    rem = max(0, safe_total - safe_comp)
    lvl_pct = int(round((safe_comp / float(safe_total)) * 100))
    if status == "COMPLETED":
        ov_pct = 100
        lvl_pct = 100
    elif level_num <= 0:
        ov_pct = 0
    else:
        ov_pct = min(99, max(1, int(round(((level_num - 1 + (safe_comp / float(safe_total))) / 9.0) * 100))))

    with _GLOBAL_JOB_LOCK:
        if vendor_id is not None:
            _GLOBAL_JOB_STATE["vendor_id"] = vendor_id
        if vendor_name is not None:
            _GLOBAL_JOB_STATE["vendor_name"] = vendor_name
        if mode is not None:
            _GLOBAL_JOB_STATE["mode"] = mode
        _GLOBAL_JOB_STATE["status"] = status
        _GLOBAL_JOB_STATE["level_num"] = level_num
        _GLOBAL_JOB_STATE["stage_title"] = stage_title
        _GLOBAL_JOB_STATE["detail"] = detail
        _GLOBAL_JOB_STATE["completed"] = safe_comp
        _GLOBAL_JOB_STATE["total"] = safe_total
        _GLOBAL_JOB_STATE["remaining"] = rem
        _GLOBAL_JOB_STATE["level_pct"] = lvl_pct
        _GLOBAL_JOB_STATE["overall_pct"] = ov_pct
        if status == "RUNNING":
            _GLOBAL_JOB_STATE["error"] = None
            if report_id is None:
                _GLOBAL_JOB_STATE["report_id"] = None
            if latest_report is None:
                _GLOBAL_JOB_STATE["latest_report"] = None
        else:
            if report_id is not None:
                _GLOBAL_JOB_STATE["report_id"] = report_id
            if latest_report is not None:
                _GLOBAL_JOB_STATE["latest_report"] = latest_report
            _GLOBAL_JOB_STATE["error"] = error
        if latest_vendor is not None:
            _GLOBAL_JOB_STATE["latest_vendor"] = latest_vendor
        snapshot = dict(_GLOBAL_JOB_STATE)

    _save_job_state_to_disk(snapshot)
    return snapshot


def _get_global_job_snapshot() -> dict[str, Any]:
    with _GLOBAL_JOB_LOCK:
        snap = dict(_GLOBAL_JOB_STATE)
    if _DEFAULT_ACTIVE_STATUS_PATH.is_file() and "PYTEST_CURRENT_TEST" not in os.environ:
        try:
            disk_data = json.loads(_DEFAULT_ACTIVE_STATUS_PATH.read_text(encoding="utf-8"))
            if isinstance(disk_data, dict):
                st_val = disk_data.get("status")
                # Auto-heal stale jobs if no update for > 3 minutes (e.g. process was killed)
                if st_val == "RUNNING" and disk_data.get("updated_at"):
                    try:
                        last_t = datetime.fromisoformat(disk_data["updated_at"])
                        if (datetime.now(timezone.utc) - last_t).total_seconds() > 180:
                            disk_data["status"] = "CANCELLED"
                            disk_data["error"] = "Previous run was interrupted or timed out."
                    except Exception:
                        pass
                if disk_data.get("status") in ("RUNNING", "COMPLETED", "FAILED", "CANCELLED"):
                    for k, v in disk_data.items():
                        if k in snap:
                            snap[k] = v
        except Exception:
            pass
    return snap


def _request_cancel_active_job() -> None:
    with _GLOBAL_JOB_LOCK:
        _GLOBAL_JOB_STATE["cancel_requested"] = True


def _init_session_state() -> None:
    repo = EvidenceRepository(db_path=_DEFAULT_DB_PATH, artifact_dir=_DEFAULT_ARTIFACT_DIR)
    st.session_state["evidence_repo"] = repo

    if "vendors_by_id" not in st.session_state:
        vendors = MeridianCSVVendorLoader().load(_DEFAULT_VENDORS_CSV)
        st.session_state["vendors_by_id"] = {v.vendor_id: v for v in vendors}

    if "criticality_engine" not in st.session_state:
        st.session_state["criticality_engine"] = MeridianCriticalityEngine()

    if "criticality_by_id" not in st.session_state:
        engine: MeridianCriticalityEngine = st.session_state["criticality_engine"]
        st.session_state["criticality_by_id"] = {
            vid: engine.evaluate(v) for vid, v in st.session_state["vendors_by_id"].items()
        }

    if "human_crit_overrides" not in st.session_state:
        st.session_state["human_crit_overrides"] = {}

    if "reports_by_vendor" not in st.session_state:
        # Load genuine live OSINT assessment reports, preserving sample reports for vendors without live runs
        all_persisted = repo.load_latest_reports_by_vendor(only_live=False)
        live_persisted = repo.load_latest_reports_by_vendor(only_live=True)
        merged = dict(all_persisted)
        merged.update(live_persisted)
        st.session_state["reports_by_vendor"] = merged
    elif "PYTEST_CURRENT_TEST" not in os.environ:
        # Sync any completed background job or newly persisted live report after browser refresh
        snap = _get_global_job_snapshot()
        latest_rpt = snap.get("latest_report")
        latest_v = snap.get("latest_vendor")
        if isinstance(latest_rpt, VendorEndToEndAssessmentReport):
            st.session_state["reports_by_vendor"][latest_rpt.vendor_id] = latest_rpt
            st.session_state["e2e_report"] = latest_rpt
        if isinstance(latest_v, MeridianVendor):
            st.session_state["vendors_by_id"][latest_v.vendor_id] = latest_v
            engine = st.session_state["criticality_engine"]
            st.session_state["criticality_by_id"][latest_v.vendor_id] = engine.evaluate(
                latest_v,
                human_review=st.session_state["human_crit_overrides"].get(latest_v.vendor_id),
            )

    if "selected_vendor_id" not in st.session_state:
        vids = sorted(st.session_state["vendors_by_id"].keys())
        st.session_state["selected_vendor_id"] = "V-002" if "V-002" in vids else (vids[0] if vids else "V-002")

    if "nav_section" not in st.session_state:
        st.session_state["nav_section"] = "Portfolio"

    if "portfolio_search" not in st.session_state:
        st.session_state["portfolio_search"] = ""
    if "portfolio_tier" not in st.session_state:
        st.session_state["portfolio_tier"] = "All Tiers"
    if "portfolio_status" not in st.session_state:
        st.session_state["portfolio_status"] = "All Statuses"
    if "portfolio_sort" not in st.session_state:
        st.session_state["portfolio_sort"] = "Criticality (High → Low)"


def _navigate_to(page: str, vendor_id: Optional[str] = None) -> None:
    if vendor_id and vendor_id in st.session_state.get("vendors_by_id", {}):
        st.session_state["selected_vendor_id"] = vendor_id
    if page in _NAV_PAGES:
        st.session_state["nav_section"] = page


def _get_vendor_assessment_status(
    vid: str,
    crit: CriticalityResult,
    report: Optional[VendorEndToEndAssessmentReport],
) -> tuple[str, str, bool]:
    """Return `(status_label, badge_variant, is_completed_real_assessment)`."""
    snap = _get_global_job_snapshot()
    if snap.get("vendor_id") == vid:
        job_st = snap.get("status")
        if job_st == "FAILED":
            return "Live Run Failed", "critical", False
        if job_st == "RUNNING":
            return "Running (Live OSINT)", "purple", False

    if report is None:
        if crit.requires_review:
            return "Awaiting Tier Review", "high", False
        return "Not Started", "neutral", False

    if report.is_synthetic_evidence_run:
        return "Sample Walkthrough", "purple", False

    verdict_val = report.ai_use_verdict.verdict.value
    if verdict_val == "INCOMPLETE_ASSESSMENT" or not report.ai_use_verdict.depth_completed:
        return "Incomplete (Blocked)", "high", False

    if report.human_review_required:
        return "Awaiting Analyst Review", "high", True

    return "Completed", "success", True


def _run_pipeline_core(
    vendor: MeridianVendor,
    *,
    mode: str,
    repo: EvidenceRepository,
    readiness: SystemReadinessReport,
    crit_engine: MeridianCriticalityEngine,
    enable_local_llm: bool = False,
    enable_triage_llm: Optional[bool] = None,
    enable_rag_embeddings: Optional[bool] = None,
    enable_classification_llm: Optional[bool] = None,
    enable_summary_llm: Optional[bool] = None,
    human_crit_override: Optional[HumanReviewRecord] = None,
    analyst_gate_overrides: Optional[dict[str, tuple[str, str]]] = None,
    human_validations: Optional[list[HumanValidationRecord]] = None,
    progress_callback: Optional[Any] = None,
) -> VendorEndToEndAssessmentReport:
    all_sources = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
    if mode == "SAMPLE":
        stub_search, stub_fetch = build_synthetic_demo_adapters_for_vendor(vendor)
        pipeline = MeridianEndToEndPipeline(
            criticality_engine=crit_engine,
            search_backend=stub_search,
            fetch_adapter=stub_fetch,
            evidence_repository=repo,
            enable_local_llm_from_env=enable_local_llm,
            enable_triage_llm=enable_triage_llm,
            enable_rag_embeddings=enable_rag_embeddings,
            enable_classification_llm=enable_classification_llm,
            enable_summary_llm=enable_summary_llm,
            readiness=readiness,
        )
        return pipeline.run_vendor_assessment(
            vendor,
            human_criticality_override=human_crit_override,
            execute_searches=True,
            execute_extraction=True,
            mark_extracted_sources_reviewed=True,
            reviewed_no_evidence_sources=all_sources,
            analyst_gate_overrides=analyst_gate_overrides,
            human_validations=human_validations,
            persist_to_repository=True,
            progress_callback=progress_callback,
        )

    live_search = build_search_backend_from_env()
    live_fetch = build_fetch_adapter_from_env(allow_live_direct=True)
    pipeline = MeridianEndToEndPipeline(
        criticality_engine=crit_engine,
        search_backend=live_search,
        fetch_adapter=live_fetch,
        evidence_repository=repo,
        enable_local_llm_from_env=enable_local_llm,
        enable_triage_llm=enable_triage_llm,
        enable_rag_embeddings=enable_rag_embeddings,
        enable_classification_llm=enable_classification_llm,
        enable_summary_llm=enable_summary_llm,
        readiness=readiness,
    )
    return pipeline.run_vendor_assessment(
        vendor,
        human_criticality_override=human_crit_override,
        execute_searches=True,
        execute_extraction=True,
        mark_extracted_sources_reviewed=False,
        analyst_gate_overrides=analyst_gate_overrides,
        human_validations=human_validations,
        persist_to_repository=True,
        progress_callback=progress_callback,
    )


def _execute_vendor_assessment(
    vendor: MeridianVendor,
    *,
    mode: str,
    enable_local_llm: bool = False,
    enable_triage_llm: Optional[bool] = None,
    enable_rag_embeddings: Optional[bool] = None,
    enable_classification_llm: Optional[bool] = None,
    enable_summary_llm: Optional[bool] = None,
    human_crit_override: Optional[HumanReviewRecord] = None,
    analyst_gate_overrides: Optional[dict[str, tuple[str, str]]] = None,
    human_validations: Optional[list[HumanValidationRecord]] = None,
    progress_callback: Optional[Any] = None,
) -> VendorEndToEndAssessmentReport:
    """Execute the 9-level assessment pipeline synchronously and update session state."""
    repo: EvidenceRepository = st.session_state["evidence_repo"]
    readiness: SystemReadinessReport = st.session_state["system_readiness"]
    crit_engine: MeridianCriticalityEngine = st.session_state["criticality_engine"]

    report = _run_pipeline_core(
        vendor,
        mode=mode,
        repo=repo,
        readiness=readiness,
        crit_engine=crit_engine,
        enable_local_llm=enable_local_llm,
        enable_triage_llm=enable_triage_llm,
        enable_rag_embeddings=enable_rag_embeddings,
        enable_classification_llm=enable_classification_llm,
        enable_summary_llm=enable_summary_llm,
        human_crit_override=human_crit_override,
        analyst_gate_overrides=analyst_gate_overrides,
        human_validations=human_validations,
        progress_callback=progress_callback,
    )

    st.session_state["vendors_by_id"][vendor.vendor_id] = vendor
    st.session_state["criticality_by_id"][vendor.vendor_id] = crit_engine.evaluate(
        vendor, human_review=human_crit_override
    )
    st.session_state["reports_by_vendor"][vendor.vendor_id] = report
    st.session_state["e2e_report"] = report
    st.session_state["selected_vendor_id"] = vendor.vendor_id
    return report


def _start_background_vendor_assessment(
    vendor: MeridianVendor,
    *,
    mode: str,
    enable_local_llm: bool = False,
    enable_triage_llm: Optional[bool] = None,
    enable_rag_embeddings: Optional[bool] = None,
    enable_classification_llm: Optional[bool] = None,
    enable_summary_llm: Optional[bool] = None,
    human_crit_override: Optional[HumanReviewRecord] = None,
) -> None:
    """Launch a persistent background assessment thread whose progress survives browser refreshes."""
    readiness: SystemReadinessReport = st.session_state["system_readiness"]
    crit_engine: MeridianCriticalityEngine = st.session_state["criticality_engine"]

    with _GLOBAL_JOB_LOCK:
        _GLOBAL_JOB_STATE["cancel_requested"] = False

    _update_global_job_progress(
        status="RUNNING",
        vendor_id=vendor.vendor_id,
        vendor_name=vendor.vendor_name,
        mode=mode,
        level_num=1,
        stage_title="Level 1: Criticality Tiering",
        detail=f"Initializing 9-level assessment pipeline for {vendor.vendor_name}...",
        completed=0,
        total=1,
        report_id=None,
        error=None,
        latest_vendor=vendor,
    )

    def _worker() -> None:
        bg_repo = EvidenceRepository(db_path=_DEFAULT_DB_PATH, artifact_dir=_DEFAULT_ARTIFACT_DIR)

        def _bg_on_stage(
            lvl_num: int,
            stage_title: str,
            detail: str,
            completed: int = 0,
            total: int = 1,
        ) -> None:
            with _GLOBAL_JOB_LOCK:
                if _GLOBAL_JOB_STATE.get("cancel_requested"):
                    raise _AssessmentCancelledError("Assessment cancelled by analyst.")
            if _DEFAULT_ACTIVE_STATUS_PATH.is_file():
                try:
                    _disk_check = json.loads(_DEFAULT_ACTIVE_STATUS_PATH.read_text(encoding="utf-8"))
                    if _disk_check.get("cancel_requested"):
                        with _GLOBAL_JOB_LOCK:
                            _GLOBAL_JOB_STATE["cancel_requested"] = True
                        raise _AssessmentCancelledError("Assessment cancelled by analyst.")
                except _AssessmentCancelledError:
                    raise
                except Exception:
                    pass
            _update_global_job_progress(
                status="RUNNING",
                vendor_id=vendor.vendor_id,
                vendor_name=vendor.vendor_name,
                mode=mode,
                level_num=lvl_num,
                stage_title=stage_title,
                detail=detail,
                completed=completed,
                total=total,
            )

        try:
            rpt = _run_pipeline_core(
                vendor,
                mode=mode,
                repo=bg_repo,
                readiness=readiness,
                crit_engine=crit_engine,
                enable_local_llm=enable_local_llm,
                enable_triage_llm=enable_triage_llm,
                enable_rag_embeddings=enable_rag_embeddings,
                enable_classification_llm=enable_classification_llm,
                enable_summary_llm=enable_summary_llm,
                human_crit_override=human_crit_override,
                progress_callback=_bg_on_stage,
            )
            _update_global_job_progress(
                status="COMPLETED",
                vendor_id=vendor.vendor_id,
                vendor_name=vendor.vendor_name,
                mode=mode,
                level_num=9,
                stage_title="Assessment Completed",
                detail=f"Report {rpt.report_id} generated and persisted for {rpt.legal_entity}.",
                completed=1,
                total=1,
                report_id=rpt.report_id,
                error=None,
                latest_report=rpt,
                latest_vendor=vendor,
            )
        except _AssessmentCancelledError as exc:
            _update_global_job_progress(
                status="CANCELLED",
                vendor_id=vendor.vendor_id,
                vendor_name=vendor.vendor_name,
                mode=mode,
                level_num=_GLOBAL_JOB_STATE.get("level_num", 1),
                stage_title="Assessment Cancelled",
                detail=str(exc),
                completed=_GLOBAL_JOB_STATE.get("completed", 0),
                total=_GLOBAL_JOB_STATE.get("total", 1),
                error=str(exc),
            )
        except Exception as exc:
            _update_global_job_progress(
                status="FAILED",
                vendor_id=vendor.vendor_id,
                vendor_name=vendor.vendor_name,
                mode=mode,
                level_num=_GLOBAL_JOB_STATE.get("level_num", 1),
                stage_title="Assessment Failed",
                detail=str(exc),
                completed=_GLOBAL_JOB_STATE.get("completed", 0),
                total=_GLOBAL_JOB_STATE.get("total", 1),
                error=str(exc),
            )

    t = threading.Thread(target=_worker, name=f"UncoverAI_Assess_{vendor.vendor_id}", daemon=True)
    t.start()


# ─── Application Shell (Header + Top Navigation Bar) ──────────────────────────


def _render_app_shell(readiness: SystemReadinessReport) -> str:
    selected_vid = st.session_state.get("selected_vendor_id", "V-002")
    vendors_by_id: dict[str, MeridianVendor] = st.session_state.get("vendors_by_id", {})
    active_vendor = vendors_by_id.get(selected_vid)
    active_vendor_str = (
        f"{active_vendor.vendor_id} · {active_vendor.vendor_name}"
        if active_vendor
        else selected_vid
    )

    ext_dot = "uc-dot-green" if readiness.external_research_ready else "uc-dot-amber"
    ext_txt = "Serper Live Search Ready" if readiness.external_research_ready else "External Search Unconfigured"

    llm_dot = "uc-dot-green" if readiness.local_analysis_ready else "uc-dot-blue"
    llm_txt = "Local AI Models Ready" if readiness.local_analysis_ready else "Deterministic Rule Engine"

    st.markdown(
        f"""
        <div class="uc-shell-header">
            <div class="uc-brand-group">
                <span class="uc-brand-mark">UncoverAI</span>
                <div>
                    <div class="uc-brand-title">Meridian Financial Services — Third-Party AI Risk Governance</div>
                    <div class="uc-brand-subtitle">Vendor Criticality, Public OSINT Reconnaissance &amp; Point-of-Capture Evidentiary Risk Assessment</div>
                </div>
            </div>
            <div class="uc-header-pills">
                <span class="uc-header-pill"><span class="uc-dot {ext_dot}"></span>{html.escape(ext_txt)}</span>
                <span class="uc-header-pill"><span class="uc-dot {llm_dot}"></span>{html.escape(llm_txt)}</span>
                <span class="uc-header-pill">Active Vendor: <strong>{html.escape(active_vendor_str)}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_nav = st.session_state.get("nav_section", "Portfolio")
    if current_nav not in _NAV_PAGES:
        current_nav = "Portfolio"
        st.session_state["nav_section"] = "Portfolio"

    nav_cols = st.columns(len(_NAV_PAGES), gap="small")
    for idx, page_name in enumerate(_NAV_PAGES):
        is_active = page_name == current_nav
        if nav_cols[idx].button(
            page_name,
            key=f"uc_top_nav_{page_name}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["nav_section"] = page_name
            st.rerun()

    return st.session_state["nav_section"]


# ─── Page 1: Portfolio Risk Dashboard ─────────────────────────────────────────


def _render_portfolio_page(readiness: SystemReadinessReport) -> None:
    vendors_by_id: dict[str, MeridianVendor] = st.session_state["vendors_by_id"]
    crit_by_id: dict[str, CriticalityResult] = st.session_state["criticality_by_id"]
    reports_by_id: dict[str, VendorEndToEndAssessmentReport] = st.session_state["reports_by_vendor"]

    st.markdown(
        """
        <div class="uc-page-header">
            <div>
                <h1 class="uc-page-title">Third-Party Vendor Portfolio</h1>
                <p class="uc-page-desc">Monitor inherent vendor criticality (D/P/R/O/V &amp; O1–O5 safeguards), AI-risk assessment progress, and surfaced governance findings across Meridian's vendor inventory.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute the 5 non-redundant portfolio KPIs from real data
    total_vendors = len(vendors_by_id)
    completed_assessments = 0
    critical_vendors = 0
    high_risk_vendors = 0
    awaiting_review = 0

    for vid, v in vendors_by_id.items():
        crit = crit_by_id[vid]
        rpt = reports_by_id.get(vid)
        _, _, is_completed = _get_vendor_assessment_status(vid, crit, rpt)
        if is_completed:
            completed_assessments += 1
        final_tier = str(crit.final_criticality or "Unknown")
        if final_tier == "Critical":
            critical_vendors += 1
        elif final_tier == "High":
            high_risk_vendors += 1
        if crit.requires_review or (rpt is not None and rpt.human_Review_required):
            awaiting_review += 1

    st.markdown(
        f"""
        <div class="uc-kpi-grid">
            <div class="uc-kpi-card uc-kpi-blue">
                <div class="uc-kpi-label">Total Vendors</div>
                <div class="uc-kpi-value">{total_vendors}</div>
                <div class="uc-kpi-sub">Active third-party relationships</div>
            </div>
            <div class="uc-kpi-card uc-kpi-green">
                <div class="uc-kpi-label">Completed AI-Risk Assessments</div>
                <div class="uc-kpi-value">{completed_assessments} / {total_vendors}</div>
                <div class="uc-kpi-sub">Verified live OSINT assessments</div>
            </div>
            <div class="uc-kpi-card uc-kpi-red">
                <div class="uc-kpi-label">Critical Vendors</div>
                <div class="uc-kpi-value">{critical_vendors}</div>
                <div class="uc-kpi-sub">Comprehensive OSINT depth required</div>
            </div>
            <div class="uc-kpi-card uc-kpi-amber">
                <div class="uc-kpi-label">High-Risk Vendors</div>
                <div class="uc-kpi-value">{high_risk_vendors}</div>
                <div class="uc-kpi-sub">Enhanced OSINT depth required</div>
            </div>
            <div class="uc-kpi-card uc-kpi-slate">
                <div class="uc-kpi-label">Assessments Awaiting Review</div>
                <div class="uc-kpi-value">{awaiting_review}</div>
                <div class="uc-kpi-sub">Pending analyst governance sign-off</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter & Sort Bar
    st.markdown(
        """
        <div class="uc-section-header">
            <div>
                <h2 class="uc-section-title">Vendor Risk Inventory</h2>
                <p class="uc-section-subtitle">Filter by criticality tier, AI-risk assessment status, or keyword search. Service descriptions wrap in full without clipping.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2, f_col3, f_col4 = st.columns([2.2, 1.2, 1.4, 1.4], gap="small")
    search_query = f_col1.text_input(
        "Search Vendor ID, Name, Domain, or Service",
        value=st.session_state["portfolio_search"],
        placeholder="e.g., Fiserv, DNA Core Banking, V-002...",
        key="portfolio_search_input",
    )
    st.session_state["portfolio_search"] = search_query

    tier_options = ["All Tiers", "Critical", "High", "Medium", "Low"]
    tier_filter = f_col2.selectbox(
        "Final Criticality Tier",
        options=tier_options,
        index=tier_options.index(st.session_state["portfolio_tier"]) if st.session_state["portfolio_tier"] in tier_options else 0,
        key="portfolio_tier_select",
    )
    st.session_state["portfolio_tier"] = tier_filter

    status_options = [
        "All Statuses",
        "Completed",
        "Awaiting Analyst Review",
        "Incomplete (Blocked)",
        "Sample Walkthrough",
        "Awaiting Tier Review",
        "Not Started",
    ]
    status_filter = f_col3.selectbox(
        "AI-Risk Assessment Status",
        options=status_options,
        index=status_options.index(st.session_state["portfolio_status"]) if st.session_state["portfolio_status"] in status_options else 0,
        key="portfolio_status_select",
    )
    st.session_state["portfolio_status"] = status_filter

    sort_options = [
        "Criticality (High → Low)",
        "Base Score (High → Low)",
        "Vendor ID (A → Z)",
        "Vendor Name (A → Z)",
    ]
    sort_by = f_col4.selectbox(
        "Sort Inventory By",
        options=sort_options,
        index=sort_options.index(st.session_state["portfolio_sort"]) if st.session_state["portfolio_sort"] in sort_options else 0,
        key="portfolio_sort_select",
    )
    st.session_state["portfolio_sort"] = sort_by

    # Apply filters
    filtered_rows: list[tuple[MeridianVendor, CriticalityResult, Optional[VendorEndToEndAssessmentReport], str, str]] = []
    q_low = search_query.strip().lower()

    for vid, v in vendors_by_id.items():
        crit = crit_by_id[vid]
        rpt = reports_by_id.get(vid)
        status_lbl, status_var, _ = _get_vendor_assessment_status(vid, crit, rpt)

        if tier_filter != "All Tiers" and str(crit.final_criticality or "") != tier_filter:
            continue
        if status_filter != "All Statuses" and status_lbl != status_filter:
            continue
        if q_low:
            haystack = " ".join([
                v.vendor_id,
                v.vendor_name,
                v.domain or "",
                v.category or "",
                v.service_product_provided or "",
                v.business_process_supported or "",
            ]).lower()
            if q_low not in haystack:
                continue

        filtered_rows.append((v, crit, rpt, status_lbl, status_var))

    # Apply sorting
    if sort_by == "Criticality (High → Low)":
        filtered_rows.sort(
            key=lambda item: (
                _TIER_SORT_ORDER.get(str(item[1].final_criticality or "Unknown"), 9),
                -(item[1].base_score or 0.0),
                item[0].vendor_id,
            )
        )
    elif sort_by == "Base Score (High → Low)":
        filtered_rows.sort(key=lambda item: (-(item[1].base_score or 0.0), item[0].vendor_id))
    elif sort_by == "Vendor Name (A → Z)":
        filtered_rows.sort(key=lambda item: (item[0].vendor_name.lower(), item[0].vendor_id))
    else:
        filtered_rows.sort(key=lambda item: item[0].vendor_id)

    if not filtered_rows:
        st.markdown(
            """
            <div class="uc-empty-state">
                <div class="uc-empty-title">No Vendors Match Current Filters</div>
                <div class="uc-empty-desc">No vendors in Meridian's portfolio matched your search or filter criteria. Reset the filters below to view the full vendor inventory.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear All Filters", key="btn_clear_portfolio_filters", type="primary"):
            st.session_state["portfolio_search"] = ""
            st.session_state["portfolio_tier"] = "All Tiers"
            st.session_state["portfolio_status"] = "All Statuses"
            st.rerun()
    else:
        selected_vid = st.session_state.get("selected_vendor_id", filtered_rows[0][0].vendor_id)
        table_rows_html: list[str] = []
        for v, crit, rpt, status_lbl, status_var in filtered_rows:
            row_cls = "uc-row-selected" if v.vendor_id == selected_vid else ""
            base_score_str = f"{crit.base_score:.2f}" if crit.base_score is not None else "Unknown"
            prov_str = str(crit.provisional_criticality or "Unknown")
            final_str = str(crit.final_criticality or "Unknown")
            prov_note = (
                f'<div style="font-size:0.73rem;color:#64748B;margin-top:2px;">Base: {html.escape(prov_str)}</div>'
                if prov_str != final_str
                else ""
            )
            last_date = _format_short_date(rpt.generated_at) if rpt else "—"
            verdict_sub = ""
            if rpt:
                verdict_sub = (
                    f'<div style="font-size:0.74rem;color:#475569;margin-top:3px;">'
                    f'Verdict: <strong>{html.escape(format_plain_status(rpt.ai_use_verdict.verdict))}</strong> '
                    f'({len(rpt.risk_findings)} finding(s))</div>'
                )

            table_rows_html.append(
                f'<tr class="{row_cls}">'
                f'<td style="width:85px;"><span class="uc-mono">{html.escape(v.vendor_id)}</span></td>'
                f'<td style="width:190px;"><strong>{html.escape(v.vendor_name)}</strong>'
                f'<div style="font-size:0.76rem;color:#64748B;">{html.escape(v.domain or "No domain")} · {html.escape(v.category or "Vendor")}</div></td>'
                f'<td style="min-width:260px;"><strong>{html.escape(v.service_product_provided or "UNKNOWN")}</strong>'
                f'<div style="font-size:0.77rem;color:#475569;margin-top:2px;">Process: {html.escape(v.business_process_supported or "UNKNOWN")}</div></td>'
                f'<td style="width:130px;">{_tier_badge(final_str)}{prov_note}</td>'
                f'<td style="width:95px;"><span class="uc-mono">{html.escape(base_score_str)}</span></td>'
                f'<td style="width:195px;">{_status_badge(status_lbl, status_var)}{verdict_sub}</td>'
                f'<td style="width:140px;font-size:0.78rem;color:#475569;">{html.escape(last_date)}</td>'
                f'</tr>'
            )

        st.markdown(
            f"""
            <div class="uc-table-wrap">
                <table class="uc-table">
                    <thead>
                        <tr>
                            <th>Vendor ID</th>
                            <th>Vendor Name &amp; Domain</th>
                            <th>Purchased Service &amp; Business Process</th>
                            <th>Final Tier</th>
                            <th>Base Score</th>
                            <th>AI-Risk Assessment Status</th>
                            <th>Last Assessed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(table_rows_html)}
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Selected Vendor Detail & Action Bar ──
    all_vids = sorted(vendors_by_id.keys())
    current_vid = st.session_state.get("selected_vendor_id", all_vids[0])
    if current_vid not in vendors_by_id:
        current_vid = all_vids[0]

    sel_col, act_col1, act_col2, act_col3 = st.columns([2.4, 1.2, 1.2, 1.2], gap="small")
    chosen_vid = sel_col.selectbox(
        "Inspect Selected Vendor Profile & Governance Breakdown",
        options=all_vids,
        index=all_vids.index(current_vid),
        format_func=lambda vid: f"{vid} — {vendors_by_id[vid].vendor_name} ({crit_by_id[vid].final_criticality or 'Unknown'})",
        key="portfolio_vendor_selector",
    )
    st.session_state["selected_vendor_id"] = chosen_vid

    sel_vendor = vendors_by_id[chosen_vid]
    sel_crit = crit_by_id[chosen_vid]
    sel_rpt = reports_by_id.get(chosen_vid)

    st.write("")
    if act_col1.button("Open Assessment Workspace", key="btn_port_open_assess", type="primary", use_container_width=True):
        _navigate_to("Assessments", chosen_vid)
        st.rerun()
    if act_col2.button("Inspect Evidence Register", key="btn_port_open_ev", type="secondary", use_container_width=True):
        _navigate_to("Evidence", chosen_vid)
        st.rerun()
    if act_col3.button("View Assessment Dossier", key="btn_port_open_rpt", type="secondary", use_container_width=True):
        _navigate_to("Reports", chosen_vid)
        st.rerun()

    detail_left, detail_right = st.columns([1.25, 1.0], gap="medium")

    with detail_left:
        overrides_trig = [ov.override_id for ov in sel_crit.overrides_evaluated if ov.triggered]
        ov_summary = ", ".join(overrides_trig) if overrides_trig else "None triggered"
        st.markdown(
            f"""
            <div class="uc-section-header" style="margin-top:6px;">
                <div>
                    <h3 class="uc-section-title">{html.escape(sel_vendor.vendor_name)} ({html.escape(sel_vendor.vendor_id)}) — Criticality &amp; Factor Provenance</h3>
                    <p class="uc-section-subtitle">Weighted Formula: 0.30·D + 0.20·P + 0.20·R + 0.20·O + 0.10·V with O1–O5 Override Governance</p>
                </div>
            </div>
            <div class="uc-kv-grid">
                <div class="uc-kv-box">
                    <div class="uc-kv-label">Primary Domain</div>
                    <div class="uc-kv-val">{html.escape(sel_vendor.domain or "UNKNOWN")}</div>
                </div>
                <div class="uc-kv-box">
                    <div class="uc-kv-label">Base Score &amp; Provisional Tier</div>
                    <div class="uc-kv-val">{f"{sel_crit.base_score:.2f}" if sel_crit.base_score is not None else "Unknown"} ({html.escape(str(sel_crit.provisional_criticality or "Unknown"))})</div>
                </div>
                <div class="uc-kv-box">
                    <div class="uc-kv-label">Final Criticality Tier</div>
                    <div class="uc-kv-val">{_tier_badge(str(sel_crit.final_criticality or "Unknown"))}</div>
                </div>
                <div class="uc-kv-box">
                    <div class="uc-kv-label">O1–O5 Overrides Triggered</div>
                    <div class="uc-kv-val">{html.escape(ov_summary)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # D/P/R/O/V Factor Table
        factor_rows_html: list[str] = []
        factor_names = {
            "D": "Data Sensitivity (0.30)",
            "P": "Process Criticality (0.20)",
            "R": "Regulatory Scope (0.20)",
            "O": "Operational Dependency (0.20)",
            "V": "Data Volume (0.10)",
        }
        for f_code, f_obj in (
            ("D", sel_crit.D),
            ("P", sel_crit.P),
            ("R", sel_crit.R),
            ("O", sel_crit.O),
            ("V", sel_crit.V),
        ):
            sc_txt = str(f_obj.score) if f_obj.score is not None else "UNKNOWN"
            wc_txt = f"{f_obj.weighted_contribution:.2f}" if f_obj.weighted_contribution is not None else "—"
            factor_rows_html.append(
                f"<tr>"
                f"<td><strong>{html.escape(f_code)}</strong> — {html.escape(factor_names[f_code])}</td>"
                f"<td><span class='uc-mono'>{html.escape(sc_txt)}</span></td>"
                f"<td><span class='uc-mono'>{html.escape(wc_txt)}</span></td>"
                f"<td>{html.escape(format_plain_status(f_obj.determination_method.value))}</td>"
                f"<td>{html.escape(f_obj.rationale)}</td>"
                f"</tr>"
            )

        st.markdown(
            f"""
            <div class="uc-table-wrap">
                <table class="uc-table">
                    <thead>
                        <tr>
                            <th>Factor</th>
                            <th>Score</th>
                            <th>Weighted</th>
                            <th>Provenance</th>
                            <th>Analytical Rationale</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(factor_rows_html)}
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_right:
        st.markdown(
            """
            <div class="uc-section-header" style="margin-top:6px;">
                <div>
                    <h3 class="uc-section-title">AI-Risk Assessment Status &amp; Key Findings</h3>
                    <p class="uc-section-subtitle">Current OSINT investigation status, G1–G4 gate determination, and surfaced R1–R8 risks</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if sel_rpt is None:
            st.markdown(
                f"""
                <div class="uc-empty-state">
                    <div class="uc-empty-title">No AI-Risk Assessment Executed Yet for {html.escape(sel_vendor.vendor_name)}</div>
                    <div class="uc-empty-desc">
                        Inherent criticality tiering is complete (<strong>{html.escape(str(sel_crit.final_criticality or "Unknown"))}</strong>),
                        but public OSINT reconnaissance and evidentiary AI-risk evaluation have not been run yet.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c_btn1, c_btn2 = st.columns(2, gap="small")
            if c_btn1.button(
                f"Configure Live Assessment ({sel_vendor.vendor_id})",
                key="btn_port_start_live",
                type="primary",
                use_container_width=True,
            ):
                _navigate_to("Assessments", chosen_vid)
                st.rerun()
            if c_btn2.button(
                f"Run Sample Walkthrough ({sel_vendor.vendor_id})",
                key="btn_port_start_sample",
                type="secondary",
                use_container_width=True,
            ):
                _execute_vendor_assessment(sel_vendor, mode="SAMPLE", enable_local_llm=False)
                _navigate_to("Assessments", chosen_vid)
                st.rerun()
        else:
            mode_tag = (
                _status_badge("Sample Walkthrough (Synthetic Fixtures)", "purple")
                if sel_rpt.is_synthetic_evidence_run
                else _status_badge("Live OSINT Assessment", "success")
            )
            st.markdown(
                f"""
                <div class="uc-kv-grid">
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">Assessment Run Mode</div>
                        <div class="uc-kv-val">{mode_tag}</div>
                    </div>
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">AI-Use Verdict</div>
                        <div class="uc-kv-val">{_verdict_badge(sel_rpt.ai_use_verdict.verdict.value)}</div>
                    </div>
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">Captured Evidence</div>
                        <div class="uc-kv-val">{len(sel_rpt.evidence_records)} verbatim excerpt(s)</div>
                    </div>
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">R1–R8 Risk Findings</div>
                        <div class="uc-kv-val">{len(sel_rpt.risk_findings)} surfaced finding(s)</div>
                    </div>
                </div>
                <div class="uc-callout uc-callout-info">
                    <strong>Determination Basis:</strong> {html.escape(sel_rpt.ai_use_verdict.basis_summary)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if sel_rpt.risk_findings:
                rf_html = []
                for rf in sel_rpt.risk_findings[:4]:
                    rf_html.append(
                        f"<tr>"
                        f"<td><strong>{html.escape(rf.risk_rule_id)}</strong>: {html.escape(rf.risk_title)}</td>"
                        f"<td>{_severity_badge(rf.severity.value)}</td>"
                        f"<td>{html.escape(rf.finding_description)}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"""
                    <div class="uc-table-wrap">
                        <table class="uc-table">
                            <thead>
                                <tr><th>Risk Rule</th><th>Severity</th><th>Surfaced Exposure</th></tr>
                            </thead>
                            <tbody>{"".join(rf_html)}</tbody>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # O1-O5 Override summary table
        ov_rows_html = []
        for ov in sel_crit.overrides_evaluated:
            trig_badge = _status_badge("Triggered", "critical") if ov.triggered else _status_badge("Not Triggered", "neutral")
            ov_rows_html.append(
                f"<tr>"
                f"<td><span class='uc-mono'>{html.escape(ov.override_id)}</span></td>"
                f"<td>{trig_badge}</td>"
                f"<td>{html.escape(str(ov.resulting_floor or '—'))}</td>"
                f"<td>{html.escape(ov.rationale)}</td>"
                f"</tr>"
            )
        with st.expander("View O1–O5 Governance Override Evaluation", expanded=False):
            st.markdown(
                f"""
                <div class="uc-table-wrap">
                    <table class="uc-table">
                        <thead>
                            <tr><th>Rule</th><th>Status</th><th>Resulting Tier</th><th>Evaluation Rationale</th></tr>
                        </thead>
                        <tbody>{"".join(ov_rows_html)}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── Page 2: Assessment Workspace (Setup -> Execution -> Results) ─────────────


def _render_assessments_page(readiness: SystemReadinessReport) -> None:
    vendors_by_id: dict[str, MeridianVendor] = st.session_state["vendors_by_id"]
    reports_by_id: dict[str, VendorEndToEndAssessmentReport] = st.session_state["reports_by_vendor"]

    st.markdown(
        """
        <div class="uc-page-header">
            <div>
                <h1 class="uc-page-title">Vendor AI-Risk Assessment Workspace</h1>
                <p class="uc-page-desc">Configure vendor scope and criticality inputs, execute bounded OSINT reconnaissance and evidence extraction, and validate G1–G4 gates and R1–R8 risk findings.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── AREA A: ASSESSMENT SETUP ──
    st.markdown(
        """
        <div class="uc-section-header">
            <div>
                <h2 class="uc-section-title">A. Assessment Setup &amp; Scope Configuration</h2>
                <p class="uc-section-subtitle">Select an existing portfolio vendor or enter a new third-party vendor. Unknown inputs are preserved explicitly as UNKNOWN for analyst review.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_col1, setup_col2 = st.columns([1.3, 1.7], gap="medium")
    with setup_col1:
        source_mode = st.radio(
            "Vendor Input Source",
            options=["Portfolio Vendor", "New / Custom Vendor"],
            horizontal=True,
            key="assess_vendor_source_mode",
        )
        all_vids = sorted(vendors_by_id.keys())
        cur_vid = st.session_state.get("selected_vendor_id", all_vids[0])
        if cur_vid not in vendors_by_id:
            cur_vid = all_vids[0]

        if source_mode == "Portfolio Vendor":
            chosen_vid = st.selectbox(
                "Select Portfolio Vendor",
                options=all_vids,
                index=all_vids.index(cur_vid),
                format_func=lambda vid: f"{vid} — {vendors_by_id[vid].vendor_name}",
                key="assess_portfolio_vendor_select",
            )
            st.session_state["selected_vendor_id"] = chosen_vid
            base_v = vendors_by_id[chosen_vid]
        else:
            base_v = MeridianVendor(
                vendor_id="V-007",
                vendor_name="",
                domain="",
                category="Enterprise SaaS",
                service_product_provided="UNKNOWN",
                business_process_supported="UNKNOWN",
                data_classification_accessed="UNKNOWN",
                operational_dependency="UNKNOWN",
                data_volume_annual="UNKNOWN",
                is_sole_source=False,
            )

        exec_mode_choice = st.selectbox(
            "Assessment Execution Mode",
            options=[
                "Live OSINT Assessment (Serper Search + Direct HTTP / ScraperAPI)",
                "Sample Assessment (Pre-Collected Demonstration Fixtures)",
            ],
            index=0 if readiness.external_research_ready else 1,
            key="assess_execution_mode_select",
        )
        is_sample_mode = exec_mode_choice.startswith("Sample")

        st.markdown(
            "<div style='font-size:0.78rem;font-weight:600;color:#334155;margin:6px 0 4px 0;'>"
            "Stage-by-Stage Local AI Model Routing Controls"
            "</div>",
            unsafe_allow_html=True,
        )
        ai_c1, ai_c2 = st.columns(2, gap="small")
        use_ai_l3b = ai_c1.checkbox(
            "Level 3B: URL Triage (llama3.2:latest)",
            value=False,
            key="assess_ai_l3b",
            help="Use local Ollama llama3.2:latest to score and prioritize candidate search URLs in Level 3B.",
        )
        use_ai_l5 = ai_c2.checkbox(
            "Level 5: Semantic RAG (all-MiniLM-L6-v2)",
            value=True,
            key="assess_ai_l5",
            help="Use local sentence-transformers/all-MiniLM-L6-v2 embeddings to retrieve top risk-concept passages in Level 5.",
        )
        ai_c3, ai_c4 = st.columns(2, gap="small")
        use_ai_l6 = ai_c3.checkbox(
            "Level 6: Signal Classification (mistral:latest)",
            value=False,
            key="assess_ai_l6",
            help="Use local Ollama mistral:latest to classify candidate AI/automation excerpts in Level 6.",
        )
        use_ai_l8 = ai_c4.checkbox(
            "Level 8: Risk Summarization (mistral:latest)",
            value=False,
            key="assess_ai_l8",
            help="Use local Ollama mistral:latest to draft evidence-grounded R1–R8 risk explanations in Level 8.",
        )
        use_local_ai = bool(use_ai_l3b or use_ai_l6 or use_ai_l8)

    with setup_col2:
        c1, c2, c3 = st.columns([1.0, 1.4, 1.4], gap="small")
        in_vid = c1.text_input("Vendor ID", value=base_v.vendor_id, key=f"in_vid_{source_mode}_{base_v.vendor_id}")
        in_name = c2.text_input("Vendor Legal / Display Name", value=base_v.vendor_name, key=f"in_name_{source_mode}_{base_v.vendor_id}")
        in_dom = c3.text_input("Primary Domain", value=base_v.domain or "", placeholder="e.g. fiserv.com", key=f"in_dom_{source_mode}_{base_v.vendor_id}")

        c4, c5 = st.columns(2, gap="small")
        in_srv = c4.text_input(
            "Service Meridian Purchases",
            value=base_v.service_product_provided or "UNKNOWN",
            key=f"in_srv_{source_mode}_{base_v.vendor_id}",
        )
        in_bp = c5.text_input(
            "Business Process Supported",
            value=base_v.business_process_supported or "UNKNOWN",
            key=f"in_bp_{source_mode}_{base_v.vendor_id}",
        )

        c6, c7, c8, c9 = st.columns([1.2, 1.2, 1.2, 0.9], gap="small")
        in_dc = c6.text_input(
            "Data Classification (D)",
            value=base_v.data_classification_accessed or "UNKNOWN",
            key=f"in_dc_{source_mode}_{base_v.vendor_id}",
        )
        in_od = c7.text_input(
            "Operational Dependency (O)",
            value=base_v.operational_dependency or "UNKNOWN",
            key=f"in_od_{source_mode}_{base_v.vendor_id}",
        )
        in_dv = c8.text_input(
            "Annual Data Volume (V)",
            value=base_v.data_volume_annual or "UNKNOWN",
            key=f"in_dv_{source_mode}_{base_v.vendor_id}",
        )
        in_sole = c9.checkbox(
            "Sole Source",
            value=bool(getattr(base_v, "is_sole_source", False)),
            key=f"in_sole_{source_mode}_{base_v.vendor_id}",
            help="Trigger Override O2 (Critical floor) when Vendor is a Sole Source AND Operational Dependency (O) = 3.",
        )

    validated_vendor, val_errors, val_warnings = validate_vendor_assessment_input(
        vendor_id=in_vid,
        vendor_name=in_name,
        domain=in_dom,
        category=base_v.category,
        vendor_description=base_v.vendor_description,
        service_product_provided=in_srv,
        business_process_supported=in_bp,
        data_classification_accessed=in_dc,
        operational_dependency=in_od,
        data_volume_annual=in_dv,
        is_sole_source=in_sole,
    )

    if val_errors:
        for err in val_errors:
            st.markdown(f'<div class="uc-callout uc-callout-danger"><strong>Validation Error:</strong> {html.escape(err)}</div>', unsafe_allow_html=True)
    elif val_warnings:
        st.markdown(
            f'<div class="uc-callout uc-callout-warn"><strong>Input Governance Notice:</strong> {html.escape(" | ".join(val_warnings))}</div>',
            unsafe_allow_html=True,
        )

    # ── AREA B: EXECUTION ──
    st.markdown(
        """
        <div class="uc-section-header">
            <div>
                <h2 class="uc-section-title">B. Assessment Execution &amp; Pipeline Controls</h2>
                <p class="uc-section-subtitle">Execute all nine assessment stages in sequence with dual-bar stage/item telemetry and persistent progress across browser refreshes.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_sample_mode and not readiness.external_research_ready:
        st.markdown(
            f"""
            <div class="uc-callout uc-callout-warn">
                <strong>Live Search Credentials Not Configured:</strong> {html.escape(readiness.search_status_message)}
                Running a Live OSINT Assessment without search credentials will execute Levels 1–2 and mark Level 3 as
                <code>BLOCKED BY CONFIG</code> and the AI-use determination as <code>INCOMPLETE_ASSESSMENT</code>.
                To evaluate a full end-to-end walkthrough offline, select <strong>Sample Assessment</strong> above.
            </div>
            """,
            unsafe_allow_html=True,
        )

    job_snap = _get_global_job_snapshot()
    is_job_running = (
        job_snap.get("status") == "RUNNING" and "PYTEST_CURRENT_TEST" not in os.environ
    )

    exec_col1, exec_col2, exec_col3 = st.columns([1.5, 1.5, 2.0], gap="small")
    primary_label = (
        f"Run Sample Assessment ({in_vid or 'Vendor'})"
        if is_sample_mode
        else f"Run Live OSINT Assessment ({in_vid or 'Vendor'})"
    )
    run_clicked = exec_col1.button(
        primary_label,
        key="btn_execute_assessment_primary",
        type="primary",
        disabled=(validated_vendor is None or is_job_running),
        use_container_width=True,
    )

    alt_sample_clicked = False
    if not is_sample_mode:
        alt_sample_clicked = exec_col2.button(
            "Load Sample Walkthrough Instead",
            key="btn_execute_assessment_sample_alt",
            type="secondary",
            disabled=(validated_vendor is None or is_job_running),
            use_container_width=True,
        )

    cancel_clicked = False
    if is_job_running:
        cancel_clicked = exec_col3.button(
            f"Stop Active Run ({job_snap.get('vendor_id')})",
            key="btn_cancel_active_assessment",
            type="secondary",
            use_container_width=True,
        )
        if cancel_clicked:
            _request_cancel_active_job()

    overall_progress_container = st.empty()
    level_progress_container = st.empty()
    status_container = st.empty()

    if (run_clicked or alt_sample_clicked) and validated_vendor is not None:
        target_mode = "SAMPLE" if (is_sample_mode or alt_sample_clicked) else "LIVE"

        prog_overall = overall_progress_container.progress(
            0,
            text=f"Overall Pipeline (Level 1 of 9 — 0%): Initializing 9-level assessment pipeline for {validated_vendor.vendor_name}...",
        )
        prog_level = level_progress_container.progress(
            0,
            text="Current Level Progress: 0 / 1 completed (1 left — 0%) — Starting Level 1...",
        )

        with _GLOBAL_JOB_LOCK:
            _GLOBAL_JOB_STATE["cancel_requested"] = False

        def _on_stage(
            lvl_num: int,
            stage_title: str,
            detail: str,
            completed: int = 0,
            total: int = 1,
        ) -> None:
            with _GLOBAL_JOB_LOCK:
                if _GLOBAL_JOB_STATE.get("cancel_requested"):
                    raise _AssessmentCancelledError("Assessment cancelled by analyst.")
            safe_tot = max(1, int(total))
            safe_comp = max(0, min(int(completed), safe_tot))
            rem = max(0, safe_tot - safe_comp)
            lvl_pct = int(round((safe_comp / float(safe_tot)) * 100))
            ov_pct = min(99, max(1, int(round(((lvl_num - 1 + (safe_comp / float(safe_tot))) / 9.0) * 100))))
            prog_overall.progress(
                ov_pct,
                text=f"Overall Pipeline (Level {lvl_num} of 9 — {ov_pct}%): {stage_title}",
            )
            prog_level.progress(
                lvl_pct,
                text=f"Current Level Progress: {safe_comp} / {safe_tot} completed ({rem} left — {lvl_pct}%) — {detail}",
            )
            _update_global_job_progress(
                status="RUNNING",
                vendor_id=validated_vendor.vendor_id,
                vendor_name=validated_vendor.vendor_name,
                mode=target_mode,
                level_num=lvl_num,
                stage_title=stage_title,
                detail=detail,
                completed=safe_comp,
                total=safe_tot,
            )

        _update_global_job_progress(
            status="RUNNING",
            vendor_id=validated_vendor.vendor_id,
            vendor_name=validated_vendor.vendor_name,
            mode=target_mode,
            level_num=1,
            stage_title="Level 1: Criticality Tiering",
            detail=f"Initializing 9-level assessment pipeline for {validated_vendor.vendor_name}...",
            completed=0,
            total=1,
            latest_vendor=validated_vendor,
        )

        try:
            report = _execute_vendor_assessment(
                validated_vendor,
                mode=target_mode,
                enable_local_llm=use_local_ai,
                enable_triage_llm=use_ai_l3b,
                enable_rag_embeddings=use_ai_l5,
                enable_classification_llm=use_ai_l6,
                enable_summary_llm=use_ai_l8,
                human_crit_override=st.session_state["human_crit_overrides"].get(validated_vendor.vendor_id),
                progress_callback=_on_stage,
            )
            prog_overall.progress(100, text=f"Overall Pipeline (9 of 9 Levels — 100%): Assessment {report.report_id} completed.")
            prog_level.progress(100, text="Current Level Progress: 1 / 1 completed (0 left — 100%) — All stages finished.")
            status_container.markdown(
                f'<div class="uc-callout uc-callout-success"><strong>Assessment Execution Finished:</strong> '
                f'Report <code>{html.escape(report.report_id)}</code> generated for <strong>{html.escape(report.legal_entity)}</strong>.</div>',
                unsafe_allow_html=True,
            )
            _update_global_job_progress(
                status="COMPLETED",
                vendor_id=validated_vendor.vendor_id,
                vendor_name=validated_vendor.vendor_name,
                mode=target_mode,
                level_num=9,
                stage_title="Assessment Completed",
                detail=f"Report {report.report_id} generated and persisted for {report.legal_entity}.",
                completed=1,
                total=1,
                report_id=report.report_id,
                latest_report=report,
                latest_vendor=validated_vendor,
            )
        except _AssessmentCancelledError as exc:
            _update_global_job_progress(
                status="CANCELLED",
                vendor_id=validated_vendor.vendor_id,
                vendor_name=validated_vendor.vendor_name,
                mode=target_mode,
                level_num=1,
                stage_title="Assessment Cancelled",
                detail=str(exc),
                error=str(exc),
            )
            status_container.markdown(
                f'<div class="uc-callout uc-callout-warn"><strong>Assessment Cancelled:</strong> {html.escape(str(exc))}</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            _update_global_job_progress(
                status="FAILED",
                vendor_id=validated_vendor.vendor_id,
                vendor_name=validated_vendor.vendor_name,
                mode=target_mode,
                level_num=1,
                stage_title="Assessment Failed",
                detail=str(exc),
                error=str(exc),
            )
            status_container.markdown(
                f'<div class="uc-callout uc-callout-danger"><strong>Assessment Failed:</strong> {html.escape(str(exc))}</div>',
                unsafe_allow_html=True,
            )
    elif "PYTEST_CURRENT_TEST" not in os.environ:
        job_snap = _get_global_job_snapshot()
        j_status = job_snap.get("status")
        j_vid = job_snap.get("vendor_id")
        j_vname = job_snap.get("vendor_name") or j_vid or "Vendor"
        j_lvl = int(job_snap.get("level_num") or 1)
        j_title = str(job_snap.get("stage_title") or f"Level {j_lvl}")
        j_detail = str(job_snap.get("detail") or "")
        j_comp = int(job_snap.get("completed") or 0)
        j_tot = max(1, int(job_snap.get("total") or 1))
        j_rem = max(0, int(job_snap.get("remaining") or (j_tot - j_comp)))
        j_lvl_pct = max(0, min(100, int(job_snap.get("level_pct") or 0)))
        j_ov_pct = max(0, min(100, int(job_snap.get("overall_pct") or 0)))

        if j_status == "RUNNING":
            overall_progress_container.progress(
                j_ov_pct,
                text=f"Overall Pipeline [{j_vid} · {j_vname}] (Level {j_lvl} of 9 — {j_ov_pct}%): {j_title}",
            )
            level_progress_container.progress(
                j_lvl_pct,
                text=f"Current Level Progress: {j_comp} / {j_tot} completed ({j_rem} left — {j_lvl_pct}%) — {j_detail}",
            )
            status_container.markdown(
                f'<div class="uc-callout uc-callout-info"><strong>Live Assessment Running in Background ({html.escape(str(j_vid))}):</strong> '
                f'Progress persists automatically across browser refreshes and tab switches.</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.75)
            st.rerun()
        elif j_status == "COMPLETED" and j_vid == (validated_vendor.vendor_id if validated_vendor else cur_vid):
            overall_progress_container.progress(
                100,
                text=f"Overall Pipeline [{j_vid} · {j_vname}] (9 of 9 Levels — 100%): Assessment {job_snap.get('report_id') or ''} completed.",
            )
            level_progress_container.progress(
                100,
                text=f"Current Level Progress: {j_tot} / {j_tot} completed (0 left — 100%) — {j_detail}",
            )
            status_container.markdown(
                f'<div class="uc-callout uc-callout-success"><strong>Assessment Execution Finished:</strong> '
                f'{html.escape(j_detail)}</div>',
                unsafe_allow_html=True,
            )
        elif j_status in ("CANCELLED", "FAILED") and j_vid == (validated_vendor.vendor_id if validated_vendor else cur_vid):
            status_container.markdown(
                f'<div class="uc-callout uc-callout-warn"><strong>Assessment {html.escape(str(j_status))}:</strong> '
                f'{html.escape(str(job_snap.get("error") or j_detail))}</div>',
                unsafe_allow_html=True,
            )

    # ── AREA C: RESULTS ──
    active_vid = validated_vendor.vendor_id if validated_vendor else st.session_state.get("selected_vendor_id", "V-002")
    active_report = reports_by_id.get(active_vid)

    st.markdown(
        """
        <div class="uc-section-header">
            <div>
                <h2 class="uc-section-title">C. Assessment Results &amp; Analyst Governance</h2>
                <p class="uc-section-subtitle">Executive determination, G1–G4 evidentiary gates, E1–E14 evidence register, R1–R8 risk findings, and analyst sign-off.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if active_report is None:
        st.markdown(
            f"""
            <div class="uc-empty-state">
                <div class="uc-empty-title">No Assessment Results Available for {html.escape(active_vid)}</div>
                <div class="uc-empty-desc">Click <strong>{html.escape(primary_label)}</strong> above to execute the assessment workflow and populate results, evidence, and risk findings.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Clear separation for failed live runs vs historical/sample dossiers
    if job_snap.get("vendor_id") == active_vid and job_snap.get("status") == "FAILED":
        st.markdown(
            f"""
            <div class="uc-callout uc-callout-danger" style="margin-bottom:12px;">
                <strong>⚠️ Live OSINT Assessment Failed for {html.escape(active_vid)}:</strong><br>
                The run encountered an execution error:<br>
                <code>{html.escape(str(job_snap.get("error") or "Unknown execution failure"))}</code><br>
                <em>No new live assessment dossier was generated. Any dossier displayed below is historical or demonstration fixture data.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Mode banner (Sample vs Live)
    if active_report.is_synthetic_evidence_run:
        if job_snap.get("vendor_id") == active_vid and job_snap.get("status") == "FAILED":
            st.markdown(
                """
                <div class="uc-callout uc-callout-warn">
                    <strong>Notice — Displaying Baseline Demonstration Fixture:</strong>
                    The assessment results shown below are from a <em>pre-computed synthetic demonstration fixture</em>,
                    <strong>NOT</strong> from the live assessment that failed above.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="uc-callout uc-callout-warn">
                    <strong>Sample Assessment Mode (Pre-Collected Demonstration Fixtures):</strong>
                    This report was generated using deterministic sample fixtures to demonstrate the nine-level workflow.
                    It is labeled as synthetic sample data and is never counted as a completed live vendor assessment.
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif job_snap.get("vendor_id") == active_vid and job_snap.get("status") == "FAILED":
        st.markdown(
            f"""
            <div class="uc-callout uc-callout-info">
                <strong>Notice — Displaying Prior Historical Assessment ({html.escape(active_report.report_id)}):</strong>
                The assessment results shown below are from a <em>prior completed run</em> ({html.escape(active_report.generated_at)}).
                They do <strong>NOT</strong> reflect the failed run above.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif active_report.ai_use_verdict.verdict.value == "INCOMPLETE_ASSESSMENT":
        st.markdown(
            f"""
            <div class="uc-callout uc-callout-danger">
                <strong>Incomplete Live Assessment:</strong>
                {html.escape(active_report.ai_use_verdict.basis_summary)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    crit_sum = active_report.criticality_summary or {}
    ds = active_report.deterministic_risk_score
    pos_gates = sum(1 for g in active_report.gates if g.status == "QUALIFYING_POSITIVE_EVIDENCE")

    st.markdown(
        f"""
        <div class="uc-kv-grid">
            <div class="uc-kv-box">
                <div class="uc-kv-label">Effective Criticality Tier</div>
                <div class="uc-kv-val">{_tier_badge(str(crit_sum.get("final_criticality", "Unknown")))} (Base: {html.escape(str(crit_sum.get("base_score", "N/A")))})</div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">AI-Use Verdict &amp; Scope</div>
                <div class="uc-kv-val">{_verdict_badge(active_report.ai_use_verdict.verdict.value)}<div style="font-size:0.75rem;color:#64748B;margin-top:3px;">Scope: {html.escape(format_plain_status(active_report.determined_scope))}</div></div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Evidentiary Gates (G1–G4)</div>
                <div class="uc-kv-val">{pos_gates} / 4 Positive Gates<div style="font-size:0.75rem;color:#64748B;margin-top:3px;">{len(active_report.evidence_records)} captured excerpt(s)</div></div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Surfaced Risk Findings (R1–R8)</div>
                <div class="uc-kv-val">{len(active_report.risk_findings)} finding(s)<div style="font-size:0.75rem;color:#64748B;margin-top:3px;">Supply Chain: {html.escape(format_plain_status(active_report.supply_chain_identity_status))}</div></div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Deterministic Risk Rating</div>
                <div class="uc-kv-val">{_status_badge(format_plain_status(ds.policy_status), "neutral")}<div style="font-size:0.75rem;color:#64748B;margin-top:3px;">Policy: {html.escape(ds.policy_status.value)}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res_tabs = st.tabs([
        "Executive Summary & Gates (G1–G4)",
        "Criticality Result (D/P/R/O/V)",
        f"Evidence Register ({len(active_report.evidence_records)})",
        f"Risk Findings ({len(active_report.risk_findings)}) & Open Questions",
        "Human Review & Exports",
    ])

    # Tab 1: Executive Summary & Gates
    with res_tabs[0]:
        st.markdown(
            f"""
            <div class="uc-callout uc-callout-info">
                <strong>AI-Use Verdict Statement:</strong> {html.escape(active_report.ai_use_verdict.verdict_statement)}<br/>
                <strong>Analytical Basis:</strong> {html.escape(active_report.ai_use_verdict.basis_summary)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        gate_rows = []
        for g in active_report.gates:
            g_var = (
                "critical"
                if g.status == "QUALIFYING_POSITIVE_EVIDENCE"
                else "success"
                if g.status == "QUALIFYING_NEGATIVE_EVIDENCE"
                else "high"
            )
            gate_rows.append(
                f"<tr>"
                f"<td><span class='uc-mono'>{html.escape(g.gate_id)}</span></td>"
                f"<td><strong>{html.escape(g.title)}</strong><div style='font-size:0.76rem;color:#64748B;'>{html.escape(g.question)}</div></td>"
                f"<td>{_status_badge(format_plain_status(g.status), g_var)}</td>"
                f"<td><span class='uc-mono'>{html.escape(', '.join(g.supporting_evidence_ids) or 'None')}</span></td>"
                f"<td>{html.escape(g.rationale)}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <div class="uc-table-wrap">
                <table class="uc-table">
                    <thead>
                        <tr><th>Gate</th><th>Gate Question</th><th>Determination</th><th>Evidence IDs</th><th>Rationale</th></tr>
                    </thead>
                    <tbody>{"".join(gate_rows)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 2: Criticality Result
    with res_tabs[1]:
        factors = crit_sum.get("factors", {})
        f_rows = []
        for fk in ("D", "P", "R", "O", "V"):
            fd = factors.get(fk, {})
            f_rows.append(
                f"<tr>"
                f"<td><strong>{html.escape(fk)}</strong></td>"
                f"<td><span class='uc-mono'>{html.escape(str(fd.get('score', 'UNKNOWN')))}</span></td>"
                f"<td><span class='uc-mono'>{html.escape(str(fd.get('weight', '')))}</span></td>"
                f"<td><span class='uc-mono'>{html.escape(str(fd.get('weighted_contribution', '')))}</span></td>"
                f"<td>{html.escape(format_plain_status(fd.get('determination_method', '')))}</td>"
                f"<td>{html.escape(str(fd.get('rationale', '')))}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <div class="uc-table-wrap">
                <table class="uc-table">
                    <thead>
                        <tr><th>Factor</th><th>Score</th><th>Weight</th><th>Weighted</th><th>Method</th><th>Rationale</th></tr>
                    </thead>
                    <tbody>{"".join(f_rows)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 3: Evidence Register
    with res_tabs[2]:
        mismatched = [
            d for d in active_report.extracted_documents
            if d.extraction_status == ExtractionStatus.SOURCE_IDENTITY_MISMATCH
        ]
        if mismatched:
            for m_doc in mismatched:
                st.markdown(
                    f"""
                    <div class="uc-callout uc-callout-danger">
                        <strong>Source-Identity Mismatch Rejected ({html.escape(m_doc.document_id)}):</strong>
                        Requested URL <code>{html.escape(m_doc.requested_url or m_doc.url)}</code> returned page identity
                        <code>{html.escape(m_doc.declared_og_url or m_doc.declared_canonical_url or m_doc.title or "Unknown")}</code>.
                        Document was marked <code>SOURCE_IDENTITY_MISMATCH</code> and excluded from vendor evidence.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if not active_report.evidence_records:
            st.markdown(
                """
                <div class="uc-empty-state">
                    <div class="uc-empty-title">No Evidence Records Captured</div>
                    <div class="uc-empty-desc">No public web or PDF evidence records were captured during this assessment run.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            w_map = {w.evidence_id: w for w in active_report.weighted_evidence}
            ev_rows = []
            for ev in active_report.evidence_records:
                we = w_map.get(ev.evidence_id)
                st_str = we.evidence_strength.value if we else ev.evidence_strength.value
                erules = ", ".join(we.fired_e_rule_ids if we else ev.e_rule_ids)
                sup_g = ", ".join(gates_for_evidence(active_report, ev.evidence_id))
                ev_rows.append(
                    f"<tr>"
                    f"<td><span class='uc-mono'>{html.escape(ev.evidence_id)}</span></td>"
                    f"<td><span class='uc-mono'>{html.escape(ev.source_class)}</span> (Grade {html.escape(ev.source_reliability_grade)})</td>"
                    f"<td>{html.escape(format_plain_status(st_str))}</td>"
                    f"<td><span class='uc-mono'>{html.escape(erules or '—')}</span></td>"
                    f"<td><span class='uc-mono'>{html.escape(sup_g or '—')}</span></td>"
                    f"<td><span class='uc-mono'>{html.escape(ev.retrieval_method)}</span></td>"
                    f"<td><a href='{html.escape(ev.canonical_url)}' target='_blank'>{html.escape(ev.canonical_url)}</a>"
                    f"<div style='margin-top:4px;color:#1E293B;'><em>\"{html.escape(ev.verbatim_excerpt)}\"</em></div>"
                    f"<div style='font-size:0.72rem;color:#64748B;margin-top:2px;'>SHA-256: <code>{html.escape(ev.content_sha256[:16])}...</code> | Integrity: {html.escape(ev.source_integrity_status)}</div></td>"
                    f"</tr>"
                )
            st.markdown(
                f"""
                <div class="uc-table-wrap">
                    <table class="uc-table">
                        <thead>
                            <tr><th>ID</th><th>Source</th><th>Strength</th><th>E-Rules</th><th>Gates</th><th>Method</th><th>URL &amp; Verbatim Excerpt</th></tr>
                        </thead>
                        <tbody>{"".join(ev_rows)}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tab 4: Risk Findings & Unresolved Questions
    with res_tabs[3]:
        if not active_report.risk_findings:
            st.markdown(
                """
                <div class="uc-empty-state">
                    <div class="uc-empty-title">No R1–R8 AI Security Risk Findings Surfaced</div>
                    <div class="uc-empty-desc">No evidence-backed R1–R8 risk findings were triggered for this run.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            rf_rows = []
            for rf in active_report.risk_findings:
                rf_rows.append(
                    f"<tr>"
                    f"<td><strong>{html.escape(rf.risk_rule_id)}</strong>: {html.escape(rf.risk_title)}<div style='font-size:0.74rem;color:#64748B;'>ID: {html.escape(rf.finding_id)}</div></td>"
                    f"<td>{_severity_badge(rf.severity.value)}<div style='font-size:0.74rem;color:#64748B;margin-top:3px;'>{html.escape(format_plain_status(rf.exposure_determination))}</div></td>"
                    f"<td>{html.escape(rf.finding_description)}<div style='font-size:0.75rem;color:#475569;margin-top:4px;'>Evidence: <code>{html.escape(', '.join(rf.evidence_ids))}</code></div></td>"
                    f"<td>{html.escape(rf.recommended_remediation)}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"""
                <div class="uc-table-wrap">
                    <table class="uc-table">
                        <thead>
                            <tr><th>Risk Rule</th><th>Severity &amp; Exposure</th><th>Deterministic Finding &amp; Evidence</th><th>Recommended Follow-Up</th></tr>
                        </thead>
                        <tbody>{"".join(rf_rows)}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if active_report.unresolved_questions:
            q_items = "".join(f"<li>{html.escape(q)}</li>" for q in active_report.unresolved_questions)
            st.markdown(
                f"""
                <div class="uc-callout uc-callout-warn">
                    <strong>Unresolved Diligence Questions ({len(active_report.unresolved_questions)}):</strong>
                    <ul style="margin:6px 0 0 18px;padding:0;">{q_items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Tab 5: Human Review & Exports
    with res_tabs[4]:
        rev_col, exp_col = st.columns([1.3, 1.0], gap="medium")
        with rev_col:
            st.markdown("#### Log Analyst Governance Validation / Override")
            target_type = st.selectbox(
                "Validation Target Type",
                options=["VERDICT", "GATE", "RISK_FINDING", "CRITICALITY_O5"],
                key="hval_target_type",
            )
            target_id = st.text_input("Target ID (e.g. G2, R1, VERDICT, O5)", value="VERDICT", key="hval_target_id")
            decision = st.selectbox(
                "Analyst Decision",
                options=["ACCEPTED_AS_PROPOSED", "OVERRIDDEN_BY_ANALYST", "ESCALATED_FOR_VENDOR_DILIGENCE"],
                key="hval_decision",
            )
            reviewer_id = st.text_input("Reviewer Name / Role", value="Senior TPRM Analyst", key="hval_reviewer")
            rationale_txt = st.text_area(
                "Written Governance Rationale (Required)",
                placeholder="Document the evidentiary basis for accepting, overriding, or escalating this determination...",
                key="hval_rationale",
            )
            if st.button("Record Analyst Sign-Off", key="btn_record_hval", type="primary"):
                if not rationale_txt.strip():
                    st.error("A written rationale is required to record an analyst validation decision.")
                else:
                    rec = HumanValidationService.create_validation_record(
                        vendor_id=active_report.vendor_id,
                        target_type=target_type,
                        target_id=target_id,
                        proposed_automated_finding=active_report.ai_use_verdict.verdict.value,
                        human_decision=decision,
                        rationale=rationale_txt,
                        original_status="PROPOSED",
                        updated_status=decision,
                        reviewer_id=reviewer_id,
                    )
                    active_report.human_validations.append(rec)
                    st.session_state["evidence_repo"].save_human_validation(rec)
                    st.session_state["evidence_repo"].save_vendor_report(active_report)
                    st.success(f"Recorded validation {rec.validation_id}.")

        with exp_col:
            st.markdown("#### Export Assessment Package")
            st.download_button(
                "Download Executive HTML Dossier (.html)",
                data=AssessmentReportExporter.to_html(active_report),
                file_name=f"{active_report.vendor_id}_{active_report.report_id}.html",
                mime="text/html",
                use_container_width=True,
                key="dl_assess_html",
            )
            st.download_button(
                "Download Markdown Assessment Report (.md)",
                data=AssessmentReportExporter.to_markdown(active_report),
                file_name=f"{active_report.vendor_id}_{active_report.report_id}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_assess_md",
            )
            st.download_button(
                "Download Point-of-Capture Evidence CSV (.csv)",
                data=AssessmentReportExporter.to_evidence_csv(active_report),
                file_name=f"{active_report.vendor_id}_evidence.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_assess_csv",
            )
            st.download_button(
                "Download Vendor Inventory Record (21 Cols CSV)",
                data=AssessmentReportExporter.to_inventory_csv(active_report),
                file_name=f"{active_report.vendor_id}_inventory_record.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_assess_inventory_csv",
            )
            st.download_button(
                "Download Full Audit JSON (.json)",
                data=AssessmentReportExporter.to_json(active_report),
                file_name=f"{active_report.vendor_id}_{active_report.report_id}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_assess_json",
            )

    # Single compact expandable 9-Level Stage Execution Ledger (Levels 1-9)
    tracker_rows = build_nine_level_stage_tracker(active_report)
    with st.expander("Audit Trail: Nine-Level Assessment Execution Ledger (Canonical Levels 1–9)", expanded=False):
        ledger_html: list[str] = []
        for r in tracker_rows:
            st_variant = (
                "success"
                if r.status == "COMPLETED"
                else "high"
                if r.status in ("REQUIRES_REVIEW", "PARTIAL", "NOT_CONFIGURED")
                else "neutral"
            )
            ledger_html.append(
                f"<tr>"
                f"<td><strong>{html.escape(r.level_label)}</strong></td>"
                f"<td><strong>{html.escape(r.level_name)}</strong></td>"
                f"<td>{_status_badge(r.badge_label, st_variant)}</td>"
                f"<td>{html.escape(r.analyst_method)}</td>"
                f"<td><span class='uc-mono'>{html.escape(r.analyst_counts)}</span></td>"
                f"<td>{html.escape(r.analyst_summary)}</td>"
                f"</tr>"
            )
            for sub in r.substeps:
                ledger_html.append(
                    f"<tr style='background:#F8FAFC;'>"
                    f"<td style='padding-left:24px;color:#475569;'>↳ <em>{html.escape(sub['substep_id'])}</em></td>"
                    f"<td style='color:#475569;'><em>{html.escape(sub['substep_name'])}</em></td>"
                    f"<td><span class='uc-mono'>{html.escape(sub['status'])}</span></td>"
                    f"<td>—</td>"
                    f"<td>—</td>"
                    f"<td style='color:#475569;'>{html.escape(sub['detail'])}</td>"
                    f"</tr>"
                )
        st.markdown(
            f"""
            <div class="uc-table-wrap">
                <table class="uc-table">
                    <thead>
                        <tr><th>Level</th><th>Canonical Framework Stage</th><th>Execution Status</th><th>Method</th><th>Volume</th><th>Stage Outcome</th></tr>
                    </thead>
                    <tbody>{"".join(ledger_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── Page 3: Evidence Workbench ───────────────────────────────────────────────


def _render_evidence_page() -> None:
    vendors_by_id: dict[str, MeridianVendor] = st.session_state["vendors_by_id"]
    reports_by_id: dict[str, VendorEndToEndAssessmentReport] = st.session_state["reports_by_vendor"]

    st.markdown(
        """
        <div class="uc-page-header">
            <div>
                <h1 class="uc-page-title">Point-of-Capture Evidence Workbench</h1>
                <p class="uc-page-desc">Filter and inspect verbatim evidence excerpts, cryptographic SHA-256 capture hashes, retrieval provenance (Direct HTTP vs ScraperAPI), source-integrity verification, and E1–E14 rule weightings.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_vids = sorted(vendors_by_id.keys())
    cur_vid = st.session_state.get("selected_vendor_id", all_vids[0])

    f1, f2, f3, f4, f5 = st.columns([1.6, 1.1, 1.2, 1.1, 1.2], gap="small")
    vendor_choice = f1.selectbox(
        "Filter by Vendor",
        options=["All Assessed Vendors"] + all_vids,
        index=(all_vids.index(cur_vid) + 1) if cur_vid in reports_by_id else 0,
        format_func=lambda x: x if x == "All Assessed Vendors" else f"{x} — {vendors_by_id[x].vendor_name}",
        key="ev_wb_vendor_filter",
    )
    if vendor_choice != "All Assessed Vendors":
        st.session_state["selected_vendor_id"] = vendor_choice

    source_options = ["All Sources"] + [f"S{i}" for i in range(1, 13)]
    source_filter = f2.selectbox("Source Class", options=source_options, key="ev_wb_source_filter")

    strength_options = [
        "All Strengths",
        "CONFIRMED",
        "STRONG",
        "INDICATIVE",
        "MARKETING",
        "DISCARDED_HOSTILE",
        "REQUIRES_REVIEW",
    ]
    strength_filter = f3.selectbox("Evidence Strength", options=strength_options, key="ev_wb_strength_filter")

    gate_options = ["All Gates", "G1", "G2", "G3", "G4"]
    gate_filter = f4.selectbox("Supported Gate", options=gate_options, key="ev_wb_gate_filter")

    method_options = ["All Methods", "DIRECT_HTTP", "SCRAPERAPI", "SYNTHETIC_SAMPLE"]
    method_filter = f5.selectbox("Retrieval Method", options=method_options, key="ev_wb_method_filter")

    # Collect matching evidence items across selected vendor(s)
    target_reports = (
        [reports_by_id[vendor_choice]]
        if vendor_choice != "All Assessed Vendors" and vendor_choice in reports_by_id
        else list(reports_by_id.values())
    )

    matching_items: list[tuple[VendorEndToEndAssessmentReport, Any, Any, list[str]]] = []
    rejected_docs: list[tuple[str, Any]] = []

    for rpt in target_reports:
        w_map = {w.evidence_id: w for w in rpt.weighted_evidence}
        for doc in rpt.extracted_documents:
            if doc.extraction_status == ExtractionStatus.SOURCE_IDENTITY_MISMATCH:
                rejected_docs.append((rpt.legal_entity, doc))
        for ev in rpt.evidence_records:
            we = w_map.get(ev.evidence_id)
            eff_strength = we.evidence_strength.value if we else ev.evidence_strength.value
            sup_gates = gates_for_evidence(rpt, ev.evidence_id)

            if source_filter != "All Sources" and ev.source_class != source_filter:
                continue
            if strength_filter != "All Strengths" and eff_strength != strength_filter:
                continue
            if gate_filter != "All Gates" and gate_filter not in sup_gates:
                continue
            if method_filter != "All Methods" and ev.retrieval_method != method_filter:
                continue

            matching_items.append((rpt, ev, we, sup_gates))

    if rejected_docs:
        for v_name, r_doc in rejected_docs:
            st.markdown(
                f"""
                <div class="uc-callout uc-callout-danger">
                    <strong>Rejected Upstream Document ({html.escape(v_name)} · {html.escape(r_doc.document_id)}):</strong>
                    Requested <code>{html.escape(r_doc.requested_url or r_doc.url)}</code> via <code>{html.escape(r_doc.retrieval_method)}</code>,
                    but received identity metadata for <code>{html.escape(r_doc.declared_og_url or r_doc.declared_canonical_url or r_doc.title or "unrelated site")}</code>.
                    Status: <code>SOURCE_IDENTITY_MISMATCH</code> (blocked from evidence register).
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not matching_items:
        target_vid = vendor_choice if vendor_choice != "All Assessed Vendors" else cur_vid
        st.markdown(
            f"""
            <div class="uc-empty-state">
                <div class="uc-empty-title">No Captured Evidence Records Match Current Selection</div>
                <div class="uc-empty-desc">
                    Run a Live OSINT Assessment or load a Sample Walkthrough for <strong>{html.escape(vendors_by_id[target_vid].vendor_name)}</strong>
                    to populate point-of-capture evidence excerpts, SHA-256 hashes, and E1–E14 rule evaluations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, _ = st.columns([1.4, 1.4, 2.2], gap="small")
        if c1.button(f"Load Sample Evidence for {target_vid}", key="btn_ev_load_sample", type="primary", use_container_width=True):
            _execute_vendor_assessment(vendors_by_id[target_vid], mode="SAMPLE", enable_local_llm=False)
            st.rerun()
        if c2.button("Go to Assessment Workspace", key="btn_ev_go_assess", type="secondary", use_container_width=True):
            _navigate_to("Assessments", target_vid)
            st.rerun()
        return

    # Render Evidence Workbench Table
    ev_table_rows = []
    for rpt, ev, we, sup_gates in matching_items:
        eff_strength = we.evidence_strength.value if we else ev.evidence_strength.value
        w_band = we.weight_band.value if we else "LOW"
        erules = ", ".join(we.fired_e_rule_ids if we else ev.e_rule_ids)
        origin_badge = (
            _status_badge("Sample Fixture", "purple")
            if ev.is_synthetic_test_fixture
            else _status_badge(ev.retrieval_method, "success")
        )
        ev_table_rows.append(
            f"<tr>"
            f"<td><span class='uc-mono'>{html.escape(ev.evidence_id)}</span><div style='font-size:0.74rem;color:#64748B;'>{html.escape(ev.legal_entity)}</div></td>"
            f"<td><span class='uc-mono'>{html.escape(ev.source_class)}</span> (Grade {html.escape(ev.source_reliability_grade)})</td>"
            f"<td>{_status_badge(format_plain_status(eff_strength), 'medium')} <span class='uc-mono'>{html.escape(w_band)}</span></td>"
            f"<td>{html.escape(format_plain_status(ev.scope))}</td>"
            f"<td><span class='uc-mono'>{html.escape(erules or '—')}</span></td>"
            f"<td><span class='uc-mono'>{html.escape(', '.join(sup_gates) or '—')}</span></td>"
            f"<td>{origin_badge}<div style='font-size:0.72rem;color:#64748B;margin-top:2px;'>{html.escape(ev.source_integrity_status)}</div></td>"
            f"<td><a href='{html.escape(ev.canonical_url)}' target='_blank'>{html.escape(ev.canonical_url)}</a>"
            f"<div style='margin-top:3px;'><em>\"{html.escape(ev.verbatim_excerpt)}\"</em></div></td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <div class="uc-table-wrap">
            <table class="uc-table">
                <thead>
                    <tr>
                        <th>Evidence ID</th>
                        <th>Source</th>
                        <th>Strength &amp; Weight</th>
                        <th>Product Scope</th>
                        <th>E-Rules</th>
                        <th>Gates</th>
                        <th>Retrieval &amp; Integrity</th>
                        <th>Canonical URL &amp; Full Verbatim Excerpt</th>
                    </tr>
                </thead>
                <tbody>{"".join(ev_table_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed Point-of-Capture Inspector
    st.markdown(
        """
        <div class="uc-section-header" style="margin-top:16px;">
            <div>
                <h3 class="uc-section-title">Point-of-Capture Provenance &amp; E1–E14 Rule Inspector</h3>
                <p class="uc-section-subtitle">Select any captured evidence item to inspect its full cryptographic provenance, character offsets, and rule-by-rule evaluation.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ev_options = [f"{ev.evidence_id} — {ev.legal_entity} ({ev.source_class}: {ev.page_title or ev.canonical_url})" for _, ev, _, _ in matching_items]
    sel_idx = st.selectbox(
        "Select Evidence Record to Inspect",
        options=list(range(len(ev_options))),
        format_func=lambda i: ev_options[i],
        key="ev_wb_detail_selector",
    )
    sel_rpt, sel_ev, sel_we, sel_gates = matching_items[sel_idx]

    st.markdown(
        f"""
        <div class="uc-kv-grid">
            <div class="uc-kv-box">
                <div class="uc-kv-label">Requested URL</div>
                <div class="uc-kv-val"><span class="uc-mono">{html.escape(sel_ev.original_url)}</span></div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Final Response URL</div>
                <div class="uc-kv-val"><span class="uc-mono">{html.escape(sel_ev.final_url or sel_ev.canonical_url)}</span></div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Retrieval Method &amp; Integrity</div>
                <div class="uc-kv-val"><span class="uc-mono">{html.escape(sel_ev.retrieval_method)}</span> · {html.escape(sel_ev.source_integrity_status)}</div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Captured At (UTC)</div>
                <div class="uc-kv-val"><span class="uc-mono">{html.escape(_format_short_date(sel_ev.retrieved_at))}</span></div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">SHA-256 Content Digest</div>
                <div class="uc-kv-val"><span class="uc-mono">{html.escape(sel_ev.content_sha256[:24])}...</span></div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Character Offsets / Page</div>
                <div class="uc-kv-val"><span class="uc-mono">chars {sel_ev.char_start}..{sel_ev.char_end}</span> (Page: {sel_ev.page_number or "Web"})</div>
            </div>
        </div>
        <div class="uc-callout uc-callout-info">
            <strong>Full Verbatim Excerpt ({html.escape(sel_ev.section_heading or sel_ev.page_title or "Document Body")}):</strong><br/>
            "{html.escape(sel_ev.verbatim_excerpt)}"
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sel_we and sel_we.applied_e_rules:
        er_html = []
        for er in sel_we.applied_e_rules:
            if er.triggered:
                er_html.append(
                    f"<tr>"
                    f"<td><span class='uc-mono'>{html.escape(er.rule_id)}</span></td>"
                    f"<td><strong>{html.escape(er.rule_name)}</strong></td>"
                    f"<td>{_status_badge(er.resulting_strength.value if er.resulting_strength else 'APPLIED', 'medium')}</td>"
                    f"<td>{html.escape(er.rationale)}</td>"
                    f"</tr>"
                )
        if er_html:
            st.markdown(
                f"""
                <div class="uc-table-wrap">
                    <table class="uc-table">
                        <thead>
                            <tr><th>Rule ID</th><th>Evidentiary Rule Name</th><th>Resulting Disposition</th><th>Rule Evaluation Rationale</th></tr>
                        </thead>
                        <tbody>{"".join(er_html)}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── Page 4: Reports & Export Center ──────────────────────────────────────────


def _render_reports_page() -> None:
    vendors_by_id: dict[str, MeridianVendor] = st.session_state["vendors_by_id"]
    reports_by_id: dict[str, VendorEndToEndAssessmentReport] = st.session_state["reports_by_vendor"]

    st.markdown(
        """
        <div class="uc-page-header">
            <div>
                <h1 class="uc-page-title">Vendor Assessment Dossier &amp; Export Center</h1>
                <p class="uc-page-desc">Review the complete, executive-ready third-party AI risk dossier and export auditable artifacts in HTML, Markdown, CSV, or JSON format.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_vids = sorted(vendors_by_id.keys())
    cur_vid = st.session_state.get("selected_vendor_id", all_vids[0])
    if cur_vid not in vendors_by_id:
        cur_vid = all_vids[0]

    sel_col, dl1, dl2, dl3, dl4, dl5 = st.columns([1.8, 0.8, 0.8, 0.8, 0.8, 1.2], gap="small")
    chosen_vid = sel_col.selectbox(
        "Select Vendor Dossier",
        options=all_vids,
        index=all_vids.index(cur_vid),
        format_func=lambda vid: f"{vid} — {vendors_by_id[vid].vendor_name}",
        key="reports_vendor_selector",
    )
    st.session_state["selected_vendor_id"] = chosen_vid
    report = reports_by_id.get(chosen_vid)

    if report is None:
        st.markdown(
            f"""
            <div class="uc-empty-state">
                <div class="uc-empty-title">No Assessment Dossier Generated for {html.escape(vendors_by_id[chosen_vid].vendor_name)}</div>
                <div class="uc-empty-desc">Execute a Live OSINT Assessment or Sample Walkthrough to generate a complete vendor risk dossier and enable exports.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1.4, 1.4, 2.2], gap="small")
        if b1.button(f"Open Assessment Workspace ({chosen_vid})", key="btn_rpt_go_assess", type="primary", use_container_width=True):
            _navigate_to("Assessments", chosen_vid)
            st.rerun()
        if b2.button(f"Generate Sample Dossier ({chosen_vid})", key="btn_rpt_gen_sample", type="secondary", use_container_width=True):
            _execute_vendor_assessment(vendors_by_id[chosen_vid], mode="SAMPLE", enable_local_llm=False)
            st.rerun()
        return

    st.write("")
    dl1.download_button(
        "Export HTML",
        data=AssessmentReportExporter.to_html(report),
        file_name=f"{report.vendor_id}_dossier.html",
        mime="text/html",
        use_container_width=True,
        key="btn_rpt_dl_html",
    )
    dl2.download_button(
        "Export Markdown",
        data=AssessmentReportExporter.to_markdown(report),
        file_name=f"{report.vendor_id}_dossier.md",
        mime="text/markdown",
        use_container_width=True,
        key="btn_rpt_dl_md",
    )
    dl3.download_button(
        "Evidence CSV",
        data=AssessmentReportExporter.to_evidence_csv(report),
        file_name=f"{report.vendor_id}_evidence.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_rpt_dl_csv",
    )
    dl4.download_button(
        "Audit JSON",
        data=AssessmentReportExporter.to_json(report),
        file_name=f"{report.vendor_id}_report.json",
        mime="application/json",
        use_container_width=True,
        key="btn_rpt_dl_json",
    )
    dl5.download_button(
        "Inventory CSV (21 Cols)",
        data=AssessmentReportExporter.to_inventory_csv(report, vendors_by_id.get(chosen_vid)),
        file_name=f"{report.vendor_id}_inventory_record.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_rpt_dl_inv_csv",
    )

    if report.is_synthetic_evidence_run:
        st.markdown(
            """
            <div class="uc-callout uc-callout-warn">
                <strong>Dossier Origin — Sample Assessment (Synthetic Demonstration Fixtures):</strong>
                This dossier was generated from pre-collected demonstration fixtures and does not represent live vendor evidence.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="uc-callout uc-callout-info">
                <strong>Dossier Origin — Live OSINT Assessment:</strong>
                Generated from live public source reconnaissance and point-of-capture extraction.
            </div>
            """,
            unsafe_allow_html=True,
        )

    crit = report.criticality_summary or {}
    osint = report.osint_plan_summary or {}
    ds = report.deterministic_risk_score

    st.markdown(
        f"""
        <div class="uc-section-header">
            <div>
                <h2 class="uc-section-title">1. Executive Summary — {html.escape(report.legal_entity)} ({html.escape(report.vendor_id)})</h2>
                <p class="uc-section-subtitle">Report Run ID: {html.escape(report.report_id)} · Generated: {html.escape(_format_short_date(report.generated_at))}</p>
            </div>
        </div>
        <div class="uc-kv-grid">
            <div class="uc-kv-box">
                <div class="uc-kv-label">Service Purchased by Meridian</div>
                <div class="uc-kv-val">{html.escape(report.service_product_provided or "UNKNOWN")}</div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Business Process Supported</div>
                <div class="uc-kv-val">{html.escape(report.business_process_supported or "UNKNOWN")}</div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Effective Criticality Tier</div>
                <div class="uc-kv-val">{_tier_badge(str(crit.get("final_criticality", "Unknown")))} (Base: {html.escape(str(crit.get("base_score", "N/A")))})</div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">OSINT Depth Profile</div>
                <div class="uc-kv-val">{html.escape(str(osint.get("selected_depth_profile", "N/A")))} (Completed: {"Yes" if report.ai_use_verdict.depth_completed else "No"})</div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">AI-Use Determination</div>
                <div class="uc-kv-val">{_verdict_badge(report.ai_use_verdict.verdict.value)}</div>
            </div>
            <div class="uc-kv-box">
                <div class="uc-kv-label">Level 9 Numerical Risk Rating</div>
                <div class="uc-kv-val">{html.escape(format_plain_status(ds.policy_status))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 2: G1-G4 Gates & Supply Chain Identity
    st.markdown(
        f"""
        <div class="uc-section-header" style="margin-top:16px;">
            <div>
                <h2 class="uc-section-title">2. AI-Use Verdict, Supply-Chain Identity &amp; Gates (G1–G4)</h2>
                <p class="uc-section-subtitle">Named Providers: {html.escape(', '.join(report.named_providers) or 'Undisclosed')} · Named Models: {html.escape(', '.join(report.named_models) or 'Undisclosed')} · Hosting: {html.escape(', '.join(report.hosting_infrastructures) or 'Undisclosed')}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    gate_rows = []
    for g in report.gates:
        gate_rows.append(
            f"<tr>"
            f"<td><span class='uc-mono'>{html.escape(g.gate_id)}</span></td>"
            f"<td><strong>{html.escape(g.title)}</strong></td>"
            f"<td>{html.escape(format_plain_status(g.status))}</td>"
            f"<td><span class='uc-mono'>{html.escape(', '.join(g.supporting_evidence_ids) or 'None')}</span></td>"
            f"<td>{html.escape(g.rationale)}</td>"
            f"</tr>"
        )
    st.markdown(
        f"""
        <div class="uc-table-wrap">
            <table class="uc-table">
                <thead>
                    <tr><th>Gate</th><th>Gate Title</th><th>Status</th><th>Evidence IDs</th><th>Evaluation Rationale</th></tr>
                </thead>
                <tbody>{"".join(gate_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 3: R1-R8 Findings & Remediation
    st.markdown(
        """
        <div class="uc-section-header" style="margin-top:16px;">
            <div>
                <h2 class="uc-section-title">3. AI Security &amp; Governance Risk Findings (R1–R8)</h2>
                <p class="uc-section-subtitle">Evidence-backed risk exposures, analytical explanations, and recommended vendor follow-up actions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if report.risk_findings:
        rf_rows = []
        for rf in report.risk_findings:
            rf_rows.append(
                f"<tr>"
                f"<td><strong>{html.escape(rf.risk_rule_id)}</strong>: {html.escape(rf.risk_title)}</td>"
                f"<td>{_severity_badge(rf.severity.value)}</td>"
                f"<td>{html.escape(rf.finding_description)}<div style='font-size:0.75rem;color:#64748B;margin-top:3px;'>Evidence: <code>{html.escape(', '.join(rf.evidence_ids))}</code></div></td>"
                f"<td>{html.escape(rf.analyst_explanation or rf.finding_description)}</td>"
                f"<td>{html.escape(rf.recommended_remediation)}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <div class="uc-table-wrap">
                <table class="uc-table">
                    <thead>
                        <tr><th>Risk Rule</th><th>Severity</th><th>Surfaced Exposure</th><th>Analytical Summary</th><th>Recommended Follow-Up</th></tr>
                    </thead>
                    <tbody>{"".join(rf_rows)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<p style="font-size:0.85rem;color:#64748B;">No R1–R8 risk findings surfaced for this run.</p>', unsafe_allow_html=True)

    # Section 4: Limitations, Unresolved Questions & Level 9 Policy Status
    notices = format_analyst_unavailable_integrations(report, st.session_state.get("system_readiness"))
    st.markdown(
        f"""
        <div class="uc-section-header" style="margin-top:16px;">
            <div>
                <h2 class="uc-section-title">4. Governance Limitations, Open Questions &amp; Scoring Policy</h2>
            </div>
        </div>
        <div class="uc-callout uc-callout-info">
            <strong>Level 9 Deterministic Risk Rating Policy ({html.escape(ds.policy_status.value)}):</strong><br/>
            {html.escape(ds.status_message)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if notices or report.assessment_limitations:
        lim_html = "".join(f"<li>{html.escape(item)}</li>" for item in (notices + list(report.assessment_limitations)))
        st.markdown(f"<ul style='font-size:0.84rem;color:#334155;'>{lim_html}</ul>", unsafe_allow_html=True)

    # Section 5: Meridian Vendor Inventory Assessment Record (Canonical 21 Columns)
    inv_rec = AssessmentReportExporter.to_inventory_dict(report, vendors_by_id.get(chosen_vid))
    st.markdown(
        """
        <div class="uc-section-header" style="margin-top:20px;">
            <div>
                <h2 class="uc-section-title">5. Meridian Vendor Inventory Assessment Record (Canonical 21 Columns)</h2>
                <p class="uc-section-subtitle">Preserves all 10 supplied vendor input fields and delivers the 11 derived governance fields matching the Larkspur worked example.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("View 21-Field Canonical Inventory Record (Full Detail)", expanded=True):
        st.markdown("#### A. Preserved Vendor Input Fields (10 Columns)")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Vendor ID:** `{inv_rec['Vendor ID']}`\n\n**Vendor Name:** {inv_rec['Vendor Name']}")
        c2.markdown(f"**Public Website:** `{inv_rec['Public Website']}`\n\n**Category:** {inv_rec['Category']}")
        c3.markdown(f"**Operational Dependency:** `{inv_rec['Operational Dependency']}`\n\n**Data Volume (annual):** {inv_rec['Data Volume (annual)']}")
        st.markdown(f"**Vendor Description:** {inv_rec['Vendor Description']}")
        st.markdown(f"**Service / Product Provided to Meridian:** {inv_rec['Service / Product Provided to Meridian']}")
        st.markdown(f"**Business Process Supported:** {inv_rec['Business Process Supported']}")
        st.markdown(f"**Data Classification Accessed:** {inv_rec['Data Classification Accessed']}")

        st.markdown("---")
        st.markdown("#### B. Student-Derived Assessment Fields (11 Columns)")
        d1, d2, d3 = st.columns(3)
        d1.markdown(f"**Criticality Tier:** `{inv_rec['Criticality Tier']}`")
        d2.markdown(f"**AI Usage Detected:** `{inv_rec['AI Usage Detected']}`")
        d3.markdown(f"**AI Security Risk Classification:** `{inv_rec['AI Security Risk Classification']}`")

        st.markdown(f"**Criticality Rationale:**\n\n{inv_rec['Criticality Rationale']}")
        st.markdown(f"**Assessment Depth Applied:**\n\n{inv_rec['Assessment Depth Applied']}")
        st.markdown(f"**How Vendor May Be Using AI:**\n\n{inv_rec['How Vendor May Be Using AI']}")
        st.markdown(f"**Named AI Sub-processors / Model Providers:**\n\n{inv_rec['Named AI Sub-processors / Model Providers']}")
        st.markdown(f"**Risk Rationale:**\n\n{inv_rec['Risk Rationale']}")
        st.markdown(f"**Recommended Action:**\n\n{inv_rec['Recommended Action']}")
        st.markdown(f"**Evidence Description along with OSINT Source(s) / Evidence Link:**\n\n{inv_rec['Evidence Description along with OSINT Source(s) / Evidence Link']}")
        st.markdown(f"**Assessed By / Date:** `{inv_rec['Assessed By / Date']}`")


# ─── Page 5: Settings & Technical Diagnostics ─────────────────────────────────


def _render_settings_page(readiness: SystemReadinessReport) -> None:
    st.markdown(
        """
        <div class="uc-page-header">
            <div>
                <h1 class="uc-page-title">System Settings &amp; Technical Diagnostics</h1>
                <p class="uc-page-desc">Inspect OSINT search and retrieval adapters (Serper + ScraperAPI), source-integrity guardrails, local Ollama dual-model routing, Level 5 embeddings, and Level 9 scoring calibration.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        serper_badge = _status_badge("Configured", "success") if readiness.search_configured else _status_badge("API Key Not Set", "high")
        scraper_badge = _status_badge("Configured", "success") if readiness.scraperapi_configured else _status_badge("Optional Fallback Unconfigured", "neutral")

        st.markdown(
            f"""
            <div class="uc-panel">
                <div class="uc-section-header">
                    <div>
                        <h2 class="uc-section-title">1. OSINT Search &amp; Webpage Retrieval Pipeline</h2>
                        <p class="uc-section-subtitle">Level 3 Search Discovery (Serper) &amp; Level 4 Retrieval (Direct HTTP + ScraperAPI Fallback)</p>
                    </div>
                </div>
                <div class="uc-table-wrap">
                    <table class="uc-table">
                        <thead>
                            <tr><th>Component</th><th>Status</th><th>Configuration &amp; Endpoint</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Serper Search Adapter</strong><div style="font-size:0.75rem;color:#64748B;">MERIDIAN_SERPER_API_KEY</div></td>
                                <td>{serper_badge}</td>
                                <td><code>https://google.serper.dev/search</code> (POST JSON)<div style="font-size:0.76rem;color:#475569;margin-top:2px;">{html.escape(readiness.search_status_message)}</div></td>
                            </tr>
                            <tr>
                                <td><strong>Direct HTTP Fetcher</strong><div style="font-size:0.75rem;color:#64748B;">Primary Level 4 Fetch</div></td>
                                <td>{_status_badge("Active (Primary)", "success")}</td>
                                <td><code>UrllibPublicFetchAdapter</code> with SSRF/private-IP guard &amp; redirect tracking</td>
                            </tr>
                            <tr>
                                <td><strong>ScraperAPI Retrieval Fallback</strong><div style="font-size:0.75rem;color:#64748B;">MERIDIAN_SCRAPERAPI_KEY</div></td>
                                <td>{scraper_badge}</td>
                                <td><code>https://api.scraperapi.com</code> (Max Concurrency: 5, JS Render: Off by default)<div style="font-size:0.76rem;color:#475569;margin-top:2px;">{html.escape(readiness.scraperapi_status_message)}</div></td>
                            </tr>
                            <tr>
                                <td><strong>Source-Integrity Guard</strong><div style="font-size:0.75rem;color:#64748B;">Pre-Evidence Identity Check</div></td>
                                <td>{_status_badge("Enforced", "success")}</td>
                                <td>Validates requested URL vs. final URL, <code>rel="canonical"</code>, <code>og:url</code>, and page title; isolates <code>SOURCE_IDENTITY_MISMATCH</code> responses</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rag_badge = _status_badge("Vector Retrieval Ready", "success") if readiness.semantic_retrieval_ready else _status_badge("Deterministic Passthrough", "neutral")
        st.markdown(
            f"""
            <div class="uc-panel">
                <div class="uc-section-header">
                    <div>
                        <h2 class="uc-section-title">3. Level 5 RAG Chunking &amp; Semantic Embeddings</h2>
                        <p class="uc-section-subtitle">Separate embedding model configuration (never routed through generative Ollama models)</p>
                    </div>
                </div>
                <div class="uc-kv-grid">
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">Embedding Model</div>
                        <div class="uc-kv-val"><span class="uc-mono">{html.escape(readiness.embedding_model_name)}</span></div>
                    </div>
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">Active Level 5 Mode</div>
                        <div class="uc-kv-val">{rag_badge}</div>
                    </div>
                </div>
                <p style="font-size:0.82rem;color:#475569;margin:0;">{html.escape(readiness.rag_status_message)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        ollama_badge = _status_badge("Online", "success") if readiness.ollama_reachable else _status_badge("Unreachable", "high")
        llama_badge = _status_badge("Installed", "success") if readiness.llama3_2_installed else _status_badge("Missing", "high")
        mistral_badge = _status_badge("Installed", "success") if readiness.mistral_installed else _status_badge("Missing", "high")

        st.markdown(
            f"""
            <div class="uc-panel">
                <div class="uc-section-header">
                    <div>
                        <h2 class="uc-section-title">2. Local Ollama Server &amp; Dual-Model Routing</h2>
                        <p class="uc-section-subtitle">6 GB VRAM sequential model lifecycle management</p>
                    </div>
                </div>
                <div class="uc-table-wrap">
                    <table class="uc-table">
                        <thead>
                            <tr><th>Model / Service</th><th>Status</th><th>Assigned Pipeline Role</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Ollama Endpoint</strong></td>
                                <td>{ollama_badge}</td>
                                <td><code>{html.escape(readiness.ollama_endpoint)}</code> — {html.escape(readiness.ollama_status_message)}</td>
                            </tr>
                            <tr>
                                <td><strong>Model A: <code>{html.escape(readiness.triage_model)}</code></strong></td>
                                <td>{llama_badge}</td>
                                <td>Level 3B Candidate-URL Triage (lightweight relevance &amp; source-class filtering)</td>
                            </tr>
                            <tr>
                                <td><strong>Model B: <code>{html.escape(readiness.analysis_model)}</code></strong></td>
                                <td>{mistral_badge}</td>
                                <td>Level 6 Signal Classification &amp; Level 8 Grounded Risk Summarization</td>
                            </tr>
                            <tr>
                                <td><strong>Sequential VRAM Unload</strong></td>
                                <td>{_status_badge("Enabled (keep_alive=0)" if readiness.unload_on_model_switch else "Disabled", "neutral")}</td>
                                <td>Unloads Model A after Level 3B before loading Model B at Level 6</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="uc-panel">
                <div class="uc-section-header">
                    <div>
                        <h2 class="uc-section-title">4. Level 9 Risk-Scoring Policy &amp; Storage</h2>
                        <p class="uc-section-subtitle">Deterministic formula readiness (Risk Score = M_c × Σ(S_i × W_i)) and local persistence</p>
                    </div>
                </div>
                <div class="uc-callout uc-callout-warn">
                    <strong>Policy Status ({html.escape(readiness.scoring_policy_status)}):</strong><br/>
                    {html.escape(readiness.scoring_status_message)}
                </div>
                <div class="uc-kv-grid">
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">Evidence SQLite Repository</div>
                        <div class="uc-kv-val"><span class="uc-mono">{html.escape(str(_DEFAULT_DB_PATH))}</span></div>
                    </div>
                    <div class="uc-kv-box">
                        <div class="uc-kv-label">Point-of-Capture Artifacts</div>
                        <div class="uc-kv-val"><span class="uc-mono">{html.escape(str(_DEFAULT_ARTIFACT_DIR))}</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── Main Entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    _inject_enterprise_styles()
    _init_session_state()

    readiness = SystemReadinessInspector.check_system_readiness()
    st.session_state["system_readiness"] = readiness

    active_page = _render_app_shell(readiness)

    if active_page == "Portfolio":
        _render_portfolio_page(readiness)
    elif active_page == "Assessments":
        _render_assessments_page(readiness)
    elif active_page == "Evidence":
        _render_evidence_page()
    elif active_page == "Reports":
        _render_reports_page()
    elif active_page == "Settings":
        _render_settings_page(readiness)


if __name__ == "__main__":
    main()
