"""Meridian Vendor Risk Assessment — Streamlit Dashboard.

Supports:
  1. Meridian D/P/R/O/V Methodology (Case study dataset with full provenance tracking, O1-O4 safeguards, and O5 governance)
  2. Standard Qualitative Mode (Legacy 5-dimension model)
  3. Interactive Report Browser & Archival
  4. Portfolio Risk Visualizations & Analytics
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import datetime
import json
import tempfile
from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st

from meridian_assessment.config.env_loader import load_dotenv_if_present

load_dotenv_if_present()

from meridian_assessment.models.criticality import CriticalityAssessment, CriticalityTier

from meridian_assessment.models.assessment import AssessmentDepth
from meridian_assessment.models.factor_result import (
    CriticalityLevel,
    CriticalityResult,
    DeterminationMethod,
    HumanReviewRecord,
    ScoreStatus,
)
from meridian_assessment.services.pipeline import AssessmentPipeline
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine
from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.services.osint import (
    OSINTCoverageTracker,
    OSINTInvestigationPlanner,
    OSINTSearchCollector,
    OSINTSourceRegistry,
    StopConditionEvaluator,
    build_search_backend_from_env,
)
from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CriticalityApprovalStatus,
    CoverageState,
    GateStatus,
    OSINTInvestigationPlan,
    SearchExecutionStatus,
    TimerState,
)
from meridian_assessment.models.ai_risk import (
    ScoringPolicyStatus,
    VendorEndToEndAssessmentReport,
)
from meridian_assessment.models.meridian_vendor import MeridianVendor
from meridian_assessment.services.end_to_end_pipeline import (
    HumanValidationService,
    MeridianEndToEndPipeline,
)
from meridian_assessment.services.offline_demo_fixtures import (
    build_synthetic_demo_adapters_for_vendor,
)
from meridian_assessment.services.readiness_and_exports import (
    AssessmentReportExporter,
    CANONICAL_NINE_LEVEL_NAMES,
    SystemReadinessInspector,
    build_nine_level_stage_tracker,
    gates_for_evidence,
    validate_vendor_assessment_input,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meridian Financial | Third-Party AI Risk Assessment Console",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Enterprise Financial-Services Console Design System ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.25rem 2rem 2.25rem 2rem;
    max-width: 1520px;
}

.readiness-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.65rem;
    margin-bottom: 0.9rem;
}
.readiness-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-top: 3px solid #334155;
    border-radius: 6px;
    padding: 0.65rem 0.8rem;
}
.readiness-card.ok { border-top-color: #15803d; }
.readiness-card.warn { border-top-color: #b45309; }
.readiness-card.blocked { border-top-color: #b91c1c; }
.readiness-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #475569;
    margin-bottom: 0.25rem;
}
.readiness-status {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.readiness-detail {
    font-size: 0.71rem;
    color: #475569;
    line-height: 1.38;
}
.console-banner {
    padding: 0.65rem 0.95rem;
    border-radius: 6px;
    font-size: 0.78rem;
    line-height: 1.45;
    margin-bottom: 0.9rem;
}
.console-banner.synth {
    background: #fffbeb;
    border: 1px solid #f59e0b;
    color: #78350f;
}
.console-banner.incomplete {
    background: #fef2f2;
    border: 1px solid #ef4444;
    color: #7f1d1d;
}
.console-banner.real-ok {
    background: #f0f9ff;
    border: 1px solid #0284c7;
    color: #0c4a6e;
}
.stage-badge {
    display: inline-block;
    padding: 0.14rem 0.48rem;
    border-radius: 4px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.stage-badge.completed { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.stage-badge.fallback { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; }
.stage-badge.blocked { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.stage-badge.review { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.stage-badge.skipped { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }


/* ── Top Bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.25rem;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.topbar-logo {
    width: 36px;
    height: 36px;
    background: #0f172a;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    color: #ffffff;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.topbar-name {
    font-size: 1.12rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.topbar-sub {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 400;
}
.topbar-right {
    font-size: 0.72rem;
    color: #64748b;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    font-weight: 500;
}

/* ── Stat Cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.25rem;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.9rem 1.15rem;
    position: relative;
    overflow: hidden;
    transition: all 0.15s ease;
}
.stat-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
}
.stat-card.all::before      { background: #3b82f6; }
.stat-card.critical::before { background: #dc2626; }
.stat-card.high::before     { background: #ea580c; }
.stat-card.medium::before   { background: #d97706; }
.stat-card.low::before      { background: #16a34a; }

.stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 0.25rem;
}
.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
}
.stat-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* ── Section Headers ── */
.sec-header {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    margin: 1.1rem 0 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Criticality Pills ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.18rem 0.55rem;
    border-radius: 9999px;
    letter-spacing: 0.02em;
}
.pill.critical { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.pill.high     { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
.pill.medium   { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.pill.low      { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

.pill-dot { width: 5px; height: 5px; border-radius: 50%; }
.pill.critical .pill-dot { background: #dc2626; }
.pill.high .pill-dot     { background: #ea580c; }
.pill.medium .pill-dot   { background: #d97706; }
.pill.low .pill-dot      { background: #16a34a; }

/* ── Detail Panel ── */
.detail-header {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.detail-vendor-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
    margin-bottom: 0.35rem;
}
.detail-meta {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.detail-tag {
    font-size: 0.72rem;
    font-weight: 500;
    color: #334155;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 0.2rem 0.5rem;
}

/* ── Factor Cards ── */
.factor-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    margin-bottom: 0.5rem;
}
.factor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
}
.factor-title {
    font-size: 0.74rem;
    font-weight: 700;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.factor-rationale {
    font-size: 0.78rem;
    color: #334155;
    line-height: 1.45;
    margin-bottom: 0.35rem;
}
.factor-meta {
    font-size: 0.68rem;
    color: #64748b;
    border-top: 1px solid #f1f5f9;
    padding-top: 0.3rem;
}

/* ── Score Meter ── */
.score-meter {
    display: inline-flex;
    align-items: center;
    gap: 3px;
}
.score-tick {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
}
.score-tick.active-score {
    background: #0f172a;
    color: #ffffff;
}
.score-tick.filled {
    background: #e2e8f0;
    color: #475569;
}
.score-tick.empty {
    background: #f8fafc;
    color: #cbd5e1;
    border: 1px dashed #cbd5e1;
}
.score-fraction {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    margin-left: 5px;
}

/* ── Override / Safeguard Box ── */
.override-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.45rem;
    font-size: 0.75rem;
}
.override-box.triggered {
    background: #fff7ed;
    border-color: #fdba74;
}

/* ── Report Card ── */
.report-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 1rem;
}

/* ── OSINT Gate Cards & Queries ── */
.gate-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.85rem 1.05rem;
    margin-bottom: 0.55rem;
}
.query-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.45rem;
    font-size: 0.76rem;
    color: #0f172a;
}
.advisory-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.55rem;
    font-size: 0.76rem;
    color: #92400e;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    border: 1.5px dashed #e2e8f0;
    border-radius: 8px;
    color: #94a3b8;
}
.empty-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
.empty-title { font-size: 0.88rem; font-weight: 600; color: #475569; margin-bottom: 0.2rem; }
.empty-hint  { font-size: 0.75rem; color: #94a3b8; }

section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.25rem 1rem;
}

/* Clean tabs styling */
div[data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.15rem;
    margin-bottom: 1.25rem;
}
button[data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.4rem 0.6rem;
    color: #64748b;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0f172a;
    border-bottom: 2px solid #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants & Paths ────────────────────────────────────────────────────────
MERIDIAN_DATASET_CSV = Path("data/input/meridian_vendors.csv")
LEGACY_SAMPLE_CSV = Path("data/sample/sample_vendors.csv")
CONFIG_PATH = Path("config/criticality.yaml")
REPORTS_DIR = Path("data/output")

DEPTH_LABEL = {
    AssessmentDepth.COMPREHENSIVE: "Comprehensive Depth",
    AssessmentDepth.TARGETED: "Targeted Depth",
    AssessmentDepth.LIGHTWEIGHT: "Lightweight Depth",
}


def pill(val: str) -> str:
    norm = str(val).lower().replace("tier 1 — ", "").replace("tier 2 — ", "").replace("tier 3 — ", "").strip()
    css_class = "medium"
    if "crit" in norm:
        css_class = "critical"
    elif "high" in norm:
        css_class = "high"
    elif "med" in norm:
        css_class = "medium"
    elif "low" in norm:
        css_class = "low"
    return f'<span class="pill {css_class}"><span class="pill-dot"></span>{val}</span>'


def render_score_meter(score: Optional[int], max_score: int = 3) -> str:
    if score is None:
        return '<span style="font-size:0.7rem;font-weight:600;color:#94a3b8;background:#f1f5f9;padding:0.15rem 0.4rem;border-radius:4px;">UNKNOWN</span>'
    ticks = []
    for i in range(max_score + 1):
        if i == score:
            ticks.append(f'<span class="score-tick active-score">{i}</span>')
        elif i < score:
            ticks.append(f'<span class="score-tick filled">{i}</span>')
        else:
            ticks.append(f'<span class="score-tick empty">{i}</span>')
    return f'<div class="score-meter">{"".join(ticks)} <span class="score-fraction">({score}/{max_score})</span></div>'


def run_meridian_methodology(csv_bytes: bytes) -> list[CriticalityResult]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
        tmp.write(csv_bytes)
        tmp_path = Path(tmp.name)
    loader = MeridianCSVVendorLoader()
    engine = MeridianCriticalityEngine()
    vendors = loader.load(tmp_path)
    return [engine.evaluate(v) for v in vendors]


def run_legacy_pipeline(csv_bytes: bytes) -> list[CriticalityAssessment]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
        tmp.write(csv_bytes)
        tmp_path = Path(tmp.name)
    pipeline = AssessmentPipeline(config_path=CONFIG_PATH)
    return pipeline.run_from_source(tmp_path)


def load_and_normalize_report(file_path: Path) -> list[dict[str, Any]]:
    """Normalize any saved report JSON into a standardized structure."""
    try:
        content = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in content:
            if "factors" in item:
                # Meridian format
                crit = item.get("final_criticality") or item.get("proposed_criticality") or "Unknown"
                score = item.get("base_score")
                factors = item.get("factors", {})
                d = factors.get("D", {}).get("score")
                p = factors.get("P", {}).get("score")
                r = factors.get("R", {}).get("score")
                o = factors.get("O", {}).get("score")
                v = factors.get("V", {}).get("score")
                safeguards = any(ov.get("triggered") for ov in item.get("overrides_evaluated", []))
                normalized.append({
                    "id": item.get("vendor_id", ""),
                    "name": item.get("vendor_name", ""),
                    "criticality": crit,
                    "score": score,
                    "depth": item.get("assessment_depth", "Targeted"),
                    "D": d, "P": p, "R": r, "O": o, "V": v,
                    "safeguards_triggered": safeguards,
                    "raw": item,
                })
            elif "criteria" in item:
                # Legacy format
                tier_map = {"TIER_1": "Critical", "TIER_2": "Medium", "TIER_3": "Low"}
                crit = tier_map.get(item.get("criticality_tier", ""), item.get("criticality_tier", ""))
                score = item.get("criticality_score")
                crit_dict = item.get("criteria", {})
                normalized.append({
                    "id": item.get("vendor_id", ""),
                    "name": item.get("vendor_name", ""),
                    "criticality": crit,
                    "score": score,
                    "depth": str(item.get("assessment_depth", "TARGETED")).capitalize(),
                    "D": crit_dict.get("data_sensitivity", {}).get("score"),
                    "P": crit_dict.get("payment_flows", {}).get("score"),
                    "R": crit_dict.get("regulatory_exposure", {}).get("score"),
                    "O": crit_dict.get("operational_dependency", {}).get("score"),
                    "V": crit_dict.get("customer_data_volume", {}).get("score"),
                    "safeguards_triggered": False,
                    "raw": item,
                })
        return normalized
    except Exception as e:
        st.error(f"Error loading report: {e}")
        return []


# ─── Top Brand Bar ────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">M</div>
        <div>
            <div class="topbar-name">Meridian Financial &nbsp;|&nbsp; Third-Party AI Risk Assessment Console</div>
            <div class="topbar-sub">Nine-Level Deterministic Governance, Point-of-Capture OSINT Evidence &amp; Dual-Model Local Analysis</div>
        </div>
    </div>
    <div class="topbar-right">Levels 1–9 &nbsp;·&nbsp; Deterministic Scoring &amp; Verdict Guardrails</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Criticality Methodology")
    engine_mode = st.radio(
        "Engine Selection",
        ["Meridian D/P/R/O/V (Primary)", "Standard Qualitative (Legacy)"],
        label_visibility="collapsed",
    )

    st.markdown("#### Vendor Dataset")
    if "Primary" in engine_mode:
        data_source = st.radio(
            "Source",
            ["Meridian Case Study (6 Vendors)", "Upload Meridian CSV"],
            label_visibility="collapsed",
        )
    else:
        data_source = st.radio(
            "Source",
            ["Sample Dataset", "Upload CSV"],
            label_visibility="collapsed",
        )

    uploaded_file = None
    if "Upload" in data_source:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_btn = st.button("Load Inventory", type="primary", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("Reset State", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#475569;line-height:1.55">
    <strong style="color:#0f172a">Nine-Level Framework</strong><br>
    1. Criticality Tiering (D/P/R/O/V + O1–O5)<br>
    2. Assessment Depth Scoping (S1–S12)<br>
    3. Source Identification &amp; Recon (3B Triage)<br>
    4. Data Scraping &amp; Point-of-Capture<br>
    5. Semantic Filtering &amp; RAG Chunking<br>
    6. Signal Classification (E14 Guard)<br>
    7. Evidentiary Weighting (E1–E14, G1–G4)<br>
    8. Risk Surfacing &amp; Summarization (R1–R8)<br>
    9. Deterministic Risk Rating<br><br>
    <strong style="color:#0f172a">Criticality Formula</strong><br>
    <code>C = 0.30D + 0.20P + 0.20R + 0.20O + 0.10V</code>
    </div>
    """, unsafe_allow_html=True)

# ─── Session State Initialization ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []
if "mode_used" not in st.session_state:
    st.session_state.mode_used = ""
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "error_msg" not in st.session_state:
    st.session_state.error_msg = None
if "human_overrides" not in st.session_state:
    st.session_state.human_overrides = {}
if "osint_plans" not in st.session_state:
    st.session_state.osint_plans = OSINTInvestigationPlanner.load_all_saved_plans(pause_if_running=True)
if "filter_tier" not in st.session_state:
    st.session_state.filter_tier = "All"
if "show_raw_json" not in st.session_state:
    st.session_state.show_raw_json = False
if "e2e_reports" not in st.session_state:
    st.session_state.e2e_reports = {}
if "custom_vendors" not in st.session_state:
    st.session_state.custom_vendors = {}
if "system_readiness" not in st.session_state:
    st.session_state.system_readiness = None

# Auto-load case study on startup if empty
if not st.session_state.results and MERIDIAN_DATASET_CSV.exists():
    try:
        st.session_state.results = run_meridian_methodology(MERIDIAN_DATASET_CSV.read_bytes())
        st.session_state.mode_used = "meridian"
        if st.session_state.results:
            st.session_state.selected_id = st.session_state.results[0].vendor_id
    except Exception:
        pass

if reset_btn:
    st.session_state.results = []
    st.session_state.selected_id = None
    st.session_state.human_overrides = {}
    st.session_state.osint_plans = {}
    st.session_state.e2e_reports = {}
    st.session_state.custom_vendors = {}
    st.session_state.system_readiness = None
    st.session_state.filter_tier = "All"
    st.rerun()

if run_btn:
    st.session_state.error_msg = None
    st.session_state.selected_id = None
    st.session_state.human_overrides = {}
    st.session_state.osint_plans = {}
    st.session_state.e2e_reports = {}
    try:
        if "Primary" in engine_mode:
            st.session_state.mode_used = "meridian"
            if data_source == "Meridian Case Study (6 Vendors)":
                st.session_state.results = run_meridian_methodology(MERIDIAN_DATASET_CSV.read_bytes())
            elif uploaded_file:
                st.session_state.results = run_meridian_methodology(uploaded_file.read())
            else:
                st.session_state.error_msg = "Please upload a CSV file."
        else:
            st.session_state.mode_used = "legacy"
            if data_source == "Sample Dataset":
                st.session_state.results = run_legacy_pipeline(LEGACY_SAMPLE_CSV.read_bytes())
            elif uploaded_file:
                st.session_state.results = run_legacy_pipeline(uploaded_file.read())
            else:
                st.session_state.error_msg = "Please upload a CSV file."
        if st.session_state.results:
            st.session_state.selected_id = st.session_state.results[0].vendor_id
    except Exception as e:
        st.session_state.error_msg = str(e)

if st.session_state.error_msg:
    st.error(st.session_state.error_msg)

# ─── Navigation Tabs ──────────────────────────────────────────────────────────
tab_assess, tab_osint, tab_e2e, tab_reports, tab_analytics = st.tabs([
    "Criticality Tiering (Level 1)",
    "OSINT Investigation Planner (Levels 2–3)",
    "End-to-End AI Risk & Governance (Levels 1–9)",
    "Assessment Archive",
    "Portfolio Analytics",
])


# ==============================================================================
# TAB 1: LIVE ASSESSMENT CONSOLE
# ==============================================================================
with tab_assess:
    results = st.session_state.results
    is_meridian = st.session_state.mode_used == "meridian"

    if not results:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🛡️</div>
            <div class="empty-title">No assessment loaded</div>
            <div class="empty-hint">Select a dataset in the sidebar and click <strong>▶ Run</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Stat cards calculation
        crit_count = sum(1 for r in results if (getattr(r, "final_criticality", "") == CriticalityLevel.CRITICAL or getattr(r, "criticality_tier", "") == CriticalityTier.TIER_1))
        high_count = sum(1 for r in results if getattr(r, "final_criticality", "") == CriticalityLevel.HIGH)
        med_count  = sum(1 for r in results if (getattr(r, "final_criticality", "") == CriticalityLevel.MEDIUM or getattr(r, "criticality_tier", "") == CriticalityTier.TIER_2))
        low_count  = sum(1 for r in results if (getattr(r, "final_criticality", "") == CriticalityLevel.LOW or getattr(r, "criticality_tier", "") == CriticalityTier.TIER_3))

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card all">
                <div class="stat-label">Total Assessed</div>
                <div class="stat-value">{len(results)}</div>
                <div class="stat-sub">vendors in scope</div>
            </div>
            <div class="stat-card high">
                <div class="stat-label">Critical / High</div>
                <div class="stat-value">{crit_count + high_count}</div>
                <div class="stat-sub">Comprehensive assessment depth</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-label">Medium Risk</div>
                <div class="stat-value">{med_count}</div>
                <div class="stat-sub">Targeted assessment depth</div>
            </div>
            <div class="stat-card low">
                <div class="stat-label">Low Risk</div>
                <div class="stat-value">{low_count}</div>
                <div class="stat-sub">Lightweight assessment depth</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Filter Pills
        filter_col, action_col = st.columns([7, 3])
        with filter_col:
            selected_filter = st.pills(
                "Filter Tier",
                ["All", "Critical", "High", "Medium", "Low"],
                default=st.session_state.filter_tier,
                label_visibility="collapsed",
            )
            st.session_state.filter_tier = selected_filter or "All"

        with action_col:
            # Action button to archive current run to reports directory
            if st.button("💾 Archive Run to Reports", use_container_width=True, key="btn_archive_live"):
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_filename = f"meridian_assessment_{timestamp_str}.json"
                archive_path = REPORTS_DIR / archive_filename
                
                engine = MeridianCriticalityEngine() if is_meridian else None
                if is_meridian and engine:
                    archive_records = []
                    for r in results:
                        trail = engine.generate_audit_trail(r)
                        h_rev = st.session_state.human_overrides.get(r.vendor_id)
                        if h_rev:
                            trail["human_review"] = h_rev.model_dump()
                            trail["final_criticality"] = h_rev.final_criticality
                        archive_records.append(trail)
                else:
                    archive_records = [r.model_dump(mode="json") for r in results]
                
                with open(archive_path, "w", encoding="utf-8") as f:
                    json.dump(archive_records, f, indent=2)
                st.toast(f"Archived to {archive_filename}!", icon="💾")

        # Two-column layout: Inventory vs Detail
        col_left, col_right = st.columns([5, 5], gap="large")

        with col_left:
            st.markdown('<div class="sec-header"><span>Vendor Inventory</span></div>', unsafe_allow_html=True)

            # Build rows
            rows = []
            for r in results:
                if is_meridian:
                    h_rev = st.session_state.human_overrides.get(r.vendor_id)
                    final_c = h_rev.final_criticality if h_rev else r.final_criticality
                    
                    # Apply tier filter
                    if st.session_state.filter_tier != "All" and str(final_c).lower() != st.session_state.filter_tier.lower():
                        continue
                        
                    rows.append({
                        "ID": r.vendor_id,
                        "Vendor": r.vendor_name,
                        "Criticality": str(final_c),
                        "Base Score": f"{r.base_score:.2f}" if r.base_score is not None else "UNKNOWN",
                        "D": str(r.D.score) if r.D.score is not None else "UNKNOWN",
                        "P": str(r.P.score) if r.P.score is not None else "UNKNOWN",
                        "R": str(r.R.score) if r.R.score is not None else "UNKNOWN",
                        "O": str(r.O.score) if r.O.score is not None else "UNKNOWN",
                        "V": str(r.V.score) if r.V.score is not None else "UNKNOWN",
                        "Safeguards": "Triggered" if r.automatic_override_applied else "None",
                    })
                else:
                    tier_str = str(r.criticality_tier)
                    if st.session_state.filter_tier != "All" and st.session_state.filter_tier.lower() not in tier_str.lower():
                        continue
                    rows.append({
                        "ID": r.vendor_id,
                        "Vendor": r.vendor_name,
                        "Criticality": tier_str,
                        "Score": f"{r.criticality_score:.2f}",
                        "Depth": str(r.assessment_depth),
                    })

            if not rows:
                st.info(f"No vendors match filter: '{st.session_state.filter_tier}'")
            else:
                df = pd.DataFrame(rows)

                def _crit_style(val: str) -> str:
                    s = str(val).lower()
                    if "critical" in s or "tier_1" in s:
                        return "background-color:#fee2e2;color:#991b1b;font-weight:700;"
                    if "high" in s:
                        return "background-color:#fff7ed;color:#c2410c;font-weight:600;"
                    if "medium" in s or "tier_2" in s:
                        return "background-color:#fffbeb;color:#92400e;font-weight:600;"
                    return "background-color:#f0fdf4;color:#166534;font-weight:600;"

                styled = df.style.map(_crit_style, subset=["Criticality"])
                ev = st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="live_inventory_table",
                )

                if ev.selection and ev.selection.rows:
                    selected_idx = ev.selection.rows[0]
                    st.session_state.selected_id = rows[selected_idx]["ID"]

            # Download full audit trail
            export_engine = MeridianCriticalityEngine() if is_meridian else None
            if is_meridian and export_engine:
                full_export = []
                for r in results:
                    trail = export_engine.generate_audit_trail(r)
                    h_rev = st.session_state.human_overrides.get(r.vendor_id)
                    if h_rev:
                        trail["human_review"] = h_rev.model_dump()
                        trail["final_criticality"] = h_rev.final_criticality
                    full_export.append(trail)
            else:
                full_export = [r.model_dump(mode="json") for r in results]

            json_out = json.dumps(full_export, indent=2)
            st.download_button(
                "⬇ Download Complete Audit JSON",
                data=json_out,
                file_name="meridian_assessment_audit.json",
                mime="application/json",
                use_container_width=True,
                key="btn_download_full_json",
            )

        with col_right:
            sel_id = st.session_state.selected_id
            sel = next((r for r in results if r.vendor_id == sel_id), None)

            if sel is None:
                st.markdown('<div class="sec-header"><span>Vendor Dossier</span></div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">←</div>
                    <div class="empty-title">Select a vendor to inspect</div>
                    <div class="empty-hint">Click any row in the inventory to review factor provenance, safeguards, and sign off governance.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if is_meridian:
                    r: CriticalityResult = sel
                    user_override = st.session_state.human_overrides.get(r.vendor_id)
                    current_final_crit = user_override.final_criticality if user_override else r.final_criticality

                    # Panel Quick Actions Toolbar
                    btn_c1, btn_c2, btn_c3 = st.columns([3, 4, 3])
                    with btn_c1:
                        # Raw JSON toggle
                        if st.button("📋 Raw JSON", use_container_width=True, key=f"btn_toggle_raw_{r.vendor_id}"):
                            st.session_state.show_raw_json = not st.session_state.show_raw_json
                    with btn_c2:
                        # Download Single Dossier
                        single_engine = MeridianCriticalityEngine()
                        single_trail = single_engine.generate_audit_trail(r)
                        if user_override:
                            single_trail["human_review"] = user_override.model_dump()
                            single_trail["final_criticality"] = user_override.final_criticality
                        st.download_button(
                            "⬇ Export Dossier",
                            data=json.dumps(single_trail, indent=2),
                            file_name=f"{r.vendor_id}_{r.vendor_name.replace(' ', '_')}.json",
                            mime="application/json",
                            use_container_width=True,
                            key=f"btn_dl_dossier_{r.vendor_id}",
                        )
                    with btn_c3:
                        if user_override:
                            if st.button("🔄 Reset Review", use_container_width=True, key=f"btn_clear_ov_{r.vendor_id}"):
                                del st.session_state.human_overrides[r.vendor_id]
                                st.rerun()

                    # Collapsible Raw JSON view
                    if st.session_state.show_raw_json:
                        with st.expander("Audit Trail JSON Inspector", expanded=True):
                            st.json(single_trail)

                    # 1. Vendor Header
                    st.markdown(f"""
                    <div class="detail-header">
                        <div class="detail-vendor-name">{r.vendor_name} ({r.vendor_id})</div>
                        <div class="detail-meta">
                            {pill(current_final_crit or 'Low')}
                            <span class="detail-tag">Base: <strong>{r.base_score if r.base_score is not None else 'UNKNOWN'}</strong></span>
                            <span class="detail-tag">Status: <strong>{r.score_status.value}</strong></span>
                            <span class="detail-tag">Depth: <strong>{r.assessment_depth}</strong></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 2. Factor Scores (D / P / R / O / V)
                    st.markdown('<div class="sec-header"><span>1. Factor Scores &amp; Provenance</span></div>', unsafe_allow_html=True)
                    factors_to_show = [
                        ("D — Data Sensitivity", r.D),
                        ("P — Payment Flow Exposure", r.P),
                        ("R — Regulatory Exposure", r.R),
                        ("O — Operational Dependency", r.O),
                        ("V — Annual Data Volume", r.V),
                    ]
                    for label, f in factors_to_show:
                        contrib_str = f"{f.weighted_contribution:.2f}" if f.weighted_contribution is not None else "N/A"
                        weight_str = f"{int(f.weight * 100)}%" if f.weight is not None else ""
                        st.markdown(f"""
                        <div class="factor-card">
                            <div class="factor-header">
                                <span class="factor-title">{label} ({weight_str})</span>
                                {render_score_meter(f.score, 3)}
                            </div>
                            <div class="factor-rationale">{f.rationale}</div>
                            <div class="factor-meta">
                                Method: <code>{f.determination_method.value}</code> &nbsp;·&nbsp; Contrib: <strong>{contrib_str}</strong> &nbsp;·&nbsp; Input: "{f.raw_input or 'None'}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 3. Weighted Calculation
                    st.markdown('<div class="sec-header"><span>2. Weighted Calculation</span></div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="factor-card" style="font-size:0.78rem;line-height:1.6">
                        <strong>Formula:</strong> <code>0.30D + 0.20P + 0.20R + 0.20O + 0.10V</code><br>
                        <strong>Base Score:</strong> <strong>{r.base_score}</strong> &nbsp;·&nbsp;
                        <strong>Provisional Criticality:</strong> {pill(r.provisional_criticality or 'Low')}
                    </div>
                    """, unsafe_allow_html=True)

                    # 4. Automatic Safeguards (O1–O4)
                    st.markdown('<div class="sec-header"><span>3. Automatic Safeguards (O1–O4)</span></div>', unsafe_allow_html=True)
                    for o in r.overrides_evaluated:
                        trig_cls = "triggered" if o.triggered else ""
                        status_badge = f"<strong style='color:#c2410c;'>TRIGGERED (Floor: {o.resulting_floor})</strong>" if o.triggered else "<span style='color:#64748b;'>Cleared</span>"
                        st.markdown(f"""
                        <div class="override-box {trig_cls}">
                            <strong>[{o.override_id}] {o.description}</strong> — {status_badge}<br>
                            <span style="color:#64748b">Condition: {o.condition_evaluated} &nbsp;·&nbsp; Evidence: {o.evidence}</span><br>
                            <span style="color:#334155">{o.rationale}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="font-size:0.8rem;margin:0.4rem 0 0.8rem">
                        <strong>Proposed Criticality (Post-Safeguards):</strong> {pill(r.proposed_criticality or 'Low')}
                    </div>
                    """, unsafe_allow_html=True)

                    # 5. Governance & Human Review (O5)
                    st.markdown('<div class="sec-header"><span>4. Human Review &amp; Governance (O5)</span></div>', unsafe_allow_html=True)
                    with st.container(border=True):
                        choice = st.radio(
                            "Governance Action",
                            [f"Confirm Proposed ({r.proposed_criticality})", "Change Criticality"],
                            horizontal=True,
                            key=f"gov_choice_{r.vendor_id}",
                        )

                        if "Change" in choice:
                            new_val = st.selectbox(
                                "Override Tier Selection",
                                [CriticalityLevel.LOW, CriticalityLevel.MEDIUM, CriticalityLevel.HIGH, CriticalityLevel.CRITICAL],
                                index=2,
                                key=f"new_crit_{r.vendor_id}",
                            )
                            rationale_input = st.text_area(
                                "Mandatory Written Rationale",
                                placeholder="State regulatory justification or committee approval rationale...",
                                key=f"gov_rat_{r.vendor_id}",
                            )
                            if st.button("Apply Controlled Override", type="primary", key=f"btn_save_gov_{r.vendor_id}"):
                                if not rationale_input.strip():
                                    st.error("A written rationale is required when altering the automated criticality.")
                                else:
                                    st.session_state.human_overrides[r.vendor_id] = HumanReviewRecord(
                                        user_decision="CHANGE_CRITICALITY",
                                        original_criticality=r.proposed_criticality or "Low",
                                        final_criticality=new_val,
                                        rationale=rationale_input.strip(),
                                    )
                                    existing_plan = st.session_state.osint_plans.get(r.vendor_id)
                                    if existing_plan is not None:
                                        updated_plan = OSINTInvestigationPlanner().plan_investigation(
                                            vendor=r,
                                            criticality_input=r,
                                            human_review=st.session_state.human_overrides[r.vendor_id],
                                            existing_plan=existing_plan,
                                        )
                                        st.session_state.osint_plans[r.vendor_id] = updated_plan
                                        OSINTInvestigationPlanner.save_plan_state(updated_plan)
                                    st.success(f"Overridden to {new_val} by analyst.")
                                    st.rerun()
                        else:
                            if st.button("Confirm Proposed Criticality", key=f"btn_confirm_gov_{r.vendor_id}"):
                                st.session_state.human_overrides[r.vendor_id] = HumanReviewRecord(
                                    user_decision="KEEP_PROPOSED",
                                    original_criticality=r.proposed_criticality or "Low",
                                    final_criticality=r.proposed_criticality or "Low",
                                    rationale="Analyst confirmed proposed automated criticality.",
                                )
                                existing_plan = st.session_state.osint_plans.get(r.vendor_id)
                                if existing_plan is not None:
                                    updated_plan = OSINTInvestigationPlanner().plan_investigation(
                                        vendor=r,
                                        criticality_input=r,
                                        human_review=st.session_state.human_overrides[r.vendor_id],
                                        existing_plan=existing_plan,
                                    )
                                    st.session_state.osint_plans[r.vendor_id] = updated_plan
                                    OSINTInvestigationPlanner.save_plan_state(updated_plan)
                                st.info("Automated criticality confirmed.")
                                st.rerun()

                        if user_override:
                            st.markdown(f"""
                            <div style="font-size:0.75rem;background:#f1f5f9;border-radius:6px;padding:0.45rem 0.65rem;margin-top:0.4rem">
                                <strong>Logged Decision:</strong> <code>{user_override.user_decision}</code> &nbsp;·&nbsp;
                                Final: <strong>{user_override.final_criticality}</strong><br>
                                <em>"{user_override.rationale}"</em>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Legacy qualitative view
                    a: CriticalityAssessment = sel
                    st.markdown(f"""
                    <div class="detail-header">
                        <div class="detail-vendor-name">{a.vendor_name}</div>
                        <div class="detail-meta">
                            {pill(str(a.criticality_tier))}
                            <span class="detail-tag">Score: <strong>{a.criticality_score:.2f}</strong></span>
                            <span class="detail-tag">{DEPTH_LABEL.get(a.assessment_depth, str(a.assessment_depth))}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="sec-header"><span>Assessment Reasoning</span></div>', unsafe_allow_html=True)
                    for reason in a.reasoning:
                        st.markdown(f"• {reason}")


# ==============================================================================
# TAB 2: OSINT INVESTIGATION PLANNER
# ==============================================================================
with tab_osint:
    st.markdown('<div class="sec-header"><span>OSINT Depth &amp; Investigation Planner</span></div>', unsafe_allow_html=True)

    # Gather available vendors
    active_vendors = st.session_state.results
    vendor_lookup = {}
    if active_vendors:
        for v in active_vendors:
            vendor_lookup[f"{v.vendor_name} ({v.vendor_id})"] = v
    else:
        # Load from meridian_vendors.csv if no active run
        if MERIDIAN_DATASET_CSV.exists():
            loader = MeridianCSVVendorLoader()
            for v in loader.load(MERIDIAN_DATASET_CSV):
                vendor_lookup[f"{v.vendor_name} ({v.vendor_id})"] = v

    if not vendor_lookup:
        st.info("No vendors available. Please load a dataset in the sidebar.")
    else:
        # Vendor selector
        default_idx = 0
        if st.session_state.selected_id:
            for idx, k in enumerate(vendor_lookup.keys()):
                if f"({st.session_state.selected_id})" in k:
                    default_idx = idx
                    break

        selected_v_key = st.selectbox(
            "Select Vendor to Inspect OSINT Plan",
            options=list(vendor_lookup.keys()),
            index=default_idx,
            key="osint_vendor_selector",
        )

        selected_vendor_obj = vendor_lookup[selected_v_key]
        v_id = getattr(selected_vendor_obj, "vendor_id", "UNKNOWN")

        # Determine criticality and human override if available
        osint_crit = None
        osint_h_rev = None
        if isinstance(selected_vendor_obj, CriticalityResult):
            osint_crit = selected_vendor_obj
            osint_h_rev = st.session_state.human_overrides.get(v_id)
        elif hasattr(selected_vendor_obj, "vendor_id"):
            osint_h_rev = st.session_state.human_overrides.get(v_id)
            for r in st.session_state.results:
                if getattr(r, "vendor_id", None) == v_id:
                    osint_crit = r
                    break

        planner = OSINTInvestigationPlanner()
        eff_crit, eff_status = planner.resolve_effective_criticality(osint_crit, osint_h_rev)

        cached_plan = st.session_state.osint_plans.get(v_id)
        if cached_plan is None:
            cached_plan = OSINTInvestigationPlanner.load_plan_state(v_id, pause_if_running=True)
            if cached_plan is not None:
                st.session_state.osint_plans[v_id] = cached_plan

        if (
            cached_plan is None
            or cached_plan.input_criticality != eff_crit
            or cached_plan.approval_status != eff_status
        ):
            cached_plan = planner.plan_investigation(
                vendor=selected_vendor_obj,
                criticality_input=osint_crit,
                human_review=osint_h_rev,
                existing_plan=cached_plan,
            )
            st.session_state.osint_plans[v_id] = cached_plan
            OSINTInvestigationPlanner.save_plan_state(cached_plan)

        plan = cached_plan
        if plan.timebox.timer_state == TimerState.RUNNING:
            OSINTCoverageTracker.refresh_timer(plan)
            OSINTInvestigationPlanner.save_plan_state(plan)

        # ── Executive Header ──
        approval_badge_color = "#16a34a" if plan.approval_status == CriticalityApprovalStatus.APPROVED_BY_ANALYST else "#d97706"
        approval_bg = "#f0fdf4" if plan.approval_status == CriticalityApprovalStatus.APPROVED_BY_ANALYST else "#fffbeb"
        st.markdown(f"""
        <div class="detail-header" style="margin-top:0.5rem">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div class="detail-vendor-name">{plan.vendor_name} ({plan.vendor_id})</div>
                    <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.4rem">
                        Domain: <code>{plan.domain or 'No domain provided'}</code> &nbsp;·&nbsp;
                        Service: <em>"{plan.service_product_provided or 'N/A'}"</em>
                    </div>
                </div>
                <div style="text-align:right">
                    <span style="font-size:0.7rem;font-weight:700;color:{approval_badge_color};background:{approval_bg};padding:0.25rem 0.6rem;border-radius:20px;border:1px solid {approval_badge_color}33">
                        {plan.approval_status.value.replace('_', ' ')}
                    </span>
                </div>
            </div>
            <div class="detail-meta">
                {pill(plan.input_criticality)}
                <span class="detail-tag">OSINT Depth: <strong>{plan.selected_depth_profile}</strong></span>
                <span class="detail-tag">POC Timebox: <strong>{plan.timebox.planning_target}</strong> (Effort: {plan.timebox.actual_analyst_effort_minutes:.1f}m · Wall: {plan.timebox.elapsed_wall_clock_minutes:.1f}m · {plan.timebox.timer_state.value})</span>
                <span class="detail-tag">Reviewers: <strong>{'Analyst + 2nd Reviewer' if plan.reviewer_requirements.get('second_reviewer') else 'Primary Analyst'}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI Stat Cards ──
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card all">
                <div class="stat-label">Required Sources</div>
                <div class="stat-value">{len(plan.required_source_classes)}</div>
                <div class="stat-sub">+{len(plan.conditional_source_classes)} conditional · {len(plan.corroborative_source_classes)} corroborative</div>
            </div>
            <div class="stat-card high">
                <div class="stat-label">Planned Queries</div>
                <div class="stat-value">{len(plan.planned_queries)}</div>
                <div class="stat-sub">deduplicated templates</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-label">Discovery Tools</div>
                <div class="stat-value">{len(plan.associated_discovery_resources)}</div>
                <div class="stat-sub">mapped research resources</div>
            </div>
            <div class="stat-card low">
                <div class="stat-label">POC Timebox</div>
                <div class="stat-value">{plan.timebox.planned_minutes:.0f}m</div>
                <div class="stat-sub">Effort: {plan.timebox.actual_analyst_effort_minutes:.1f}m · Wall: {plan.timebox.elapsed_wall_clock_minutes:.1f}m</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-Column Layout ──
        osint_left, osint_right = st.columns([5, 5], gap="large")

        with osint_left:
            # 1. Stopping Condition Monitor
            st.markdown('<div class="sec-header"><span>1. Rulebook Stopping Condition &amp; POC Timebox</span></div>', unsafe_allow_html=True)
            stop_badge_color = (
                "#16a34a" if plan.stop_status == AssessmentStopStatus.COMPLETED_CONDITION_MET
                else "#dc2626" if plan.stop_status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE
                else "#2563eb"
            )
            stop_badge_bg = (
                "#f0fdf4" if plan.stop_status == AssessmentStopStatus.COMPLETED_CONDITION_MET
                else "#fef2f2" if plan.stop_status == AssessmentStopStatus.TIMEBOX_EXHAUSTED_INCOMPLETE
                else "#eff6ff"
            )
            st.markdown(f"""
            <div class="factor-card" style="line-height:1.5">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem">
                    <strong style="color:#0f172a;font-size:0.8rem">[{plan.stopping_rule_id}]</strong>
                    <span style="font-size:0.7rem;font-weight:700;color:{stop_badge_color};background:{stop_badge_bg};padding:0.15rem 0.5rem;border-radius:4px">
                        {plan.stop_status.value.replace('_', ' ')}
                    </span>
                </div>
                <div style="font-size:0.76rem;color:#334155;margin-bottom:0.35rem">
                    {plan.stopping_rule_description}
                </div>
                <div style="font-size:0.73rem;color:#0f172a;background:#f8fafc;padding:0.35rem 0.55rem;border-radius:4px;margin-bottom:0.35rem">
                    <strong>Evaluator Status:</strong> {plan.stop_progress_notes}
                </div>
                <div style="font-size:0.7rem;color:#64748b;border-top:1px solid #f1f5f9;padding-top:0.35rem">
                    <strong>Planned Timebox:</strong> {plan.timebox.planning_target} ({plan.timebox.planned_minutes:.0f} min) &nbsp;·&nbsp;
                    <strong>Active Analyst Effort (Controlling):</strong> {plan.timebox.actual_analyst_effort_minutes:.2f} min &nbsp;·&nbsp;
                    <strong>Elapsed Wall-Clock:</strong> {plan.timebox.elapsed_wall_clock_minutes:.2f} min &nbsp;·&nbsp;
                    <strong>Timer State:</strong> <code>{plan.timebox.timer_state.value}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Active Timer Controls (Start / Pause / Resume / Finish)
            tm_col1, tm_col2, tm_col3 = st.columns(3)
            with tm_col1:
                if plan.timebox.timer_state == TimerState.NOT_STARTED:
                    if st.button("▶ Start Timer", key=f"btn_tm_start_{v_id}", use_container_width=True, type="primary"):
                        OSINTCoverageTracker.start_timer(plan)
                        OSINTInvestigationPlanner.save_plan_state(plan)
                        st.rerun()
                elif plan.timebox.timer_state in (TimerState.PAUSED, TimerState.STOPPED):
                    if st.button("▶ Resume Timer", key=f"btn_tm_resume_{v_id}", use_container_width=True, type="primary"):
                        OSINTCoverageTracker.resume_timer(plan)
                        OSINTInvestigationPlanner.save_plan_state(plan)
                        st.rerun()
                else:
                    if st.button("🔄 Refresh Elapsed", key=f"btn_tm_refresh_{v_id}", use_container_width=True):
                        OSINTCoverageTracker.refresh_timer(plan)
                        OSINTInvestigationPlanner.save_plan_state(plan)
                        st.rerun()
            with tm_col2:
                if st.button(
                    "⏸ Pause Timer",
                    key=f"btn_tm_pause_{v_id}",
                    use_container_width=True,
                    disabled=(plan.timebox.timer_state != TimerState.RUNNING),
                ):
                    OSINTCoverageTracker.pause_timer(plan)
                    OSINTInvestigationPlanner.save_plan_state(plan)
                    st.rerun()
            with tm_col3:
                if st.button(
                    "⏹ Finish Timer",
                    key=f"btn_tm_stop_{v_id}",
                    use_container_width=True,
                    disabled=(plan.timebox.timer_state in (TimerState.NOT_STARTED, TimerState.STOPPED)),
                ):
                    OSINTCoverageTracker.stop_timer(plan)
                    OSINTInvestigationPlanner.save_plan_state(plan)
                    st.rerun()

            with st.expander("⏱️ Manual Timebox & Effort Adjustment", expanded=False):
                tb_c1, tb_c2 = st.columns(2)
                with tb_c1:
                    new_wall_min = st.number_input(
                        "Elapsed Wall-Clock (minutes)",
                        min_value=0.0,
                        value=float(plan.timebox.elapsed_wall_clock_minutes),
                        step=1.0,
                        key=f"tb_wall_{v_id}",
                    )
                with tb_c2:
                    new_effort_min = st.number_input(
                        "Actual Analyst Effort (minutes)",
                        min_value=0.0,
                        value=float(plan.timebox.actual_analyst_effort_minutes),
                        step=1.0,
                        key=f"tb_effort_{v_id}",
                    )
                if st.button("Update Timebox & Evaluate Stopping Rule", key=f"btn_tb_{v_id}", use_container_width=True):
                    OSINTCoverageTracker.update_plan_timebox(
                        plan,
                        elapsed_wall_clock_minutes=new_wall_min,
                        actual_analyst_effort_minutes=new_effort_min,
                    )
                    OSINTInvestigationPlanner.save_plan_state(plan)
                    st.rerun()

            # 2. Downstream G1–G4 Investigation Checklist
            st.markdown('<div class="sec-header"><span>2. Downstream G1–G4 Gates Checklist</span></div>', unsafe_allow_html=True)
            for g in plan.gate_checklist:
                st.markdown(f"""
                <div class="factor-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem">
                        <strong style="font-size:0.78rem;color:#0f172a">[{g.gate_id}] {g.title}</strong>
                        <span style="font-size:0.68rem;font-weight:600;color:#64748b;background:#f8fafc;padding:0.15rem 0.45rem;border-radius:4px;border:1px solid #e2e8f0">
                            {g.status.value.replace('_', ' ')}
                        </span>
                    </div>
                    <div style="font-size:0.75rem;color:#334155;margin-bottom:0.35rem">
                        <em>"{g.question}"</em>
                    </div>
                    <div style="font-size:0.7rem;color:#64748b">
                        <strong>Objectives:</strong> {', '.join(g.investigative_objectives[:2])}<br>
                        <strong>Qualifying Criteria:</strong> {g.qualifying_criteria}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 Update G1–G4 Gate Status", expanded=False):
                gate_ids = [g.gate_id for g in plan.gate_checklist]
                sel_gate_id = st.selectbox("Select Gate", options=gate_ids, key=f"sel_gate_{v_id}")
                target_gate = next(g for g in plan.gate_checklist if g.gate_id == sel_gate_id)
                gate_states = [s.value for s in GateStatus]
                cur_gate_idx = gate_states.index(target_gate.status.value)
                new_gate_state = st.selectbox(
                    "Gate Status",
                    options=gate_states,
                    index=cur_gate_idx,
                    key=f"sel_gate_status_{v_id}_{sel_gate_id}",
                )
                if st.button("Save Gate Status", key=f"btn_save_gate_{v_id}", use_container_width=True):
                    OSINTCoverageTracker.update_plan_gate(
                        plan,
                        gate_id=sel_gate_id,
                        new_status=GateStatus(new_gate_state),
                    )
                    OSINTInvestigationPlanner.save_plan_state(plan)
                    st.rerun()

        with osint_right:
            # 1. Planned Source Classes & Interactive Coverage Tracker
            st.markdown('<div class="sec-header"><span>3. Unified Source Coverage &amp; Discovery Catalog</span></div>', unsafe_allow_html=True)

            with st.expander("📝 Record Source Coverage Outcome", expanded=False):
                source_options = list(plan.coverage_records.keys())
                sel_src_id = st.selectbox(
                    "Source Class",
                    options=source_options,
                    format_func=lambda sid: f"[{sid}] {plan.coverage_records[sid].source_name}",
                    key=f"cov_src_sel_{v_id}",
                )
                cur_rec = plan.coverage_records[sel_src_id]
                cov_states = [s.value for s in CoverageState]
                cur_cov_idx = cov_states.index(cur_rec.status.value)
                new_cov_state = st.selectbox(
                    "Coverage Outcome",
                    options=cov_states,
                    index=cur_cov_idx,
                    key=f"cov_state_sel_{v_id}_{sel_src_id}",
                )
                c_url1, c_url2 = st.columns(2)
                with c_url1:
                    cand_urls = st.number_input(
                        "Candidate URLs Found",
                        min_value=0,
                        value=int(cur_rec.candidate_urls_count),
                        step=1,
                        key=f"cov_cand_{v_id}_{sel_src_id}",
                    )
                with c_url2:
                    rev_urls = st.number_input(
                        "URLs Reviewed",
                        min_value=0,
                        value=int(cur_rec.reviewed_urls_count),
                        step=1,
                        key=f"cov_rev_{v_id}_{sel_src_id}",
                    )
                prod_ev = st.checkbox(
                    "Produced New Qualifying Evidence",
                    value=bool(cur_rec.produced_new_evidence),
                    key=f"cov_prod_{v_id}_{sel_src_id}",
                )
                ev_refs_raw = st.text_input(
                    "Evidence References (URLs or IDs, comma-separated)",
                    value=", ".join(cur_rec.evidence_references),
                    placeholder="https://vendor.com/trust, EV-001",
                    key=f"cov_refs_{v_id}_{sel_src_id}",
                )
                cov_notes = st.text_input(
                    "Analyst Coverage Notes",
                    value=cur_rec.notes,
                    placeholder="Reason if unavailable, or summary of findings...",
                    key=f"cov_notes_{v_id}_{sel_src_id}",
                )
                if st.button("Save Source Coverage Outcome", type="primary", key=f"btn_save_cov_{v_id}", use_container_width=True):
                    parsed_refs = [x.strip() for x in ev_refs_raw.split(",") if x.strip()]
                    tracker = OSINTCoverageTracker(plan.coverage_records)
                    tracker.update_source_status(
                        source_id=sel_src_id,
                        new_state=CoverageState(new_cov_state),
                        candidate_urls=int(cand_urls),
                        reviewed_urls=int(rev_urls),
                        produced_new_evidence=bool(prod_ev),
                        evidence_references=parsed_refs,
                        notes=cov_notes.strip(),
                    )
                    tracker.sync_plan_stop_status(plan)
                    OSINTInvestigationPlanner.save_plan_state(plan)
                    st.rerun()

            with st.expander(f"View All {len(plan.coverage_records)} Planned Source Classes", expanded=True):
                for s_id, s_rec in plan.coverage_records.items():
                    s_def = planner.source_registry.get_source(s_id)
                    grade = s_def.default_evidence_grade if s_def else "C"
                    rel = s_def.reliability_rating if s_def else 3
                    nature = s_def.source_nature if s_def else "HYBRID"
                    if s_rec.is_required:
                        req_badge = "<span style='color:#dc2626;font-weight:700'>REQUIRED</span>"
                    elif s_rec.is_conditional:
                        req_badge = "<span style='color:#d97706;font-weight:700'>CONDITIONAL</span>"
                    else:
                        req_badge = "<span style='color:#64748b'>CORROBORATIVE</span>"

                    state_color = (
                        "#16a34a" if s_rec.status in (CoverageState.REVIEWED_EVIDENCE_RECORDED, CoverageState.REVIEWED_NO_EVIDENCE)
                        else "#d97706" if s_rec.status in (CoverageState.SEARCH_ATTEMPTED, CoverageState.REQUIRES_MANUAL_REVIEW)
                        else "#dc2626" if s_rec.status == CoverageState.UNAVAILABLE
                        else "#64748b"
                    )
                    cond_line = f"<div style='color:#b45309;font-size:0.68rem;margin-top:0.15rem'><em>Trigger: {s_rec.condition}</em></div>" if s_rec.condition else ""
                    refs_line = f"<div style='color:#15803d;font-size:0.68rem;margin-top:0.15rem'><strong>Evidence Refs:</strong> {', '.join(s_rec.evidence_references)}</div>" if s_rec.evidence_references else ""
                    notes_line = f"<div style='color:#334155;font-size:0.68rem;margin-top:0.15rem'><strong>Notes:</strong> {s_rec.notes}</div>" if s_rec.notes else ""
                    st.markdown(f"""
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:0.55rem 0.75rem;margin-bottom:0.4rem;font-size:0.75rem">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <strong>[{s_id}] {s_rec.source_name}</strong>
                            <span style="font-size:0.68rem">Grade {grade} (★{rel}) &nbsp;·&nbsp; {req_badge} &nbsp;·&nbsp; <span style="color:{state_color};font-weight:700">{s_rec.status.value}</span></span>
                        </div>
                        <div style="color:#64748b;font-size:0.7rem;margin-top:0.2rem">
                            Nature: <code>{nature}</code> &nbsp;·&nbsp; Tools: {', '.join(s_def.discovery_resources[:4]) if s_def else 'General Search'}
                        </div>
                        {cond_line}{refs_line}{notes_line}
                    </div>
                    """, unsafe_allow_html=True)

            # 2. Targeted Search Queries & Bounded Search Execution
            st.markdown('<div class="sec-header"><span>4. Generated Search Queries &amp; Bounded Search Execution</span></div>', unsafe_allow_html=True)

            q_planned = sum(1 for q in plan.planned_queries if q.execution_status == SearchExecutionStatus.PLANNED)
            q_executed = sum(1 for q in plan.planned_queries if q.execution_status == SearchExecutionStatus.EXECUTED)
            q_failed = sum(1 for q in plan.planned_queries if q.execution_status == SearchExecutionStatus.FAILED)
            q_unavail = sum(1 for q in plan.planned_queries if q.execution_status == SearchExecutionStatus.UNAVAILABLE)

            active_backend = build_search_backend_from_env()
            backend_ok, backend_msg = active_backend.is_configured()

            st.caption(
                f"{len(plan.planned_queries)} queries total · "
                f"PLANNED: {q_planned} · EXECUTED: {q_executed} · FAILED: {q_failed} · UNAVAILABLE: {q_unavail} · "
                f"Candidate URLs: {len(plan.candidate_urls)} | "
                f"Backend ({active_backend.provider_name}): {'Ready' if backend_ok else 'Unconfigured'}"
            )

            exec_col1, exec_col2 = st.columns([6, 4])
            with exec_col1:
                retry_failed_opt = st.checkbox(
                    "Retry FAILED / UNAVAILABLE queries",
                    value=False,
                    key=f"chk_retry_search_{v_id}",
                )
            with exec_col2:
                if st.button(
                    "🔍 Execute Planned Searches",
                    key=f"btn_exec_searches_{v_id}",
                    use_container_width=True,
                    type="primary",
                    disabled=(q_planned == 0 and not (retry_failed_opt and (q_failed > 0 or q_unavail > 0))),
                ):
                    collector = OSINTSearchCollector(backend=active_backend)
                    collector.execute_planned_queries(
                        plan,
                        retry_failed=retry_failed_opt,
                        retry_unavailable=retry_failed_opt,
                        persist=True,
                    )
                    st.rerun()

            if not backend_ok and q_unavail > 0:
                st.warning(backend_msg)

            query_search_term = st.text_input("Filter Queries", placeholder="Filter by keyword (e.g. Bedrock, DPA, LLM)...", label_visibility="collapsed")
            
            displayed_queries = plan.planned_queries
            if query_search_term.strip():
                displayed_queries = [
                    q for q in displayed_queries
                    if query_search_term.lower() in q.rendered_query.lower() or any(query_search_term.lower() in s.lower() for s in q.source_classes)
                ]

            with st.container(height=320):
                for q in displayed_queries:
                    q_status_color = (
                        "#16a34a" if q.execution_status == SearchExecutionStatus.EXECUTED
                        else "#dc2626" if q.execution_status == SearchExecutionStatus.FAILED
                        else "#d97706" if q.execution_status == SearchExecutionStatus.UNAVAILABLE
                        else "#64748b"
                    )
                    exec_meta_parts = []
                    if q.provider:
                        exec_meta_parts.append(f"Provider: <code>{q.provider}</code>")
                    if q.executed_at:
                        exec_meta_parts.append(f"Results: <strong>{q.result_count}</strong>")
                        exec_meta_parts.append(f"At: {q.executed_at[:19].replace('T', ' ')} UTC")
                    exec_meta_line = (
                        f"<div style='font-size:0.66rem;color:#475569;margin-top:0.2rem'>{' &nbsp;·&nbsp; '.join(exec_meta_parts)}</div>"
                        if exec_meta_parts else ""
                    )
                    err_line = (
                        f"<div style='font-size:0.66rem;color:#dc2626;margin-top:0.15rem'><strong>Error:</strong> {q.error_message}</div>"
                        if q.error_message else ""
                    )
                    st.markdown(f"""
                    <div class="query-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem">
                            <span style="font-size:0.68rem;font-weight:700;color:#2563eb">[{', '.join(q.source_classes)}]</span>
                            <span style="font-size:0.65rem;font-weight:700;color:{q_status_color}">STATUS: {q.execution_status.value}</span>
                        </div>
                        <code>{q.rendered_query}</code>
                        <div style="font-size:0.68rem;color:#64748b;margin-top:0.25rem">
                            Targets: {', '.join(q.expected_evidence_targets[:2])}
                        </div>
                        {exec_meta_line}{err_line}
                    </div>
                    """, unsafe_allow_html=True)

            # 3. Collected Candidate URLs (Unreviewed Candidates)
            with st.expander(f"🔗 Collected Candidate URLs ({len(plan.candidate_urls)} deduplicated — Pending Analyst Review)", expanded=bool(plan.candidate_urls)):
                if not plan.candidate_urls:
                    st.caption("No candidate URLs collected yet. Click '🔍 Execute Planned Searches' with a configured search provider to collect candidate URLs.")
                else:
                    for cand in plan.candidate_urls:
                        src_badges = ", ".join(cand.source_classes or ([cand.source_class] if cand.source_class else []))
                        q_badges = ", ".join(cand.query_ids or ([cand.query_id] if cand.query_id else []))
                        rank_str = f"#{cand.rank}" if cand.rank is not None else "N/A"
                        st.markdown(f"""
                        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:0.5rem 0.7rem;margin-bottom:0.35rem;font-size:0.73rem">
                            <div style="display:flex;justify-content:space-between;align-items:center">
                                <strong>{rank_str} · {cand.title or cand.url}</strong>
                                <span style="font-size:0.65rem;color:#2563eb;font-weight:700">[{src_badges}] · {cand.triage_status.value}</span>
                            </div>
                            <div style="font-size:0.69rem;color:#0f172a;word-break:break-all;margin-top:0.15rem">
                                <code>{cand.url}</code>
                            </div>
                            <div style="font-size:0.68rem;color:#475569;margin-top:0.15rem">
                                {cand.snippet or '<em>No snippet provided</em>'}
                            </div>
                            <div style="font-size:0.64rem;color:#64748b;margin-top:0.2rem">
                                Queries: <code>{q_badges}</code> &nbsp;·&nbsp; Provider: <code>{cand.provider or 'N/A'}</code> &nbsp;·&nbsp; Retrieved: {(cand.retrieved_at or '')[:19].replace('T', ' ')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # 4. Export Plan Action Button
            plan_json = json.dumps(plan.model_dump(mode="json"), indent=2)
            st.download_button(
                "⬇ Export Investigation Plan (JSON)",
                data=plan_json,
                file_name=f"OSINT_Plan_{plan.vendor_id}_{plan.vendor_name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
                key=f"btn_dl_osint_plan_{plan.vendor_id}",
            )


# ==============================================================================
# TAB 3: REPORT BROWSER & ARCHIVE
# ==============================================================================
with tab_reports:
    st.markdown('<div class="sec-header"><span>Report Discovery &amp; Inspection</span></div>', unsafe_allow_html=True)

    # Discover reports on disk
    report_files = sorted(list(REPORTS_DIR.glob("*.json")), key=lambda p: p.stat().st_mtime, reverse=True)

    top_rep_col1, top_rep_col2 = st.columns([7, 3])

    with top_rep_col1:
        report_options = {}
        for rf in report_files:
            mtime = datetime.datetime.fromtimestamp(rf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = rf.stat().st_size / 1024
            try:
                item_count = len(json.loads(rf.read_text(encoding="utf-8")))
            except Exception:
                item_count = "?"
            label = f"{rf.name} ({item_count} vendors · {size_kb:.1f} KB · {mtime})"
            report_options[label] = rf

        selected_report_label = st.selectbox(
            "Select Saved Assessment Report",
            options=list(report_options.keys()),
            index=0 if report_options else None,
            label_visibility="collapsed",
        )

    with top_rep_col2:
        uploaded_report = st.file_uploader("Or Upload JSON Report", type=["json"], label_visibility="collapsed")

    active_report_data: list[dict[str, Any]] = []
    report_source_name = ""

    if uploaded_report:
        try:
            raw_up = json.loads(uploaded_report.read().decode("utf-8"))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
                json.dump(raw_up, tmp)
                active_report_data = load_and_normalize_report(Path(tmp.name))
            report_source_name = uploaded_report.name
        except Exception as e:
            st.error(f"Failed to read uploaded report: {e}")
    elif selected_report_label and report_options:
        target_path = report_options[selected_report_label]
        active_report_data = load_and_normalize_report(target_path)
        report_source_name = target_path.name

    if not active_report_data:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📁</div>
            <div class="empty-title">No reports available</div>
            <div class="empty-hint">Generate an assessment or archive a run to view saved audit reports.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Report Executive Header
        rep_crit = sum(1 for v in active_report_data if "crit" in str(v.get("criticality", "")).lower())
        rep_high = sum(1 for v in active_report_data if "high" in str(v.get("criticality", "")).lower())
        rep_med  = sum(1 for v in active_report_data if "med" in str(v.get("criticality", "")).lower())
        rep_low  = sum(1 for v in active_report_data if "low" in str(v.get("criticality", "")).lower())
        valid_scores = [v["score"] for v in active_report_data if v["score"] is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        st.markdown(f"""
        <div class="stat-grid" style="margin-top:0.5rem">
            <div class="stat-card all">
                <div class="stat-label">Report Archive</div>
                <div class="stat-value">{len(active_report_data)}</div>
                <div class="stat-sub">{report_source_name}</div>
            </div>
            <div class="stat-card high">
                <div class="stat-label">Critical / High</div>
                <div class="stat-value">{rep_crit + rep_high}</div>
                <div class="stat-sub">Elevated exposure</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-label">Medium / Low</div>
                <div class="stat-value">{rep_med + rep_low}</div>
                <div class="stat-sub">Standard monitoring</div>
            </div>
            <div class="stat-card low">
                <div class="stat-label">Average Score</div>
                <div class="stat-value">{avg_score:.2f}</div>
                <div class="stat-sub">Across assessed portfolio</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Visual Graphs for Report ──
        st.markdown('<div class="sec-header"><span>Report Visual Distribution &amp; Risk Profiles</span></div>', unsafe_allow_html=True)
        g_col1, g_col2 = st.columns([4, 6], gap="medium")

        with g_col1:
            # 1. Criticality Tier Breakdown Chart
            tier_df = pd.DataFrame([
                {"Tier": "Critical", "Count": rep_crit},
                {"Tier": "High", "Count": rep_high},
                {"Tier": "Medium", "Count": rep_med},
                {"Tier": "Low", "Count": rep_low},
            ])
            tier_chart = alt.Chart(tier_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=22).encode(
                x=alt.X("Count:Q", title="Number of Vendors", axis=alt.Axis(tickMinStep=1, grid=False)),
                y=alt.Y("Tier:N", sort=["Critical", "High", "Medium", "Low"], title=""),
                color=alt.Color(
                    "Tier:N",
                    scale=alt.Scale(
                        domain=["Critical", "High", "Medium", "Low"],
                        range=["#dc2626", "#ea580c", "#d97706", "#16a34a"],
                    ),
                    legend=None,
                ),
                tooltip=["Tier", "Count"],
            ).properties(title="Portfolio Tier Distribution", height=160)
            st.altair_chart(tier_chart, use_container_width=True)

        with g_col2:
            # 2. Base Score Comparison vs Provisional Thresholds
            score_rows = [
                {"Vendor": v["name"], "Score": v["score"] or 0, "Tier": v["criticality"]}
                for v in active_report_data
            ]
            score_df = pd.DataFrame(score_rows).sort_values("Score", ascending=True)

            score_chart = alt.Chart(score_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18).encode(
                x=alt.X("Score:Q", title="Base Score (0 - 3 scale)", scale=alt.Scale(domain=[0, 3])),
                y=alt.Y("Vendor:N", sort="-x", title=""),
                color=alt.Color(
                    "Tier:N",
                    scale=alt.Scale(
                        domain=["Critical", "High", "Medium", "Low"],
                        range=["#dc2626", "#ea580c", "#d97706", "#16a34a"],
                    ),
                    legend=None,
                ),
                tooltip=["Vendor", "Score", "Tier"],
            ).properties(title="Ranked Vendor Base Scores", height=160)
            st.altair_chart(score_chart, use_container_width=True)

        # ── Interactive Report Table & Deep Dive ──
        st.markdown('<div class="sec-header"><span>Assessed Vendors in Report</span></div>', unsafe_allow_html=True)
        r_table_rows = []
        for v in active_report_data:
            r_table_rows.append({
                "ID": v["id"],
                "Vendor": v["name"],
                "Criticality": v["criticality"],
                "Base Score": f"{v['score']:.2f}" if v["score"] is not None else "N/A",
                "D": v["D"] if v["D"] is not None else "?",
                "P": v["P"] if v["P"] is not None else "?",
                "R": v["R"] if v["R"] is not None else "?",
                "O": v["O"] if v["O"] is not None else "?",
                "V": v["V"] if v["V"] is not None else "?",
                "Safeguards": "Triggered" if v["safeguards_triggered"] else "None",
                "Depth": v.get("depth", ""),
            })

        rep_df = pd.DataFrame(r_table_rows)

        def _rep_crit_style(val: str) -> str:
            s = str(val).lower()
            if "critical" in s or "tier_1" in s:
                return "background-color:#fee2e2;color:#991b1b;font-weight:700;"
            if "high" in s:
                return "background-color:#fff7ed;color:#c2410c;font-weight:600;"
            if "medium" in s or "tier_2" in s:
                return "background-color:#fffbeb;color:#92400e;font-weight:600;"
            return "background-color:#f0fdf4;color:#065f46;font-weight:600;"

        rep_styled = rep_df.style.map(_rep_crit_style, subset=["Criticality"])
        r_ev = st.dataframe(
            rep_styled,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="report_inventory_table",
        )

        # Inspector for selected vendor from report
        selected_rep_idx = r_ev.selection.rows[0] if (r_ev.selection and r_ev.selection.rows) else 0
        if 0 <= selected_rep_idx < len(active_report_data):
            inspected_vendor = active_report_data[selected_rep_idx]
            raw_v = inspected_vendor["raw"]

            st.markdown(f'<div class="sec-header"><span>Vendor Audit Record: {inspected_vendor["name"]} ({inspected_vendor["id"]})</span></div>', unsafe_allow_html=True)
            with st.container(border=True):
                i_col1, i_col2, i_col3 = st.columns(3)
                with i_col1:
                    st.markdown(f"**Final Criticality:** {pill(inspected_vendor['criticality'])}", unsafe_allow_html=True)
                with i_col2:
                    st.markdown(f"**Base Score:** `{'%.2f' % inspected_vendor['score'] if inspected_vendor['score'] is not None else 'N/A'}`")
                with i_col3:
                    st.markdown(f"**Safeguards Triggered:** `{'Yes' if inspected_vendor['safeguards_triggered'] else 'No'}`")

                # If factors dict exists, show rationale
                if "factors" in raw_v:
                    st.markdown("**Factor Provenance:**")
                    f_cols = st.columns(5)
                    for idx, (f_key, f_data) in enumerate(raw_v["factors"].items()):
                        with f_cols[idx]:
                            st.markdown(f"**{f_key}** (Score: `{f_data.get('score', '?')}`)")
                            st.caption(f_data.get("rationale", ""))
                
                with st.expander("View Full Raw Audit JSON for Vendor"):
                    st.json(raw_v)


# ==============================================================================
# TAB 4: PORTFOLIO ANALYTICS & INSIGHTS
# ==============================================================================
with tab_analytics:
    st.markdown('<div class="sec-header"><span>Multi-Factor Risk Analytics</span></div>', unsafe_allow_html=True)

    curr_results = st.session_state.results
    if not curr_results or st.session_state.mode_used != "meridian":
        # Fall back to meridian_diagnostic.json if live session has no meridian data
        diag_path = REPORTS_DIR / "meridian_diagnostic.json"
        if diag_path.exists():
            analytics_data = load_and_normalize_report(diag_path)
        else:
            analytics_data = []
    else:
        analytics_data = []
        for r in curr_results:
            h_rev = st.session_state.human_overrides.get(r.vendor_id)
            final_c = h_rev.final_criticality if h_rev else r.final_criticality
            analytics_data.append({
                "id": r.vendor_id,
                "name": r.vendor_name,
                "criticality": final_c,
                "score": r.base_score,
                "D": r.D.score,
                "P": r.P.score,
                "R": r.R.score,
                "O": r.O.score,
                "V": r.V.score,
                "D_contrib": r.D.weighted_contribution or 0,
                "P_contrib": r.P.weighted_contribution or 0,
                "R_contrib": r.R.weighted_contribution or 0,
                "O_contrib": r.O.weighted_contribution or 0,
                "V_contrib": r.V.weighted_contribution or 0,
                "safeguards_triggered": r.automatic_override_applied,
            })

    if not analytics_data:
        st.info("Run an assessment in Meridian mode to generate portfolio analytics.")
    else:
        # Chart 1: Factor Intensity Heatmap Matrix
        heatmap_rows = []
        for v in analytics_data:
            for factor_key in ["D", "P", "R", "O", "V"]:
                heatmap_rows.append({
                    "Vendor": v["name"],
                    "Factor": factor_key,
                    "Score": v[factor_key] if v[factor_key] is not None else 0,
                })
        hm_df = pd.DataFrame(heatmap_rows)

        base_hm = alt.Chart(hm_df).encode(
            x=alt.X("Factor:N", title="Factor", sort=["D", "P", "R", "O", "V"]),
            y=alt.Y("Vendor:N", title=""),
        )
        rect_hm = base_hm.mark_rect(cornerRadius=3).encode(
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(domain=[0, 3], range=["#f8fafc", "#3b82f6"]),
                title="Score (0-3)",
            ),
            tooltip=["Vendor", "Factor", "Score"],
        )
        text_hm = base_hm.mark_text(baseline="middle", fontWeight="bold").encode(
            text=alt.Text("Score:Q"),
            color=alt.condition(alt.datum.Score > 1.5, alt.value("#ffffff"), alt.value("#0f172a")),
        )
        heatmap_chart = (rect_hm + text_hm).properties(title="Factor Score Heatmap (0 - 3 Scale)", height=220)

        # Chart 2: Stacked Factor Weighted Contribution
        contrib_rows = []
        for v in analytics_data:
            d_val = v.get("D_contrib", (v["D"] or 0) * 0.30)
            p_val = v.get("P_contrib", (v["P"] or 0) * 0.20)
            r_val = v.get("R_contrib", (v["R"] or 0) * 0.20)
            o_val = v.get("O_contrib", (v["O"] or 0) * 0.20)
            v_val = v.get("V_contrib", (v["V"] or 0) * 0.10)
            contrib_rows.extend([
                {"Vendor": v["name"], "Factor": "D (30%)", "Contribution": d_val},
                {"Vendor": v["name"], "Factor": "P (20%)", "Contribution": p_val},
                {"Vendor": v["name"], "Factor": "R (20%)", "Contribution": r_val},
                {"Vendor": v["name"], "Factor": "O (20%)", "Contribution": o_val},
                {"Vendor": v["name"], "Factor": "V (10%)", "Contribution": v_val},
            ])
        contrib_df = pd.DataFrame(contrib_rows)

        stacked_chart = alt.Chart(contrib_df).mark_bar(height=20).encode(
            y=alt.Y("Vendor:N", title=""),
            x=alt.X("Contribution:Q", title="Base Score Contribution", stack="zero"),
            color=alt.Color(
                "Factor:N",
                scale=alt.Scale(
                    domain=["D (30%)", "P (20%)", "R (20%)", "O (20%)", "V (10%)"],
                    range=["#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#8b5cf6"],
                ),
                title="Factor Weight",
            ),
            tooltip=["Vendor", "Factor", alt.Tooltip("Contribution:Q", format=".2f")],
        ).properties(title="Weighted Factor Composition", height=220)

        a_col1, a_col2 = st.columns(2, gap="large")
        with a_col1:
            st.altair_chart(heatmap_chart, use_container_width=True)
        with a_col2:
            st.altair_chart(stacked_chart, use_container_width=True)

        # Safeguard Impact Summary
        st.markdown('<div class="sec-header"><span>Safeguard Protection Impact</span></div>', unsafe_allow_html=True)
        safeguard_vendors = [v for v in analytics_data if v.get("safeguards_triggered")]
        if safeguard_vendors:
            st.markdown(f"**{len(safeguard_vendors)} vendor(s)** elevated to a higher criticality tier by automated policy safeguards (O1–O4):")
            for sv in safeguard_vendors:
                st.markdown(f"• **{sv['name']} ({sv['id']})** — Elevated to **{sv['criticality']}** (Base Score: `{sv['score']:.2f}`)")
        else:
            st.markdown("No vendors in the current dataset required safeguard elevation.")


# ==============================================================================
# TAB 5: END-TO-END AI RISK & GOVERNANCE CONSOLE (LEVELS 1–9)
# ==============================================================================
with tab_e2e:
    # ── 1. Honest Pre-Search & Model Readiness Panel ──
    rd_hdr_c1, rd_hdr_c2 = st.columns([8, 2])
    with rd_hdr_c1:
        st.markdown(
            '<div class="sec-header"><span>System &amp; Integration Readiness (Pre-Search Status)</span></div>',
            unsafe_allow_html=True,
        )
    with rd_hdr_c2:
        if st.button("Refresh Readiness Check", use_container_width=True, key="btn_refresh_readiness"):
            st.session_state.system_readiness = SystemReadinessInspector.check_system_readiness()

    if st.session_state.system_readiness is None:
        st.session_state.system_readiness = SystemReadinessInspector.check_system_readiness()
    readiness = st.session_state.system_readiness

    ollama_cls = "ok" if (readiness.ollama_reachable and readiness.llama3_2_installed and readiness.mistral_installed) else "warn"
    ollama_lbl = (
        "ONLINE · 2/2 Models Ready"
        if (readiness.ollama_reachable and readiness.llama3_2_installed and readiness.mistral_installed)
        else ("ONLINE · Model Missing" if readiness.ollama_reachable else "OFFLINE · Deterministic Fallback")
    )
    routing_cls = "ok" if readiness.llm_routing_enabled_in_env else "warn"
    search_cls = "ok" if readiness.search_configured else "blocked"
    rag_cls = "ok" if readiness.rag_active_mode == "SEMANTIC_VECTOR_RETRIEVAL" else "warn"
    score_cls = "ok" if readiness.scoring_policy_ready else "blocked"

    st.markdown(
        f"""
        <div class="readiness-grid">
            <div class="readiness-card {ollama_cls}">
                <div class="readiness-title">1. Local Ollama Server</div>
                <div class="readiness-status">{ollama_lbl}</div>
                <div class="readiness-detail">
                    Endpoint: <code>{readiness.ollama_endpoint}</code><br>
                    <code>llama3.2:latest</code>: <strong>{'INSTALLED' if readiness.llama3_2_installed else 'MISSING'}</strong> ·
                    <code>mistral:latest</code>: <strong>{'INSTALLED' if readiness.mistral_installed else 'MISSING'}</strong>
                </div>
            </div>
            <div class="readiness-card {routing_cls}">
                <div class="readiness-title">2. Dual-Model Routing (6 GB VRAM)</div>
                <div class="readiness-status">{'ENABLED IN .ENV' if readiness.llm_routing_enabled_in_env else 'DISABLED'}</div>
                <div class="readiness-detail">
                    L3B Triage: <code>{readiness.triage_model}</code><br>
                    L6 &amp; L8 Analysis: <code>{readiness.analysis_model}</code><br>
                    Unload on switch (<code>keep_alive=0</code>): <strong>{'ON' if readiness.unload_on_model_switch else 'OFF'}</strong>
                </div>
            </div>
            <div class="readiness-card {search_cls}">
                <div class="readiness-title">3. External OSINT Search API</div>
                <div class="readiness-status">{'CONFIGURED' if readiness.search_configured else 'NOT CONFIGURED'} ({readiness.search_provider})</div>
                <div class="readiness-detail">
                    {readiness.search_status_message}
                </div>
            </div>
            <div class="readiness-card {rag_cls}">
                <div class="readiness-title">4. Level 5 RAG &amp; Embeddings</div>
                <div class="readiness-status">{'SEMANTIC ACTIVE' if readiness.rag_active_mode == 'SEMANTIC_VECTOR_RETRIEVAL' else 'CHUNK-ONLY PASSTHROUGH'}</div>
                <div class="readiness-detail">
                    <code>MERIDIAN_ENABLE_RAG_EMBEDDINGS={'true' if readiness.rag_embeddings_enabled_in_env else 'false'}</code> ·
                    <code>sentence-transformers</code>: <strong>{'INSTALLED' if readiness.sentence_transformers_installed else 'NOT INSTALLED'}</strong><br>
                    All extracted evidence passes directly to Level 6.
                </div>
            </div>
            <div class="readiness-card {score_cls}">
                <div class="readiness-title">5. Level 9 Risk-Scoring Policy</div>
                <div class="readiness-status">{readiness.scoring_policy_status}</div>
                <div class="readiness-detail">
                    Final calibrated Level 9 numerical risk score is withheld until Meridian's approved scoring policy is supplied.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Vendor Inventory & Custom Single-Vendor Input Flow ──
    meridian_vendor_map: dict[str, MeridianVendor] = {}
    if MERIDIAN_DATASET_CSV.exists():
        for mv in MeridianCSVVendorLoader().load(MERIDIAN_DATASET_CSV):
            meridian_vendor_map[mv.vendor_id] = mv
    for cust_id, cust_v in st.session_state.custom_vendors.items():
        meridian_vendor_map[cust_id] = cust_v

    with st.expander("Single-Vendor Scope Input & Custom Vendor Validator (Add or Edit Vendor Inputs)", expanded=False):
        st.caption(
            "Enter or modify a vendor's contractual scope and D/P/R/O/V inputs. Unknown fields are preserved as "
            "`UNKNOWN` (never coerced to zero) and evaluated transparently against deterministic rules and O1–O5 safeguards."
        )
        vf_c1, vf_c2, vf_c3 = st.columns(3)
        with vf_c1:
            in_vid = st.text_input("Vendor ID *", value="V-007", key="cust_vid")
            in_vname = st.text_input("Vendor Legal Name *", value="Acme FinTech AI Corp", key="cust_vname")
            in_domain = st.text_input("Primary Domain", value="acmefintech.example.com", key="cust_domain")
        with vf_c2:
            in_service = st.text_input(
                "Service Purchased by Meridian (P & R input)",
                value="Automated customer dispute triage and statement notice generation",
                key="cust_service",
            )
            in_bproc = st.text_input(
                "Business Process Supported (P & R input)",
                value="Retail banking dispute resolution and regulatory notice fulfilment",
                key="cust_bproc",
            )
            in_cat = st.text_input("Vendor Category (Contextual)", value="Financial Operations SaaS", key="cust_cat")
        with vf_c3:
            in_d = st.text_input(
                "Data Classification Accessed (D input)",
                value="Customer PII, account numbers, and transaction dispute records",
                key="cust_d",
            )
            in_o = st.selectbox(
                "Operational Dependency (O input)",
                options=["Critical", "High", "Moderate", "Low", "UNKNOWN"],
                index=1,
                key="cust_o",
            )
            in_v = st.text_input(
                "Annual Data Volume (V input)",
                value="4.5 million records annually",
                key="cust_v",
            )
            in_sole = st.checkbox("Sole-Source Provider (Evaluated under O2 Safeguard)", value=False, key="cust_sole")

        val_col1, val_col2 = st.columns([3, 7])
        with val_col1:
            save_custom_btn = st.button(
                "Validate & Register Vendor for Assessment",
                type="secondary",
                use_container_width=True,
                key="btn_validate_custom_vendor",
            )
        if save_custom_btn:
            cand_vendor, v_errors, v_warnings = validate_vendor_assessment_input(
                vendor_id=in_vid,
                vendor_name=in_vname,
                domain=in_domain,
                category=in_cat,
                service_product_provided=in_service,
                business_process_supported=in_bproc,
                data_classification_accessed=in_d,
                operational_dependency=in_o,
                data_volume_annual=in_v,
                is_sole_source=in_sole,
            )
            for err in v_errors:
                st.error(err)
            for warn in v_warnings:
                st.warning(warn)
            if cand_vendor is not None:
                st.session_state.custom_vendors[cand_vendor.vendor_id] = cand_vendor
                meridian_vendor_map[cand_vendor.vendor_id] = cand_vendor
                preview_crit = MeridianCriticalityEngine().evaluate(cand_vendor)
                st.success(
                    f"Registered {cand_vendor.vendor_name} ({cand_vendor.vendor_id}) — "
                    f"Deterministic Base Score: {preview_crit.base_score if preview_crit.base_score is not None else 'UNKNOWN'}, "
                    f"Final Criticality: {preview_crit.final_criticality} "
                    f"(Status: {preview_crit.score_status.value}). Select it below to run Levels 1–9."
                )

    if not meridian_vendor_map:
        st.info("Meridian vendor dataset (`data/input/meridian_vendors.csv`) not found.")
    else:
        st.markdown(
            '<div class="sec-header"><span>Assessment Execution Controls (Single-Vendor &amp; Portfolio Batch)</span></div>',
            unsafe_allow_html=True,
        )
        e2e_top_c1, e2e_top_c2, e2e_top_c3, e2e_top_c4 = st.columns([3.4, 4.2, 1.2, 1.2])
        with e2e_top_c1:
            e2e_vid = st.selectbox(
                "Target Vendor",
                options=list(meridian_vendor_map.keys()),
                format_func=lambda vid: f"{meridian_vendor_map[vid].vendor_name} ({vid}) — {meridian_vendor_map[vid].domain or 'no-domain'}",
                key="e2e_vendor_select",
            )
        with e2e_top_c2:
            e2e_exec_mode = st.selectbox(
                "Assessment Data Source & Model Routing Mode",
                options=[
                    "Real OSINT Mode — Dual-Model Local Ollama (llama3.2:latest + mistral:latest)",
                    "Real OSINT Mode — Deterministic Rules Only",
                    "Synthetic Fixture Demo Mode — Dual-Model Local Ollama (llama3.2:latest + mistral:latest)",
                    "Synthetic Fixture Demo Mode — Deterministic Rules Only",
                ],
                index=3,
                key="e2e_exec_mode_select",
            )
        with e2e_top_c3:
            st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
            run_e2e_btn = st.button("Run Vendor", type="primary", use_container_width=True, key="btn_run_e2e")
        with e2e_top_c4:
            st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
            run_batch_btn = st.button("Run All (Batch)", use_container_width=True, key="btn_run_e2e_batch")

        selected_mv = meridian_vendor_map[e2e_vid]
        is_synthetic_demo = e2e_exec_mode.startswith("Synthetic Fixture Demo Mode")
        use_dual_ollama = "Dual-Model Local Ollama" in e2e_exec_mode
        mode_tag = (
            "synth_ollama"
            if (is_synthetic_demo and use_dual_ollama)
            else "synth_det"
            if is_synthetic_demo
            else "real_ollama"
            if use_dual_ollama
            else "real_det"
        )
        cache_key = f"{e2e_vid}::{mode_tag}"

        def _execute_single_vendor_pipeline(
            target_vendor: MeridianVendor,
            *,
            synthetic_mode: bool,
            dual_ollama_mode: bool,
            status_container: Optional[Any] = None,
        ) -> VendorEndToEndAssessmentReport:
            h_rev = st.session_state.human_overrides.get(target_vendor.vendor_id)
            all_source_ids = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]

            def _on_progress(lvl_num: int, stage_title: str, detail: str) -> None:
                if status_container is not None:
                    status_container.write(f"**[{lvl_num}/9] {stage_title}** — {detail}")

            if synthetic_mode:
                stub_search, stub_fetch = build_synthetic_demo_adapters_for_vendor(target_vendor)
                pipeline = MeridianEndToEndPipeline(
                    search_backend=stub_search,
                    fetch_adapter=stub_fetch,
                    enable_local_llm_from_env=dual_ollama_mode,
                )
                return pipeline.run_vendor_assessment(
                    target_vendor,
                    human_criticality_override=h_rev,
                    execute_searches=True,
                    execute_extraction=True,
                    mark_extracted_sources_reviewed=True,
                    reviewed_no_evidence_sources=all_source_ids,
                    persist_to_repository=True,
                    progress_callback=_on_progress,
                )

            pipeline = MeridianEndToEndPipeline(
                enable_local_llm_from_env=dual_ollama_mode,
            )
            return pipeline.run_vendor_assessment(
                target_vendor,
                human_criticality_override=h_rev,
                execute_searches=True,
                execute_extraction=True,
                persist_to_repository=True,
                progress_callback=_on_progress,
            )

        if run_batch_btn:
            with st.status(
                f"Executing batch 9-level assessment across {len(meridian_vendor_map)} vendors ({e2e_exec_mode})...",
                expanded=True,
            ) as batch_status:
                for idx_v, (b_vid, b_mv) in enumerate(meridian_vendor_map.items(), start=1):
                    batch_status.write(f"Assessing vendor {idx_v}/{len(meridian_vendor_map)}: **{b_mv.vendor_name} ({b_vid})**...")
                    b_rep = _execute_single_vendor_pipeline(
                        b_mv,
                        synthetic_mode=is_synthetic_demo,
                        dual_ollama_mode=use_dual_ollama,
                        status_container=None,
                    )
                    st.session_state.e2e_reports[f"{b_vid}::{mode_tag}"] = b_rep
                batch_status.update(
                    label=f"Batch assessment completed for {len(meridian_vendor_map)} vendors.",
                    state="complete",
                    expanded=False,
                )

        if run_e2e_btn or cache_key not in st.session_state.e2e_reports:
            with st.status(
                f"Running 9-Level Assessment for {selected_mv.vendor_name} ({selected_mv.vendor_id})...",
                expanded=run_e2e_btn,
            ) as run_status:
                rep = _execute_single_vendor_pipeline(
                    selected_mv,
                    synthetic_mode=is_synthetic_demo,
                    dual_ollama_mode=use_dual_ollama,
                    status_container=run_status,
                )
                st.session_state.e2e_reports[cache_key] = rep
                run_status.update(
                    label=(
                        f"Completed 9-level execution for {selected_mv.vendor_name} "
                        f"(Verdict: {rep.ai_use_verdict.verdict.value} | "
                        f"Criticality: {rep.criticality_summary.get('final_criticality')})"
                    ),
                    state="complete",
                    expanded=False,
                )

        e2e_rep: VendorEndToEndAssessmentReport = st.session_state.e2e_reports[cache_key]

        # Batch comparison table if multiple vendors have been run in this mode
        batch_keys_for_mode = [
            f"{vid}::{mode_tag}"
            for vid in meridian_vendor_map
            if f"{vid}::{mode_tag}" in st.session_state.e2e_reports
        ]
        if len(batch_keys_for_mode) > 1:
            with st.expander(f"Portfolio Batch Comparison ({len(batch_keys_for_mode)} Vendors Assessed in Current Mode)", expanded=False):
                b_rows = []
                for bk in batch_keys_for_mode:
                    brep: VendorEndToEndAssessmentReport = st.session_state.e2e_reports[bk]
                    b_rows.append({
                        "Vendor ID": brep.vendor_id,
                        "Vendor": brep.legal_entity,
                        "Criticality": brep.criticality_summary.get("final_criticality", "UNKNOWN"),
                        "Base Score": brep.criticality_summary.get("base_score"),
                        "OSINT Depth": brep.osint_plan_summary.get("selected_depth_profile", ""),
                        "Depth Completed": brep.ai_use_verdict.depth_completed,
                        "Evidence Items": len(brep.evidence_records),
                        "AI-Use Verdict": brep.ai_use_verdict.verdict.value,
                        "Determined Scope": brep.determined_scope,
                        "R1–R8 Findings": len(brep.risk_findings),
                        "Level 9 Status": brep.deterministic_risk_score.policy_status.value,
                    })
                st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

        # ── 3. Explicit Data-Provenance & Assessment Completeness Banner ──
        verdict_val = e2e_rep.ai_use_verdict.verdict.value if e2e_rep.ai_use_verdict else "INCOMPLETE_ASSESSMENT"
        if e2e_rep.is_synthetic_evidence_run or is_synthetic_demo:
            st.markdown(
                """
                <div class="console-banner synth">
                    <strong>SYNTHETIC FIXTURE / DEMO WALKTHROUGH MODE ACTIVE — NOT REAL VENDOR EVIDENCE:</strong>
                    Candidate URLs, HTML/PDF documents, and verbatim excerpts in this run come from labelled offline synthetic test fixtures
                    used to verify Levels 1–9 end-to-end (including prompt-injection isolation under E14 and dual-model routing).
                    Switch to <strong>Real OSINT Mode</strong> for production vendor reconnaissance.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not readiness.search_configured:
            unavail_q = e2e_rep.osint_plan_summary.get("unavailable_queries_count", 0)
            plan_q = e2e_rep.osint_plan_summary.get("planned_queries_count", 0)
            st.markdown(
                f"""
                <div class="console-banner incomplete">
                    <strong>REAL OSINT MODE BLOCKED BY UNCONFIGURED SEARCH — ASSESSMENT INCOMPLETE (NOT A 'NO AI' VERDICT):</strong>
                    Levels 1 and 2 completed deterministically, and {plan_q} targeted OSINT queries were planned.
                    However, all {unavail_q}/{plan_q} search queries returned <code>UNAVAILABLE</code> because
                    <code>MERIDIAN_SERPAPI_KEY</code> is not configured in the local <code>.env</code> file.
                    In accordance with the rulebook, an unexecuted search cannot support a <code>NO</code> AI-use verdict;
                    the assessment status is <strong><code>{verdict_val}</code></strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="console-banner real-ok">
                    <strong>REAL OSINT ASSESSMENT MODE:</strong>
                    Run ID <code>{e2e_rep.report_id}</code> · Generated <code>{e2e_rep.generated_at}</code> ·
                    AI-Use Verdict: <strong><code>{verdict_val}</code></strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── 4. Top-Level KPI Strip ──
        score_status_val = (
            e2e_rep.deterministic_risk_score.policy_status.value
            if e2e_rep.deterministic_risk_score
            else "SCORING_POLICY_UNCONFIGURED"
        )
        crit_tier_str = e2e_rep.criticality_summary.get("final_criticality", "Unknown")
        base_score_val = e2e_rep.criticality_summary.get("base_score")
        base_score_str = f"{base_score_val:.2f}" if isinstance(base_score_val, (int, float)) else "UNKNOWN"
        depth_str = e2e_rep.osint_plan_summary.get("selected_depth_profile", "Unknown")
        timebox_str = e2e_rep.osint_plan_summary.get("timebox", {}).get("planning_target", "N/A")

        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-card all">
                    <div class="stat-label">Level 1–2 Criticality &amp; Depth</div>
                    <div class="stat-value" style="font-size:1.2rem">{crit_tier_str} <span style="font-size:0.85rem;font-weight:500;color:#475569">(Base: {base_score_str})</span></div>
                    <div class="stat-sub">Depth: {depth_str} · POC Timebox: {timebox_str}</div>
                </div>
                <div class="stat-card high">
                    <div class="stat-label">Level 3–6 OSINT &amp; Evidence</div>
                    <div class="stat-value" style="font-size:1.2rem">{len(e2e_rep.evidence_records)} Evidence Items</div>
                    <div class="stat-sub">{len(e2e_rep.url_triage_decisions)} URLs triaged · {e2e_rep.extracted_documents_count} docs fetched</div>
                </div>
                <div class="stat-card medium">
                    <div class="stat-label">Level 7 AI-Use Verdict</div>
                    <div class="stat-value" style="font-size:1.1rem">{verdict_val}</div>
                    <div class="stat-sub">Scope: {e2e_rep.determined_scope} · Depth Complete: {e2e_rep.ai_use_verdict.depth_completed}</div>
                </div>
                <div class="stat-card low">
                    <div class="stat-label">Level 8–9 Risk &amp; Rating</div>
                    <div class="stat-value" style="font-size:1.1rem">{len(e2e_rep.risk_findings)} R1–R8 Findings</div>
                    <div class="stat-sub">{score_status_val}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 5. Canonical Nine-Level Stage Execution Tracker (Levels 1–9 + Level 3B Substep) ──
        st.markdown(
            '<div class="sec-header"><span>Canonical Nine-Level Framework Stage Tracker</span></div>',
            unsafe_allow_html=True,
        )
        tracker_rows = build_nine_level_stage_tracker(e2e_rep)
        stage_table_data: list[dict[str, str]] = []
        for tr in tracker_rows:
            stage_table_data.append({
                "Stage": f"{tr.level_label}",
                "Framework Level Name": tr.level_name,
                "Status": tr.badge_label,
                "Execution Method": tr.execution_method,
                "Relevant Counts": tr.relevant_counts,
                "Concise Output / Blocker": tr.blocker if tr.blocker else tr.concise_output,
            })
            for sub in tr.substeps:
                stage_table_data.append({
                    "Stage": f"↳ {sub['substep_id']}",
                    "Framework Level Name": sub["substep_name"],
                    "Status": sub["status"],
                    "Execution Method": "Substep of Level 3",
                    "Relevant Counts": "—",
                    "Concise Output / Blocker": sub["detail"],
                })
        st.dataframe(pd.DataFrame(stage_table_data), use_container_width=True, hide_index=True)

        # ── 6. Analyst-Friendly Results Area (Six Structured Sub-Tabs) ──
        res_tab1, res_tab2, res_tab3, res_tab4, res_tab5, res_tab6 = st.tabs([
            "1. Executive Summary & Status",
            "2. Criticality & Overrides (Levels 1–2)",
            "3. AI-Use Verdict & Gates G1–G4 (Level 7)",
            "4. Evidence Register (Levels 3–6, E1–E14)",
            "5. Risk Findings & Follow-Up (Levels 8–9, R1–R8)",
            "6. Stage Audit, Telemetry, Sign-Off & Exports",
        ])

        # ── SUB-TAB 1: EXECUTIVE SUMMARY & ASSESSMENT STATUS ──
        with res_tab1:
            sum_c1, sum_c2 = st.columns([6, 4], gap="large")
            with sum_c1:
                st.markdown("##### Vendor & Service Scope")
                scope_df = pd.DataFrame([
                    {"Field": "Report Run ID", "Value": e2e_rep.report_id},
                    {"Field": "Generated Timestamp (UTC)", "Value": e2e_rep.generated_at},
                    {"Field": "Data Provenance Mode", "Value": "SYNTHETIC DEMO FIXTURES" if e2e_rep.is_synthetic_evidence_run else "REAL OSINT ASSESSMENT"},
                    {"Field": "Vendor Legal Entity", "Value": f"{e2e_rep.legal_entity} ({e2e_rep.vendor_id})"},
                    {"Field": "Primary Domain", "Value": e2e_rep.domain or "N/A"},
                    {"Field": "Service Purchased by Meridian", "Value": e2e_rep.service_product_provided or "N/A"},
                    {"Field": "Business Process Supported", "Value": e2e_rep.business_process_supported or "N/A"},
                    {"Field": "Effective Criticality Tier", "Value": f"{crit_tier_str} (Base Score: {base_score_str})"},
                    {"Field": "OSINT Depth Profile", "Value": f"{depth_str} (Completed: {e2e_rep.ai_use_verdict.depth_completed})"},
                    {"Field": "Rulebook AI-Use Verdict", "Value": f"{verdict_val} — {e2e_rep.ai_use_verdict.verdict_statement}"},
                    {"Field": "Determined AI Scope & Types", "Value": f"Scope: {e2e_rep.determined_scope} | AI Types: {', '.join(e2e_rep.determined_ai_types) or 'NONE'}"},
                    {"Field": "Fourth-Party AI Supply Chain (G4)", "Value": f"Status: {e2e_rep.supply_chain_identity_status} | Providers: {', '.join(e2e_rep.named_providers) or 'None'} | Models: {', '.join(e2e_rep.named_models) or 'None'} | Hosting: {', '.join(e2e_rep.hosting_infrastructures) or 'None'}"},
                    {"Field": "Level 9 Risk Rating Status", "Value": f"{score_status_val} (Final Score: {e2e_rep.deterministic_risk_score.final_risk_score if e2e_rep.deterministic_risk_score.final_risk_score is not None else 'None — Unconfigured Policy'})"},
                ])
                st.dataframe(scope_df, use_container_width=True, hide_index=True)

            with sum_c2:
                st.markdown("##### Assessment Limitations & Blockers")
                if e2e_rep.unavailable_integrations:
                    st.markdown("**Unconfigured / Unavailable External Integrations:**")
                    for ui_msg in e2e_rep.unavailable_integrations:
                        st.markdown(f"- {ui_msg}")
                if e2e_rep.assessment_limitations:
                    st.markdown("**Recorded Assessment Limitations:**")
                    for lim_msg in e2e_rep.assessment_limitations:
                        st.markdown(f"- {lim_msg}")
                if not e2e_rep.unavailable_integrations and not e2e_rep.assessment_limitations:
                    st.caption("No integration blockers or assessment limitations recorded.")

                if e2e_rep.unresolved_questions:
                    st.markdown("##### Open Diligence Questions for Vendor")
                    for oq in e2e_rep.unresolved_questions:
                        st.markdown(f"- {oq}")

        # ── SUB-TAB 2: CRITICALITY FACTORS, SCORE, TIER & TRIGGERED OVERRIDES (LEVELS 1–2) ──
        with res_tab2:
            crit_data = e2e_rep.criticality_summary or {}
            st.markdown(
                f"##### Level 1 Deterministic Criticality Breakdown (`C = 0.30D + 0.20P + 0.20R + 0.20O + 0.10V`)\n"
                f"**Base Score:** `{base_score_str}` &nbsp;|&nbsp; "
                f"**Base Tier:** `{crit_data.get('base_tier', 'UNKNOWN')}` &nbsp;|&nbsp; "
                f"**Override Tier:** `{crit_data.get('override_tier') or 'None'}` &nbsp;|&nbsp; "
                f"**Effective Criticality:** **`{crit_tier_str}`** (`{crit_data.get('score_status', 'COMPLETE')}`)"
            )
            factor_rows = []
            factors_dict = crit_data.get("factors", {})
            for fk in ("D", "P", "R", "O", "V"):
                fd = factors_dict.get(fk, {})
                factor_rows.append({
                    "Factor": fk,
                    "Score (0–3)": str(fd.get("score", "UNKNOWN")),
                    "Weight": f"{fd.get('weight', 0.0):.2f}" if isinstance(fd.get("weight"), (int, float)) else "—",
                    "Weighted Contribution": (
                        f"{fd.get('weighted_contribution'):.2f}"
                        if isinstance(fd.get("weighted_contribution"), (int, float))
                        else "UNKNOWN"
                    ),
                    "Determination Method": fd.get("determination_method", ""),
                    "Source Fields": ", ".join(fd.get("source_fields", [])),
                    "Matched Concepts / Phrases": ", ".join(fd.get("matched_concepts", []) or fd.get("deterministic_matches", [])),
                    "Rationale": fd.get("rationale", ""),
                })
            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)

            st.markdown("##### Evaluated Governance Safeguards & Overrides (`O1–O5`)")
            ov_rows = []
            for ov in crit_data.get("overrides_evaluated", []):
                ov_rows.append({
                    "Override ID": ov.get("override_id", ""),
                    "Rule Description": ov.get("description", ""),
                    "Condition Evaluated": ov.get("condition_evaluated", ""),
                    "Triggered": "YES" if ov.get("triggered") else "NO",
                    "Resulting Tier": ov.get("resulting_tier") or "—",
                    "Rationale": ov.get("rationale", ""),
                })
            if ov_rows:
                st.dataframe(pd.DataFrame(ov_rows), use_container_width=True, hide_index=True)

            st.markdown("##### Level 2 OSINT Depth Scoping & S1–S12 Source Coverage")
            osint_sum = e2e_rep.osint_plan_summary or {}
            cond_items = [
                c.get("source_id", str(c)) if isinstance(c, dict) else str(c)
                for c in osint_sum.get("conditional_source_classes", [])
            ]
            st.markdown(
                f"- **Selected Depth Profile:** `{osint_sum.get('selected_depth_profile', 'N/A')}` — {osint_sum.get('selection_reason', '')}\n"
                f"- **Required Source Classes:** `{', '.join(osint_sum.get('required_source_classes', []))}` | "
                f"**Conditional:** `{', '.join(cond_items) or 'None'}` | "
                f"**Corroborative:** `{', '.join(osint_sum.get('corroborative_source_classes', []))}`\n"
                f"- **Stopping Rule (`{osint_sum.get('stopping_rule_id', '')}`):** `{osint_sum.get('stop_status', '')}`"
            )
            cov_details = osint_sum.get("coverage_summary", {}).get("source_details", [])
            if cov_details:
                st.dataframe(pd.DataFrame(cov_details), use_container_width=True, hide_index=True)

        # ── SUB-TAB 3: AI-USE VERDICT & G1–G4 GATE RESULTS (LEVEL 7) ──
        with res_tab3:
            v_obj = e2e_rep.ai_use_verdict
            st.markdown("##### Level 7 Rulebook AI-Use Verdict")
            st.markdown(
                f"- **Verdict:** **`{v_obj.verdict.value}`** (Investigation Depth Completed: `{v_obj.depth_completed}`)\n"
                f"- **Verdict Statement:** {v_obj.verdict_statement}\n"
                f"- **Evidentiary Basis:** {v_obj.basis_summary}\n"
                f"- **Supporting Evidence IDs:** `{', '.join(v_obj.supporting_evidence_ids) or 'None'}`\n"
                f"- **Corroborating Source Classes:** `{', '.join(v_obj.corroborating_source_classes) or 'None'}`"
            )
            if v_obj.incomplete_reasons:
                st.warning("Completeness Guardrail Reasons: " + " | ".join(v_obj.incomplete_reasons))

            st.markdown("##### Evidentiary Gates (`G1–G4`)")
            gate_rows = []
            for g in e2e_rep.gates:
                gate_rows.append({
                    "Gate ID": g.gate_id,
                    "Gate Title": g.title,
                    "Evaluation Question": g.question,
                    "Status": g.status,
                    "Supporting Evidence IDs": ", ".join(g.supporting_evidence_ids) or "None",
                    "Rationale": g.rationale,
                    "Open Questions": "; ".join(g.open_questions) or "None",
                })
            st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

        # ── SUB-TAB 4: EVIDENCE REGISTER (LEVELS 3–6, E1–E14) ──
        with res_tab4:
            st.markdown("##### Point-of-Capture Evidence Register (`E1–E14` & `G1–G4`)")
            if not e2e_rep.evidence_records:
                st.info(
                    "No evidence records captured in this run. "
                    + (
                        "External search is currently unconfigured (`MERIDIAN_SERPAPI_KEY` is blank)."
                        if not is_synthetic_demo and not readiness.search_configured
                        else "No qualifying excerpts were extracted."
                    )
                )
            else:
                weight_by_id = {w.evidence_id: w for w in e2e_rep.weighted_evidence}
                cls_by_id = {c.evidence_id: c for c in e2e_rep.signal_classifications}

                all_sources = sorted({ev.source_class for ev in e2e_rep.evidence_records})
                all_strengths = sorted({
                    (weight_by_id[ev.evidence_id].evidence_strength.value if ev.evidence_id in weight_by_id else ev.evidence_strength.value)
                    for ev in e2e_rep.evidence_records
                })
                all_scopes = sorted({ev.scope.value for ev in e2e_rep.evidence_records})

                f_c1, f_c2, f_c3, f_c4, f_c5 = st.columns([2, 2, 2, 2, 3])
                with f_c1:
                    flt_source = st.multiselect("Source Class (S1–S12)", options=all_sources, default=[], key=f"ev_flt_src_{cache_key}")
                with f_c2:
                    flt_strength = st.multiselect("Evidence Strength", options=all_strengths, default=[], key=f"ev_flt_str_{cache_key}")
                with f_c3:
                    flt_gate = st.multiselect("Supports Gate", options=["G1", "G2", "G3", "G4"], default=[], key=f"ev_flt_gate_{cache_key}")
                with f_c4:
                    flt_scope = st.multiselect("Product Scope", options=all_scopes, default=[], key=f"ev_flt_scp_{cache_key}")
                with f_c5:
                    flt_kw = st.text_input("Keyword Filter (URL, Excerpt, E-Rule)", value="", key=f"ev_flt_kw_{cache_key}")

                filtered_evs = []
                for ev in e2e_rep.evidence_records:
                    we = weight_by_id.get(ev.evidence_id)
                    eff_strength = we.evidence_strength.value if we else ev.evidence_strength.value
                    eff_erules = we.fired_e_rule_ids if we else ev.e_rule_ids
                    ev_gates = gates_for_evidence(e2e_rep, ev.evidence_id)
                    if flt_source and ev.source_class not in flt_source:
                        continue
                    if flt_strength and eff_strength not in flt_strength:
                        continue
                    if flt_gate and not any(g in ev_gates for g in flt_gate):
                        continue
                    if flt_scope and ev.scope.value not in flt_scope:
                        continue
                    if flt_kw.strip():
                        kw_low = flt_kw.strip().lower()
                        searchable = f"{ev.evidence_id} {ev.canonical_url} {ev.verbatim_excerpt} {' '.join(eff_erules)}".lower()
                        if kw_low not in searchable:
                            continue
                    filtered_evs.append(ev)

                st.caption(f"Showing {len(filtered_evs)} of {len(e2e_rep.evidence_records)} evidence record(s).")

                ev_table_rows = []
                for ev in filtered_evs:
                    we = weight_by_id.get(ev.evidence_id)
                    cls_r = cls_by_id.get(ev.evidence_id)
                    ev_gates = gates_for_evidence(e2e_rep, ev.evidence_id)
                    ev_table_rows.append({
                        "Evidence ID": ev.evidence_id,
                        "Source Class": f"{ev.source_class} (Grade {ev.source_reliability_grade})",
                        "Source URL": ev.canonical_url,
                        "Retrieved At (UTC)": ev.retrieved_at,
                        "Scope": ev.scope.value,
                        "AI Type": ev.ai_type.value,
                        "Evidence Strength": we.evidence_strength.value if we else ev.evidence_strength.value,
                        "Weight Band": we.weight_band.value if we else "—",
                        "E1–E14 Rules": ", ".join(we.fired_e_rule_ids if we else ev.e_rule_ids),
                        "Gates": ", ".join(ev_gates),
                        "L6 Classifier": (
                            f"{cls_r.classification_method.value} ({cls_r.model_used or 'DETERMINISTIC'} [{cls_r.model_execution_status or 'OK'}])"
                            if cls_r else "—"
                        ),
                        "Verbatim Excerpt": ev.verbatim_excerpt,
                    })

                st.dataframe(
                    pd.DataFrame(ev_table_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Source URL": st.column_config.LinkColumn("Source URL", display_text="Open Source URL"),
                    },
                )

                with st.expander("Inspect Full Verbatim Excerpts, SHA-256 Hashes & E1–E14 Rule Audit", expanded=True):
                    for ev in filtered_evs:
                        we = weight_by_id.get(ev.evidence_id)
                        cls_r = cls_by_id.get(ev.evidence_id)
                        erules_str = ", ".join(we.fired_e_rule_ids if we else ev.e_rule_ids)
                        gates_str = ", ".join(gates_for_evidence(e2e_rep, ev.evidence_id)) or "None"
                        hostile_badge = " · **HOSTILE PROMPT INJECTION ISOLATED (E14)**" if ev.hostile_instruction_detected else ""
                        st.markdown(
                            f"**`{ev.evidence_id}`** · Source **`{ev.source_class}`** (Grade `{ev.source_reliability_grade}`) · "
                            f"Retrieved: `{ev.retrieved_at}` · Scope: `{ev.scope.value}` · "
                            f"Strength: **`{we.evidence_strength.value if we else ev.evidence_strength.value}`** · "
                            f"E-Rules: `{erules_str}` · Gates: `{gates_str}`{hostile_badge}\n\n"
                            f"- **Source URL:** [{ev.canonical_url}]({ev.canonical_url})  |  "
                            f"**SHA-256:** `{ev.content_sha256}`  |  **Offsets:** `{ev.char_start}:{ev.char_end}`\n"
                            f"- **Verbatim Excerpt:** *\"{ev.verbatim_excerpt}\"*"
                        )
                        if cls_r:
                            st.caption(
                                f"Level 6 Classification: `{cls_r.signal_category.value}` | Method: `{cls_r.classification_method.value}` | "
                                f"Model: `{cls_r.model_used or 'DETERMINISTIC'}` [`{cls_r.model_execution_status or 'DETERMINISTIC_ONLY'}`] — {cls_r.rationale}"
                            )
                        st.divider()

            with st.expander(f"Level 3B Candidate-URL Triage Log ({len(e2e_rep.url_triage_decisions)} URLs)", expanded=False):
                if not e2e_rep.url_triage_decisions:
                    st.caption("No candidate URLs queued for triage in this run.")
                else:
                    t_rows = [
                        {
                            "Candidate ID": td.candidate_id,
                            "Action": getattr(td.action, "value", str(td.action)),
                            "Source Class": td.candidate_source_class,
                            "Scope Relevance": getattr(td.product_scope_relevance, "value", str(td.product_scope_relevance)),
                            "Target Gates": ", ".join(td.target_gates),
                            "Confidence": td.confidence,
                            "Method": getattr(td.decision_method, "value", str(td.decision_method)),
                            "Model Used": td.model_used or "DETERMINISTIC",
                            "Model Status": td.model_execution_status or "DETERMINISTIC_ONLY",
                            "URL": td.url,
                            "Rationale": td.rationale,
                        }
                        for td in e2e_rep.url_triage_decisions
                    ]
                    st.dataframe(
                        pd.DataFrame(t_rows),
                        use_container_width=True,
                        hide_index=True,
                        column_config={"URL": st.column_config.LinkColumn("URL")},
                    )

            with st.expander("Level 5 Semantic Filtering & RAG Chunking Summary", expanded=False):
                rag_s = e2e_rep.semantic_retrieval
                st.markdown(
                    f"- **Status:** `{rag_s.status}` | **Embedding Model:** `{rag_s.model_name}` (Available: `{rag_s.model_available}`)\n"
                    f"- **Chunk Size:** `{rag_s.chunk_size_tokens}` tokens (`{rag_s.chunk_overlap_tokens}` token overlap)\n"
                    f"- **Chunks Created:** `{rag_s.chunks_created}` | **Chunks Embedded:** `{rag_s.chunks_embedded}` | **Concept Matches:** `{len(rag_s.matches)}`"
                )
                for lim in rag_s.limitations:
                    st.caption(f"Note: {lim}")

        # ── SUB-TAB 5: RISK FINDINGS (R1–R8) & RECOMMENDED FOLLOW-UP ACTIONS (LEVELS 8–9) ──
        with res_tab5:
            st.markdown("##### Level 8 AI Security Risk Findings (`R1–R8`) & Recommended Follow-Up Actions")
            st.caption(
                "Rule-determined fields (Rule ID, Severity, Exposure Determination, Supporting Evidence IDs, Verbatim Quotes, and Remediation) "
                "are strictly deterministic. Optional Model B (`mistral:latest`) narrative explanations are separated below and validated against cited evidence."
            )
            if not e2e_rep.risk_findings:
                st.info("No R1–R8 AI risk findings triggered for this assessment run.")
            else:
                rf_c1, rf_c2 = st.columns(2)
                all_sev = sorted({rf.severity.value for rf in e2e_rep.risk_findings})
                all_det = sorted({rf.exposure_determination.value for rf in e2e_rep.risk_findings})
                with rf_c1:
                    flt_sev = st.multiselect("Filter by Severity", options=all_sev, default=[], key=f"rf_flt_sev_{cache_key}")
                with rf_c2:
                    flt_det = st.multiselect("Filter by Exposure Determination", options=all_det, default=[], key=f"rf_flt_det_{cache_key}")

                shown_rfs = [
                    rf for rf in e2e_rep.risk_findings
                    if (not flt_sev or rf.severity.value in flt_sev)
                    and (not flt_det or rf.exposure_determination.value in flt_det)
                ]

                for rf in shown_rfs:
                    llm_status_str = rf.model_execution_status or "DETERMINISTIC_ONLY"
                    is_llm_fallback = rf.summary_method != "LOCAL_LLM_VALIDATED"
                    st.markdown(
                        f"""
                        <div class="factor-card">
                            <div style="display:flex;justify-content:space-between;align-items:center">
                                <strong>[{rf.risk_rule_id}] {rf.risk_title} (<code>{rf.finding_id}</code>)</strong>
                                <span>
                                    Rule Severity: <strong><code>{rf.severity.value}</code></strong> &nbsp;·&nbsp;
                                    Rule Exposure: <code>{rf.exposure_determination.value}</code> &nbsp;·&nbsp;
                                    Confidence: <code>{rf.confidence}</code>
                                </span>
                            </div>
                            <div style="font-size:0.76rem;color:#0f172a;margin-top:0.35rem">
                                <strong>Deterministic Rule Finding:</strong> {rf.finding_description}
                            </div>
                            <div style="font-size:0.73rem;color:#334155;margin-top:0.25rem">
                                <strong>Supporting Evidence IDs:</strong> <code>{', '.join(rf.evidence_ids)}</code> &nbsp;|&nbsp;
                                <strong>Fired E-Rules:</strong> <code>{', '.join(rf.fired_e_rules) or 'None'}</code> &nbsp;|&nbsp;
                                <strong>Gates:</strong> <code>{', '.join(rf.fired_gate_rules) or 'None'}</code>
                            </div>
                            <div style="font-size:0.73rem;color:#1e293b;background:#f8fafc;border-left:3px solid {'#0284c7' if not is_llm_fallback else '#64748b'};padding:0.4rem 0.6rem;border-radius:4px;margin-top:0.35rem">
                                <strong>Analyst Explanation (Method: <code>{rf.summary_method}</code> · Model: <code>{rf.model_used or 'DETERMINISTIC'}</code> [<code>{llm_status_str}</code>]):</strong><br>
                                {rf.analyst_explanation or rf.finding_description}
                            </div>
                            <div style="font-size:0.74rem;color:#0f172a;margin-top:0.35rem">
                                <strong>Recommended Follow-Up / Remediation Action:</strong> {rf.recommended_remediation}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("##### Level 9 Deterministic Risk Rating (`Risk Score = M_c * sum(S_i * W_i)`)")
            drs = e2e_rep.deterministic_risk_score
            st.markdown(
                f"- **Policy Readiness Status:** **`{drs.policy_status.value}`** (Policy Version: `{drs.policy_version}`)\n"
                f"- **Calculation Formula:** `{drs.calculation_method}`\n"
                f"- **Vendor Criticality Tier (`M_c`):** `{drs.criticality_tier}` (Multiplier: `{drs.criticality_multiplier if drs.criticality_multiplier is not None else 'UNCONFIGURED'}`)\n"
                f"- **Distinct Findings Evaluated:** `{drs.distinct_findings_scored}` (`{drs.duplicates_suppressed}` duplicate exposure(s) suppressed)\n"
                f"- **Final Numerical Risk Score:** **`{drs.final_risk_score if drs.final_risk_score is not None else 'None (SCORING_POLICY_UNCONFIGURED)'}`** | **Overall Rating:** `{drs.overall_risk_rating or 'UNCONFIGURED'}`\n"
                f"- **Status Explanation:** {drs.status_message}"
            )
            if drs.finding_breakdowns:
                bd_rows = [
                    {
                        "Finding ID": b.finding_id,
                        "Rule": b.risk_rule_id,
                        "Deduplication Key": b.dedup_exposure_key,
                        "Severity": b.severity_label,
                        "S_i (Numeric)": b.severity_numeric if b.severity_numeric is not None else "UNCONFIGURED",
                        "W_i (Weight)": b.evidentiary_weight if b.evidentiary_weight is not None else "UNCONFIGURED",
                        "S_i * W_i": b.weighted_contribution if b.weighted_contribution is not None else "UNCONFIGURED",
                        "Primary (Not Duplicate)": "YES" if b.is_deduplicated_primary else "NO (Suppressed)",
                        "Evidence IDs": ", ".join(b.source_evidence_ids),
                    }
                    for b in drs.finding_breakdowns
                ]
                st.dataframe(pd.DataFrame(bd_rows), use_container_width=True, hide_index=True)

        # ── SUB-TAB 6: STAGE AUDIT, TELEMETRY, HUMAN VALIDATION & CLEAN EXPORTS ──
        with res_tab6:
            st.markdown("##### Clean Assessment Report & Evidence Exports")
            st.caption(
                "Export the complete assessment report or filtered evidence register. Exports include run ID, timestamp, "
                "provenance hashes, stage statuses, and human-validation records, and never include API keys or secrets."
            )
            dl_c1, dl_c2, dl_c3, dl_c4 = st.columns(4)
            with dl_c1:
                st.download_button(
                    "Download Assessment JSON (.json)",
                    data=AssessmentReportExporter.to_json(e2e_rep),
                    file_name=f"Meridian_Assessment_{e2e_rep.vendor_id}_{e2e_rep.report_id}.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"dl_json_{cache_key}",
                )
            with dl_c2:
                st.download_button(
                    "Download Evidence CSV (.csv)",
                    data=AssessmentReportExporter.to_evidence_csv(e2e_rep),
                    file_name=f"Meridian_Evidence_{e2e_rep.vendor_id}_{e2e_rep.report_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_csv_{cache_key}",
                )
            with dl_c3:
                st.download_button(
                    "Download Markdown Report (.md)",
                    data=AssessmentReportExporter.to_markdown(e2e_rep),
                    file_name=f"Meridian_Report_{e2e_rep.vendor_id}_{e2e_rep.report_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"dl_md_{cache_key}",
                )
            with dl_c4:
                st.download_button(
                    "Download HTML Report (.html)",
                    data=AssessmentReportExporter.to_html(e2e_rep),
                    file_name=f"Meridian_Report_{e2e_rep.vendor_id}_{e2e_rep.report_id}.html",
                    mime="text/html",
                    use_container_width=True,
                    key=f"dl_html_{cache_key}",
                )

            st.divider()
            st.markdown("##### Dual-Model Routing & 6 GB VRAM Lifecycle Telemetry")
            if e2e_rep.model_routing_telemetry:
                tel = e2e_rep.model_routing_telemetry
                assignments = tel.get("model_assignments", {})
                vram_info = tel.get("vram_management", {})
                tasks_tel = tel.get("tasks", {})
                tel_rows = []
                for t_key, t_label in [
                    ("URL_TRIAGE", "Level 3B: Candidate-URL Triage (Model A)"),
                    ("RAG_EMBEDDING", "Level 5: Semantic Chunk Embedding (Separate Model)"),
                    ("EVIDENCE_CLASSIFICATION", "Level 6: Signal Classification (Model B)"),
                    ("RISK_SUMMARIZATION", "Level 8: Risk Finding Summarization (Model B)"),
                ]:
                    td = tasks_tel.get(t_key, {})
                    tel_rows.append({
                        "Pipeline Task": t_label,
                        "Assigned Model": assignments.get(t_key, "N/A"),
                        "Overall Status": td.get("overall_status", "NOT_CALLED"),
                        "Calls Total": td.get("calls_total", 0),
                        "Calls Succeeded": td.get("calls_succeeded", 0),
                        "Total Latency (ms)": td.get("total_latency_ms", 0.0),
                        "Errors / Fallback Notes": "; ".join(td.get("errors", [])) or "None",
                    })
                st.dataframe(pd.DataFrame(tel_rows), use_container_width=True, hide_index=True)
                unload_evts = vram_info.get("vram_unload_events", [])
                st.caption(
                    f"Ollama Endpoint: `{tel.get('ollama_endpoint')}` | Enabled: `{tel.get('ollama_enabled')}` | "
                    f"Sequential VRAM Unload (`keep_alive=0`): `{vram_info.get('unload_on_model_switch')}` "
                    f"({len(unload_evts)} unload event(s) recorded in this run)."
                )

            st.divider()
            st.markdown("##### Human Validation & Governance Sign-Off")
            if e2e_rep.human_Review_required:
                st.warning(
                    "Human validation is required before final governance sign-off: "
                    + " | ".join(e2e_rep.human_review_triggers)
                )
            hv_f1, hv_f2 = st.columns([5, 5], gap="large")
            with hv_f1:
                hv_reviewer = st.text_input("Reviewer Name / Role", value="Primary TPRM Analyst", key=f"hv_rev_{cache_key}")
                hv_target_type = st.selectbox(
                    "Validation Target Type",
                    options=["VERDICT", "GATE", "RISK_FINDING", "CRITICALITY_O5"],
                    key=f"hv_ttype_{cache_key}",
                )
                hv_decision = st.selectbox(
                    "Validation Decision",
                    options=["CONFIRM_ASSESSMENT", "OVERRIDE_AI_VERDICT", "ACCEPT_WITH_CONDITIONS"],
                    key=f"hv_dec_{cache_key}",
                )
            with hv_f2:
                hv_verdict = st.selectbox(
                    "Updated Status / Verdict",
                    options=["YES", "INCONCLUSIVE", "NO", "INCOMPLETE_ASSESSMENT"],
                    index=0,
                    key=f"hv_verd_{cache_key}",
                )
                hv_rat = st.text_area(
                    "Mandatory Written Rationale",
                    placeholder="Document evidence review, compensating controls, and validation rationale...",
                    key=f"hv_rat_{cache_key}",
                )
                if st.button("Record Human Validation Decision", type="primary", key=f"btn_save_hv_{cache_key}", use_container_width=True):
                    try:
                        val_rec = HumanValidationService.create_validation_record(
                            vendor_id=e2e_rep.vendor_id,
                            target_type=hv_target_type,
                            target_id="AI_USE_VERDICT" if hv_target_type == "VERDICT" else hv_target_type,
                            proposed_automated_finding=verdict_val,
                            human_decision=hv_decision,
                            rationale=hv_rat,
                            original_status=verdict_val,
                            updated_status=hv_verdict if hv_decision != "CONFIRM_ASSESSMENT" else verdict_val,
                            reviewer_id=hv_reviewer,
                        )
                        e2e_rep.human_validations.append(val_rec)
                        st.success(f"Recorded human validation `{val_rec.validation_id}`.")
                        st.rerun()
                    except ValueError as err:
                        st.error(str(err))

            if e2e_rep.human_validations:
                st.markdown("**Logged Human Validation Records:**")
                hv_rows = [
                    {
                        "Validation ID": hv.validation_id,
                        "Target": f"{hv.target_type}:{hv.target_id}",
                        "Decision": hv.human_decision,
                        "Original Status": hv.original_status,
                        "Updated Status": hv.updated_status,
                        "Reviewer": hv.reviewer_id,
                        "Reviewed At (UTC)": hv.reviewed_at,
                        "Rationale": hv.rationale,
                    }
                    for hv in e2e_rep.human_validations
                ]
                st.dataframe(pd.DataFrame(hv_rows), use_container_width=True, hide_index=True)



