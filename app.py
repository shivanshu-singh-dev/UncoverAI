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
from meridian_assessment.services.osint import OSINTInvestigationPlanner, OSINTSourceRegistry
from meridian_assessment.models.osint_plan import (
    AssessmentStopStatus,
    CriticalityApprovalStatus,
    CoverageState,
    GateStatus,
    OSINTInvestigationPlan,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meridian | Vendor Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Pure Minimalist Design System ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2.2rem 2.5rem 2.2rem;
    max-width: 1480px;
}

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
            <div class="topbar-name">Meridian &nbsp; Vendor Risk Assessment</div>
            <div class="topbar-sub">Deterministic Criticality Engine — D/P/R/O/V Methodology with Safeguards &amp; Governance</div>
        </div>
    </div>
    <div class="topbar-right">Critical / High / Medium / Low &nbsp;·&nbsp; Zero Generative LLM Scoring</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Methodology Mode")
    engine_mode = st.radio(
        "Engine Selection",
        ["Meridian D/P/R/O/V (Primary)", "Standard Qualitative (Legacy)"],
        label_visibility="collapsed",
    )

    st.markdown("#### Vendor Input")
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
        run_btn = st.button("▶ Run", type="primary", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#64748b;line-height:1.6">
    <strong style="color:#0f172a">Formula</strong><br>
    <code>C = 0.30D + 0.20P + 0.20R + 0.20O + 0.10V</code><br><br>
    <strong style="color:#0f172a">Safeguards (O1–O4)</strong><br>
    • <strong>O1</strong>: Privileged Access → Min High<br>
    • <strong>O2</strong>: Sole-Source + O3 → Critical<br>
    • <strong>O3</strong>: D3 + V3 → Min High<br>
    • <strong>O4</strong>: P3 + O≥2 → Min High<br><br>
    <strong style="color:#0f172a">Governance (O5)</strong><br>
    Audited human analyst review &amp; written rationale.
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
if "filter_tier" not in st.session_state:
    st.session_state.filter_tier = "All"
if "show_raw_json" not in st.session_state:
    st.session_state.show_raw_json = False

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
    st.session_state.filter_tier = "All"
    st.rerun()

if run_btn:
    st.session_state.error_msg = None
    st.session_state.selected_id = None
    st.session_state.human_overrides = {}
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
tab_assess, tab_osint, tab_reports, tab_analytics = st.tabs([
    "⚡ Live Assessment",
    "🎯 OSINT Investigation Planner",
    "📁 Report Browser",
    "📊 Portfolio Analytics",
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
                        "D": r.D.score if r.D.score is not None else "?",
                        "P": r.P.score if r.P.score is not None else "?",
                        "R": r.R.score if r.R.score is not None else "?",
                        "O": r.O.score if r.O.score is not None else "?",
                        "V": r.V.score if r.V.score is not None else "?",
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

        # Determine criticality and human override if available
        osint_crit = None
        osint_h_rev = None
        if isinstance(selected_vendor_obj, CriticalityResult):
            osint_crit = selected_vendor_obj
            osint_h_rev = st.session_state.human_overrides.get(selected_vendor_obj.vendor_id)
        elif hasattr(selected_vendor_obj, "vendor_id"):
            osint_h_rev = st.session_state.human_overrides.get(selected_vendor_obj.vendor_id)
            for r in st.session_state.results:
                if getattr(r, "vendor_id", None) == selected_vendor_obj.vendor_id:
                    osint_crit = r
                    break

        planner = OSINTInvestigationPlanner()
        plan = planner.plan_investigation(
            vendor=selected_vendor_obj,
            criticality_input=osint_crit,
            human_review=osint_h_rev,
        )

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
                <span class="detail-tag">Rulebook Timebox: <strong>{plan.timebox.planning_target}</strong></span>
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
                <div class="stat-sub">S1–S12 source classes</div>
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
                <div class="stat-label">Downstream Gates</div>
                <div class="stat-value">4</div>
                <div class="stat-sub">G1–G4 checklist objectives</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-Column Layout ──
        osint_left, osint_right = st.columns([5, 5], gap="large")

        with osint_left:
            # 1. Stopping Condition Monitor
            st.markdown('<div class="sec-header"><span>1. Rulebook Stopping Condition</span></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="factor-card" style="line-height:1.5">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem">
                    <strong style="color:#0f172a;font-size:0.8rem">[{plan.stopping_rule_id}]</strong>
                    <span style="font-size:0.7rem;font-weight:700;color:#2563eb;background:#eff6ff;padding:0.15rem 0.5rem;border-radius:4px">
                        {plan.stop_status.value.replace('_', ' ')}
                    </span>
                </div>
                <div style="font-size:0.76rem;color:#334155;margin-bottom:0.4rem">
                    {plan.stopping_rule_description}
                </div>
                <div style="font-size:0.7rem;color:#64748b;border-top:1px solid #f1f5f9;padding-top:0.35rem">
                    <strong>Timebox Target:</strong> {plan.timebox.planning_target} (Provisional planning estimate · Expiry does not equal completion)
                </div>
            </div>
            """, unsafe_allow_html=True)

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

            # 3. Policy Conflicts (Fixed Sources vs Depth Profile)
            if plan.policy_conflicts:
                st.markdown('<div class="sec-header"><span>3. Policy Alignment Advisory</span></div>', unsafe_allow_html=True)
                for pc in plan.policy_conflicts:
                    st.markdown(f"""
                    <div class="advisory-box">
                        <strong>⚠️ {pc.source_name} ({pc.source_id}) Policy Variance</strong><br>
                        <span style="color:#78350f">{pc.issue}</span><br>
                        <em>Impact: {pc.impact_notes}</em>
                    </div>
                    """, unsafe_allow_html=True)

        with osint_right:
            # 1. Planned Source Classes
            st.markdown('<div class="sec-header"><span>3. Source Classes &amp; Discovery Catalog</span></div>', unsafe_allow_html=True)
            
            with st.expander(f"View All {len(plan.coverage_records)} Planned Source Classes", expanded=True):
                for s_id, s_rec in plan.coverage_records.items():
                    s_def = planner.source_registry.get_source(s_id)
                    grade = s_def.default_evidence_grade if s_def else "C"
                    rel = s_def.reliability_rating if s_def else 3
                    req_badge = "<span style='color:#dc2626;font-weight:700'>REQUIRED</span>" if s_rec.is_required else "<span style='color:#64748b'>CORROBORATIVE</span>"
                    st.markdown(f"""
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;padding:0.55rem 0.75rem;margin-bottom:0.4rem;font-size:0.75rem">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <strong>[{s_id}] {s_rec.source_name}</strong>
                            <span style="font-size:0.68rem">Grade {grade} (★{rel}) &nbsp;·&nbsp; {req_badge}</span>
                        </div>
                        <div style="color:#64748b;font-size:0.7rem;margin-top:0.2rem">
                            Tools: {', '.join(s_def.discovery_resources[:4]) if s_def else 'General Search'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # 2. Targeted Search Queries
            st.markdown('<div class="sec-header"><span>4. Generated Search Queries (Deduplicated)</span></div>', unsafe_allow_html=True)
            st.caption(f"{len(plan.planned_queries)} queries generated. Status: PLANNED (Ready for search adapter execution).")
            
            query_search_term = st.text_input("Filter Queries", placeholder="Filter by keyword (e.g. Bedrock, DPA, LLM)...", label_visibility="collapsed")
            
            displayed_queries = plan.planned_queries
            if query_search_term.strip():
                displayed_queries = [
                    q for q in displayed_queries
                    if query_search_term.lower() in q.rendered_query.lower() or any(query_search_term.lower() in s.lower() for s in q.source_classes)
                ]

            with st.container(height=320):
                for q in displayed_queries:
                    st.markdown(f"""
                    <div class="query-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem">
                            <span style="font-size:0.68rem;font-weight:700;color:#2563eb">[{', '.join(q.source_classes)}]</span>
                            <span style="font-size:0.65rem;color:#64748b">STATUS: {q.execution_status.value}</span>
                        </div>
                        <code>{q.rendered_query}</code>
                        <div style="font-size:0.68rem;color:#64748b;margin-top:0.25rem">
                            Targets: {', '.join(q.expected_evidence_targets[:2])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # 3. Export Plan Action Button
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
