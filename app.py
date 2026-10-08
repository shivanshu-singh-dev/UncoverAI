"""Meridian Vendor Assessment — Streamlit Dashboard.

Supports both:
  1. Meridian D/P/R/O/V Methodology (Case study dataset with full provenance tracking)
  2. Standard Assessment Mode (legacy 5-dimension qualitative model)
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import json
import tempfile
from typing import Optional, Union

import streamlit as st

from meridian_assessment.models.criticality import CriticalityAssessment, CriticalityTier
from meridian_assessment.models.assessment import AssessmentDepth
from meridian_assessment.models.factor_result import CriticalityResult
from meridian_assessment.services.pipeline import AssessmentPipeline
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine
from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.utils.exceptions import MeridianAssessmentError

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meridian | Vendor Risk",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design system ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1400px; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 2rem;
}
.topbar-brand { display: flex; align-items: center; gap: 0.6rem; }
.topbar-logo {
    width: 32px; height: 32px;
    background: #111827;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; color: white; font-weight: 700;
}
.topbar-name { font-size: 1.05rem; font-weight: 700; color: #111827; letter-spacing: -0.01em; }
.topbar-sub  { font-size: 0.72rem; color: #6b7280; font-weight: 400; }
.topbar-right { font-size: 0.7rem; color: #9ca3af; }

/* ── Stat cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.1rem 1.25rem 1rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
}
.stat-card.all::before  { background: #6366f1; }
.stat-card.t1::before   { background: #ef4444; }
.stat-card.t2::before   { background: #f59e0b; }
.stat-card.t3::before   { background: #10b981; }
.stat-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 0.35rem; }
.stat-value { font-size: 2rem; font-weight: 700; color: #111827; line-height: 1; }
.stat-sub   { font-size: 0.72rem; color: #9ca3af; margin-top: 0.2rem; }

/* ── Section headers ── */
.sec-header {
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #9ca3af; margin: 1.5rem 0 0.75rem;
}

/* ── Tier pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.68rem; font-weight: 600;
    padding: 0.2rem 0.6rem; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.pill.t1 { background: #fef2f2; color: #b91c1c; }
.pill.t2 { background: #fffbeb; color: #92400e; }
.pill.t3 { background: #ecfdf5; color: #065f46; }
.pill-dot { width: 6px; height: 6px; border-radius: 50%; }
.pill.t1 .pill-dot { background: #ef4444; }
.pill.t2 .pill-dot { background: #f59e0b; }
.pill.t3 .pill-dot { background: #10b981; }

/* ── Detail panel ── */
.detail-header {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.detail-vendor-name { font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 0.15rem; }
.detail-domain { font-size: 0.78rem; color: #6b7280; font-family: monospace; margin-bottom: 0.75rem; }
.detail-meta { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
.detail-score {
    font-size: 0.78rem; font-weight: 600; color: #374151;
    background: #f3f4f6; border-radius: 6px;
    padding: 0.2rem 0.55rem;
}

/* ── Factor Cards ── */
.factor-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
}
.factor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.factor-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #111827;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.factor-score-badge {
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    background: #f3f4f6;
    color: #111827;
}
.factor-rationale {
    font-size: 0.78rem;
    color: #4b5563;
    line-height: 1.4;
    margin-bottom: 0.4rem;
}
.factor-meta {
    font-size: 0.68rem;
    color: #9ca3af;
}

/* ── Bar chart ── */
.bar-row { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.55rem; }
.bar-lbl { font-size: 0.72rem; color: #374151; width: 145px; flex-shrink: 0; }
.bar-track { flex: 1; background: #f3f4f6; border-radius: 4px; height: 7px; }
.bar-fill  { border-radius: 4px; height: 7px; transition: width 0.3s ease; }
.bar-num   { font-size: 0.72rem; font-weight: 600; color: #374151; width: 32px; text-align: right; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3.5rem 1rem;
    border: 1.5px dashed #e5e7eb; border-radius: 12px;
    color: #9ca3af;
}
.empty-icon { font-size: 2rem; margin-bottom: 0.75rem; }
.empty-title { font-size: 0.9rem; font-weight: 600; color: #6b7280; margin-bottom: 0.35rem; }
.empty-hint  { font-size: 0.78rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #f9fafb; border-right: 1px solid #e5e7eb; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MERIDIAN_DATASET_CSV = Path("data/input/meridian_vendors.csv")
LEGACY_SAMPLE_CSV = Path("data/sample/sample_vendors.csv")
CONFIG_PATH = Path("config/criticality.yaml")

TIER_LABEL = {
    "TIER_1": "Tier 1 — High",
    "TIER_2": "Tier 2 — Medium",
    "TIER_3": "Tier 3 — Low",
    CriticalityTier.TIER_1: "Tier 1 — High",
    CriticalityTier.TIER_2: "Tier 2 — Medium",
    CriticalityTier.TIER_3: "Tier 3 — Low",
}
TIER_CSS = {
    "TIER_1": "t1",
    "TIER_2": "t2",
    "TIER_3": "t3",
    CriticalityTier.TIER_1: "t1",
    CriticalityTier.TIER_2: "t2",
    CriticalityTier.TIER_3: "t3",
}
DEPTH_LABEL = {
    "COMPREHENSIVE": "Comprehensive",
    "TARGETED": "Targeted",
    "LIGHTWEIGHT": "Lightweight",
    AssessmentDepth.COMPREHENSIVE: "Comprehensive",
    AssessmentDepth.TARGETED: "Targeted",
    AssessmentDepth.LIGHTWEIGHT: "Lightweight",
}


def pill(tier_val: str) -> str:
    c = TIER_CSS.get(tier_val, "t3")
    lbl = TIER_LABEL.get(tier_val, str(tier_val))
    return f'<span class="pill {c}"><span class="pill-dot"></span>{lbl}</span>'


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


# ─── Top bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">M</div>
        <div>
            <div class="topbar-name">Meridian &nbsp; Vendor Risk Assessment</div>
            <div class="topbar-sub">Deterministic Criticality Engine — D/P/R/O/V Methodology</div>
        </div>
    </div>
    <div class="topbar-right">Evidence-Driven · Zero-LLM Scoring</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Methodology Mode")
    engine_mode = st.radio(
        "Engine Selection",
        ["Meridian D/P/R/O/V (New)", "Standard Qualitative (Legacy)"],
        label_visibility="collapsed",
    )

    st.markdown("#### Vendor Input")
    if engine_mode == "Meridian D/P/R/O/V (New)":
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

    run_btn = st.button("Run Assessment", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#6b7280;line-height:1.65">
    <strong style="color:#111827">D/P/R/O/V Formula</strong><br>
    C = 0.30D + 0.20P + 0.20R + 0.20O + 0.10V<br><br>
    <strong style="color:#111827">Factor Scale</strong><br>
    0 to 3 per factor with explicit provenance.<br><br>
    <strong style="color:#111827">Overrides</strong><br>
    O1–O5 deterministic floor triggers.
    </div>
    """, unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []
if "mode_used" not in st.session_state:
    st.session_state.mode_used = ""
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "error_msg" not in st.session_state:
    st.session_state.error_msg = None

if run_btn:
    st.session_state.error_msg = None
    st.session_state.selected_id = None
    try:
        if engine_mode == "Meridian D/P/R/O/V (New)":
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
    except Exception as e:
        st.session_state.error_msg = str(e)

if st.session_state.error_msg:
    st.error(st.session_state.error_msg)
    st.stop()

results = st.session_state.results

if not results:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🛡</div>
            <div class="empty-title">No assessment loaded</div>
            <div class="empty-hint">Select a dataset in the sidebar and click <strong>Run Assessment</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─── Stat cards ───────────────────────────────────────────────────────────────
t1 = sum(1 for r in results if getattr(r, "criticality_tier", "") == "TIER_1" or getattr(r, "criticality_tier", None) == CriticalityTier.TIER_1)
t2 = sum(1 for r in results if getattr(r, "criticality_tier", "") == "TIER_2" or getattr(r, "criticality_tier", None) == CriticalityTier.TIER_2)
t3 = sum(1 for r in results if getattr(r, "criticality_tier", "") == "TIER_3" or getattr(r, "criticality_tier", None) == CriticalityTier.TIER_3)

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card all">
        <div class="stat-label">Total Assessed</div>
        <div class="stat-value">{len(results)}</div>
        <div class="stat-sub">vendors</div>
    </div>
    <div class="stat-card t1">
        <div class="stat-label">Tier 1 — High Risk</div>
        <div class="stat-value">{t1}</div>
        <div class="stat-sub">Comprehensive review</div>
    </div>
    <div class="stat-card t2">
        <div class="stat-label">Tier 2 — Medium Risk</div>
        <div class="stat-value">{t2}</div>
        <div class="stat-sub">Targeted review</div>
    </div>
    <div class="stat-card t3">
        <div class="stat-label">Tier 3 — Low Risk</div>
        <div class="stat-value">{t3}</div>
        <div class="stat-sub">Lightweight review</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Two-column layout ────────────────────────────────────────────────────────
col_left, col_right = st.columns([5, 5], gap="large")

with col_left:
    st.markdown('<div class="sec-header">Assessment Results</div>', unsafe_allow_html=True)
    import pandas as pd

    rows = []
    is_meridian = st.session_state.mode_used == "meridian"

    for r in results:
        if is_meridian:
            rows.append({
                "ID": r.vendor_id,
                "Vendor": r.vendor_name,
                "Tier": TIER_LABEL.get(r.criticality_tier, str(r.criticality_tier)),
                "Score": f"{r.base_score:.2f}" if r.base_score is not None else "UNKNOWN",
                "Status": r.score_status.value,
                "D": r.D.score if r.D.score is not None else "?",
                "P": r.P.score if r.P.score is not None else "?",
                "R": r.R.score if r.R.score is not None else "?",
                "O": r.O.score if r.O.score is not None else "?",
                "V": r.V.score if r.V.score is not None else "?",
            })
        else:
            rows.append({
                "ID": r.vendor_id,
                "Vendor": r.vendor_name,
                "Tier": TIER_LABEL.get(r.criticality_tier, str(r.criticality_tier)),
                "Score": f"{r.criticality_score:.2f}",
                "Depth": DEPTH_LABEL.get(r.assessment_depth, str(r.assessment_depth)),
            })

    df = pd.DataFrame(rows)

    def _tier_style(val: str) -> str:
        if "Tier 1" in str(val):
            return "background-color:#fef2f2;color:#b91c1c;font-weight:600;"
        if "Tier 2" in str(val):
            return "background-color:#fffbeb;color:#92400e;font-weight:600;"
        return "background-color:#ecfdf5;color:#065f46;font-weight:600;"

    styled = df.style.map(_tier_style, subset=["Tier"])
    ev = st.dataframe(styled, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")

    if ev.selection and ev.selection.rows:
        st.session_state.selected_id = results[ev.selection.rows[0]].vendor_id

    # JSON export
    json_out = json.dumps([r.model_dump(mode="json") for r in results], indent=2)
    st.download_button("⬇ Download Results JSON", data=json_out, file_name="criticality_results.json", mime="application/json", use_container_width=True)

with col_right:
    sel_id = st.session_state.selected_id
    sel = next((r for r in results if r.vendor_id == sel_id), None)

    if sel is None:
        st.markdown('<div class="sec-header">Vendor Factor Detail</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="empty-state" style="padding:2.5rem 1rem">
            <div class="empty-icon" style="font-size:1.4rem">←</div>
            <div class="empty-title">Select a vendor</div>
            <div class="empty-hint">Click any row in the results table to view its full factor breakdown and provenance audit trail.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Render Selected Vendor
        if is_meridian:
            # ── MERIDIAN D/P/R/O/V VIEW ──
            r: CriticalityResult = sel
            st.markdown(f"""
            <div class="detail-header">
                <div class="detail-vendor-name">{r.vendor_name} ({r.vendor_id})</div>
                <div class="detail-meta">
                    {pill(r.criticality_tier or 'TIER_3')}
                    <span class="detail-score">Base Score: <strong>{r.base_score}</strong></span>
                    <span class="detail-score">Status: <strong>{r.score_status.value}</strong></span>
                    <span class="detail-score">Depth: <strong>{r.assessment_depth}</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if r.override_applied:
                triggered_ids = [o.override_id for o in r.overrides_evaluated if o.triggered]
                st.warning(f"**Override Triggered:** Floor applied via {triggered_ids} -> Tier 1 (Comprehensive)")

            if r.requires_review:
                st.info(f"**Review Notes:** {', '.join(r.review_notes)}")

            st.markdown('<div class="sec-header">Factor Provenance Breakdown</div>', unsafe_allow_html=True)

            factors_to_show = [
                ("D — Data Sensitivity", r.D),
                ("P — Payment Flow Exposure", r.P),
                ("R — Regulatory Exposure", r.R),
                ("O — Operational Dependency", r.O),
                ("V — Annual Data Volume", r.V),
            ]

            for label, f in factors_to_show:
                score_str = str(f.score) if f.score is not None else "UNKNOWN"
                contrib_str = f"{f.weighted_contribution:.2f}" if f.weighted_contribution is not None else "N/A"
                st.markdown(f"""
                <div class="factor-card">
                    <div class="factor-header">
                        <span class="factor-title">{label}</span>
                        <span class="factor-score-badge">Score: {score_str} (Contrib: {contrib_str})</span>
                    </div>
                    <div class="factor-rationale"><strong>Rationale:</strong> {f.rationale}</div>
                    <div class="factor-meta">
                        Method: <code>{f.determination_method.value}</code> &nbsp;·&nbsp; Source: {', '.join(f.source_fields)} &nbsp;·&nbsp; Input: "{f.raw_input or 'None'}"
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            # ── LEGACY 5-DIMENSION VIEW ──
            a: CriticalityAssessment = sel
            st.markdown(f"""
            <div class="detail-header">
                <div class="detail-vendor-name">{a.vendor_name}</div>
                <div class="detail-domain">{a.domain}</div>
                <div class="detail-meta">
                    {pill(a.criticality_tier)}
                    <span class="detail-score">Score: {a.criticality_score}</span>
                    <span class="detail-score">{DEPTH_LABEL[a.assessment_depth]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="sec-header">Assessment Reasoning</div>', unsafe_allow_html=True)
            for reason in a.reasoning:
                st.markdown(f"• {reason}")
