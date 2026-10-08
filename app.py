"""Meridian Vendor Assessment — Streamlit Dashboard."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import json
import tempfile
from typing import Optional

import streamlit as st

from meridian_assessment.models.criticality import CriticalityAssessment, CriticalityTier
from meridian_assessment.models.assessment import AssessmentDepth
from meridian_assessment.services.pipeline import AssessmentPipeline
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

/* ── Criterion cards ── */
.crit-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; }
.crit-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
.crit-name  { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #6b7280; margin-bottom: 0.35rem; }
.crit-badge {
    display: inline-block;
    font-size: 0.7rem; font-weight: 700;
    padding: 0.15rem 0.45rem; border-radius: 4px;
    margin-bottom: 0.4rem;
}
.crit-badge.critical { background:#fef2f2; color:#b91c1c; }
.crit-badge.high     { background:#fff7ed; color:#c2410c; }
.crit-badge.medium   { background:#fffbeb; color:#92400e; }
.crit-badge.low      { background:#f0fdf4; color:#166534; }
.crit-badge.none     { background:#f9fafb; color:#6b7280; }
.crit-stats { font-size: 0.72rem; color: #9ca3af; }
.crit-stats strong { color: #374151; }

/* ── Bar chart ── */
.bar-row { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.55rem; }
.bar-lbl { font-size: 0.72rem; color: #374151; width: 145px; flex-shrink: 0; }
.bar-track { flex: 1; background: #f3f4f6; border-radius: 4px; height: 7px; }
.bar-fill  { border-radius: 4px; height: 7px; transition: width 0.3s ease; }
.bar-num   { font-size: 0.72rem; font-weight: 600; color: #374151; width: 32px; text-align: right; }

/* ── Reasoning ── */
.reason-item {
    display: flex; align-items: flex-start; gap: 0.55rem;
    padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;
    font-size: 0.8rem; color: #374151; line-height: 1.45;
}
.reason-chevron { color: #6366f1; font-size: 0.65rem; margin-top: 0.1rem; flex-shrink: 0; }

/* ── OSINT pending ── */
.osint-box {
    background: #f9fafb;
    border: 1px dashed #d1d5db;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
.osint-tag {
    display: inline-block;
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em;
    background: #f3f4f6; color: #6b7280;
    border-radius: 4px; padding: 0.15rem 0.45rem; margin-bottom: 0.6rem;
}
.osint-text { font-size: 0.78rem; color: #6b7280; line-height: 1.6; }

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

/* ── Upload area ── */
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #d1d5db !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_CSV = Path("data/sample/sample_vendors.csv")
CONFIG_PATH = Path("config/criticality.yaml")

TIER_LABEL = {
    CriticalityTier.TIER_1: "Tier 1 — High",
    CriticalityTier.TIER_2: "Tier 2 — Medium",
    CriticalityTier.TIER_3: "Tier 3 — Low",
}
TIER_CSS = {
    CriticalityTier.TIER_1: "t1",
    CriticalityTier.TIER_2: "t2",
    CriticalityTier.TIER_3: "t3",
}
DEPTH_LABEL = {
    AssessmentDepth.COMPREHENSIVE: "Comprehensive",
    AssessmentDepth.TARGETED: "Targeted",
    AssessmentDepth.LIGHTWEIGHT: "Lightweight",
}
CRIT_DISPLAY = {
    "data_sensitivity":      "Data Sensitivity",
    "payment_flows":         "Payment Flows",
    "regulatory_exposure":   "Regulatory Exposure",
    "operational_dependency":"Operational Dependency",
    "customer_data_volume":  "Customer Data Volume",
}
BAR_COLOR = {
    CriticalityTier.TIER_1: "#ef4444",
    CriticalityTier.TIER_2: "#f59e0b",
    CriticalityTier.TIER_3: "#10b981",
}
MAX_SCORE = 5.0  # max weighted score ceiling for bar scaling (all-critical)


def pill(tier: CriticalityTier) -> str:
    c = TIER_CSS[tier]
    return (
        f'<span class="pill {c}">'
        f'<span class="pill-dot"></span>'
        f'{TIER_LABEL[tier]}'
        f'</span>'
    )


def crit_badge(val: str) -> str:
    return f'<span class="crit-badge {val}">{val.upper()}</span>'


def run_pipeline(csv_bytes: bytes) -> list[CriticalityAssessment]:
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
            <div class="topbar-name">Meridian &nbsp; Vendor Risk</div>
            <div class="topbar-sub">Evidence-driven third-party security assessment</div>
        </div>
    </div>
    <div class="topbar-right">Criticality Assessment · v1.0</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Vendor Input")
    mode = st.radio("Source", ["Upload CSV", "Use sample dataset"], label_visibility="collapsed")

    uploaded_file = None
    use_sample = False

    if mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Drop vendor CSV",
            type=["csv"],
            label_visibility="collapsed",
            help=(
                "Required columns: vendor_id, vendor_name, domain, "
                "data_sensitivity, payment_flows, regulatory_exposure, "
                "operational_dependency, customer_data_volume\n\n"
                "Accepted values: critical / high / medium / low / none"
            ),
        )
    else:
        use_sample = True

    run_btn = st.button(
        "Run Assessment",
        type="primary",
        use_container_width=True,
        disabled=(not use_sample and uploaded_file is None),
    )

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#9ca3af;line-height:1.65">
    <strong style="color:#374151">Five scoring dimensions</strong><br>
    Data Sensitivity · Payment Flows<br>
    Regulatory Exposure<br>
    Operational Dependency<br>
    Customer Data Volume<br><br>
    <strong style="color:#374151">Tiers</strong><br>
    Tier 1 (≥ 3.5) → Comprehensive<br>
    Tier 2 (≥ 2.0) → Targeted<br>
    Tier 3 (&lt; 2.0) → Lightweight
    </div>
    """, unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "assessments" not in st.session_state:
    st.session_state.assessments: list[CriticalityAssessment] = []
if "error" not in st.session_state:
    st.session_state.error: Optional[str] = None
if "selected" not in st.session_state:
    st.session_state.selected: Optional[str] = None

# ─── Run pipeline ─────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.error = None
    st.session_state.selected = None
    try:
        if use_sample:
            if not SAMPLE_CSV.is_file():
                st.session_state.error = f"Sample file not found: {SAMPLE_CSV}"
            else:
                st.session_state.assessments = run_pipeline(SAMPLE_CSV.read_bytes())
        elif uploaded_file:
            st.session_state.assessments = run_pipeline(uploaded_file.read())
    except MeridianAssessmentError as exc:
        st.session_state.error = str(exc)
    except Exception as exc:
        st.session_state.error = f"Unexpected error — check your input file.\n\n{exc}"

# ─── Error ────────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(st.session_state.error)
    st.stop()

assessments = st.session_state.assessments

# ─── Empty state ──────────────────────────────────────────────────────────────
if not assessments:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🛡</div>
            <div class="empty-title">No assessment loaded</div>
            <div class="empty-hint">Upload a vendor CSV or choose the sample dataset,<br>then click <strong>Run Assessment</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─── Stat cards ───────────────────────────────────────────────────────────────
t1 = sum(1 for a in assessments if a.criticality_tier == CriticalityTier.TIER_1)
t2 = sum(1 for a in assessments if a.criticality_tier == CriticalityTier.TIER_2)
t3 = sum(1 for a in assessments if a.criticality_tier == CriticalityTier.TIER_3)

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card all">
        <div class="stat-label">Total Assessed</div>
        <div class="stat-value">{len(assessments)}</div>
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
col_left, col_right = st.columns([5, 4], gap="large")

# ── Left: results table ──
with col_left:
    st.markdown('<div class="sec-header">Assessment Results</div>', unsafe_allow_html=True)

    import pandas as pd

    rows = []
    for a in assessments:
        rows.append({
            "ID": a.vendor_id,
            "Vendor": a.vendor_name,
            "Tier": TIER_LABEL[a.criticality_tier],
            "Score": a.criticality_score,
            "Depth": DEPTH_LABEL[a.assessment_depth],
        })
    df = pd.DataFrame(rows)

    # pandas 2.x uses .map() not .applymap()
    def _tier_style(val: str) -> str:
        if "Tier 1" in val:
            return "background-color:#fef2f2;color:#b91c1c;font-weight:600;"
        if "Tier 2" in val:
            return "background-color:#fffbeb;color:#92400e;font-weight:600;"
        return "background-color:#ecfdf5;color:#065f46;font-weight:600;"

    styled = df.style.map(_tier_style, subset=["Tier"])

    ev = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if ev.selection and ev.selection.rows:
        st.session_state.selected = assessments[ev.selection.rows[0]].vendor_id

    # JSON export
    json_out = json.dumps([a.model_dump(mode="json") for a in assessments], indent=2)
    st.download_button(
        "⬇  Download JSON",
        data=json_out,
        file_name="assessment_results.json",
        mime="application/json",
        use_container_width=True,
    )

# ── Right: detail panel ──
with col_right:
    sel_id = st.session_state.selected
    sel: Optional[CriticalityAssessment] = next((a for a in assessments if a.vendor_id == sel_id), None)

    if sel is None:
        st.markdown('<div class="sec-header">Vendor Detail</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="empty-state" style="padding:2.5rem 1rem">
            <div class="empty-icon" style="font-size:1.4rem">←</div>
            <div class="empty-title">Select a vendor</div>
            <div class="empty-hint">Click any row in the results table<br>to see the full assessment breakdown.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        a = sel
        tier_css = TIER_CSS[a.criticality_tier]
        bar_color = BAR_COLOR[a.criticality_tier]

        # ── Vendor header ──
        st.markdown(f"""
        <div class="detail-header">
            <div class="detail-vendor-name">{a.vendor_name}</div>
            <div class="detail-domain">{a.domain}</div>
            <div class="detail-meta">
                {pill(a.criticality_tier)}
                <span class="detail-score">Score {a.criticality_score}</span>
                <span class="detail-score">{DEPTH_LABEL[a.assessment_depth]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Criterion grid ──
        st.markdown('<div class="sec-header">Dimension Scores</div>', unsafe_allow_html=True)

        cards_html = '<div class="crit-grid">'
        for k, res in a.criteria.items():
            name = CRIT_DISPLAY.get(k, k)
            val = res.value.value
            cards_html += f"""
            <div class="crit-card">
                <div class="crit-name">{name}</div>
                {crit_badge(val)}
                <div class="crit-stats">
                    Score <strong>{res.score}</strong> &nbsp;·&nbsp;
                    Weight <strong>{int(res.weight * 100)}%</strong> &nbsp;·&nbsp;
                    Contribution <strong>{res.weighted_score:.2f}</strong>
                </div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # ── Contribution bars ──
        st.markdown('<div class="sec-header">Weighted Contributions</div>', unsafe_allow_html=True)
        bars_html = ""
        for k, res in a.criteria.items():
            name = CRIT_DISPLAY.get(k, k)
            pct = min(res.weighted_score / (MAX_SCORE * res.weight) * 100, 100)
            bars_html += f"""
            <div class="bar-row">
                <div class="bar-lbl">{name}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
                </div>
                <div class="bar-num">{res.weighted_score:.2f}</div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

        # ── Reasoning ──
        st.markdown('<div class="sec-header">Assessment Reasoning</div>', unsafe_allow_html=True)
        reasons_html = ""
        for r in a.reasoning:
            reasons_html += f'<div class="reason-item"><span class="reason-chevron">▸</span><span>{r}</span></div>'
        st.markdown(reasons_html, unsafe_allow_html=True)

        # ── OSINT placeholder ──
        st.markdown('<div class="sec-header">External Investigation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="osint-box">
            <div class="osint-tag">Pending</div>
            <div class="osint-text">
                Automated discovery is not yet active for this vendor.<br><br>
                Planned capabilities:<br>
                &nbsp;· Public source footprint analysis<br>
                &nbsp;· Domain security posture (DNS, TLS, MX)<br>
                &nbsp;· Credential exposure monitoring<br>
                &nbsp;· Regulatory enforcement search<br>
                &nbsp;· Evidence collection &amp; AI-assisted triage
            </div>
        </div>
        """, unsafe_allow_html=True)
