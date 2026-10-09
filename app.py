"""Meridian Vendor Assessment — Streamlit Dashboard.

Supports both:
  1. Meridian D/P/R/O/V Methodology (Case study dataset with full provenance tracking, O1-O4 safeguards, and O5 governance)
  2. Standard Qualitative Mode (Legacy 5-dimension model)
"""

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
from meridian_assessment.models.factor_result import CriticalityLevel, CriticalityResult, HumanReviewRecord
from meridian_assessment.services.pipeline import AssessmentPipeline
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine
from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader

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

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.25rem 2rem 2.25rem; max-width: 1440px; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
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
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
}
.stat-card.all::before      { background: #6366f1; }
.stat-card.critical::before { background: #991b1b; }
.stat-card.high::before     { background: #ef4444; }
.stat-card.medium::before   { background: #f59e0b; }
.stat-card.low::before      { background: #10b981; }

.stat-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 0.3rem; }
.stat-value { font-size: 1.85rem; font-weight: 700; color: #111827; line-height: 1; }
.stat-sub   { font-size: 0.72rem; color: #9ca3af; margin-top: 0.2rem; }

/* ── Section headers ── */
.sec-header {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: #4b5563; margin: 1.25rem 0 0.6rem;
}

/* ── Criticality pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.7rem; font-weight: 600;
    padding: 0.2rem 0.6rem; border-radius: 20px;
    letter-spacing: 0.03em;
}
.pill.critical { background: #fee2e2; color: #991b1b; }
.pill.high     { background: #fef2f2; color: #b91c1c; }
.pill.medium   { background: #fffbeb; color: #92400e; }
.pill.low      { background: #ecfdf5; color: #065f46; }

.pill-dot { width: 6px; height: 6px; border-radius: 50%; }
.pill.critical .pill-dot { background: #991b1b; }
.pill.high .pill-dot     { background: #ef4444; }
.pill.medium .pill-dot   { background: #f59e0b; }
.pill.low .pill-dot      { background: #10b981; }

/* ── Detail container ── */
.detail-header {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.85rem;
}
.detail-vendor-name { font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 0.35rem; }
.detail-meta { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.detail-tag {
    font-size: 0.75rem; font-weight: 600; color: #374151;
    background: #f3f4f6; border-radius: 6px;
    padding: 0.2rem 0.55rem;
}

/* ── Factor Cards ── */
.factor-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    margin-bottom: 0.5rem;
}
.factor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
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
    color: #374151;
    line-height: 1.4;
    margin-bottom: 0.25rem;
}
.factor-meta {
    font-size: 0.68rem;
    color: #9ca3af;
}

/* ── Override box ── */
.override-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.45rem;
    font-size: 0.75rem;
}
.override-box.triggered {
    background: #fff7ed;
    border-color: #fdba74;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3rem 1rem;
    border: 1.5px dashed #e5e7eb; border-radius: 12px;
    color: #9ca3af;
}
.empty-icon { font-size: 1.75rem; margin-bottom: 0.5rem; }
.empty-title { font-size: 0.9rem; font-weight: 600; color: #6b7280; margin-bottom: 0.25rem; }
.empty-hint  { font-size: 0.78rem; }

section[data-testid="stSidebar"] { background: #f9fafb; border-right: 1px solid #e5e7eb; }
section[data-testid="stSidebar"] .block-container { padding: 1.25rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MERIDIAN_DATASET_CSV = Path("data/input/meridian_vendors.csv")
LEGACY_SAMPLE_CSV = Path("data/sample/sample_vendors.csv")
CONFIG_PATH = Path("config/criticality.yaml")


def pill(val: str) -> str:
    norm = val.lower().replace("tier 1 — ", "").replace("tier 2 — ", "").replace("tier 3 — ", "").strip()
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
            <div class="topbar-sub">Deterministic Criticality Engine — D/P/R/O/V Methodology with Safeguards &amp; Governance</div>
        </div>
    </div>
    <div class="topbar-right">Critical / High / Medium / Low Model · Zero Generative LLM Scoring</div>
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

    run_btn = st.button("Run Assessment", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#6b7280;line-height:1.6">
    <strong style="color:#111827">Formula</strong><br>
    C = 0.30D + 0.20P + 0.20R + 0.20O + 0.10V<br><br>
    <strong style="color:#111827">Safeguards (O1–O4)</strong><br>
    • <strong>O1</strong>: Privileged Access (Min High)<br>
    • <strong>O2</strong>: Sole-Source + O3 (Critical)<br>
    • <strong>O3</strong>: D3 + V3 (Min High)<br>
    • <strong>O4</strong>: P3 + O≥2 (Min High)<br><br>
    <strong style="color:#111827">Governance (O5)</strong><br>
    Human analyst approval &amp; rationalized changes.
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
if "human_overrides" not in st.session_state:
    st.session_state.human_overrides = {}

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

# ─── Stat cards ───
is_meridian = st.session_state.mode_used == "meridian"
crit_count = sum(1 for r in results if (getattr(r, "final_criticality", "") == CriticalityLevel.CRITICAL or getattr(r, "criticality_tier", "") == CriticalityTier.TIER_1))
high_count = sum(1 for r in results if getattr(r, "final_criticality", "") == CriticalityLevel.HIGH)
med_count  = sum(1 for r in results if (getattr(r, "final_criticality", "") == CriticalityLevel.MEDIUM or getattr(r, "criticality_tier", "") == CriticalityTier.TIER_2))
low_count  = sum(1 for r in results if (getattr(r, "final_criticality", "") == CriticalityLevel.LOW or getattr(r, "criticality_tier", "") == CriticalityTier.TIER_3))

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card all">
        <div class="stat-label">Total Assessed</div>
        <div class="stat-value">{len(results)}</div>
        <div class="stat-sub">vendors</div>
    </div>
    <div class="stat-card high">
        <div class="stat-label">Critical / High</div>
        <div class="stat-value">{crit_count + high_count}</div>
        <div class="stat-sub">Comprehensive depth</div>
    </div>
    <div class="stat-card medium">
        <div class="stat-label">Medium Risk</div>
        <div class="stat-value">{med_count}</div>
        <div class="stat-sub">Targeted depth</div>
    </div>
    <div class="stat-card low">
        <div class="stat-label">Low Risk</div>
        <div class="stat-value">{low_count}</div>
        <div class="stat-sub">Lightweight depth</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Two-column layout ────────────────────────────────────────────────────────
col_left, col_right = st.columns([5, 5], gap="large")

with col_left:
    st.markdown('<div class="sec-header">Assessment Inventory</div>', unsafe_allow_html=True)
    import pandas as pd

    rows = []
    for r in results:
        if is_meridian:
            # apply any session-stored human review
            h_rev = st.session_state.human_overrides.get(r.vendor_id)
            final_c = h_rev.final_criticality if h_rev else r.final_criticality
            rows.append({
                "ID": r.vendor_id,
                "Vendor": r.vendor_name,
                "Criticality": final_c,
                "Base Score": f"{r.base_score:.2f}" if r.base_score is not None else "UNKNOWN",
                "D": r.D.score if r.D.score is not None else "?",
                "P": r.P.score if r.P.score is not None else "?",
                "R": r.R.score if r.R.score is not None else "?",
                "O": r.O.score if r.O.score is not None else "?",
                "V": r.V.score if r.V.score is not None else "?",
                "Safeguards": "Triggered" if r.automatic_override_applied else "None",
            })
        else:
            rows.append({
                "ID": r.vendor_id,
                "Vendor": r.vendor_name,
                "Criticality": str(r.criticality_tier),
                "Score": f"{r.criticality_score:.2f}",
                "Depth": str(r.assessment_depth),
            })

    df = pd.DataFrame(rows)

    def _crit_style(val: str) -> str:
        s = str(val).lower()
        if "critical" in s:
            return "background-color:#fee2e2;color:#991b1b;font-weight:700;"
        if "high" in s:
            return "background-color:#fef2f2;color:#b91c1c;font-weight:600;"
        if "medium" in s:
            return "background-color:#fffbeb;color:#92400e;font-weight:600;"
        return "background-color:#ecfdf5;color:#065f46;font-weight:600;"

    styled = df.style.map(_crit_style, subset=["Criticality"])
    ev = st.dataframe(styled, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")

    if ev.selection and ev.selection.rows:
        st.session_state.selected_id = results[ev.selection.rows[0]].vendor_id

    # JSON export
    json_out = json.dumps([r.model_dump(mode="json") for r in results], indent=2)
    st.download_button("⬇ Download Audit Trail JSON", data=json_out, file_name="meridian_assessment_results.json", mime="application/json", use_container_width=True)

with col_right:
    sel_id = st.session_state.selected_id
    sel = next((r for r in results if r.vendor_id == sel_id), None)

    if sel is None:
        st.markdown('<div class="sec-header">Vendor Audit &amp; Governance Detail</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="empty-state" style="padding:2.5rem 1rem">
            <div class="empty-icon">←</div>
            <div class="empty-title">Select a vendor</div>
            <div class="empty-hint">Click any vendor in the table to view its complete factor scores, weighted calculation, automatic safeguards, and human review controls.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if is_meridian:
            r: CriticalityResult = sel
            # Apply human review if recorded in session
            user_override = st.session_state.human_overrides.get(r.vendor_id)
            current_final_crit = user_override.final_criticality if user_override else r.final_criticality

            # ── 1. Vendor Header ──
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

            # ── 2. Factor Scores (D / P / R / O / V) ──
            st.markdown('<div class="sec-header">1. Factor Scores &amp; Provenance</div>', unsafe_allow_html=True)
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
                    <div class="factor-rationale">{f.rationale}</div>
                    <div class="factor-meta">
                        Method: <code>{f.determination_method.value}</code> &nbsp;·&nbsp; Source: {', '.join(f.source_fields)} &nbsp;·&nbsp; Input: "{f.raw_input or 'None'}"
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── 3. Weighted Calculation & Provisional Criticality ──
            st.markdown('<div class="sec-header">2. Weighted Calculation &amp; Provisional Criticality</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="factor-card" style="font-size:0.8rem;line-height:1.6">
                <strong>Formula:</strong> <code>0.30D + 0.20P + 0.20R + 0.20O + 0.10V</code><br>
                <strong>Base Score:</strong> <strong>{r.base_score}</strong><br>
                <strong>Provisional Criticality:</strong> {pill(r.provisional_criticality or 'Low')}
            </div>
            """, unsafe_allow_html=True)

            # ── 4. Automatic Safeguards (O1-O4) ──
            st.markdown('<div class="sec-header">3. Automatic Safeguards (O1–O4 Overrides)</div>', unsafe_allow_html=True)
            for o in r.overrides_evaluated:
                trig_cls = "triggered" if o.triggered else ""
                status_txt = f"<strong style='color:#c2410c;'>TRIGGERED (Floor: {o.resulting_floor})</strong>" if o.triggered else "<span style='color:#6b7280;'>Not triggered</span>"
                st.markdown(f"""
                <div class="override-box {trig_cls}">
                    <strong>[{o.override_id}] {o.description}</strong> — {status_txt}<br>
                    <span style="color:#64748b">Condition: {o.condition_evaluated} &nbsp;·&nbsp; Evidence: {o.evidence}</span><br>
                    <span style="color:#334155">{o.rationale}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-size:0.8rem;margin:0.4rem 0 0.8rem">
                <strong>Proposed Criticality (post-safeguards):</strong> {pill(r.proposed_criticality or 'Low')}
            </div>
            """, unsafe_allow_html=True)

            # ── 5. Human Review (O5 Governance) ──
            st.markdown('<div class="sec-header">4. Human Review &amp; Governance (O5)</div>', unsafe_allow_html=True)
            
            with st.container(border=True):
                choice = st.radio(
                    "Governance Action",
                    [f"Keep Proposed ({r.proposed_criticality})", "Change Criticality"],
                    horizontal=True,
                    key=f"gov_choice_{r.vendor_id}",
                )

                if "Change" in choice:
                    new_val = st.selectbox(
                        "Controlled Criticality Selection",
                        [CriticalityLevel.LOW, CriticalityLevel.MEDIUM, CriticalityLevel.HIGH, CriticalityLevel.CRITICAL],
                        index=2,
                        key=f"new_crit_{r.vendor_id}",
                    )
                    rationale_input = st.text_area(
                        "Mandatory Audit Rationale",
                        placeholder="Provide formal audit / committee rationale for overriding calculated criticality...",
                        key=f"gov_rat_{r.vendor_id}",
                    )
                    if st.button("Save Governance Override", type="primary", key=f"btn_save_{r.vendor_id}"):
                        if not rationale_input.strip():
                            st.error("A written rationale is required when changing the criticality.")
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
                    if st.button("Confirm Proposed Criticality", key=f"btn_confirm_{r.vendor_id}"):
                        st.session_state.human_overrides[r.vendor_id] = HumanReviewRecord(
                            user_decision="KEEP_PROPOSED",
                            original_criticality=r.proposed_criticality or "Low",
                            final_criticality=r.proposed_criticality or "Low",
                            rationale="Analyst confirmed proposed automated criticality.",
                        )
                        st.info("Proposed criticality confirmed.")
                        st.rerun()

                if user_override:
                    st.markdown(f"""
                    <div style="font-size:0.75rem;background:#f1f5f9;border-radius:6px;padding:0.4rem 0.6rem;margin-top:0.4rem">
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
                <div class="detail-domain">{a.domain}</div>
                <div class="detail-meta">
                    {pill(str(a.criticality_tier))}
                    <span class="detail-tag">Score: {a.criticality_score}</span>
                    <span class="detail-tag">{DEPTH_LABEL.get(a.assessment_depth, str(a.assessment_depth))}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="sec-header">Assessment Reasoning</div>', unsafe_allow_html=True)
            for reason in a.reasoning:
                st.markdown(f"• {reason}")
