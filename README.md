# Meridian Vendor Assessment

Evidence-driven, deterministic third-party vendor risk assessment for Meridian Financial.

---

## Overview

A vendor criticality classification system evaluating third-party vendors for risk. Criticality scoring is deterministic, explainable, and configuration-driven — without generative LLM scoring.

### Supported Criticality Methodologies

1. **Meridian D/P/R/O/V Methodology** (Primary):
   - Evaluates 5 factors on a **0 to 3 scale**:
     - **D (Data Sensitivity)** (30%): Classified strictly from data classification accessed; uses highest applicable level.
     - **P (Payment Flow Exposure)** (20%): Derived from actual service scope and business process; distinguishes direct actions (initiate, transmit, authorize, clear, settle) from payment reporting and transaction handling; supports deterministic negation.
     - **R (Regulatory Exposure)** (20%): Evaluates direct material regulatory obligations (notices, reporting, filings) vs indirect compliance or critical infrastructure.
     - **O (Operational Dependency)** (20%): Direct structured mapping (`Low`=0, `Moderate`=1, `High`=2, `Critical`=3).
     - **V (Annual Data Volume)** (10%): Count parser with unit scaling; isolates frequency indicators (e.g. monthly cycles) and population counts as `UNKNOWN` rather than misinterpreting them.
   - **Formula**:
     $$C = 0.30D + 0.20P + 0.20R + 0.20O + 0.10V$$
   - **Deterministic Overrides (O1–O5)**: Rule-based floor evaluations (e.g. privileged service credentials, direct payment clearing/settlement, critical financial infrastructure) that elevate vendors to Tier 1 when conditions are met.
   - **Provenance Tracking**: Every factor records its raw input, normalized input, determination method (`DIRECT`, `DETERMINISTIC_RULE`, `SEMANTIC_MATCH`, `HUMAN_REVIEW`, or `UNKNOWN`), matched concepts, and audit rationale.

2. **Standard Qualitative Assessment Mode** (Legacy):
   - 5 qualitative dimensions (`data_sensitivity`, `payment_flows`, `regulatory_exposure`, `operational_dependency`, `customer_data_volume`) on qualitative levels (`critical`, `high`, `medium`, `low`, `none`).

---

## Getting Started

### 1. Environment Setup

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows CMD**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### CLI Execution
```bash
python main.py --input data/input/vendors.csv
```

**CLI Options**:
```
  --input   PATH   Vendor CSV path (default: data/input/vendors.csv)
  --config  PATH   Scoring config YAML (default: config/criticality.yaml)
  --output  PATH   JSON results path (default: data/output/criticality_results.json)
  --log-level      DEBUG | INFO | WARNING | ERROR
```

### Streamlit Web Dashboard
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`. Allows switching between the **Meridian D/P/R/O/V Methodology** (with full factor provenance breakdown, override detection, and review flags) and the legacy assessment mode.

### Diagnostic Script (Meridian Dataset)
```bash
python scripts/generate_diagnostic.py
```
Runs the D/P/R/O/V engine against the case study vendor inventory (`data/input/meridian_vendors.csv`) and exports factor provenance details to `data/output/meridian_diagnostic.json`.

---

## Running Tests
```bash
pytest
```
Run with coverage:
```bash
pytest --cov=src/meridian_assessment
```

---

## Project Structure

```
.
├── AGENTS.md                  # Workspace rules (hygiene, gitignore, requirements, README accuracy)
├── config/
│   ├── factor_weights.yaml                 # D/P/R/O/V formula weights
│   ├── data_sensitivity_taxonomy.yaml      # D factor concept taxonomy
│   ├── operational_dependency_mapping.yaml # O factor structured mapping
│   ├── volume_thresholds.yaml              # V factor threshold parameters & unit definitions
│   ├── payment_ontology.yaml               # P factor ontology, verbs & negation rules
│   ├── regulatory_taxonomy.yaml            # R factor taxonomy & reference statements
│   ├── semantic_config.yaml                # Semantic embedding fallback configuration
│   └── criticality.yaml                    # Thresholds & legacy scoring configuration
├── data/
│   ├── input/
│   │   ├── meridian_vendors.csv            # Meridian case study vendor inventory (6 vendors)
│   │   └── vendors.csv                     # Standard vendor dataset
│   ├── sample/
│   │   └── sample_vendors.csv              # Minimal sample dataset
│   └── output/                             # Generated output files (gitignored)
├── src/meridian_assessment/
│   ├── config/                # YAML configuration loaders
│   ├── ingestion/             # CSV loaders (Meridian & Standard formats)
│   ├── models/                # Pydantic schemas (MeridianVendor, FactorResult, Vendor)
│   ├── osint/                 # OSINT contract interface
│   ├── services/              # Criticality engines, scoping, factor classifiers
│   │   └── factors/           # D, P, R, O, V modular factor classifiers & text utils
│   └── utils/                 # Logging and exception hierarchy
├── tests/
│   ├── unit/                  # Factor unit tests (D, P, R, O, V), config, models, engine
│   └── integration/           # Pipeline & CLI integration tests
├── scripts/
│   └── generate_diagnostic.py # Full diagnostic evaluation runner
├── app.py                     # Streamlit application dashboard
├── main.py                    # Command-line interface
├── pyproject.toml             # Project metadata & pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```