# Meridian Vendor Assessment

Evidence-driven, deterministic third-party vendor risk assessment for Meridian Financial.

---

## Overview

Classifies third-party vendors into risk tiers using five structured criteria. Every tier assignment traces back to specific weights, scores, and configuration values — no black-box scoring.

**Five scoring dimensions:**
- `data_sensitivity` — access to regulated financial records, PII, IP, auth/security systems
- `payment_flows` — payment processing, routing, reconciliation, monetary transactions
- `regulatory_exposure` — SEC, FINRA, GDPR, NYDFS, PCI-DSS exposure
- `operational_dependency` — business criticality and disruption potential
- `customer_data_volume` — scale of customer identity records processed

**Tiers (inclusive lower bounds):**
| Tier | Min Score | Assessment Depth |
|---|---|---|
| Tier 1 — High | ≥ 3.5 | Comprehensive |
| Tier 2 — Medium | ≥ 2.0 | Targeted |
| Tier 3 — Low | < 2.0 | Lightweight |

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

### CLI
```bash
python main.py --input data/input/vendors.csv
```

Options:
```
  --input   PATH   Vendor CSV path (default: data/input/vendors.csv)
  --config  PATH   Scoring config YAML (default: config/criticality.yaml)
  --output  PATH   JSON results path (default: data/output/criticality_results.json)
  --log-level      DEBUG | INFO | WARNING | ERROR
```

### Web UI
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a vendor CSV or use the built-in sample dataset.

---

## Running Tests
```bash
pytest
pytest --cov=src/meridian_assessment
```

---

## CSV Input Format

| Column | Required values |
|---|---|
| `vendor_id` | Unique string |
| `vendor_name` | Display name |
| `domain` | e.g. `example.com` |
| `data_sensitivity` | `critical` / `high` / `medium` / `low` / `none` |
| `payment_flows` | `critical` / `high` / `medium` / `low` / `none` |
| `regulatory_exposure` | `critical` / `high` / `medium` / `low` / `none` |
| `operational_dependency` | `critical` / `high` / `medium` / `low` / `none` |
| `customer_data_volume` | `critical` / `high` / `medium` / `low` / `none` |

---

## Scoring Configuration

All weights, severity scales, and tier thresholds are in [`config/criticality.yaml`](config/criticality.yaml). Modify without touching Python:

```yaml
criterion_weights:
  data_sensitivity: 0.25
  payment_flows: 0.20
  regulatory_exposure: 0.20
  operational_dependency: 0.20
  customer_data_volume: 0.15

severity_scores:
  critical: 5.0
  high: 4.0
  medium: 2.5
  low: 1.0
  none: 0.0
```

**Verified example:**
```
data_sensitivity:       critical (5.0) × 0.25 = 1.25
payment_flows:          critical (5.0) × 0.20 = 1.00
regulatory_exposure:    high     (4.0) × 0.20 = 0.80
operational_dependency: critical (5.0) × 0.20 = 1.00
customer_data_volume:   high     (4.0) × 0.15 = 0.60
                                          Total = 4.65 → Tier 1
```

---

## Project Structure

```
.
├── config/
│   └── criticality.yaml
├── data/
│   ├── input/vendors.csv
│   └── sample/sample_vendors.csv
├── src/meridian_assessment/
│   ├── config/       # Config loader
│   ├── ingestion/    # CSV loader
│   ├── models/       # Pydantic schemas
│   ├── osint/        # OSINT interface contract
│   ├── services/     # Criticality engine, scoping, pipeline
│   └── utils/        # Logger, exceptions
├── tests/
│   ├── unit/
│   └── integration/
├── app.py            # Streamlit UI
└── main.py           # CLI
```