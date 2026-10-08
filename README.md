# Meridian Vendor Assessment POC — Phase 1: Foundation

Evidence-driven, deterministic third-party vendor risk assessment foundation for **Meridian Financial**.

Phase 1 provides vendor ingestion, data validation, deterministic criticality scoring, assessment-depth scoping, and a future-facing OSINT interface abstraction.

---

## Features (Phase 1)
- **Vendor Ingestion & Validation**: Ingests vendor lists (CSV) with schema enforcement, domain validation, categorical level normalization, and duplicate ID detection.
- **5 Core Criticality Dimensions**:
  1. **Data Sensitivity**: Access to regulated financial records, PII, intellectual property, authentication/security systems.
  2. **Payment Flows**: Involvement in payment processing, routing, reconciliation, or monetary transactions.
  3. **Regulatory Exposure**: Exposure to SEC, FINRA, GDPR, NYDFS, and PCI-DSS compliance frameworks.
  4. **Operational Dependency**: Business criticality and disruption potential on operational paths.
  5. **Customer Data Volume**: Scale of customer identity records processed.
- **Configurable Scoring & Thresholds**: Weights, severity scales, tier thresholds, and reasoning rules are fully decoupled in `config/criticality.yaml`.
- **Explainable & Deterministic Output**: Generates machine-readable JSON results with per-criterion breakdowns and human-readable audit justification drivers.
- **Assessment Depth Scoping**: Deterministically assigns depth (`COMPREHENSIVE`, `TARGETED`, `LIGHTWEIGHT`) based on criticality tier.
- **Future-Proof OSINT Contract**: Abstract OSINT interface ready for Phase 2 search sources, LLM analysis, and RAG pipelines without refactoring core services.

---

## Directory Structure

```text
UncoverAI/
├── config/
│   └── criticality.yaml          # Configurable scoring rules & thresholds
├── data/
│   ├── input/
│   │   └── vendors.csv           # Default input vendor list
│   ├── sample/
│   │   └── sample_vendors.csv    # Sample input dataset
│   └── output/
│       └── criticality_results.json # Generated structured results
├── docs/
│   └── architecture.md           # Architecture documentation
├── src/
│   └── meridian_assessment/
│       ├── __init__.py
│       ├── config/               # Configuration loading & validation
│       ├── ingestion/            # Vendor data ingestion loaders
│       ├── models/               # Pydantic schemas (Vendor, Criticality, Scope)
│       ├── osint/                # Future-facing OSINT abstraction contract
│       ├── services/             # Scoring, scoping, and pipeline orchestrator
│       └── utils/                # Logging, custom exceptions
├── tests/
│   ├── unit/                     # Unit test suite
│   └── integration/              # End-to-end and CLI integration tests
├── .env.example                  # Environment variable template
├── .gitignore                    # Project hygiene rules
├── main.py                       # CLI entrypoint
├── pyproject.toml                # Project build & test configuration
├── README.md                     # Project documentation
└── requirements.txt              # Pinned lightweight dependencies
```

---

## Getting Started

### 1. Environment Setup

Create and activate a dedicated virtual environment:

#### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Linux / macOS
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

### Run CLI Assessment
Execute the assessment pipeline against a vendor CSV:

```bash
python main.py --input data/input/vendors.csv
```

#### Custom Options
```bash
python main.py --input data/sample/sample_vendors.csv \
               --config config/criticality.yaml \
               --output data/output/criticality_results.json \
               --log-level INFO
```

### Example CLI Output
```text
Meridian Vendor Assessment
--------------------------

Loaded vendors: 10

V001 | Apex Core Banking Cloud
Criticality: Tier 1 (Score: 4.65)
Assessment Depth: Comprehensive
Key Drivers:
  - Accesses sensitive financial, PII, or security-critical data (Rating: CRITICAL, Score: 5.0)
  - Directly involved in payment processing, routing, or monetary transactions (Rating: CRITICAL, Score: 5.0)
  - Significant regulatory compliance exposure (SEC, FINRA, GDPR, NYDFS, PCI-DSS) (Rating: HIGH, Score: 4.0)
  - Critical operational dependency on critical business paths (Rating: CRITICAL, Score: 5.0)
  - Processes large volume or high scale of customer identity records (Rating: HIGH, Score: 4.0)

V002 | PayStream Global
Criticality: Tier 1 (Score: 4.2)
Assessment Depth: Comprehensive
Key Drivers:
  - Accesses sensitive financial, PII, or security-critical data (Rating: HIGH, Score: 4.0)
  - Directly involved in payment processing, routing, or monetary transactions (Rating: CRITICAL, Score: 5.0)
  - Significant regulatory compliance exposure (SEC, FINRA, GDPR, NYDFS, PCI-DSS) (Rating: HIGH, Score: 4.0)
  - Critical operational dependency on critical business paths (Rating: HIGH, Score: 4.0)
  - Processes large volume or high scale of customer identity records (Rating: HIGH, Score: 4.0)

Assessment completed successfully.
```

---

## Configuration (`config/criticality.yaml`)

To customize scoring weights, severity scales, or tier thresholds without modifying code, edit `config/criticality.yaml`:

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

tier_thresholds:
  TIER_1:
    min_score: 3.5
    label: "Tier 1 — High"
  TIER_2:
    min_score: 2.0
    label: "Tier 2 — Medium"
  TIER_3:
    min_score: 0.0
    label: "Tier 3 — Low"
```

---

## Running Tests

Run the full test suite with `pytest`:

```bash
pytest
```

To run with test coverage reporting:
```bash
pytest --cov=src/meridian_assessment
```

---

## Phase 2 Roadmap
- **OSINT Discovery & Scraping**: Ingest public data sources, corporate registries, certificate transparency logs, and domain hygiene endpoints based on `AssessmentScope`.
- **Local LLM & RAG Integration**: Evidence extraction and correlation using embeddings and local LLM models.
- **Threat & Vulnerability Scoring**: Deterministic risk aggregation combining criticality tiering and OSINT findings.