# Meridian Vendor Assessment Architecture (Phase 1)

## Overview
The Meridian Vendor Risk Assessment Pipeline is designed to perform evidence-driven, explainable cybersecurity assessments of third-party vendors. 

**Phase 1** establishes the core foundation:
- Ingestion & schema validation of vendor profiles
- Configuration-driven, deterministic criticality scoring
- Scoping and assessment-depth determination
- Export of machine-readable structured results
- Future-facing OSINT interface abstraction

```
                 +-----------------------+
                 |  Vendor Input (CSV)   |
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |   CSV Vendor Loader   |
                 |   (Schema Validation) |
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |  Criticality Engine   | <--- config/criticality.yaml
                 | (Deterministic Scores)|
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |    Scoping Engine     |
                 | (Depth Specification) |
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 | Structured JSON Export|
                 +-----------------------+
                             |
                   [ Phase 2 Plug-in ]
                             |
                             v
                 +-----------------------+
                 |   OSINT Assessment    |
                 | (Placeholder Interface|
                 +-----------------------+
```

---

## Core Components

### 1. Data Models (`src/meridian_assessment/models/`)
- `Vendor`: Immutable Pydantic model enforcing validation on required fields, domains, and the 5 criticality dimensions.
- `CriticalityAssessment`: Complete result payload with overall score, assigned tier, assessment depth, per-criterion score breakdown, and explainability reasoning list.
- `AssessmentScope`: Boundaries and recommended focus areas for subsequent investigation stages.
- `CriticalityConfig`: Strongly-typed validation for scoring weights, severity scales, tier thresholds, and reasoning templates.

### 2. Ingestion Layer (`src/meridian_assessment/ingestion/`)
- `BaseVendorLoader`: Abstract base class allowing future loaders (API, database, JSON) without refactoring the pipeline.
- `CSVVendorLoader`: Concrete implementation handling CSV parsing, column alias normalization (`payment_involvement` vs `payment_flows`), duplicate vendor ID rejection, and row validation error aggregation.

### 3. Criticality & Scoping Engines (`src/meridian_assessment/services/`)
- `CriticalityEngine`: Computes deterministic weighted scores based on `config/criticality.yaml`. Tiers are assigned dynamically from thresholds:
  - **Tier 1 (High)**: min score $\ge 3.5$ $\rightarrow$ Comprehensive Assessment
  - **Tier 2 (Medium)**: min score $\ge 2.0$ $\rightarrow$ Targeted Assessment
  - **Tier 3 (Low)**: min score $< 2.0$ $\rightarrow$ Lightweight Baseline
- `ScopingEngine`: Maps criticality tiers to assessment scope definitions and specific focus areas.
- `AssessmentPipeline`: High-level orchestrator executing batch runs and exporting machine-readable JSON artifacts.

### 4. OSINT Interface Contract (`src/meridian_assessment/osint/`)
- `BaseOSINTAssessment`: Abstract contract for Phase 2 OSINT discovery. Accepts `Vendor`, `CriticalityTier`, and `AssessmentScope`.
- `OSINTAssessmentPlaceholder`: Concrete placeholder returning explicit status `NOT_IMPLEMENTED_PHASE_1` until Phase 2 search sources and LLM/RAG strategies are provided.

---

## Explainability & Determinism
Criticality determination is completely non-blackbox:
1. Each criterion has a quantitative weight and qualitative-to-numerical severity mapping in YAML.
2. Criterion score = $\text{Weight}_i \times \text{SeverityScore}_i$.
3. Total score = $\sum_{i} \text{CriterionScore}_i$.
4. Any criterion exceeding the mention threshold ($2.5$) automatically generates human-readable audit justification drivers in `reasoning`.
