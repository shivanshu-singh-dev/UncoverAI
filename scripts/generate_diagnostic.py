"""Diagnostic generator script for the Meridian Vendor dataset under the new D/P/R/O/V methodology."""
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from meridian_assessment.ingestion.meridian_csv_loader import MeridianCSVVendorLoader
from meridian_assessment.services.meridian_criticality_engine import MeridianCriticalityEngine


def main():
    loader = MeridianCSVVendorLoader()
    engine = MeridianCriticalityEngine()

    csv_path = Path("data/input/meridian_vendors.csv")
    vendors = loader.load(csv_path)

    results = []
    print("\n" + "=" * 120)
    print("MERIDIAN FINANCIAL — VENDOR CRITICALITY ASSESSMENT DIAGNOSTIC")
    print("=" * 120 + "\n")

    for v in vendors:
        res = engine.evaluate(v)
        trail = engine.generate_audit_trail(res)
        results.append(trail)

        print(f"Vendor: {v.vendor_id} | {v.vendor_name}")
        print(f"  Operational Dependency: {v.operational_dependency} -> O={res.O.score} ({res.O.determination_method.value}) | Rationale: {res.O.rationale}")
        print(f"  Data Classification:   {v.data_classification_accessed} -> D={res.D.score} ({res.D.determination_method.value}) | Rationale: {res.D.rationale}")
        print(f"  Annual Data Volume:    {v.data_volume_annual} -> V={res.V.score if res.V.score is not None else 'UNKNOWN'} ({res.V.determination_method.value}) | Type: {res.V.volume_type} | Rationale: {res.V.rationale}")
        print(f"  Payment Flow:          P={res.P.score if res.P.score is not None else 'UNKNOWN'} ({res.P.determination_method.value}) | Rationale: {res.P.rationale}")
        print(f"  Regulatory Exposure:   R={res.R.score if res.R.score is not None else 'UNKNOWN'} ({res.R.determination_method.value}) | Rationale: {res.R.rationale}")
        print(f"  -- Contributions: D={res.D.weighted_contribution} (w={res.D.weight}), P={res.P.weighted_contribution} (w={res.P.weight}), R={res.R.weighted_contribution} (w={res.R.weight}), O={res.O.weighted_contribution} (w={res.O.weight}), V={res.V.weighted_contribution} (w={res.V.weight})")
        print(f"  -- Base Score: {res.base_score} | Score Status: {res.score_status.value}")
        print(f"  -- Provisional Criticality: {res.provisional_criticality}")
        
        # Overrides detail
        for o in res.overrides_evaluated:
            status = "TRIGGERED" if o.triggered else "NOT TRIGGERED"
            floor_text = f" (Floor: {o.resulting_floor})" if o.triggered else ""
            print(f"     [{o.override_id}] {status}{floor_text}: {o.description} | Evidence: {o.evidence} | Rationale: {o.rationale}")
            
        print(f"  -- Proposed Criticality (after O1-O4): {res.proposed_criticality}")
        print(f"  -- O5 User Decision: {res.human_review.user_decision if res.human_review else 'PROPOSED_ACCEPTED'} | Final Criticality: {res.final_criticality} | Depth: {res.assessment_depth}")
        if res.requires_review:
            print(f"  -- [!] REVIEW FLAGGED: {res.review_notes}")
        print("-" * 120)

    # Save to json
    out_path = Path("data/output/meridian_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Diagnostic saved to {out_path}\n")


if __name__ == "__main__":
    main()
