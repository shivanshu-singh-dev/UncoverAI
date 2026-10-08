"""Meridian Vendor Assessment — CLI."""

import argparse
import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from meridian_assessment.services.pipeline import AssessmentPipeline
from meridian_assessment.utils.exceptions import MeridianAssessmentError
from meridian_assessment.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meridian Financial — Vendor Risk Assessment")
    parser.add_argument("--input", "-i", default="data/input/vendors.csv", help="Vendor CSV file path")
    parser.add_argument("--config", "-c", default="config/criticality.yaml", help="Criticality configuration YAML")
    parser.add_argument("--output", "-o", default="data/output/criticality_results.json", help="JSON output path")
    parser.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(level=args.log_level)

    print("\nMeridian Vendor Assessment")
    print("--------------------------\n")

    try:
        pipeline = AssessmentPipeline(config_path=args.config)
        assessments = pipeline.run_from_source(args.input)

        print(f"Loaded vendors: {len(assessments)}\n")

        for a in assessments:
            tier_label = a.criticality_tier.value.replace("_", " ").title()
            depth_label = a.assessment_depth.value.capitalize()
            print(f"{a.vendor_id} | {a.vendor_name}")
            print(f"Criticality: {tier_label} (Score: {a.criticality_score})")
            print(f"Assessment Depth: {depth_label}")
            if a.reasoning:
                print("Key Drivers:")
                for reason in a.reasoning:
                    print(f"  - {reason}")
            print()

        output_file = pipeline.export_results_json(assessments, args.output)
        print(f"Results exported to: {output_file}")
        print("\nAssessment completed successfully.\n")
        return 0

    except MeridianAssessmentError as exc:
        logger.error("Assessment failed: %s", exc)
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
