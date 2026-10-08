"""Meridian Vendor Assessment CLI - Phase 1 Foundation."""

import argparse
import sys
from pathlib import Path

# Ensure src is in sys.path when running main.py directly
src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from meridian_assessment.services.pipeline import AssessmentPipeline
from meridian_assessment.utils.exceptions import MeridianAssessmentError
from meridian_assessment.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Meridian Financial - Vendor Risk Assessment Pipeline (Phase 1)"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/input/vendors.csv",
        help="Path to input vendor CSV file (default: data/input/vendors.csv)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/criticality.yaml",
        help="Path to criticality configuration YAML (default: config/criticality.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/output/criticality_results.json",
        help="Path to write structured JSON assessment results (default: data/output/criticality_results.json)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )
    return parser.parse_args()


def format_tier_display(tier_val: str) -> str:
    """Format tier identifier to user-friendly label (e.g. TIER_1 -> Tier 1)."""
    clean = tier_val.replace("_", " ").title()
    return clean


def format_depth_display(depth_val: str) -> str:
    """Format assessment depth identifier (e.g. COMPREHENSIVE -> Comprehensive)."""
    return depth_val.capitalize()


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_args()
    logger = setup_logger(level=args.log_level)

    print("\nMeridian Vendor Assessment")
    print("--------------------------\n")

    try:
        pipeline = AssessmentPipeline(config_path=args.config)
        assessments = pipeline.run_from_source(args.input)

        print(f"Loaded vendors: {len(assessments)}\n")

        for assessment in assessments:
            tier_str = format_tier_display(assessment.criticality_tier.value)
            depth_str = format_depth_display(assessment.assessment_depth.value)
            print(f"{assessment.vendor_id} | {assessment.vendor_name}")
            print(f"Criticality: {tier_str} (Score: {assessment.criticality_score})")
            print(f"Assessment Depth: {depth_str}")
            if assessment.reasoning:
                print("Key Drivers:")
                for reason in assessment.reasoning:
                    print(f"  - {reason}")
            print()

        # Export JSON results
        output_file = pipeline.export_results_json(assessments, args.output)
        print(f"Results exported to: {output_file}")
        print("\nAssessment completed successfully.\n")
        return 0

    except MeridianAssessmentError as exc:
        logger.error("Assessment failed: %s", exc)
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error occurred during execution: %s", exc)
        print(f"\n[FATAL] Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
