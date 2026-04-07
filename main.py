"""
Wedding Photo Selector - Main Entry Point
Usage: python main.py --input ./input --output ./output --top 50
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import WeddingPhotoSelectorPipeline
from src.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Wedding Photo Selector - Automatically picks your best wedding photos"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./input",
        help="Path to folder containing wedding images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Path to output folder for selected photos and reports"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=50,
        help="Number of top photos to select (default: 50)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./config/settings.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level"
    )
    parser.add_argument(
        "--copy-photos",
        action="store_true",
        default=True,
        help="Copy selected photos to output folder"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate HTML/CSV analysis report"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("   AI Wedding Photo Selector")
    logger.info("=" * 60)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input folder not found: {input_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "top_photos").mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)

    pipeline = WeddingPhotoSelectorPipeline(
        config_path=args.config,
        output_dir=output_path
    )

    results = pipeline.run(
        input_dir=input_path,
        top_n=args.top,
        copy_photos=args.copy_photos,
        generate_report=args.generate_report
    )

    logger.info("=" * 60)
    logger.info(f"  ✅ Done! Processed {results['total_processed']} images")
    logger.info(f"  🏆 Selected top {len(results['selected_photos'])} photos")
    logger.info(f"  📁 Output saved to: {output_path}")
    if args.generate_report:
        logger.info(f"  📊 Report: {output_path}/reports/report.html")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
