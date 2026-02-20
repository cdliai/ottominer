import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ottominer",
        description="Ottoman Miner - Text Mining Toolkit for Ottoman Turkish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_extract_subparser(subparsers)
    _add_analyze_subparser(subparsers)
    _add_visualize_subparser(subparsers)
    _add_run_subparser(subparsers)

    return parser


def _add_extract_subparser(subparsers):
    """Add the extract subcommand."""
    extract_parser = subparsers.add_parser(
        "extract", help="Extract text from PDF documents"
    )
    extract_parser.add_argument(
        "-i", "--input", type=Path, required=True, help="Input PDF file or directory"
    )
    extract_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    extract_parser.add_argument(
        "--ocr", action="store_true", help="Enable OCR for image-based PDFs"
    )
    extract_parser.add_argument(
        "--ocr-backend",
        choices=["auto", "surya", "ollama"],
        default="auto",
        help="OCR backend to use (default: auto)",
    )
    extract_parser.add_argument(
        "--ocr-model",
        default="deepseek-ocr",
        help="Ollama model for OCR (default: deepseek-ocr)",
    )
    extract_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )


def _add_analyze_subparser(subparsers):
    """Add the analyze subcommand."""
    analyze_parser = subparsers.add_parser("analyze", help="Analyze extracted text")
    analyze_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input JSON file or directory from extract",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    analyze_parser.add_argument(
        "--analyzers",
        type=str,
        default="all",
        help="Comma-separated analyzers: semantic,morphology,genre,similarity (default: all)",
    )
    analyze_parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Use embeddings for similarity (requires sentence-transformers)",
    )


def _add_visualize_subparser(subparsers):
    """Add the visualize subcommand."""
    visualize_parser = subparsers.add_parser(
        "visualize", help="Generate visualizations from analysis results"
    )
    visualize_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input JSON file or directory from analyze",
    )
    visualize_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    visualize_parser.add_argument(
        "--html",
        action="store_true",
        default=True,
        help="Generate HTML report (default: True)",
    )
    visualize_parser.add_argument(
        "--figures", action="store_true", help="Generate static figures (PNG/PDF)"
    )
    visualize_parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for static figures (default: 300)"
    )


def _add_run_subparser(subparsers):
    """Add the run subcommand (full pipeline)."""
    run_parser = subparsers.add_parser(
        "run", help="Run full pipeline: extract → analyze → visualize"
    )
    run_parser.add_argument(
        "-i", "--input", type=Path, required=True, help="Input PDF file or directory"
    )
    run_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    run_parser.add_argument(
        "--analyzers",
        type=str,
        default="all",
        help="Comma-separated analyzers (default: all)",
    )
    run_parser.add_argument(
        "--ocr", action="store_true", help="Enable OCR for image-based PDFs"
    )
    run_parser.add_argument(
        "--ocr-backend",
        choices=["auto", "surya", "ollama"],
        default="auto",
        help="OCR backend to use (default: auto)",
    )
    run_parser.add_argument(
        "--html", action="store_true", default=True, help="Generate HTML report"
    )
    run_parser.add_argument(
        "--figures", action="store_true", help="Generate static figures"
    )
    run_parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )


def cmd_extract(args) -> int:
    """Execute the extract command."""
    from ..extractors.universal import UniversalVDI

    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "vdi": {
            "stream_a": args.ocr_backend if args.ocr_backend != "auto" else "kraken",
            "stream_b": "mistral",
            "stream_a_config": {
                "model": args.ocr_model,
            },
        }
    }

    pdf_files = _collect_pdf_files(input_path)
    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return 1

    print(f"Extracting text from {len(pdf_files)} PDF(s) using UniversalVDI...")

    extractor = UniversalVDI(config)
    results = extractor.batch_extract(pdf_files)

    successful = sum(1 for v in results.values() if v)
    
    # Save the output files
    for file_path, text in results.items():
        if text:
            # We assume extractors know how to save but we can just save it directly here
            # using BaseExtractor's save_output method which may be inherited
            out_file = output_dir / f"{Path(file_path).stem}.txt"
            out_file.write_text(text, encoding="utf-8")

    print(f"Extracted {successful}/{len(pdf_files)} documents")

    return 0


def cmd_analyze(args) -> int:
    """Execute the analyze command."""
    from ..pipeline import Pipeline
    from ..analyzers.base import AnalyzedDocument, TokenizedDocument
    import json

    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyzers == "all":
        analyzers = None
    else:
        analyzers = [a.strip() for a in args.analyzers.split(",")]

    json_files = _collect_json_files(input_path)
    if not json_files:
        print(f"No JSON files found in {input_path}")
        print("Run 'ottominer extract' first or provide JSON output.")
        return 1

    print(f"Analyzing {len(json_files)} document(s)...")

    docs = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            tok = TokenizedDocument(
                source_path=data.get("source_path", str(jf)),
                raw_text=data.get("raw_text", ""),
                tokens=data.get("tokens", []),
                lemmas=data.get("lemmas", []),
                offsets=[tuple(o) for o in data.get("offsets", [])],
                filtered_tokens=data.get("filtered_tokens", []),
            )
            docs.append(AnalyzedDocument(tokenized=tok))
        except Exception as e:
            print(f"Warning: Could not load {jf}: {e}")

    pipeline = Pipeline(
        output_dir=output_dir, analyzers=analyzers, use_embeddings=args.embeddings
    )

    for doc in docs:
        pipeline._process_one_direct(doc)

    if "similarity" in (analyzers or ["similarity"]):
        pipeline._similarity.compute_batch(docs)

    for doc in docs:
        pipeline._save_json(doc)

    print(f"Analyzed {len(docs)} document(s)")
    return 0


def cmd_visualize(args) -> int:
    """Execute the visualize command."""
    import json

    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = _collect_json_files(input_path)
    if not json_files:
        print(f"No JSON files found in {input_path}")
        return 1

    docs = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            docs.append(data)
        except Exception as e:
            print(f"Warning: Could not load {jf}: {e}")

    if args.html:
        from ..visualizers.html_report import generate_html_report

        report_path = generate_html_report(docs, output_dir)
        print(f"HTML report: {report_path}")

    if args.figures:
        from ..visualizers.static_figures import generate_figures

        figure_paths = generate_figures(docs, output_dir, dpi=args.dpi)
        print(f"Generated {len(figure_paths)} figures")

    return 0


def cmd_run(args) -> int:
    """Execute the full pipeline."""
    from ..pipeline import Pipeline

    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = _collect_pdf_files(input_path)
    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return 1

    if args.analyzers == "all":
        analyzers = None
    else:
        analyzers = [a.strip() for a in args.analyzers.split(",")]

    config = {
        "pdf_extraction": {
            "ocr_backend": args.ocr_backend,
            "enable_ocr": args.ocr,
            "workers": args.workers,
        }
    }

    print(f"Processing {len(pdf_files)} PDF(s)...")

    pipeline = Pipeline(
        output_dir=output_dir,
        analyzers=analyzers,
        extractor_config=config,
    )

    results = pipeline.run(pdf_files)

    successful = sum(1 for r in results if r is not None)
    print(f"Processed {successful}/{len(pdf_files)} documents")

    if args.html or args.figures:
        import json

        docs = []
        for jf in output_dir.glob("*.json"):
            try:
                docs.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                pass

        if args.html:
            from ..visualizers.html_report import generate_html_report

            report_path = generate_html_report(docs, output_dir)
            print(f"HTML report: {report_path}")

        if args.figures:
            from ..visualizers.static_figures import generate_figures

            figure_paths = generate_figures(docs, output_dir, dpi=args.dpi)
            print(f"Generated {len(figure_paths)} figures")

    return 0


def _collect_pdf_files(path: Path) -> List[Path]:
    """Collect PDF files from a path."""
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    return list(path.glob("*.pdf"))


def _collect_json_files(path: Path) -> List[Path]:
    """Collect JSON files from a path."""
    if path.is_file():
        return [path] if path.suffix.lower() == ".json" else []
    return list(path.glob("*.json"))


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "extract": cmd_extract,
        "analyze": cmd_analyze,
        "visualize": cmd_visualize,
        "run": cmd_run,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
