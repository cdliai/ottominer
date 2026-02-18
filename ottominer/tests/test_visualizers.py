import pytest
from pathlib import Path
import json


def make_sample_docs(tmp_path):
    """Create sample analysis JSON files for testing."""
    docs = [
        {
            "source_path": "doc1.pdf",
            "raw_text": "namaz ibadet din ticaret",
            "tokens": ["namaz", "ibadet", "din", "ticaret"],
            "lemmas": ["namaz", "ibadet", "din", "ticaret"],
            "filtered_tokens": ["namaz", "ibadet", "din", "ticaret"],
            "semantic_labels": {"namaz": ["religious"], "ticaret": ["economic"]},
            "morphology": {"ibadet": ["verbal"]},
            "genre_scores": {"formal_marker_count": 2, "informal_marker_count": 0},
            "register": "formal",
            "similarity_vector": [0.1, 0.2, 0.3, 0.4],
        },
        {
            "source_path": "doc2.pdf",
            "raw_text": "selam naber dostum kanka",
            "tokens": ["selam", "naber", "dostum", "kanka"],
            "lemmas": ["selam", "naber", "dost", "kanka"],
            "filtered_tokens": ["selam", "naber", "dostum", "kanka"],
            "semantic_labels": {},
            "morphology": {},
            "genre_scores": {"formal_marker_count": 0, "informal_marker_count": 3},
            "register": "informal",
            "similarity_vector": [0.5, 0.6, 0.7, 0.8],
        },
    ]

    for i, doc in enumerate(docs):
        path = tmp_path / f"doc{i + 1}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    return docs


class TestHTMLReport:
    def test_generate_html_report_creates_file(self, tmp_path):
        from ottominer.visualizers.html_report import generate_html_report

        docs = make_sample_docs(tmp_path)
        output_dir = tmp_path / "output"

        report_path = generate_html_report(docs, output_dir)

        assert report_path.exists()
        assert report_path.suffix == ".html"

    def test_html_report_contains_expected_content(self, tmp_path):
        from ottominer.visualizers.html_report import generate_html_report

        docs = make_sample_docs(tmp_path)
        output_dir = tmp_path / "output"

        report_path = generate_html_report(docs, output_dir)
        content = report_path.read_text(encoding="utf-8")

        assert "OttoMiner" in content
        assert "2" in content  # Document count

    def test_html_report_with_empty_docs(self, tmp_path):
        from ottominer.visualizers.html_report import generate_html_report

        output_dir = tmp_path / "output"
        report_path = generate_html_report([], output_dir)

        assert report_path.exists()


class TestStaticFigures:
    def test_generate_figures_creates_files(self, tmp_path):
        from ottominer.visualizers.static_figures import generate_figures

        docs = make_sample_docs(tmp_path)
        output_dir = tmp_path / "output"

        figure_paths = generate_figures(docs, output_dir, dpi=150)

        assert len(figure_paths) > 0
        for path in figure_paths:
            assert path.exists()

    def test_generate_figures_png_and_pdf(self, tmp_path):
        from ottominer.visualizers.static_figures import generate_figures

        docs = make_sample_docs(tmp_path)
        output_dir = tmp_path / "output"

        figure_paths = generate_figures(docs, output_dir, dpi=150)

        png_files = [p for p in figure_paths if p.suffix == ".png"]
        pdf_files = [p for p in figure_paths if p.suffix == ".pdf"]

        assert len(png_files) > 0
        assert len(pdf_files) > 0

    def test_generate_figures_empty_docs(self, tmp_path):
        from ottominer.visualizers.static_figures import generate_figures

        output_dir = tmp_path / "output"
        figure_paths = generate_figures([], output_dir, dpi=150)

        assert figure_paths == []

    def test_figure_files_have_content(self, tmp_path):
        from ottominer.visualizers.static_figures import generate_figures

        docs = make_sample_docs(tmp_path)
        output_dir = tmp_path / "output"

        figure_paths = generate_figures(docs, output_dir, dpi=150)

        for path in figure_paths:
            assert path.stat().st_size > 0


class TestVisualizerIntegration:
    def test_visualize_command_integration(self, tmp_path):
        from ottominer.cli.main import cmd_visualize
        import argparse

        make_sample_docs(tmp_path)
        output_dir = tmp_path / "output"

        args = argparse.Namespace(
            input=tmp_path, output=output_dir, html=True, figures=True, dpi=150
        )

        result = cmd_visualize(args)

        assert result == 0
        assert (output_dir / "report.html").exists()
