import pytest
import json
from pathlib import Path
from reportlab.pdfgen import canvas


def make_pdf(path: Path, text: str = "Osmanlı devleti namaz ticaret efendi paşa"):
    c = canvas.Canvas(str(path))
    c.drawString(50, 750, text)
    c.save()
    return path


class TestPipeline:
    def test_pipeline_runs_on_single_pdf(self, tmp_path):
        from ottominer.pipeline import Pipeline

        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run([pdf])
        assert len(results) == 1

    def test_output_json_is_created(self, tmp_path):
        from ottominer.pipeline import Pipeline

        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        p.run([pdf])
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 1

    def test_output_json_has_expected_keys(self, tmp_path):
        from ottominer.pipeline import Pipeline

        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        p.run([pdf])
        json_file = next(output_dir.glob("*.json"))
        data = json.loads(json_file.read_text())
        assert "source_path" in data
        assert "tokens" in data
        assert "semantic_labels" in data
        assert "register" in data

    def test_pipeline_with_multiple_pdfs(self, tmp_path):
        from ottominer.pipeline import Pipeline

        pdfs = [
            make_pdf(tmp_path / f"doc{i}.pdf", f"metin {i} namaz") for i in range(3)
        ]
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run(pdfs)
        assert len(results) == 3

    def test_pipeline_skips_failed_extraction_gracefully(self, tmp_path):
        from ottominer.pipeline import Pipeline

        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run([bad_pdf])
        assert isinstance(results, list)

    def test_extract_only_mode(self, tmp_path):
        from ottominer.pipeline import Pipeline

        pdf = make_pdf(tmp_path / "test.pdf")
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir, analyzers=[])
        results = p.run([pdf])
        assert len(results) == 1

    def test_similarity_vectors_populated_in_batch(self, tmp_path):
        from ottominer.pipeline import Pipeline

        pdfs = [
            make_pdf(tmp_path / f"doc{i}.pdf", "namaz ibadet ticaret") for i in range(2)
        ]
        output_dir = tmp_path / "output"
        p = Pipeline(output_dir=output_dir)
        results = p.run(pdfs)
        vectors = [r.similarity_vector for r in results if r and r.similarity_vector]
        assert len(vectors) > 0
