import pytest
from pathlib import Path
from ottominer.extractors.pdf import (
    PDFExtractor,
    ParallelPDFExtractor,
    _chardet_available,
    _surya_available,
    _ollama_available,
)
from reportlab.pdfgen import canvas
import time
from ottominer.utils.progress import ProgressTracker


def create_test_pdf(tmp_path, content="Test Document", filename="test.pdf"):
    """Create a small test PDF file."""
    pdf_path = tmp_path / filename
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, content)
    c.save()
    return pdf_path


@pytest.fixture
def test_pdfs(tmp_path):
    """Create multiple test PDFs."""
    return [
        create_test_pdf(tmp_path, f"Test Doc {i}", f"test_{i}.pdf") for i in range(3)
    ]


@pytest.fixture
def pdf_config():
    """Test configuration for PDF extraction."""
    return {
        "pdf_extraction": {
            "dpi": 300,
            "margins": (50, 50, 0, 0),
            "table_strategy": "lines_strict",
            "fontsize_limit": 4,
            "workers": 2,
            "batch_size": 10,
        }
    }


class TestBaseExtractor:
    @pytest.mark.timeout(5)
    def test_validate_file_exists(self, tmp_path):
        pdf_path = create_test_pdf(tmp_path)
        extractor = PDFExtractor()
        assert extractor.validate_file(pdf_path) == pdf_path

    @pytest.mark.timeout(5)
    def test_validate_file_not_exists(self, tmp_path):
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.validate_file(tmp_path / "nonexistent.pdf")

    @pytest.mark.timeout(5)
    def test_validate_file_not_file(self, tmp_path):
        extractor = PDFExtractor()
        with pytest.raises(ValueError):
            extractor.validate_file(tmp_path)


class TestPDFExtractor:
    @pytest.mark.timeout(5)
    def test_extract_single_pdf(self, tmp_path, pdf_config):
        pdf_path = create_test_pdf(tmp_path)
        extractor = PDFExtractor(pdf_config)
        result = extractor.extract(pdf_path)
        assert isinstance(result, str)
        assert "Test Document" in result

    @pytest.mark.timeout(5)
    def test_save_output(self, tmp_path):
        extractor = PDFExtractor()
        output = "Test content"
        output_file = tmp_path / "output.txt"
        extractor.save_output(output, output_file)
        assert output_file.read_text(encoding="utf-8") == output

    @pytest.mark.timeout(5)
    def test_extract_invalid_pdf(self, tmp_path):
        pdf_path = tmp_path / "invalid.pdf"
        pdf_path.write_bytes(b"Not a PDF")
        extractor = PDFExtractor()
        with pytest.raises(ValueError):
            extractor.extract(pdf_path)


class TestParallelPDFExtractor:
    @pytest.mark.timeout(10)
    def test_batch_extract(self, test_pdfs):
        """Test batch extraction of PDFs."""
        extractor = ParallelPDFExtractor()
        results = extractor.batch_extract(test_pdfs)

        # Verify results
        assert results is not None
        assert len(results) == len(test_pdfs)

        # Check content of successful extractions
        successful_results = [v for v in results.values() if v is not None]
        assert len(successful_results) > 0
        assert all("Test Doc" in v for v in successful_results)

    @pytest.mark.timeout(5)
    def test_resource_management(self):
        extractor = ParallelPDFExtractor()
        assert extractor.max_workers > 0


@pytest.mark.timeout(10)
def test_integration(test_pdfs):
    """Integration test for PDF extraction."""
    config = {
        "pdf_extraction": {
            "dpi": 300,
            "margins": (50, 50, 0, 0),
            "table_strategy": "lines_strict",
            "fontsize_limit": 4,
            "workers": 2,
        }
    }

    # Force stop any existing progress
    ProgressTracker().force_stop()

    # Test single extraction
    extractor = PDFExtractor(config)
    result = extractor.extract(test_pdfs[0])
    assert isinstance(result, str)
    assert "Test Doc 0" in result

    # Force stop before parallel extraction
    ProgressTracker().force_stop()
    time.sleep(0.1)  # Give time for cleanup

    # Test parallel extraction
    parallel = ParallelPDFExtractor(config)
    results = parallel.batch_extract(test_pdfs)
    assert results is not None
    assert len(results) == len(test_pdfs)

    # Check content of successful extractions
    successful_results = [v for v in results.values() if v is not None]
    assert len(successful_results) > 0
    assert all("Test Doc" in v for v in successful_results)


class TestTieredExtraction:
    def test_assess_quality_good_extraction(self):
        extractor = PDFExtractor()
        long_text = "This is good text with many characters per page. " * 5
        quality = extractor._assess_quality(long_text, 1)
        assert quality["needs_ocr"] is False
        assert quality["needs_encoding_repair"] is False

    def test_assess_quality_low_chars_per_page(self):
        extractor = PDFExtractor()
        quality = extractor._assess_quality("short", 1)
        assert quality["needs_ocr"] is True
        assert "low chars/page" in quality["reason"]

    def test_assess_quality_high_replacement_ratio(self):
        extractor = PDFExtractor()
        text_with_replacements = "test\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd text" * 10
        quality = extractor._assess_quality(text_with_replacements, 1)
        assert quality["needs_encoding_repair"] is True

    def test_assess_quality_empty_text(self):
        extractor = PDFExtractor()
        quality = extractor._assess_quality("", 1)
        assert quality["needs_ocr"] is True
        assert quality["reason"] == "empty output"

    def test_chardet_availability_check(self):
        result = _chardet_available()
        assert isinstance(result, bool)

    def test_surya_availability_check(self):
        result = _surya_available()
        assert isinstance(result, bool)

    def test_ollama_availability_check(self):
        result = _ollama_available()
        assert isinstance(result, bool)

    def test_valid_text_pdf_uses_tier1(self, tmp_path, pdf_config):
        from ottominer.extractors.pdf import PDFExtractor

        pdf_path = create_test_pdf(tmp_path, "Osmanlı devleti büyük imparatorluk")
        extractor = PDFExtractor(pdf_config)
        result = extractor.extract(pdf_path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_page_count(self, tmp_path):
        extractor = PDFExtractor()
        pdf_path = create_test_pdf(tmp_path, "Test content")
        count = extractor._get_page_count(pdf_path)
        assert count == 1

    def test_is_valid_pdf(self, tmp_path):
        extractor = PDFExtractor()
        pdf_path = create_test_pdf(tmp_path, "Test")
        assert extractor._is_valid_pdf(pdf_path) is True

        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a pdf")
        assert extractor._is_valid_pdf(bad_pdf) is False

    def test_ocr_backend_config(self):
        config = {
            "pdf_extraction": {
                "ocr_backend": "surya",
                "ocr_model": "test-model",
                "enable_ocr": True,
            }
        }
        extractor = PDFExtractor(config)
        assert extractor.ocr_backend == "surya"
        assert extractor.ocr_model == "test-model"
        assert extractor.enable_ocr is True

    def test_extract_tiered_fallback_on_failure(self, tmp_path):
        extractor = PDFExtractor()
        pdf_path = create_test_pdf(tmp_path, "Short")
        result = extractor._extract_tiered(pdf_path)
        assert isinstance(result, str)
