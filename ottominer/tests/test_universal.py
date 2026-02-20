import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import asyncio

from ottominer.extractors.universal import UniversalVDI

class TestUniversalVDI:

    def test_estimate_confidence(self):
        vdi = UniversalVDI()
        
        assert vdi._estimate_confidence("") == 0.0
        assert vdi._estimate_confidence("short") == 0.1
        assert vdi._estimate_confidence("this is a valid piece of text with no strange replacements") >= 0.95
        assert vdi._estimate_confidence("testtesttest\ufffd\ufffd\ufffd") < 0.95

    @patch("ottominer.extractors.universal._run_ocr_backend_sync")
    def test_process_page_async_high_confidence(self, mock_ocr, tmp_path):
        import fitz
        pdf_path = tmp_path / "dummy.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()
        
        # Stream A returns high confidence text
        mock_ocr.side_effect = ["High confidence text that should bypass stream B.", "Stream B text"]
        
        vdi = UniversalVDI()
        result = asyncio.run(vdi._process_page_async(str(pdf_path)))
        
        assert result == "High confidence text that should bypass stream B."
        # Because Stream B is cancelled or its result is ignored

    @patch("ottominer.extractors.universal._run_ocr_backend_sync")
    def test_process_page_async_low_confidence(self, mock_ocr, tmp_path):
        import fitz
        pdf_path = tmp_path / "dummy.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()
        
        # Stream A returns low confidence, Stream B returns good text
        # To simulate this, we need the first call to return low confidence text
        def side_effect(backend_name, *args, **kwargs):
            if backend_name == "kraken":
                return "low"
            return "Good stream B text."
            
        mock_ocr.side_effect = side_effect
        
        vdi = UniversalVDI(config={"vdi": {"stream_a": "kraken", "stream_b": "mistral"}})
        # Disable layout to simplify test
        vdi.enable_layout = False
        
        result = asyncio.run(vdi._process_page_async(str(pdf_path)))
        
        assert result == "Good stream B text."
