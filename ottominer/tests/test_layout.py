import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import fitz
import sys

from ottominer.extractors.layout import LayoutSegmenter


class TestLayoutSegmenter:
    
    def test_basic_fallback_segmentation(self, tmp_path):
        # Create a dummy PDF
        pdf_path = tmp_path / "dummy.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test Text")
        doc.save(str(pdf_path))
        doc.close()
        
        # Test basic backend explicitly
        segmenter = LayoutSegmenter(config={"backend": "basic"})
        zones = segmenter.segment_page(pdf_path)
        
        assert len(zones) >= 1
        assert zones[0]["type"] == "text"
        assert len(zones[0]["bbox"]) == 4

    def test_docling_segmentation(self, tmp_path):
        from ottominer.extractors.layout import DOCLING_AVAILABLE
        if not DOCLING_AVAILABLE:
            pytest.skip("DocLING not installed")
            
        pdf_path = tmp_path / "docling_dummy.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        # Mock docling dynamically if it exists
        with patch("docling.document_converter.DocumentConverter") as mock_converter_class:
            mock_instance = mock_converter_class.return_value
            mock_result = MagicMock()
            mock_element1 = MagicMock(type="text", bbox=[10, 20, 100, 200])
            mock_element2 = MagicMock(type="illustration", bbox=[50, 60, 150, 260])
            mock_result.document.elements = [mock_element1, mock_element2]
            mock_instance.convert.return_value = mock_result
            
            segmenter = LayoutSegmenter(config={"backend": "docling"})
            zones = segmenter.segment_page(pdf_path)
            
            assert len(zones) == 2
            assert zones[0]["type"] == "text"
            assert zones[1]["type"] == "illustration"
            assert zones[0]["bbox"] == [10, 20, 100, 200]
