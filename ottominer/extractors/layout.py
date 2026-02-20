import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    # Attempt to load docling if installed
    import docling.document_converter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class LayoutSegmenter:
    """
    Layout Segmentation Engine.
    
    Uses DocLING (or falls back to other tools) to detect document zones
    (Main Text, Marginalia, Illustrations). This provides the "Map" for 
    high-fidelity Vision-Language Models (VLMs) to process complex pages
    without getting overwhelmed by mixed layouts.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.backend = self.config.get("backend", "docling" if DOCLING_AVAILABLE else "basic")

    def is_available(self) -> bool:
        if self.backend == "docling":
            return DOCLING_AVAILABLE
        return True # Basic fallback is always available

    def segment_page(self, pdf_page_path: Path) -> List[Dict[str, Any]]:
        """
        Segment a single PDF page into logical zones.
        
        Returns:
            A list of zone dictionaries, e.g.,
            [
                {"type": "text", "bbox": [x0, y0, x1, y1]},
                {"type": "illustration", "bbox": [...]},
                {"type": "marginalia", "bbox": [...]}
            ]
        """
        if self.backend == "docling" and DOCLING_AVAILABLE:
            return self._segment_docling(pdf_page_path)
            
        return self._segment_basic(pdf_page_path)

    def _segment_docling(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run DocLING to extract layout bounding boxes."""
        zones = []
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            
            # This is a conceptual implementation of DocLING's API.
            # DocLING's output format provides elements with bounding boxes.
            for item in result.document.elements:
                item_type = getattr(item, "type", "text")
                bbox = getattr(item, "bbox", None)
                if bbox:
                    # Convert bbox to list if necessary
                    zones.append({
                        "type": str(item_type),
                        "bbox": bbox
                    })
                    
            logger.info(f"DocLING segmented {len(zones)} zones for {file_path}")
        except Exception as e:
            logger.error(f"DocLING segmentation failed: {e}")
            return self._segment_basic(file_path)
            
        # If no zones found, treat the whole page as a single zone
        if not zones:
            return self._segment_basic(file_path)
            
        return zones

    def _segment_basic(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Basic fallback: treats the whole page as a single text zone.
        In a more advanced fallback, we could use fitz (PyMuPDF) to find image blocks vs text blocks.
        """
        zones = []
        try:
            import fitz
            doc = fitz.open(str(file_path))
            page = doc[0]
            
            # Get images on the page
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                rect = page.get_image_bbox(img)
                zones.append({
                    "type": "illustration",
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                })
            
            # Add a generic full-page text block if we didn't do full layout analysis
            # In a real fallback, we might use page.get_text("blocks")
            blocks = page.get_text("blocks")
            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                # block_type 0 is text, 1 is image
                if block_type == 0:
                    zones.append({
                        "type": "text",
                        "bbox": [x0, y0, x1, y1]
                    })
            
            doc.close()
            logger.info(f"Basic layout segmented {len(zones)} zones for {file_path}")
        except Exception as e:
            logger.warning(f"Basic layout segmentation failed: {e}")
            zones.append({
                "type": "text",
                "bbox": [0, 0, 1000, 1000] # arbitrary full page bbox
            })
            
        return zones
