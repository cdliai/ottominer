from pathlib import Path
from typing import List, Tuple, Optional
import logging
import tempfile
import os

from .image_preprocessing import preprocess_image, PreprocessingConfig

logger = logging.getLogger(__name__)


def preprocess_pdf(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
    config: Optional[PreprocessingConfig] = None,
    dpi: int = 300,
) -> List[Path]:
    """
    Preprocess all pages of a PDF for improved OCR.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save preprocessed images
        config: Preprocessing configuration
        dpi: DPI for PDF to image conversion

    Returns:
        List of paths to preprocessed page images
    """
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF not available. Install: pip install pymupdf")
        return []

    config = config or PreprocessingConfig()
    output_dir = output_dir or Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    output_paths = []

    for page_num in range(doc.page_count):
        page = doc[page_num]

        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")

        import numpy as np

        nparr = np.frombuffer(img_bytes, np.uint8)

        try:
            import cv2

            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except ImportError:
            logger.warning("OpenCV not available, skipping preprocessing")
            output_path = output_dir / f"page_{page_num:03d}.png"
            output_path.write_bytes(img_bytes)
            output_paths.append(output_path)
            continue

        processed = preprocess_image(img, config)

        output_path = output_dir / f"page_{page_num:03d}.png"
        try:
            import cv2

            cv2.imwrite(str(output_path), processed)
        except Exception as e:
            logger.error(f"Failed to save preprocessed image: {e}")

        output_paths.append(output_path)

    doc.close()
    return output_paths


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[Tuple[int, bytes]]:
    """
    Convert PDF pages to images without preprocessing.

    Returns:
        List of (page_number, image_bytes) tuples
    """
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF not available")
        return []

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        pages.append((page_num, img_bytes))

    doc.close()
    return pages


def pdf_page_to_numpy(pdf_path: Path, page_num: int = 0, dpi: int = 300):
    """Convert a single PDF page to numpy array."""
    import numpy as np

    pages = pdf_to_images(pdf_path, dpi)
    if not pages or page_num >= len(pages):
        return None

    _, img_bytes = pages[page_num]
    nparr = np.frombuffer(img_bytes, np.uint8)

    try:
        import cv2

        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except ImportError:
        return None


class PDFPreprocessor:
    """High-level PDF preprocessor with caching."""

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.config = config or PreprocessingConfig()
        self.cache_dir = cache_dir

    def preprocess(self, pdf_path: Path) -> List[Path]:
        """Preprocess PDF, using cache if available."""
        if self.cache_dir:
            cache_key = f"{pdf_path.stem}_{pdf_path.stat().st_mtime}"
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                return list(cache_path.glob("*.png"))
            return preprocess_pdf(pdf_path, cache_path, self.config)

        return preprocess_pdf(pdf_path, config=self.config)
