import logging
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseOCRBackend

logger = logging.getLogger(__name__)


class TesseractBackend(BaseOCRBackend):
    """Tesseract OCR backend - fast, local, CPU-based.

    Install: apt-get install tesseract-ocr tesseract-ocr-tur
             pip install pytesseract pdf2image

    Pros: Fast, no GPU needed, well-tested
    Cons: Lower accuracy for Ottoman script
    """

    name = "tesseract"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pytesseract  # noqa: F401
            from pdf2image import convert_from_path  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(self, file_path: Path) -> Optional[str]:
        if not self.is_available():
            logger.warning(
                "Tesseract not available. Install: pip install pytesseract pdf2image"
            )
            return None

        try:
            import pytesseract
            from pdf2image import convert_from_path

            lang = self.config.get("lang", "tur")
            dpi = self.config.get("dpi", 300)

            logger.info(f"Running Tesseract OCR on {file_path} (lang={lang})")

            images = convert_from_path(str(file_path), dpi=dpi)

            texts = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=lang)
                if text.strip():
                    texts.append(text.strip())

            if texts:
                return "\n\n".join(texts)

            return None

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return None
