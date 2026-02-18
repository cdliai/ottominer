import logging
from pathlib import Path
from typing import Optional

from .base import BaseOCRBackend

logger = logging.getLogger(__name__)


class EasyOCRBackend(BaseOCRBackend):
    """EasyOCR backend - multilingual, GPU/CPU, 80+ languages.

    Install: pip install easyocr

    Pros: Good multilingual support, GPU optional
    Cons: Slower than Tesseract, larger model
    """

    name = "easyocr"
    _reader = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_reader(self):
        if self.__class__._reader is None:
            import easyocr

            langs = self.config.get("langs", ["tr", "ar"])
            gpu = self.config.get("gpu", True)
            self.__class__._reader = easyocr.Reader(langs, gpu=gpu)
        return self.__class__._reader

    def extract(self, file_path: Path) -> Optional[str]:
        if not self.is_available():
            logger.warning("EasyOCR not available. Install: pip install easyocr")
            return None

        try:
            import fitz

            logger.info(f"Running EasyOCR on {file_path}")

            reader = self._get_reader()

            doc = fitz.open(str(file_path))
            texts = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.config.get("dpi", 200))

                import numpy as np

                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                results = reader.readtext(img)
                page_text = " ".join([r[1] for r in results if r[1].strip()])
                if page_text:
                    texts.append(page_text)

            doc.close()

            if texts:
                return "\n\n".join(texts)

            return None

        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return None
