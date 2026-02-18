import logging
from pathlib import Path
from typing import Optional

from .base import BaseOCRBackend

logger = logging.getLogger(__name__)


class KrakenBackend(BaseOCRBackend):
    """Kraken OCR backend - specialized for historical documents.

    Install: pip install kraken

    Kraken is designed for:
    - Historical manuscripts
    - Arabic/Persian/Ottoman script
    - Non-Latin scripts
    - Trainable on custom corpora

    Models:
    - arabic_notalibertal.mlmodel - Arabic script
    - bentinck.mlmodel - General historical
    - Custom models can be trained

    Pros: Best open-source for historical docs, trainable, good Arabic support
    Cons: Slower than Tesseract, requires model download
    """

    name = "kraken"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import kraken  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(self, file_path: Path) -> Optional[str]:
        if not self.is_available():
            logger.warning("Kraken not available. Install: pip install kraken")
            return None

        try:
            from kraken import pageseg
            from kraken import rpred
            from kraken.lib import models
            import fitz

            model_name = self.config.get("model", "arabic_notalibertal.mlmodel")

            logger.info(f"Running Kraken OCR on {file_path} (model={model_name})")

            try:
                model = models.load_any(model_name)
            except Exception as e:
                logger.info(f"Downloading Kraken model {model_name}...")
                from kraken.lib import vgsl

                model = vgsl.TorchVGSLModel.load_model(model_name)

            doc = fitz.open(str(file_path))
            texts = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.config.get("dpi", 300))

                import tempfile
                import numpy as np

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(pix.tobytes("png"))
                    tmp_path = tmp.name

                try:
                    im = pageseg.read_image(tmp_path)

                    segments = pageseg.segment(im)

                    predictions = rpred.rpred(model, im, segments)

                    page_text = "\n".join(p.prediction for p in predictions)
                    if page_text.strip():
                        texts.append(page_text.strip())

                finally:
                    import os

                    os.unlink(tmp_path)

            doc.close()

            if texts:
                return "\n\n".join(texts)

            return None

        except Exception as e:
            logger.error(f"Kraken OCR failed: {e}")
            return None
