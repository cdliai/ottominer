import base64
import logging
from pathlib import Path
from typing import Optional

from .base import BaseOCRBackend

logger = logging.getLogger(__name__)


class OllamaBackend(BaseOCRBackend):
    """Ollama vision model backend - local GPU-accelerated OCR.

    Install: pip install ollama
    Models: deepseek-ocr, llava, qwen2-vl, minicpm-v

    Pros: High accuracy, local, GPU accelerated
    Cons: Requires GPU, slower for many pages
    """

    name = "ollama"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import ollama  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(self, file_path: Path) -> Optional[str]:
        if not self.is_available():
            logger.warning("Ollama not available. Install: pip install ollama")
            return None

        try:
            import ollama
            import fitz

            model = self.config.get("model", "deepseek-ocr")

            logger.info(f"Running Ollama OCR ({model}) on {file_path}")

            doc = fitz.open(str(file_path))
            texts = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.config.get("dpi", 150))

                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                response = ollama.chat(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Extract all text from this document page. Output only the text, no explanations.",
                            "images": [img_b64],
                        }
                    ],
                )

                if response and "message" in response:
                    content = response["message"].get("content", "")
                    if content.strip():
                        texts.append(content.strip())

            doc.close()

            if texts:
                return "\n\n".join(texts)

            return None

        except Exception as e:
            logger.error(f"Ollama OCR failed: {e}")
            return None
