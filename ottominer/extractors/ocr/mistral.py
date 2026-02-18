import base64
import logging
import os
from pathlib import Path
from typing import Optional

from .base import BaseOCRBackend

logger = logging.getLogger(__name__)


class MistralBackend(BaseOCRBackend):
    """Mistral Vision API backend - cloud OCR with high accuracy.

    Install: pip install mistralai
    API Key: Set MISTRAL_API_KEY environment variable

    Pros: High accuracy, fast, no GPU needed
    Cons: Requires API key, costs money
    """

    name = "mistral"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mistralai  # noqa: F401

            return bool(os.environ.get("MISTRAL_API_KEY"))
        except ImportError:
            return False

    def extract(self, file_path: Path) -> Optional[str]:
        if not self.is_available():
            logger.warning(
                "Mistral not available. Install: pip install mistralai and set MISTRAL_API_KEY"
            )
            return None

        try:
            from mistralai import Mistral
            import fitz

            api_key = os.environ.get("MISTRAL_API_KEY") or self.config.get("api_key")
            model = self.config.get("model", "pixtral-12b-2409")

            if not api_key:
                logger.error("MISTRAL_API_KEY not set")
                return None

            client = Mistral(api_key=api_key)

            logger.info(f"Running Mistral OCR ({model}) on {file_path}")

            doc = fitz.open(str(file_path))
            texts = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.config.get("dpi", 150))

                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                response = client.chat.complete(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this document page. Output only the extracted text.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                )

                if response and response.choices:
                    content = response.choices[0].message.content
                    if content and content.strip():
                        texts.append(content.strip())

            doc.close()

            if texts:
                return "\n\n".join(texts)

            return None

        except Exception as e:
            logger.error(f"Mistral OCR failed: {e}")
            return None
