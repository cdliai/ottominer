from typing import Optional

from .base import BaseOCRBackend
from .tesseract import TesseractBackend
from .easyocr_backend import EasyOCRBackend
from .ollama import OllamaBackend
from .mistral import MistralBackend
from .kraken import KrakenBackend

__all__ = [
    "BaseOCRBackend",
    "TesseractBackend",
    "EasyOCRBackend",
    "OllamaBackend",
    "MistralBackend",
    "KrakenBackend",
    "get_backend",
]


def get_backend(name: str, config: Optional[dict] = None):
    """Factory function to get OCR backend by name."""
    backends = {
        "tesseract": TesseractBackend,
        "easyocr": EasyOCRBackend,
        "ollama": OllamaBackend,
        "mistral": MistralBackend,
        "kraken": KrakenBackend,
    }

    backend_class = backends.get(name.lower())
    if not backend_class:
        raise ValueError(
            f"Unknown OCR backend: {name}. Available: {list(backends.keys())}"
        )

    return backend_class(config if config else {})
