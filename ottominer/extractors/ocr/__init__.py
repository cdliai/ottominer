from .base import BaseOCRBackend
from .tesseract import TesseractBackend
from .easyocr_backend import EasyOCRBackend
from .ollama import OllamaBackend
from .mistral import MistralBackend

__all__ = [
    "BaseOCRBackend",
    "TesseractBackend",
    "EasyOCRBackend",
    "OllamaBackend",
    "MistralBackend",
    "get_backend",
]


def get_backend(name: str, config: dict = None):
    """Factory function to get OCR backend by name."""
    backends = {
        "tesseract": TesseractBackend,
        "easyocr": EasyOCRBackend,
        "ollama": OllamaBackend,
        "mistral": MistralBackend,
    }

    backend_class = backends.get(name.lower())
    if not backend_class:
        raise ValueError(
            f"Unknown OCR backend: {name}. Available: {list(backends.keys())}"
        )

    return backend_class(config or {})
