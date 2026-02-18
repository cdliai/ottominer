from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseOCRBackend(ABC):
    """Abstract base class for OCR backends."""

    name: str = "base"

    def __init__(self, config: dict = None):
        self.config = config or {}

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if the OCR backend is installed and available."""
        pass

    @abstractmethod
    def extract(self, file_path: Path) -> Optional[str]:
        """Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text or None if extraction failed
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(available={self.is_available()})"
