from pathlib import Path
from typing import Dict, Union, List, Optional, Tuple
import pymupdf4llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import fitz  # PyMuPDF

from ..utils.logger import setup_logger
from ..utils.progress import ProgressTracker
from ..utils.decorators import handle_exceptions
from .base import BaseExtractor

logger = setup_logger(__name__)

# Tier thresholds
MIN_CHARS_PER_PAGE = 100
MAX_REPLACEMENT_RATIO = 0.05


def _chardet_available() -> bool:
    try:
        import chardet  # noqa: F401

        return True
    except ImportError:
        return False


def _surya_available() -> bool:
    try:
        from surya.ocr import run_ocr  # noqa: F401

        return True
    except ImportError:
        return False


def _ollama_available() -> bool:
    try:
        import ollama  # noqa: F401

        return True
    except ImportError:
        return False


class PDFExtractor(BaseExtractor):
    """PDF extraction with tiered fallback strategy.

    Tiers:
      1. pymupdf4llm (always on) - text-native PDFs
      2. chardet encoding repair - fix corrupted text
      3. Surya OCR (optional) - image-based PDFs
      4. Ollama vision (optional) - GPU-accelerated OCR
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.config = config.get("pdf_extraction", {}) if config else {}
        self.pdf_config = {
            "dpi": self.config.get("dpi", 300),
            "margins": self.config.get("margins", (50, 50, 0, 0)),
            "table_strategy": self.config.get("table_strategy", "lines_strict"),
            "fontsize_limit": self.config.get("fontsize_limit", 4),
        }
        self.progress = ProgressTracker()
        self.ocr_backend = self.config.get("ocr_backend", "auto")
        self.ocr_model = self.config.get("ocr_model", "deepseek-ocr")
        self.enable_ocr = self.config.get("enable_ocr", False)

    def batch_extract(self, file_paths: List[Union[str, Path]]) -> Dict[str, str]:
        """Implement required abstract method."""
        return {str(path): self.extract(path) for path in file_paths}

    @handle_exceptions
    def extract(self, file_path: Union[str, Path], timeout: int = 30) -> str:
        """Extract text from PDF with tiered fallback."""
        from rich.progress import Progress, SpinnerColumn

        with Progress(
            SpinnerColumn(), *Progress.get_default_columns(), transient=True
        ) as progress:
            task = progress.add_task(f"Processing {file_path}...", total=1)
            try:
                result = self._extract_tiered(file_path)
                progress.update(task, advance=1)
                return result
            except Exception as e:
                progress.update(task, description=f"Error: {str(e)}")
                raise

    def _extract_tiered(self, file_path: Union[str, Path]) -> str:
        """Run tiered extraction with fallback logic."""
        path = self.validate_file(file_path)

        if not self._is_valid_pdf(path):
            raise ValueError(f"Invalid or corrupted PDF file: {file_path}")

        page_count = self._get_page_count(path)

        # Tier 1: Standard extraction
        text = self._extract_tier1(path)

        if text is None:
            logger.warning(f"Tier 1 failed for {path}, trying OCR")
            return self._extract_with_ocr(path)

        # Check extraction quality
        quality = self._assess_quality(text, page_count)
        logger.debug(f"Extraction quality for {path}: {quality}")

        if quality["needs_ocr"]:
            logger.info(
                f"Low quality extraction ({quality['reason']}), escalating to OCR"
            )
            ocr_text = self._extract_with_ocr(path)
            if ocr_text:
                return ocr_text

        # Tier 2: Try encoding repair if needed
        if quality["needs_encoding_repair"] and _chardet_available():
            repaired = self._repair_encoding(path, text)
            if repaired:
                text = repaired

        return text or ""

    def _extract_tier1(self, file_path: Path) -> Optional[str]:
        """Tier 1: Standard pymupdf4llm extraction."""
        try:
            md_text = pymupdf4llm.to_markdown(str(file_path), **self.pdf_config)
            return md_text
        except Exception as e:
            logger.error(f"Tier 1 extraction failed: {e}")
            return None

    def _get_page_count(self, file_path: Path) -> int:
        """Get PDF page count."""
        try:
            doc = fitz.open(str(file_path))
            count = doc.page_count
            doc.close()
            return count
        except Exception:
            return 1

    def _assess_quality(self, text: str, page_count: int) -> Dict:
        """Assess extraction quality and determine if escalation is needed."""
        if not text:
            return {
                "needs_ocr": True,
                "needs_encoding_repair": False,
                "reason": "empty output",
            }

        total_chars = len(text)
        chars_per_page = total_chars / max(page_count, 1)

        replacement_count = text.count("\ufffd")
        replacement_ratio = replacement_count / max(total_chars, 1)

        needs_ocr = chars_per_page < MIN_CHARS_PER_PAGE
        needs_encoding_repair = replacement_ratio > MAX_REPLACEMENT_RATIO

        reason = None
        if needs_ocr:
            reason = f"low chars/page ({chars_per_page:.1f} < {MIN_CHARS_PER_PAGE})"
        elif needs_encoding_repair:
            reason = f"high replacement ratio ({replacement_ratio:.2%})"

        return {
            "needs_ocr": needs_ocr,
            "needs_encoding_repair": needs_encoding_repair,
            "reason": reason,
            "chars_per_page": chars_per_page,
            "replacement_ratio": replacement_ratio,
        }

    def _repair_encoding(self, file_path: Path, text: str) -> Optional[str]:
        """Attempt to repair encoding issues using chardet."""
        if not _chardet_available():
            logger.warning("chardet not available, skipping encoding repair")
            return None

        try:
            import chardet

            raw_bytes = text.encode("utf-8", errors="replace")
            detected = chardet.detect(raw_bytes)

            if detected["confidence"] > 0.7 and detected["encoding"]:
                try:
                    repaired = raw_bytes.decode(detected["encoding"])
                    if repaired.count("\ufffd") < text.count("\ufffd"):
                        logger.info(f"Encoding repaired: {detected['encoding']}")
                        return repaired
                except Exception:
                    pass

            return None
        except Exception as e:
            logger.error(f"Encoding repair failed: {e}")
            return None

    def _extract_with_ocr(self, file_path: Path) -> Optional[str]:
        """Extract using OCR backend."""
        if self.ocr_backend == "ollama":
            return self._ocr_ollama(file_path)
        elif self.ocr_backend == "surya":
            return self._ocr_surya(file_path)
        else:
            if _ollama_available():
                result = self._ocr_ollama(file_path)
                if result:
                    return result
            if _surya_available():
                result = self._ocr_surya(file_path)
                if result:
                    return result
            if not self.enable_ocr:
                logger.info("OCR not enabled and no OCR backend available")
            return None

    def _ocr_surya(self, file_path: Path) -> Optional[str]:
        """Extract using Surya OCR."""
        if not _surya_available():
            logger.warning(
                "Surya OCR not available. Install with: pip install surya-ocr"
            )
            return None

        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.recognition.model import load_model as load_rec_model

            logger.info(f"Running Surya OCR on {file_path}")

            det_model = load_det_model()
            rec_model = load_rec_model()

            results = run_ocr([str(file_path)], [None], det_model, rec_model)

            if results:
                text = "\n".join(
                    line.text for page in results for line in page.text_lines
                )
                return text

            return None
        except Exception as e:
            logger.error(f"Surya OCR failed: {e}")
            return None

    def _ocr_ollama(self, file_path: Path) -> Optional[str]:
        """Extract using Ollama vision model."""
        if not _ollama_available():
            logger.warning("Ollama not available. Install with: pip install ollama")
            return None

        try:
            import ollama
            import base64

            logger.info(f"Running Ollama OCR ({self.ocr_model}) on {file_path}")

            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

            response = ollama.chat(
                model=self.ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract all text from this PDF document. Output only the extracted text, no explanations.",
                        "images": [pdf_b64],
                    }
                ],
            )

            if response and "message" in response:
                return response["message"].get("content", "")

            return None
        except Exception as e:
            logger.error(f"Ollama OCR failed: {e}")
            return None

    def _is_valid_pdf(self, file_path: Path) -> bool:
        """Check if file is a valid PDF."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)
                return header.startswith(b"%PDF-")
        except Exception:
            return False

    def _save_output(self, content: str, source_path: Path) -> Path:
        """Save extracted content to output directory."""
        output_dir = Path(self.config.get("output_dir", "output"))
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f"{source_path.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Saved output to: {output_path}")
        return output_path

    def save_output(self, content: str, output_path: Union[str, Path]) -> None:
        """Save extracted content to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")


class ParallelPDFExtractor(PDFExtractor):
    """Parallel PDF extraction with tiered fallback."""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.max_workers = self.config.get("workers", max(2, os.cpu_count() // 2))

    def _extract_single(
        self, file_path: Path, progress=None, task_id=None
    ) -> Optional[str]:
        """Extract text from a single PDF with tiered fallback."""
        try:
            result = self._extract_tiered(file_path)
            if progress and task_id:
                progress.update(task_id, advance=1)
            return result
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None

    @handle_exceptions
    def batch_extract(
        self, file_paths: List[Union[str, Path]]
    ) -> Dict[str, Optional[str]]:
        """Extract text from multiple PDFs in parallel with tiered fallback."""
        file_paths = [self.validate_file(f) for f in file_paths]
        results = {}

        with self.progress as progress:
            task_id = progress.add_task(
                f"Processing {len(file_paths)} PDFs", total=len(file_paths)
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self._extract_single, path, progress, task_id): path
                    for path in file_paths
                }

                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[str(path)] = result
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
                        results[str(path)] = None

        return results or None
