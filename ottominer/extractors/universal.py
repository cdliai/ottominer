import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Union, Dict
import fitz  # PyMuPDF
import tempfile
import os

from ..utils.logger import setup_logger
from .ocr import get_backend
from .base import BaseExtractor
from .layout import LayoutSegmenter

logger = setup_logger(__name__)


def _run_ocr_backend_sync(backend_name: str, config: dict, file_path: str) -> Optional[str]:
    """Helper function to run OCR backend in a process pool."""
    try:
        backend = get_backend(backend_name, config)
        if backend.is_available():
            return backend.extract(Path(file_path))
        return None
    except Exception as e:
        logger.error(f"Error running OCR backend {backend_name} on {file_path}: {e}")
        return None


class UniversalVDI(BaseExtractor):
    """
    Universal Visual Document Intelligence (VDI) Pipeline.
    
    Implements a Dual-Stream parallel engine using asyncio and multiprocessing:
    - Stream A (Fast/Local): Uses Kraken or Tesseract on local CPU/GPU.
    - Stream B (Accurate/Remote): Uses VLM (Mistral/Ollama) for complex layout.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config or {})
        self.config = (config or {}).get("vdi", {})
        
        # Configure streams
        self.stream_a_name = self.config.get("stream_a", "kraken")
        self.stream_b_name = self.config.get("stream_b", "mistral")
        
        self.stream_a_config = self.config.get("stream_a_config", {})
        self.stream_b_config = self.config.get("stream_b_config", {})
        
        # Threshold for cancelling Stream B (if Stream A is good enough)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.95)
        
        self.enable_layout = self.config.get("enable_layout", True)
        self.layout_segmenter = LayoutSegmenter()
        
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() // 2))
        self.thread_pool = ThreadPoolExecutor(max_workers=4) # For IO bound API calls

    def _estimate_confidence(self, text: Optional[str]) -> float:
        """
        Estimate the confidence of the OCR output based on text heuristics.
        Returns a float between 0.0 and 1.0.
        """
        if not text:
            return 0.0
            
        # Basic heuristic: 
        # - High number of '' (replacement character) means low confidence
        # - Too short text means low confidence
        # - High ratio of alphabetic/Arabic characters to symbols means high confidence
        if len(text.strip()) < 10:
            return 0.1
            
        replacement_count = text.count("\ufffd")
        if replacement_count > 0:
            ratio = replacement_count / len(text)
            if ratio > 0.05:
                return max(0.1, 1.0 - (ratio * 10))
        
        # In a real implementation, this could use durak or a language model
        # to calculate a perplexity score. For now, if it extracted clean text,
        # we give it a decent score.
        import re
        # Count alphanumeric + Arabic chars vs total non-space chars
        valid_chars = len(re.findall(r'[\w\u0600-\u06FF]', text))
        non_space_chars = len(re.findall(r'\S', text))
        
        if non_space_chars == 0:
            return 0.0
            
        ratio = valid_chars / non_space_chars
        # Scale ratio to be slightly optimistic
        return min(1.0, ratio * 1.1)

    def _crop_pdf_page(self, pdf_path: str, bbox: list) -> str:
        """Create a temporary PDF containing only the cropped bounding box area."""
        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = fitz.Rect(*bbox)
        # Ensure cropbox doesn't exceed mediabox
        rect.intersect(page.rect)
        page.set_cropbox(rect)
        
        fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix="ottominer_crop_")
        os.close(fd)
        
        # Save to temp path (preserves the cropbox setting for PyMuPDF rendering)
        doc.save(temp_path)
        doc.close()
        return temp_path

    async def _run_stream_b_with_layout_async(self, page_pdf_path: str) -> Optional[str]:
        """Run Stream B on layout-segmented zones of the page."""
        loop = asyncio.get_running_loop()
        
        if not self.enable_layout:
            return await loop.run_in_executor(
                self.thread_pool,
                _run_ocr_backend_sync,
                self.stream_b_name,
                self.stream_b_config,
                page_pdf_path
            )
            
        # 1. Segment Page
        zones = await loop.run_in_executor(
            self.thread_pool,
            self.layout_segmenter.segment_page,
            Path(page_pdf_path)
        )
        
        if not zones:
            return await loop.run_in_executor(
                self.thread_pool,
                _run_ocr_backend_sync,
                self.stream_b_name,
                self.stream_b_config,
                page_pdf_path
            )
            
        # 2. Crop and run OCR for each zone
        tasks = []
        crop_paths = []
        
        for zone in zones:
            # We skip pure illustrations unless the VLM can caption them.
            # Mistral Pixtral can caption illustrations. We'll pass all zones.
            bbox = zone.get("bbox")
            if bbox:
                crop_path = self._crop_pdf_page(page_pdf_path, bbox)
                crop_paths.append(crop_path)
                
                # Start OCR task for this crop
                task = loop.run_in_executor(
                    self.thread_pool,
                    _run_ocr_backend_sync,
                    self.stream_b_name,
                    self.stream_b_config,
                    crop_path
                )
                tasks.append(task)
                
        if not tasks:
            return None
            
        # 3. Gather results and merge
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cleanup crops
        for cp in crop_paths:
            try:
                os.unlink(cp)
            except Exception:
                pass
                
        valid_texts = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Stream B zone OCR failed: {res}")
            elif res and res.strip():
                valid_texts.append(res.strip())
                
        return "\\n\\n".join(valid_texts)

    async def _process_page_async(self, page_pdf_path: str) -> Optional[str]:
        """
        Run the dual-stream logic on a single page.
        """
        loop = asyncio.get_running_loop()
        
        # Start Stream A (Local Process Pool)
        stream_a_task = loop.run_in_executor(
            self.process_pool, 
            _run_ocr_backend_sync, 
            self.stream_a_name, 
            self.stream_a_config, 
            page_pdf_path
        )
        
        # Start Stream B (Thread Pool with Layout segmentation)
        stream_b_task = asyncio.create_task(self._run_stream_b_with_layout_async(page_pdf_path))
        
        # Wait for Stream A to finish first (it should be faster)
        # We use asyncio.wait to see which finishes first, but typically A is local
        done, pending = await asyncio.wait(
            [stream_a_task, stream_b_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        if stream_a_task in done:
            try:
                local_text = stream_a_task.result()
                confidence = self._estimate_confidence(local_text)
                logger.debug(f"Stream A ({self.stream_a_name}) finished with confidence {confidence:.2f}")
                
                if confidence >= self.confidence_threshold:
                    logger.info(f"Stream A confidence high enough ({confidence:.2f} >= {self.confidence_threshold}). Cancelling Stream B.")
                    # In python futures, cancelling a running thread/process is hard, 
                    # but we can ignore the result.
                    if not stream_b_task.done():
                        stream_b_task.cancel()
                    return local_text
            except Exception as e:
                logger.error(f"Stream A failed: {e}")
        
        # If we reach here, Stream A was not confident enough or failed. Wait for Stream B.
        try:
            if not stream_b_task.done():
                logger.info("Waiting for Stream B (Remote VLM) to finish...")
            remote_text = await stream_b_task
            
            # Simple hybrid merge: If Stream B returned text, use it. Else fallback to A.
            if remote_text and remote_text.strip():
                return remote_text
            
            # If Stream B failed or returned empty, we must await Stream A
            if not stream_a_task.done():
                logger.info("Stream B returned no text. Waiting for Stream A to finish...")
                return await stream_a_task
            elif stream_a_task.exception() is None:
                return stream_a_task.result()
                
        except Exception as e:
            logger.error(f"Stream B failed: {e}")
            if not stream_a_task.done():
                return await stream_a_task
            elif stream_a_task.exception() is None:
                return stream_a_task.result()
                
        return None

    def _split_pdf_to_pages(self, file_path: Path) -> List[str]:
        """Split a PDF into single-page temporary PDFs."""
        doc = fitz.open(str(file_path))
        page_paths = []
        
        # Create a temp dir for this document
        temp_dir = Path(tempfile.mkdtemp(prefix="ottominer_vdi_"))
        
        for i in range(doc.page_count):
            page_doc = fitz.open()
            page_doc.insert_pdf(doc, from_page=i, to_page=i)
            page_path = temp_dir / f"page_{i}.pdf"
            page_doc.save(str(page_path))
            page_doc.close()
            page_paths.append(str(page_path))
            
        doc.close()
        return page_paths

    async def _extract_async(self, file_path: Path) -> str:
        """Asynchronously extract text from all pages in a PDF."""
        logger.info(f"Starting Universal VDI extraction for {file_path}")
        
        # Fast path: try text extraction first
        try:
            import pymupdf4llm
            text = pymupdf4llm.to_markdown(str(file_path))
            if text and len(text.strip()) > 10:
                logger.info(f"Native text extraction successful for {file_path}")
                return text
        except Exception as e:
            logger.debug(f"Native text extraction failed: {e}")
            
        page_paths = self._split_pdf_to_pages(file_path)
        
        # Process pages concurrently
        tasks = [self._process_page_async(p) for p in page_paths]
        
        # Gather results in order
        results = await asyncio.gather(*tasks)
        
        # Cleanup temp files
        for p in page_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        try:
            os.rmdir(Path(page_paths[0]).parent)
        except Exception:
            pass
            
        # Combine texts and normalize to NFC for strict diacritic preservation
        import unicodedata
        valid_texts = [t for t in results if t is not None]
        combined_text = "\\n\\n".join(valid_texts)
        return unicodedata.normalize("NFC", combined_text)

    def extract(self, file_path: Union[str, Path]) -> str:
        """Synchronous wrapper for the async extraction."""
        path = self.validate_file(file_path)
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self._extract_async(path))

    def batch_extract(self, file_paths: List[Union[str, Path]]) -> Dict[str, str]:
        """Extract multiple PDFs sequentially (each PDF is parallelized internally)."""
        results = {}
        for path in file_paths:
            try:
                results[str(path)] = self.extract(path)
            except Exception as e:
                logger.error(f"Batch extraction failed for {path}: {e}")
                results[str(path)] = ""
        return results

    def __del__(self):
        """Cleanup thread and process pools."""
        self.process_pool.shutdown(wait=False)
        self.thread_pool.shutdown(wait=False)
