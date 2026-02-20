import time
import fitz
import asyncio
from pathlib import Path
from typing import Tuple

from ottominer.extractors.universal import UniversalVDI
from ottominer.utils.logger import setup_logger

logger = setup_logger("benchmark")

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_metrics(ground_truth: str, extracted: str) -> Tuple[float, float]:
    """Calculate Character Error Rate (CER) and Word Error Rate (WER)."""
    if not extracted:
        extracted = ""
        
    gt_clean = ground_truth.strip().replace("\\n", " ")
    ext_clean = extracted.strip().replace("\\n", " ")
    
    # Character Error Rate (CER)
    char_distance = levenshtein_distance(gt_clean, ext_clean)
    cer = char_distance / max(len(gt_clean), 1)
    
    # Word Error Rate (WER)
    gt_words = gt_clean.split()
    ext_words = ext_clean.split()
    
    word_distance = levenshtein_distance(gt_words, ext_words)
    wer = word_distance / max(len(gt_words), 1)
    
    return cer, wer

def create_synthetic_pdf(text: str, output_path: str):
    """Create a purely image-based PDF using PyMuPDF (fitz) for benchmarking OCR."""
    import tempfile
    import os
    
    # Create text PDF first
    doc_temp = fitz.open()
    page = doc_temp.new_page()
    page.insert_text(fitz.Point(50, 50), text, fontsize=14)
    
    # Render to image
    pix = page.get_pixmap(dpi=150)
    w, h = page.rect.width, page.rect.height
    fd, temp_img = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    pix.save(temp_img)
    doc_temp.close()
    
    # Create final image-based PDF
    doc = fitz.open()
    img_page = doc.new_page(width=w, height=h)
    img_page.insert_image(img_page.rect, filename=temp_img)
    
    doc.save(output_path)
    doc.close()
    
    try:
        os.unlink(temp_img)
    except:
        pass

def run_benchmark():
    logger.info("Initializing UniversalVDI Benchmark...")
    
    # 1. Setup Test Data
    ground_truth_text = (
        "Ottoman Text Analysis Benchmark\\n"
        "Transliteration: Muḥammad, ʿAlī, ṣalāh, āyāt.\\n"
        "This is a test to verify diacritic preservation and extraction pace."
    )
    
    test_pdf_path = "benchmark_sample.pdf"
    create_synthetic_pdf(ground_truth_text, test_pdf_path)
    logger.info(f"Created synthetic PDF at {test_pdf_path}")
    
    # 2. Setup Extractor
    # Using native 'auto' or basic config to measure baseline performance
    extractor = UniversalVDI(config={
        "vdi": {
            "stream_a": "tesseract",
            "stream_b": "mistral"
        }
    })
    
    # 3. Run Extraction and Measure Pace
    logger.info("Starting extraction...")
    start_time = time.time()
    
    # Using the sync wrapper which manages its own event loop
    extracted_text = extractor.extract(test_pdf_path)
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    # 4. Calculate Accuracy (CER / WER)
    cer, wer = calculate_metrics(ground_truth_text, extracted_text)
    
    # 5. Report Results
    print("\\n" + "="*50)
    print(" " * 15 + "BENCHMARK RESULTS")
    print("="*50)
    print(f"Pace (Time Taken):      {time_elapsed:.3f} seconds")
    print(f"Character Error Rate:   {cer*100:.2f}%")
    print(f"Word Error Rate:        {wer*100:.2f}%")
    print(f"Extracted Length:       {len(extracted_text)} characters")
    print("="*50)
    
    # Clean up
    try:
        Path(test_pdf_path).unlink()
    except Exception as e:
        logger.warning(f"Could not clean up {test_pdf_path}: {e}")

if __name__ == "__main__":
    run_benchmark()
