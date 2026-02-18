import pytest
from pathlib import Path
import tempfile
import numpy as np


class TestPreprocessing:
    def test_preprocessing_config_defaults(self):
        from ottominer.preprocessing.image_preprocessing import PreprocessingConfig

        config = PreprocessingConfig()
        assert config.denoise is True
        assert config.binarize is True
        assert config.deskew is True
        assert config.target_dpi == 300

    def test_preprocessing_config_custom(self):
        from ottominer.preprocessing.image_preprocessing import PreprocessingConfig

        config = PreprocessingConfig(
            denoise=False, binarization_method="adaptive", target_dpi=200
        )
        assert config.denoise is False
        assert config.binarization_method == "adaptive"
        assert config.target_dpi == 200

    def test_opencv_availability(self):
        try:
            import cv2

            available = True
        except ImportError:
            available = False

        from ottominer.preprocessing.image_preprocessing import _opencv_available

        assert _opencv_available() == available

    def test_preprocess_image_returns_array(self):
        from ottominer.preprocessing.image_preprocessing import (
            preprocess_image,
            _opencv_available,
        )

        if not _opencv_available():
            pytest.skip("OpenCV not available")

        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 255

        result = preprocess_image(img)

        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2

    def test_estimate_image_quality(self):
        from ottominer.preprocessing.image_preprocessing import (
            estimate_image_quality,
            _opencv_available,
        )

        if not _opencv_available():
            pytest.skip("OpenCV not available")

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        quality = estimate_image_quality(img)

        assert "sharpness" in quality
        assert "contrast" in quality
        assert "noise_level" in quality


class TestOCRBackends:
    def test_base_ocr_backend_abstract(self):
        from ottominer.extractors.ocr.base import BaseOCRBackend

        with pytest.raises(TypeError):
            BaseOCRBackend()

    def test_get_backend_factory(self):
        from ottominer.extractors.ocr import get_backend

        backend = get_backend("tesseract")
        assert backend.name == "tesseract"

        backend = get_backend("kraken")
        assert backend.name == "kraken"

    def test_get_backend_unknown_raises(self):
        from ottominer.extractors.ocr import get_backend

        with pytest.raises(ValueError):
            get_backend("unknown_backend")

    def test_tesseract_backend_available(self):
        from ottominer.extractors.ocr.tesseract import TesseractBackend

        backend = TesseractBackend()
        assert isinstance(backend.is_available(), bool)

    def test_easyocr_backend_available(self):
        from ottominer.extractors.ocr.easyocr_backend import EasyOCRBackend

        backend = EasyOCRBackend()
        assert isinstance(backend.is_available(), bool)

    def test_ollama_backend_available(self):
        from ottominer.extractors.ocr.ollama import OllamaBackend

        backend = OllamaBackend()
        assert isinstance(backend.is_available(), bool)

    def test_kraken_backend_available(self):
        from ottominer.extractors.ocr.kraken import KrakenBackend

        backend = KrakenBackend()
        assert isinstance(backend.is_available(), bool)

    def test_mistral_backend_available(self):
        from ottominer.extractors.ocr.mistral import MistralBackend

        backend = MistralBackend()
        assert isinstance(backend.is_available(), bool)

    def test_backend_repr(self):
        from ottominer.extractors.ocr.tesseract import TesseractBackend

        backend = TesseractBackend()
        repr_str = repr(backend)
        assert "TesseractBackend" in repr_str
        assert "available" in repr_str
