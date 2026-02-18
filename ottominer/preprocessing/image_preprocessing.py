from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def _opencv_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def _skimage_available() -> bool:
    try:
        from skimage import filters, morphology  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class PreprocessingConfig:
    """Configuration for document image preprocessing."""

    denoise: bool = True
    denoise_strength: int = 10

    binarize: bool = True
    binarization_method: str = "otsu"
    block_size: int = 11
    c: int = 2

    deskew: bool = True
    deskew_angle_limit: float = 45.0

    remove_borders: bool = True
    border_size: int = 10

    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0

    remove_lines: bool = False
    line_thickness: int = 1

    dilate_text: bool = False
    dilation_iterations: int = 1

    target_dpi: int = 300


def preprocess_image(
    image: np.ndarray, config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """
    Preprocess a document image for improved OCR accuracy.

    Pipeline:
        1. Denoise (remove noise and artifacts)
        2. Deskew (correct rotation)
        3. Enhance contrast
        4. Binarize (convert to black and white)
        5. Remove borders/lines
        6. Optional morphological operations

    Args:
        image: Input image as numpy array (BGR or grayscale)
        config: Preprocessing configuration

    Returns:
        Preprocessed image ready for OCR
    """
    if not _opencv_available():
        logger.warning("OpenCV not available. Install: pip install opencv-python")
        return image

    import cv2

    config = config or PreprocessingConfig()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if config.enhance_contrast:
        gray = _enhance_contrast(gray, config)

    if config.denoise:
        gray = _denoise(gray, config)

    if config.deskew:
        gray = _deskew(gray, config)

    if config.binarize:
        gray = _binarize(gray, config)

    if config.remove_lines:
        gray = _remove_lines(gray, config)

    if config.remove_borders:
        gray = _remove_borders(gray, config)

    if config.dilate_text:
        gray = _dilate_text(gray, config)

    return gray


def _enhance_contrast(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Enhance image contrast using CLAHE."""
    import cv2

    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit)
    return clahe.apply(image)


def _denoise(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Remove noise from image."""
    import cv2

    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=config.denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def _deskew(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Correct image rotation/skew."""
    import cv2

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -config.deskew_angle_limit:
        angle = -(90 + angle)
    elif angle > config.deskew_angle_limit:
        angle = 90 - angle
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def _binarize(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Convert image to binary (black and white)."""
    import cv2

    if config.binarization_method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif config.binarization_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            config.block_size,
            config.c,
        )
    elif config.binarization_method == "sauvola" and _skimage_available():
        from skimage.filters import threshold_sauvola

        thresh = threshold_sauvola(image, window_size=config.block_size)
        binary = (image > thresh).astype(np.uint8) * 255
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return binary


def _remove_lines(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Remove horizontal and vertical lines from document."""
    import cv2

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, config.line_thickness))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (config.line_thickness, 40))

    lines_h = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_h)
    lines_v = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_v)

    lines = cv2.add(lines_h, lines_v)
    result = cv2.subtract(image, lines)

    return result


def _remove_borders(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Remove border artifacts."""
    h, w = image.shape
    return image[
        config.border_size : h - config.border_size,
        config.border_size : w - config.border_size,
    ]


def _dilate_text(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Dilate text to make it thicker/more visible."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.dilate(image, kernel, iterations=config.dilation_iterations)


def estimate_image_quality(image: np.ndarray) -> dict:
    """
    Estimate image quality metrics for preprocessing decisions.

    Returns:
        Dictionary with quality metrics:
        - sharpness: Estimated sharpness (higher is better)
        - contrast: Contrast level (0-1)
        - noise_level: Estimated noise level (0-1)
        - skew_angle: Estimated skew angle in degrees
    """
    if not _opencv_available():
        return {"error": "OpenCV not available"}

    import cv2

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    contrast = gray.std() / 128.0

    h, w = gray.shape
    regions = [
        gray[0 : h // 4, 0 : w // 4],
        gray[0 : h // 4, 3 * w // 4 : w],
        gray[3 * h // 4 : h, 0 : w // 4],
        gray[3 * h // 4 : h, 3 * w // 4 : w],
    ]
    corner_mean = np.mean([r.mean() for r in regions])
    center_mean = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    noise_level = abs(corner_mean - center_mean) / 255.0

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
    else:
        angle = 0

    return {
        "sharpness": float(sharpness),
        "contrast": float(contrast),
        "noise_level": float(noise_level),
        "skew_angle": float(angle),
        "recommended_preprocessing": {
            "denoise": noise_level > 0.1,
            "deskew": abs(angle) > 0.5,
            "enhance_contrast": contrast < 0.3,
        },
    }


class ImagePreprocessor:
    """High-level image preprocessor with automatic quality detection."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def preprocess(
        self, image: np.ndarray, auto_detect: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image with optional automatic quality detection.

        Returns:
            Tuple of (preprocessed_image, quality_metrics)
        """
        quality = estimate_image_quality(image)

        config = self.config
        if auto_detect and "recommended_preprocessing" in quality:
            rec = quality["recommended_preprocessing"]
            config = PreprocessingConfig(
                denoise=rec.get("denoise", self.config.denoise),
                deskew=rec.get("deskew", self.config.deskew),
                enhance_contrast=rec.get(
                    "enhance_contrast", self.config.enhance_contrast
                ),
                binarize=self.config.binarize,
                remove_borders=self.config.remove_borders,
            )

        result = preprocess_image(image, config)
        return result, quality
