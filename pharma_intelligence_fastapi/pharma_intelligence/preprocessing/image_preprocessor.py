from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from PIL import Image  # noqa: F401


logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] in {3, 4}:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def upscale_image(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
    height, width = image.shape[:2]
    if width >= 1500 and height >= 1500:
        return image
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def deskew_image(image: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return image

    angles = []
    for line in lines:
        rho, theta = line[0]
        _ = rho
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return image

    angle = float(np.median(angles))
    if abs(angle) <= 0.5 or abs(angle) >= 45:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize_image(image: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_pixels = int(np.sum(binary == 0))
    white_pixels = int(np.sum(binary == 255))
    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)
    return binary


def sharpen_image(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def save_preprocessed_image(image: np.ndarray, original_path: str) -> str:
    root, extension = os.path.splitext(original_path)
    output_path = f"{root}_preprocessed{extension or '.png'}"
    cv2.imwrite(output_path, image)
    return output_path


def preprocess_image(image_path: str) -> str:
    steps = [
        ("load_image", lambda current: load_image(image_path)),
        ("convert_to_grayscale", convert_to_grayscale),
        ("upscale_image", lambda current: upscale_image(current, scale_factor=2.0)),
        ("deskew_image", deskew_image),
        ("denoise_image", denoise_image),
        ("binarize_image", binarize_image),
        ("sharpen_image", sharpen_image),
        ("save_preprocessed_image", lambda current: save_preprocessed_image(current, image_path)),
    ]
    current = None
    for step_name, step in steps:
        try:
            current = step(current)
        except Exception as exc:
            logger.warning("Image preprocessing step %s failed for %s: %s", step_name, image_path, exc)
            return image_path
    return str(current)
