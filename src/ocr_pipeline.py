import cv2
import numpy as np


def preprocess_image(image_path: str):
    """Read and preprocess a document image using OpenCV."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return image, processed


def run_pytorch_document_encoder(image: np.ndarray):
    """Process the document image using PyTorch."""
    import torch

    tensor = torch.from_numpy(image).float() / 255.0

    mean_value = tensor.mean()
    std_value = tensor.std()

    return {
        "mean": float(mean_value.item()),
        "std": float(std_value.item())
    }


def extract_text(image_path: str):
    """Extract text using Tesseract OCR."""
    import pytesseract

    _, processed = preprocess_image(image_path)

    text = pytesseract.image_to_string(
        processed,
        config="--psm 6"
    )

    return text


def process_image(image_path: str):
    """Complete OCR document processing pipeline."""

    image, processed = preprocess_image(image_path)

    pytorch_features = run_pytorch_document_encoder(processed)

    text = extract_text(image_path)

    return {
        "text": text,
        "shape": tuple(image.shape),
        "pytorch_features": pytorch_features
    }
