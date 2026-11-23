import os
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import IMAGE_SIZE, FACE_CASCADE_PATH


def preprocess_and_prepare_data(input_dir, output_dir=None, image_size=None, augment=True, normalize=True):
    """Single-function pipeline for labeling, face cropping, grayscale, resizing, and augmentation."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_path}")

    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    if image_size is None:
        image_size = IMAGE_SIZE

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if FACE_CASCADE_PATH and os.path.exists(FACE_CASCADE_PATH) else None
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in allowed_extensions])

    images = []
    labels = []

    for image_file in image_files:
        try:
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_region = image[y:y + h, x:x + w]
                else:
                    face_region = image
            else:
                face_region = image

            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if face_region.ndim == 3 else face_region
            resized = cv2.resize(gray_face, image_size)

            if normalize:
                resized = resized.astype("float32") / 255.0

            parts = image_file.stem.split(".")
            label = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            processed_images = [resized]
            processed_labels = [label]

            if augment:
                flipped = cv2.flip(resized, 1)
                processed_images.append(flipped)
                processed_labels.append(label)

            for processed, processed_label in zip(processed_images, processed_labels):
                images.append(processed)
                labels.append(processed_label)

                if output_path != input_path:
                    save_path = output_path / f"{image_file.stem}_{processed_label}.jpg"
                    save_image = (processed * 255).astype(np.uint8) if normalize else processed
                    cv2.imwrite(str(save_path), save_image)
        except Exception as exc:
            print(f"Skipped {image_file.name}: {exc}")

    if not images:
        return np.array([]), np.array([])

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    sample_dir = os.path.join(PROJECT_ROOT, "dataset")
    images, labels = preprocess_and_prepare_data(sample_dir)
    print(f"Jumlah citra: {len(images)}")
    print(f"Bentuk data: {images.shape}")
    print(f"Label: {labels[:10]}")
