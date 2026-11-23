import argparse
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from config import DATASET_PATH, FACE_CASCADE_PATH, IMAGE_SIZE, MODEL_PATH
from training.model import model


def load_labels(labels_path):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


def detect_face(image, cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Tidak dapat memuat cascade classifier dari: {cascade_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return faces


def preprocess_face(image, face_box):
    x, y, w, h = face_box
    face = image[y:y + h, x:x + w]
    if face is None or face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, IMAGE_SIZE)
    gray = gray.astype('float32') / 255.0
    return gray.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


def build_recognizer(num_classes):
    recognizer = model(IMAGE_SIZE + (1,), num_classes)
    recognizer.load_weights(MODEL_PATH)
    return recognizer


def identify_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    labels = load_labels(os.path.join(DATASET_PATH, 'labels.txt'))
    if not labels:
        raise ValueError('File labels kosong atau tidak valid.')

    recognizer = build_recognizer(len(labels))
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    faces = detect_face(image, FACE_CASCADE_PATH)
    if len(faces) == 0:
        print('Tidak ditemukan wajah pada gambar.')
        return

    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    face_box = faces[0]
    input_tensor = preprocess_face(image, face_box)
    if input_tensor is None:
        print('Gagal memproses wajah dari gambar.')
        return

    prediction = recognizer.predict(input_tensor, verbose=0)[0]
    class_idx = int(np.argmax(prediction))
    confidence = float(prediction[class_idx])

    print('Hasil identifikasi:')
    print(f'  Label       : {labels[class_idx]}')
    print(f'  Class index : {class_idx}')
    print(f'  Confidence  : {confidence:.4f}')

    top_indices = np.argsort(prediction)[::-1][:3]
    print('\nTop 3 prediksi:')
    for rank, idx in enumerate(top_indices, start=1):
        label_name = labels[idx] if 0 <= idx < len(labels) else 'Unknown'
        print(f'  {rank}. {label_name} ({idx}) = {prediction[idx]:.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Identifikasi gambar wajah dari foto test.')
    parser.add_argument('image_path', help='Path file gambar test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    identify_image(args.image_path)
