import os
import sys
import time

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)
from config import *


def open_camera(max_attempts=5):
    backends = []
    if hasattr(cv2, 'CAP_DSHOW'):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, 'CAP_MSMF'):
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    for backend in backends:
        for _ in range(max_attempts):
            cam = cv2.VideoCapture(0, backend)
            if cam.isOpened():
                return cam
            cam.release()
            time.sleep(0.5)
    return None


face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
cam = open_camera()
if cam is None or not cam.isOpened():
    print('[ERROR] Kamera tidak bisa dibuka. Pastikan kamera sudah terhubung dan izin akses kamera diberikan.')
    sys.exit(1)

face_id = input('\nMasukkan user id: ').strip()
face_name = input(f'Nama untuk user {face_id}: ').strip()

print('\n[INFO] Mengambil data wajah. Posisikan wajah di depan kamera...')

os.makedirs(DATASET_PATH, exist_ok=True)
file_path = os.path.join(DATASET_PATH, 'labels.txt')
file_exists = os.path.exists(file_path)
with open(file_path, 'a', encoding='utf-8') as file:
    if file_exists:
        file.write('\n')
    file.write(face_name)

count = 0
target_samples = NUM_SAMPLES
frame_count = 0
start_time = time.time()
fps = 0.0

while count < target_samples:
    ret, img = cam.read()
    if not ret or img is None or img.size == 0:
        time.sleep(0.1)
        continue

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        cv2.putText(img, 'Wajah tidak terdeteksi', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'Progress: {count}/{target_samples}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('image', img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        continue

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w + 20, y + h + 20), (255, 0, 0), 2)
        face_gray = gray[y:y + h, x:x + w]
        if face_gray.size == 0:
            continue

        face_gray = cv2.resize(face_gray, (128, 128))
        out_path = os.path.join(DATASET_PATH, f'User.{face_id}.{count}.jpg')
        if cv2.imwrite(out_path, face_gray):
            count += 1
            print(f'[INFO] Menyimpan {out_path}')
        else:
            print(f'[WARN] Gagal menyimpan {out_path}')

        frame_count += 1
        if frame_count >= 1:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            frame_count = 0
            start_time = time.time()

        cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'Progress: {count}/{target_samples}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'Img Captured: {count}', (img.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if count >= target_samples:
            break

    cv2.imshow('image', img)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break

print('\n[INFO] Proses selesai. Kamera ditutup.')
cam.release()
cv2.destroyAllWindows()
