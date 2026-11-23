import base64
import os
import sqlite3
import sys
import threading
import cv2
import numpy as np
from datetime import datetime
import time
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from training.model import model as build_cnn_model

from config import (
    BATCH_SIZE,
    DATASET_PATH,
    EPOCHS,
    FACE_CASCADE_PATH,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    NUM_SAMPLES,
    RANDOM_STATE,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'face-attendance-secret'
app.config['DATABASE'] = os.path.join(SCRIPT_DIR, 'attendance.db')
app.config['TRAINING_STATUS'] = {
    'running': False,
    'message': 'Siap',
    'progress': 0,
}

DB_PATH = app.config['DATABASE']

VIDEO_CAPTURE_SOURCE = 0
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
VIDEO_CAPTURE_FPS = 15
VIDEO_PROCESS_WIDTH = 320
VIDEO_SKIP_FRAMES = 2
VIDEO_CAPTURE_OPEN_ATTEMPTS = 5
VIDEO_CAPTURE_READ_FAILURE_THRESHOLD = 30

VIDEO_CAPTURE_BACKENDS = []
for backend_name in ('CAP_DSHOW', 'CAP_MSMF'):
    if hasattr(cv2, backend_name):
        VIDEO_CAPTURE_BACKENDS.append(getattr(cv2, backend_name))
VIDEO_CAPTURE_BACKENDS.append(cv2.CAP_ANY)

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX

cnn_model = None
cnn_student_class_map = {}
cnn_class_to_student = {}

current_prediction = {
    'name': 'Menunggu wajah...',
    'confidence': 0.0,
    'class_idx': -1,
}
prediction_lock = threading.Lock()


def get_current_prediction_snapshot():
    with prediction_lock:
        return current_prediction.copy()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def open_video_capture(source, max_attempts=VIDEO_CAPTURE_OPEN_ATTEMPTS):
    for backend in VIDEO_CAPTURE_BACKENDS:
        for attempt in range(max_attempts):
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                cap.release()
                time.sleep(0.2)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                return cap

            cap.release()
            time.sleep(0.2)

    return None


def create_error_frame(message='Kamera tidak tersedia'):
    blank = np.zeros((VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank, message, (10, VIDEO_FRAME_HEIGHT // 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return None
    return (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def login_required(func):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


def detect_faces_with_haar(frame):
    if face_cascade is None or face_cascade.empty():
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
    )


def count_student_face_samples(student_id):
    if not os.path.exists(DATASET_PATH):
        return 0

    count = 0
    for filename in os.listdir(DATASET_PATH):
        if filename.startswith(f'User.{student_id}.') and filename.lower().endswith('.jpg'):
            count += 1
    return count


def crop_face_region(frame, face_box, padding_ratio=0.25):
    x, y, w, h = face_box
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    return frame[y1:y2, x1:x2]


def preprocess_face_image(image):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, IMAGE_SIZE)
    gray = gray.astype('float32') / 255.0
    return gray


def preprocess_for_storage(image):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, IMAGE_SIZE)
    return gray


def preprocess_for_inference(frame, face_box):
    x, y, w, h = face_box
    face = frame[y:y + h, x:x + w]
    if face is None or face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, IMAGE_SIZE)
    gray = gray.astype('float32') / 255.0
    return gray.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


def preprocess_for_cnn(face):
    if face is None or face.size == 0:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, IMAGE_SIZE)
    gray = gray.astype('float32') / 255.0
    return gray.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


DATASET_CAPTURE_STEPS = [
    {'id': 'front', 'label': 'Wajah depan', 'instruction': 'Lihat lurus ke kamera'},
    {'id': 'left', 'label': 'Miring kiri', 'instruction': 'Putar kepala sedikit ke kiri'},
    {'id': 'right', 'label': 'Miring kanan', 'instruction': 'Putar kepala sedikit ke kanan'},
    {'id': 'up', 'label': 'Menengok atas', 'instruction': 'Lihat sedikit ke atas'},
    {'id': 'down', 'label': 'Menengok bawah', 'instruction': 'Lihat sedikit ke bawah'},
    {'id': 'smile', 'label': 'Senyum', 'instruction': 'Beri ekspresi senyum'},
    {'id': 'open_mouth', 'label': 'Mulut terbuka', 'instruction': 'Buka mulut sedikit'},
]


def get_cnn_training_config():
    return {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
    }


def load_cnn_model():
    global cnn_model, cnn_class_to_student
    if cnn_model is not None:
        return cnn_model
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        students = get_registered_students()
        num_classes = max(2, len(students))
        recognizer = build_cnn_model(IMAGE_SIZE + (1,), num_classes, learning_rate=LEARNING_RATE)
        recognizer.load_weights(MODEL_PATH)
        cnn_model = recognizer
        if students and not cnn_class_to_student:
            student_ids = sorted(student['student_id'] for student in students)
            cnn_class_to_student = {idx: student_id for idx, student_id in enumerate(student_ids)}
        return cnn_model
    except Exception as exc:
        print(f'[CNN] Gagal memuat model: {exc}')
        return None


def build_cnn_student_class_map():
    global cnn_student_class_map, cnn_class_to_student
    students = get_registered_students()
    if not students:
        cnn_student_class_map = {}
        cnn_class_to_student = {}
        return {}

    student_ids = sorted(student['student_id'] for student in students)
    cnn_class_to_student = {idx: student_id for idx, student_id in enumerate(student_ids)}
    student_lookup = {student['student_id']: student['name'] for student in students}
    cnn_student_class_map = {
        student_id: {
            'name': student_lookup.get(student_id, student_id),
            'class_idx': idx,
        }
        for idx, student_id in enumerate(student_ids)
    }
    return cnn_student_class_map


def set_training_status(running, message, progress=0):
    app.config['TRAINING_STATUS'] = {
        'running': running,
        'message': message,
        'progress': progress,
    }


def train_cnn_model():
    global cnn_model, cnn_class_to_student, cnn_student_class_map

    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras import callbacks as keras_callbacks
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    set_training_status(True, 'Menyiapkan data pelatihan...', 10)
    students = get_registered_students()
    if not students:
        set_training_status(False, 'Tidak ada siswa terdaftar', 0)
        print('[CNN] Tidak ada siswa terdaftar, pelatihan model dibatalkan.')
        return False

    student_ids = sorted(student['student_id'] for student in students)
    student_lookup = {student['student_id']: student['name'] for student in students}
    class_index_by_student = {student_id: idx for idx, student_id in enumerate(student_ids)}

    samples = []
    labels = []
    for filename in sorted(os.listdir(DATASET_PATH)):
        if not filename.lower().endswith('.jpg') or not filename.startswith('User.'):
            continue
        parts = filename.split('.')
        if len(parts) < 3:
            continue
        student_id = parts[1]
        if student_id not in class_index_by_student:
            continue

        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        image = image.astype('float32') / 255.0
        samples.append(image)
        labels.append(class_index_by_student[student_id])

    if len(samples) < 2:
        set_training_status(False, f'Tidak cukup sampel untuk melatih model: {len(samples)} sampel ditemukan.', 0)
        print(f'[CNN] Tidak cukup sampel untuk melatih model: {len(samples)} sampel ditemukan.')
        return False

    samples_array = np.array(samples, dtype='float32')[..., np.newaxis]
    labels_array = np.array(labels, dtype='int32')
    effective_num_classes = max(2, len(student_ids))
    labels_categorical = to_categorical(labels_array, num_classes=effective_num_classes)

    set_training_status(True, f'Memulai pelatihan model dari {len(samples)} sampel...', 25)
    print(f'[CNN] Memulai pelatihan model dari {len(samples)} sampel untuk {len(student_ids)} siswa...')
    if len(np.unique(labels_array)) > 1:
        val_test_split = VALIDATION_SPLIT + TEST_SPLIT
        x_train, x_temp, y_train, y_temp, _, labels_temp = train_test_split(
            samples_array,
            labels_categorical,
            labels_array,
            test_size=val_test_split,
            random_state=RANDOM_STATE,
            stratify=labels_array,
        )
        val_fraction = TEST_SPLIT / val_test_split if val_test_split > 0 else 0.5
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=val_fraction,
            random_state=RANDOM_STATE,
            stratify=labels_temp,
        )
    else:
        x_train, x_val, x_test = samples_array, samples_array, samples_array
        y_train, y_val, y_test = labels_categorical, labels_categorical, labels_categorical

    training_config = get_cnn_training_config()
    set_training_status(True, 'Melatih jaringan CNN dengan augmentasi...', 40)
    K.clear_session()
    recognizer = build_cnn_model(
        IMAGE_SIZE + (1,),
        effective_num_classes,
        learning_rate=training_config['learning_rate'],
    )
    checkpoint = keras_callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )

    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    train_generator = augmentor.flow(
        x_train,
        y_train,
        batch_size=training_config['batch_size'],
    )
    steps_per_epoch = max(1, len(x_train) // training_config['batch_size'])

    recognizer.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=training_config['epochs'],
        validation_data=(x_val, y_val),
        callbacks=[checkpoint],
    )
    set_training_status(True, 'Menyimpan model hasil pelatihan...', 90)
    recognizer.save_weights(MODEL_PATH)
    recognizer.load_weights(MODEL_PATH)

    cnn_model = recognizer
    cnn_class_to_student = {idx: student_id for idx, student_id in enumerate(student_ids)}
    cnn_student_class_map = {
        student_id: {
            'name': student_lookup.get(student_id, student_id),
            'class_idx': idx,
        }
        for idx, student_id in enumerate(student_ids)
    }
    set_training_status(False, f'Pelatihan selesai dan model disimpan ke {MODEL_PATH}', 100)
    print(f'[CNN] Pelatihan selesai dan model disimpan ke {MODEL_PATH}')
    return True


def recognize_student_with_cnn(frame, face_box):
    model = load_cnn_model()
    if model is None:
        return None

    if not cnn_student_class_map:
        build_cnn_student_class_map()

    x, y, w, h = face_box
    face = frame[y:y + h, x:x + w]
    if face is None or face.size == 0:
        return None

    tensor = preprocess_for_cnn(face)
    if tensor is None:
        return None

    prediction = model.predict(tensor, verbose=0)[0]
    class_idx = int(np.argmax(prediction))
    confidence = float(prediction[class_idx])

    student_id = cnn_class_to_student.get(class_idx)
    if student_id is None:
        return None

    student_meta = cnn_student_class_map.get(student_id)
    if student_meta is None:
        return None

    return {
        'student_id': student_id,
        'student_name': student_meta['name'],
        'confidence': confidence,
        'class_idx': class_idx,
    }


def trigger_training_if_ready(student_id, target_samples):
    global cnn_model, cnn_student_class_map, cnn_class_to_student

    existing_count = count_student_face_samples(student_id)
    if existing_count < max(1, int(target_samples)):
        return False

    cnn_model = None
    cnn_student_class_map = {}
    cnn_class_to_student = {}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    def run_training():
        try:
            train_cnn_model()
        except Exception as exc:
            print(f'[CNN] Pelatihan gagal dijalankan di background: {exc}')

    threading.Thread(target=run_training, daemon=True).start()
    return True


def save_registered_faces(student_id, images, label_name=None, target_samples=None):
    global cnn_model, cnn_student_class_map, cnn_class_to_student

    os.makedirs(DATASET_PATH, exist_ok=True)

    if label_name:
        labels_path = os.path.join(DATASET_PATH, 'labels.txt')
        existing_labels = []
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as file:
                existing_labels = [line.strip() for line in file.read().splitlines() if line.strip()]
        if label_name not in existing_labels:
            with open(labels_path, 'a', encoding='utf-8') as file:
                if existing_labels:
                    file.write('\n')
                file.write(label_name)

    existing_count = count_student_face_samples(student_id)

    target_samples = target_samples or NUM_SAMPLES
    target_samples = max(1, min(int(target_samples), NUM_SAMPLES))
    remaining_slots = max(0, target_samples - existing_count)
    if remaining_slots <= 0:
        trigger_training_if_ready(student_id, target_samples)
        build_cnn_student_class_map()
        return []

    images = images[:remaining_slots]
    saved_paths = []
    next_index = existing_count
    for image in images:
        if image is None or image.size == 0:
            continue
        out_path = os.path.join(DATASET_PATH, f'User.{student_id}.{next_index}.jpg')
        if cv2.imwrite(out_path, image):
            saved_paths.append(out_path)
            next_index += 1

    if saved_paths:
        total_saved = existing_count + len(saved_paths)
        if total_saved >= target_samples:
            trigger_training_if_ready(student_id, target_samples)
        else:
            build_cnn_student_class_map()
    else:
        build_cnn_student_class_map()
    return saved_paths


def save_registered_face(student_id, image, label_name=None):
    return save_registered_faces(student_id, [image], label_name=label_name)


def get_registered_students():
    conn = get_db()
    students = conn.execute('SELECT * FROM students ORDER BY id DESC').fetchall()
    conn.close()
    return students


def get_student_face_samples(student_id):
    dataset_files = [
        os.path.join(DATASET_PATH, f)
        for f in os.listdir(DATASET_PATH)
        if f.lower().endswith('.jpg') and f.startswith(f'User.{student_id}.')
    ]
    return sorted(set(dataset_files))


def recognize_student(frame, face_box):
    return recognize_student_with_cnn(frame, face_box)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Username atau password salah')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        confirm = request.form['confirm_password']
        if not username or not password:
            return render_template('register.html', error='Semua field wajib diisi')
        if password != confirm:
            return render_template('register.html', error='Password tidak cocok')
        conn = get_db()
        existing = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if existing:
            conn.close()
            return render_template('register.html', error='Username sudah ada')
        conn.execute(
            'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
            (username, generate_password_hash(password), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    students = get_registered_students()
    conn = get_db()
    attendance_rows = conn.execute('SELECT * FROM attendance ORDER BY id DESC LIMIT 10').fetchall()
    attendance_today = conn.execute(
        'SELECT COUNT(*) AS count FROM attendance WHERE date(recorded_at) = ?',
        (datetime.now().strftime('%Y-%m-%d'),)
    ).fetchone()['count']
    conn.close()
    return render_template(
        'dashboard.html',
        students=students,
        attendance_rows=attendance_rows,
        total_students=len(students),
        attendance_today=attendance_today,
    )


@app.route('/students/register', methods=['GET', 'POST'])
@login_required
def register_student():
    if request.method == 'POST':
        student_id = request.form['student_id'].strip()
        name = request.form['name'].strip()
        if not student_id or not name:
            return render_template('register_student.html', error='Student ID dan nama wajib diisi')
        conn = get_db()
        existing = conn.execute('SELECT id FROM students WHERE student_id = ?', (student_id,)).fetchone()
        if existing:
            conn.close()
            return render_template('register_student.html', error='Student ID sudah ada')
        conn.execute(
            'INSERT INTO students (student_id, name, created_by, created_at) VALUES (?, ?, ?, ?)',
            (student_id, name, session['user_id'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        conn.close()
        session['current_student_id'] = student_id
        session['current_student_name'] = name
        return redirect(url_for('register_student'))
    return render_template('register_student.html', student_id=session.get('current_student_id'), student_name=session.get('current_student_name'))


@app.route('/haarcascade_frontalface_default.xml')
def haarcascade_file():
    return send_file(FACE_CASCADE_PATH, mimetype='application/xml')


@app.route('/students/register/capture', methods=['POST'])
@login_required
def capture_student_face():
    data = request.get_json()
    if not data:
        return {'status': 'error', 'message': 'Gambar tidak diterima'}, 400

    student_id = session.get('current_student_id')
    if not student_id:
        return {'status': 'error', 'message': 'Silakan daftarkan data siswa dahulu'}, 400

    image_payloads = data.get('images') if isinstance(data.get('images'), list) else []
    if not image_payloads and 'image' in data:
        image_payloads = [data['image']]
    if not image_payloads:
        return {'status': 'error', 'message': 'Gambar tidak diterima'}, 400

    capture_mode = data.get('capture_mode', 'manual')
    target_samples = data.get('target_samples', NUM_SAMPLES)
    try:
        target_samples = int(target_samples)
    except (TypeError, ValueError):
        target_samples = NUM_SAMPLES
    target_samples = max(1, min(target_samples, NUM_SAMPLES))

    processed_faces = []
    for image_payload in image_payloads:
        if len(processed_faces) >= target_samples:
            break
        if not isinstance(image_payload, str) or ',' not in image_payload:
            continue
        image_data = base64.b64decode(image_payload.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        faces = detect_faces_with_haar(image)
        if len(faces) == 0:
            continue

        x, y, w, h = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
        face_img = image[y:y + h, x:x + w]
        face_gray = preprocess_for_storage(face_img)
        if face_gray is not None:
            processed_faces.append(face_gray)

    current_count = len(get_student_face_samples(student_id))
    if not processed_faces:
        return {
            'status': 'warning',
            'message': 'Wajah tidak terdeteksi, lanjutkan mengarahkan wajah ke kamera',
            'saved_count': 0,
            'total_count': current_count,
            'target_samples': target_samples,
            'completed': current_count >= target_samples,
        }, 200

    target_for_save = max(target_samples, current_count + len(processed_faces))
    saved_paths = save_registered_faces(
        student_id,
        processed_faces,
        label_name=session.get('current_student_name'),
        target_samples=target_for_save,
    )
    if not saved_paths:
        return {'status': 'error', 'message': 'Gagal menyimpan gambar wajah'}, 400

    total_count = current_count + len(saved_paths)
    return {
        'status': 'ok',
        'message': f'{len(saved_paths)} gambar wajah berhasil ditambahkan',
        'saved_count': len(saved_paths),
        'total_count': total_count,
        'target_samples': target_samples,
        'completed': total_count >= target_samples,
        'capture_mode': capture_mode,
        'steps': DATASET_CAPTURE_STEPS,
    }


@app.route('/attendance', methods=['GET', 'POST'])
@login_required
def attendance():
    message = None
    if request.method == 'POST':
        try:
            prediction = get_current_prediction_snapshot()
            if prediction['name'] != 'Menunggu wajah...' and prediction['confidence'] >= 0.6:
                conn = get_db()
                student = conn.execute('SELECT * FROM students WHERE name = ?', (prediction['name'],)).fetchone()
                if student:
                    conn.execute(
                        'INSERT INTO attendance (student_id, student_name, recorded_at, confidence) VALUES (?, ?, ?, ?)',
                        (student['student_id'], student['name'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), prediction['confidence'])
                    )
                    conn.commit()
                    conn.close()
                    message = f'Absensi berhasil untuk {student["name"]}'
                    status = 'ok'
                else:
                    conn.close()
                    message = 'Siswa belum terdaftar'
                    status = 'warning'
            else:
                message = 'Prediksi belum cukup stabil'
                status = 'warning'
        except Exception as exc:
            message = f'Gagal melakukan absensi: {exc}'
            status = 'error'

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': status, 'message': message})

    return render_template('attendance.html', message=message, current_prediction=current_prediction)


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    global current_prediction
    cap = open_video_capture(VIDEO_CAPTURE_SOURCE)
    if cap is None or not cap.isOpened():
        error_frame = create_error_frame('Kamera tidak bisa dibuka')
        if error_frame:
            yield error_frame
        return

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, VIDEO_CAPTURE_FPS)

        frame_count = 0
        start_time = time.time()
        fps = 0.0
        processed_count = 0
        failed_reads = 0
        face_box = None
        last_prediction = {'name': 'Menunggu wajah...', 'confidence': 0.0, 'class_idx': -1}

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                failed_reads += 1
                if failed_reads >= VIDEO_CAPTURE_READ_FAILURE_THRESHOLD:
                    cap.release()
                    cap = open_video_capture(VIDEO_CAPTURE_SOURCE)
                    if cap is None or not cap.isOpened():
                        break
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, VIDEO_CAPTURE_FPS)
                    failed_reads = 0
                time.sleep(0.1)
                continue

            failed_reads = 0
            if frame.shape[1] != VIDEO_PROCESS_WIDTH:
                process_frame = cv2.resize(frame, (VIDEO_PROCESS_WIDTH, int(frame.shape[0] * VIDEO_PROCESS_WIDTH / frame.shape[1])))
            else:
                process_frame = frame.copy()

            if processed_count % VIDEO_SKIP_FRAMES == 0:
                faces = detect_faces_with_haar(process_frame)
                if len(faces) > 0:
                    face_box = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
                    x, y, w, h = face_box
                    x = int(x * frame.shape[1] / process_frame.shape[1])
                    y = int(y * frame.shape[0] / process_frame.shape[0])
                    w = int(w * frame.shape[1] / process_frame.shape[1])
                    h = int(h * frame.shape[0] / process_frame.shape[0])
                    face_box = (x, y, w, h)
                else:
                    face_box = None

            if face_box is not None:
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if processed_count % VIDEO_SKIP_FRAMES == 0:
                    match = recognize_student(frame, face_box)
                    if match:
                        last_prediction = {
                            'name': match['student_name'],
                            'confidence': match['confidence'],
                            'class_idx': match.get('class_idx', -1),
                        }
                    else:
                        last_prediction = {'name': 'Menunggu wajah...', 'confidence': 0.0, 'class_idx': -1}

                with prediction_lock:
                    current_prediction = last_prediction.copy()
                label_text = f"{last_prediction['name']} {last_prediction['confidence']:.2f}"
                cv2.putText(frame, label_text, (x + 5, y - 10), font, 0.8, (255, 255, 255), 2)
            else:
                last_prediction = {'name': 'Menunggu wajah...', 'confidence': 0.0, 'class_idx': -1}
                with prediction_lock:
                    current_prediction = last_prediction.copy()

            frame_count += 1
            processed_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed

            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), font, 0.8, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.01)
    finally:
        if cap is not None:
            cap.release()


@app.route('/api/dataset-info')
@login_required
def api_dataset_info():
    """API endpoint untuk info dataset"""
    students = get_registered_students()
    
    dataset_info = {
        'total_students': len(students),
        'students': []
    }
    
    for student in students:
        samples = get_student_face_samples(student['student_id'])
        dataset_info['students'].append({
            'student_id': student['student_id'],
            'name': student['name'],
            'sample_count': len(samples),
            'target_samples': NUM_SAMPLES
        })
    
    dataset_info['total_samples'] = sum(s['sample_count'] for s in dataset_info['students'])
    
    return dataset_info


@app.route('/export-attendance', methods=['GET'])
@login_required
def export_attendance():
    """Export attendance data ke CSV"""
    import csv
    from io import StringIO
    
    conn = get_db()
    rows = conn.execute('SELECT * FROM attendance ORDER BY recorded_at DESC').fetchall()
    conn.close()
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Student ID', 'Student Name', 'Recorded At', 'Confidence'])
    
    for row in rows:
        writer.writerow([row['student_id'], row['student_name'], row['recorded_at'], f"{row['confidence']:.4f}"])
    
    output.seek(0)
    return send_file(
        StringIO(output.getvalue()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )


@app.route('/clear-attendance', methods=['POST'])
@login_required
def clear_attendance():
    """Clear semua attendance records"""
    conn = get_db()
    conn.execute('DELETE FROM attendance')
    conn.commit()
    conn.close()
    
    return {'status': 'ok', 'message': 'Attendance data cleared'}


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
