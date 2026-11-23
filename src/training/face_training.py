import numpy as np
from PIL import Image
import os
import io
import datetime
import itertools
import sys

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, average_precision_score, precision_recall_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from model import model

from config import *


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def save_text_to_file(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def save_metrics_to_file(path, metrics):
    with open(path, 'w', encoding='utf-8') as f:
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")


def save_model_summary(path, model):
    stream = io.StringIO()
    model.summary(print_fn=lambda line: stream.write(line + '\n'))
    save_text_to_file(path, stream.getvalue())


def smooth_curve(values, window=5):
    values = np.asarray(values, dtype=float)
    if values.size < 2 or window <= 1:
        return values
    window = min(window, values.size)
    weights = np.ones(window, dtype=float) / window
    smooth = np.convolve(values, weights, mode='same')
    half = window // 2
    if half > 0 and values.size >= half:
        smooth[:half] = np.cumsum(values[:half]) / np.arange(1, half + 1)
        smooth[-half:] = np.cumsum(values[-half:][::-1])[::-1] / np.arange(half, 0, -1)
    return smooth


class ValidationMetricsCallback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        if x_val is None or y_val is None:
            return
        y_prob = self.model.predict(x_val, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_val, axis=1)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        self.val_precision.append(precision)
        self.val_recall.append(recall)
        self.val_f1.append(f1)
        if logs is not None:
            logs['val_precision'] = precision
            logs['val_recall'] = recall
            logs['val_f1'] = f1


def get_history_series(history, keys):
    for key in keys:
        if key in history.history:
            return history.history[key]
    return []


def save_history_arrays(path, history, val_precision=None, val_recall=None, val_f1=None):
    np.savez_compressed(
        path,
        loss=np.array(get_history_series(history, ['loss']), dtype=float),
        val_loss=np.array(get_history_series(history, ['val_loss']), dtype=float),
        accuracy=np.array(get_history_series(history, ['accuracy', 'acc', 'categorical_accuracy', 'sparse_categorical_accuracy']), dtype=float),
        val_accuracy=np.array(get_history_series(history, ['val_accuracy', 'val_acc', 'val_categorical_accuracy', 'val_sparse_categorical_accuracy']), dtype=float),
        val_precision=np.array(val_precision or [], dtype=float),
        val_recall=np.array(val_recall or [], dtype=float),
        val_f1=np.array(val_f1 or [], dtype=float),
    )


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_epoch_curve(epochs, values, title, ylabel, path, label, color):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, values, marker='o', linestyle='-', color=color, label=label, linewidth=2, markersize=6)
    if len(values) > 1:
        ax.plot(epochs, smooth_curve(values, window=5), linestyle='--', color=color, alpha=0.75, label=f'{label} (smooth)')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=False)
    if len(epochs) <= 20:
        ax.set_xticks(epochs)
    save_figure(fig, path)


def plot_comparison_curve(epochs, train_values, val_values, title, ylabel, path, train_label, val_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_values, marker='o', linestyle='-', color='#1f77b4', label=train_label, linewidth=2, markersize=6)
    ax.plot(epochs, val_values, marker='o', linestyle='-', color='#ff7f0e', label=val_label, linewidth=2, markersize=6)
    if len(train_values) > 1:
        ax.plot(epochs, smooth_curve(train_values, window=5), linestyle='--', color='#1f77b4', alpha=0.75, label=f'{train_label} (smooth)')
    if len(val_values) > 1:
        ax.plot(epochs, smooth_curve(val_values, window=5), linestyle='--', color='#ff7f0e', alpha=0.75, label=f'{val_label} (smooth)')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=False)
    if len(epochs) <= 20:
        ax.set_xticks(epochs)
    save_figure(fig, path)


def plot_training_history(history, metrics_callback, results_dir):
    loss = get_history_series(history, ['loss'])
    val_loss = get_history_series(history, ['val_loss'])
    accuracy = get_history_series(history, ['accuracy', 'acc', 'categorical_accuracy', 'sparse_categorical_accuracy'])
    val_accuracy = get_history_series(history, ['val_accuracy', 'val_acc', 'val_categorical_accuracy', 'val_sparse_categorical_accuracy'])
    precision = metrics_callback.val_precision
    recall = metrics_callback.val_recall
    f1 = metrics_callback.val_f1
    epochs = np.arange(1, max(len(loss), len(accuracy), len(precision)) + 1)

    save_history_arrays(os.path.join(results_dir, 'training_history.npz'), history, precision, recall, f1)

    if len(accuracy) > 0:
        plot_epoch_curve(epochs[:len(accuracy)], accuracy, 'Training Accuracy', 'Accuracy', os.path.join(results_dir, 'training_accuracy.png'), 'Train Accuracy', '#1f77b4')
    if len(val_accuracy) > 0:
        plot_epoch_curve(epochs[:len(val_accuracy)], val_accuracy, 'Validation Accuracy', 'Accuracy', os.path.join(results_dir, 'validation_accuracy.png'), 'Validation Accuracy', '#ff7f0e')
    if len(loss) > 0:
        plot_epoch_curve(epochs[:len(loss)], loss, 'Training Loss', 'Loss', os.path.join(results_dir, 'training_loss.png'), 'Train Loss', '#2ca02c')
    if len(val_loss) > 0:
        plot_epoch_curve(epochs[:len(val_loss)], val_loss, 'Validation Loss', 'Loss', os.path.join(results_dir, 'validation_loss.png'), 'Validation Loss', '#d62728')
    if len(accuracy) > 0 and len(val_accuracy) > 0:
        plot_comparison_curve(epochs[:min(len(accuracy), len(val_accuracy))], accuracy[:len(val_accuracy)], val_accuracy, 'Training vs Validation Accuracy', 'Accuracy', os.path.join(results_dir, 'accuracy_comparison.png'), 'Train Accuracy', 'Validation Accuracy')
    if len(loss) > 0 and len(val_loss) > 0:
        plot_comparison_curve(epochs[:min(len(loss), len(val_loss))], loss[:len(val_loss)], val_loss, 'Training vs Validation Loss', 'Loss', os.path.join(results_dir, 'loss_comparison.png'), 'Train Loss', 'Validation Loss')
    if len(precision) > 0:
        plot_epoch_curve(epochs[:len(precision)], precision, 'Validation Precision', 'Precision', os.path.join(results_dir, 'precision_curve.png'), 'Precision', '#9467bd')
    if len(recall) > 0:
        plot_epoch_curve(epochs[:len(recall)], recall, 'Validation Recall', 'Recall', os.path.join(results_dir, 'recall_curve.png'), 'Recall', '#8c564b')
    if len(f1) > 0:
        plot_epoch_curve(epochs[:len(f1)], f1, 'Validation F1 Score', 'F1 Score', os.path.join(results_dir, 'f1_score_curve.png'), 'F1 Score', '#17becf')


def plot_confusion_matrix(cm, labels, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if labels is None or len(labels) != cm.shape[0]:
        labels = [str(i) for i in range(cm.shape[0])]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label',
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.max() != 0 else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > thresh else 'black'
        ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color=color)

    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def plot_confidence_boxplot(confidences, y_true, y_pred, path):
    correct = confidences[y_true == y_pred]
    incorrect = confidences[y_true != y_pred]
    plt.figure(figsize=(8, 6))
    plt.boxplot([correct, incorrect], labels=['Correct', 'Incorrect'], patch_artist=True,
                boxprops=dict(facecolor='#a6cee3'), medianprops=dict(color='black'))
    plt.title('Confidence Distribution for Correct and Incorrect Predictions')
    plt.ylabel('Max Prediction Confidence')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metric_summary(metrics, path):
    keys = list(metrics.keys())
    values = [metrics[k] for k in keys]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f'{height:.3f}', ha='center', va='bottom')
    plt.ylim(0, 1)
    plt.title('Summary Metrics: Accuracy, Precision, Recall, F1, mAP')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_classification_report_chart(y_true, y_pred, label_names, path):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_keys = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    precisions = [report[k]['precision'] for k in class_keys]
    recalls = [report[k]['recall'] for k in class_keys]
    f1s = [report[k]['f1-score'] for k in class_keys]
    labels = [str(k) for k in class_keys]

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(14, 7))
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1-score')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.title('Classification Report per Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_class_counts(y_true, labels, path):
    unique, counts = np.unique(y_true, return_counts=True)
    if labels is None or len(labels) < len(unique):
        labels = [str(i) for i in unique]
    else:
        labels = [labels[int(i)] if int(i) < len(labels) else str(i) for i in unique]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='#1f77b4')
    plt.title('Object / Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_threshold_metrics(y_true, y_pred, confidences, num_thresholds=101):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precision_vals = []
    recall_vals = []
    f1_vals = []

    for threshold in thresholds:
        mask = confidences >= threshold
        if np.sum(mask) == 0:
            precision_vals.append(np.nan)
            recall_vals.append(np.nan)
            f1_vals.append(np.nan)
            continue

        precision_vals.append(precision_score(y_true[mask], y_pred[mask], average='weighted', zero_division=0))
        recall_vals.append(recall_score(y_true[mask], y_pred[mask], average='weighted', zero_division=0))
        f1_vals.append(f1_score(y_true[mask], y_pred[mask], average='weighted', zero_division=0))

    return thresholds, precision_vals, recall_vals, f1_vals


def plot_threshold_curves(thresholds, precision_vals, recall_vals, f1_vals, base_path):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_vals, marker='o', markersize=3)
    plt.title('F1 Score vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path.format(metric='f1_vs_confidence'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_vals, marker='o', markersize=3, color='#2ca02c')
    plt.title('Precision vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path.format(metric='precision_vs_confidence'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recall_vals, marker='o', markersize=3, color='#d62728')
    plt.title('Recall vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path.format(metric='recall_vs_confidence'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, marker='o', markersize=3, color='#9467bd')
    plt.title('Precision vs Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path.format(metric='precision_vs_recall'))
    plt.close()


def plot_precision_recall_curve_multi(y_test, y_prob, path):
    y_test_flat = y_test.ravel()
    y_prob_flat = y_prob.ravel()
    precision, recall, _ = precision_recall_curve(y_test_flat, y_prob_flat)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='#1f77b4')
    plt.title('Precision-Recall Curve (micro-average)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_label_names():
    labels_path = os.path.join(PROJECT_ROOT, 'dataset', 'labels.txt')
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return None


def downsample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    return np.array(img)


def apply_data_augmentation(x_train, y_train):
    """
    Apply data augmentation to training data using ImageDataGenerator

    Augmentations applied:
    - Rotation: ±15 degrees
    - Width shift: 10%
    - Height shift: 10%
    - Horizontal flip
    - Zoom: 15%
    - Fill mode: nearest

    Returns:
        Augmented training data (x_train, y_train)
    """
    print("[INFO] Applying data augmentation to training data...")

    augmentor = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode='nearest',
        rescale=None,
    )

    augmented_images = []
    augmented_labels = []
    batches_per_image = 2

    for img, label in zip(x_train, y_train):
        augmented_images.append(img)
        augmented_labels.append(label)

        img_reshaped = img.reshape((1,) + img.shape)
        augmented_batch = augmentor.flow(img_reshaped, batch_size=1, shuffle=False)

        for _ in range(batches_per_image):
            augmented_img = next(augmented_batch)[0]
            augmented_images.append(augmented_img)
            augmented_labels.append(label)

    print(f"✓ Augmentation completed!")
    print(f"  Original training data: {len(x_train)} samples")
    print(f"  Augmented training data: {len(augmented_images)} samples")
    print(f"  Augmentation factor: {len(augmented_images) / len(x_train):.1f}x")

    return np.array(augmented_images), np.array(augmented_labels)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faceSamples.append(downsample_image(img_numpy))
            ids.append(id)
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
            continue

    return np.array(faceSamples), np.array(ids)


def create_train_val_test_splits(x, y, train_split=0.7, validation_split=0.15, test_split=0.15, random_state=RANDOM_STATE):
    if not 0 < train_split < 1 or not 0 < validation_split < 1 or not 0 < test_split < 1:
        raise ValueError('train_split, validation_split, dan test_split harus berada di antara 0 dan 1')

    if abs((train_split + validation_split + test_split) - 1.0) > 1e-9:
        raise ValueError('train_split + validation_split + test_split harus sama dengan 1')

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        train_size=train_split,
        random_state=random_state,
        stratify=y,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_split / (validation_split + test_split),
        random_state=random_state,
        stratify=y_temp,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def save_evaluation_summary(path, dataset_info, training_info, metrics, label_names=None):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Evaluation Summary\n')
        f.write('===================\n')
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write('\nDataset information:\n')
        for key, value in dataset_info.items():
            f.write(f"- {key}: {value}\n")

        f.write('\nTraining configuration:\n')
        for key, value in training_info.items():
            f.write(f"- {key}: {value}\n")

        if label_names:
            f.write('\nLabel names:\n')
            for idx, label in enumerate(label_names[:dataset_info.get('Number of classes', 0)]):
                f.write(f"  {idx}: {label}\n")

        f.write('\nEvaluation metrics:\n')
        for key, value in metrics.items():
            f.write(f"- {key}: {value:.6f}\n")


def main():
    print('\n [INFO] Training faces now.')
    faces, ids = getImagesAndLabels(DATASET_PATH)

    K.clear_session()
    n_faces = len(set(ids))
    classifier = model(INPUT_SHAPE, n_faces)

    faces = np.array([downsample_image(ab) for ab in faces])
    faces = faces[:, :, :, np.newaxis]
    print('Shape of Data: ' + str(faces.shape))
    print('Number of unique faces : ' + str(n_faces))

    faces = faces.astype('float32')
    faces /= 255.

    x_train, x_val, x_test, y_train, y_val, y_test = create_train_val_test_splits(
        faces,
        ids,
        train_split=0.7,
        validation_split=0.15,
        test_split=0.15,
        random_state=RANDOM_STATE,
    )

    original_train_count = len(x_train)
    x_train, y_train = apply_data_augmentation(x_train, y_train)
    augmented_train_count = len(x_train)

    y_train = to_categorical(y_train, num_classes=n_faces)
    y_val = to_categorical(y_val, num_classes=n_faces)
    y_test = to_categorical(y_test, num_classes=n_faces)

    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    # Ensure model directory exists so checkpoint can write weights
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:
        ensure_directory(model_dir)

    validation_callback = ValidationMetricsCallback(validation_data=(x_val, y_val))

    history = classifier.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[checkpoint, validation_callback],
    )

    save_model_summary(os.path.join(PROJECT_ROOT, 'evaluation_results', 'model_summary.txt'), classifier)

    # Load weights if available; otherwise continue (weights will be created by checkpoint)
    if os.path.exists(MODEL_PATH):
        try:
            classifier.load_weights(MODEL_PATH)
            print(f"Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: failed to load weights from {MODEL_PATH}: {e}")
    else:
        print(f"No existing weights found at {MODEL_PATH}; continuing without loading.")

    results_dir = os.path.join(PROJECT_ROOT, 'evaluation_results')
    ensure_directory(results_dir)

    plot_training_history(history, validation_callback, results_dir)
    print('Saved enhanced training curves to', results_dir)

    y_prob = classifier.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    map_score = average_precision_score(y_test, y_prob, average='macro')

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'mAP': map_score,
    }

    save_metrics_to_file(os.path.join(results_dir, 'evaluation_metrics.txt'), metrics)
    print(f"Saved evaluation metrics to {os.path.join(results_dir, 'evaluation_metrics.txt')}")

    class_report = classification_report(y_true, y_pred, zero_division=0)
    save_text_to_file(os.path.join(results_dir, 'classification_report.txt'), class_report)
    print(f"Saved classification report to {os.path.join(results_dir, 'classification_report.txt')}")

    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(results_dir, 'confusion_matrix.npy'), cm)
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    class_names = [str(i) for i in range(n_faces)]
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"Saved confusion matrix plot to {cm_path}")

    plot_classification_report_chart(y_true, y_pred, class_names, os.path.join(results_dir, 'classification_report_chart.png'))
    print(f"Saved classification report chart to {os.path.join(results_dir, 'classification_report_chart.png')}")

    plot_class_counts(y_true, class_names, os.path.join(results_dir, 'class_distribution.png'))
    print(f"Saved class distribution chart to {os.path.join(results_dir, 'class_distribution.png')}")

    plot_metric_summary(metrics, os.path.join(results_dir, 'metric_summary.png'))
    print(f"Saved metric summary chart to {os.path.join(results_dir, 'metric_summary.png')}")

    confidences = np.max(y_prob, axis=1)
    threshold_path_template = os.path.join(results_dir, '{metric}.png')
    thresholds, precision_vals, recall_vals, f1_vals = compute_threshold_metrics(y_true, y_pred, confidences)
    plot_threshold_curves(thresholds, precision_vals, recall_vals, f1_vals, threshold_path_template)
    print(f"Saved threshold curves to {results_dir}")

    plot_precision_recall_curve_multi(y_test, y_prob, os.path.join(results_dir, 'precision_recall_curve.png'))
    print(f"Saved precision-recall curve to {os.path.join(results_dir, 'precision_recall_curve.png')}")

    summary_path = os.path.join(results_dir, 'evaluation_summary.txt')
    dataset_info = {
        'Number of classes': n_faces,
        'Train samples (after augmentation)': augmented_train_count,
        'Train samples (original)': original_train_count,
        'Validation samples': len(x_val),
        'Test samples': len(x_test),
    }
    training_info = {
        'Image size': IMAGE_SIZE,
        'Batch size': BATCH_SIZE,
        'Epochs': EPOCHS,
        'Learning rate': LEARNING_RATE,
        'Validation split': VALIDATION_SPLIT,
        'Test split': TEST_SPLIT,
        'Augmentation factor': f'{augmented_train_count / original_train_count:.2f}',
    }
    save_evaluation_summary(summary_path, dataset_info, training_info, metrics, class_names)
    print(f"Saved evaluation summary to {summary_path}")

    print('\n=== Evaluation summary ===')
    print(f'Accuracy: {accuracy:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'F1 Score: {f1:.6f}')
    print(f'mAP: {map_score:.6f}')
    print('\nClassification report:')
    print(class_report)


if __name__ == '__main__':
    main()
