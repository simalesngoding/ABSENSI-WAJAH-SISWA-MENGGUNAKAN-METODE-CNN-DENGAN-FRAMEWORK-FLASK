import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EVAL_DIR = os.path.join(PROJECT_ROOT, 'evaluation_results')
HISTORY_FILE = os.path.join(EVAL_DIR, 'training_history.npz')
METRICS_FILE = os.path.join(EVAL_DIR, 'evaluation_metrics.txt')
CM_FILE = os.path.join(EVAL_DIR, 'confusion_matrix.npy')


def smooth_curve(values, window=5):
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2 or window <= 1:
        return values
    window = min(window, n)
    weights = np.ones(window, dtype=float) / window
    smooth = np.convolve(values, weights, mode='same')
    half = window // 2
    if half > 0:
        smooth[:half] = np.cumsum(values[:half]) / np.arange(1, half + 1)
        smooth[-half:] = np.cumsum(values[-half:][::-1])[::-1] / np.arange(half, 0, -1)
    return smooth


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_epoch_curve(epochs, values, title, ylabel, path, label, color):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, values, marker='o', linestyle='-', color=color, label=label, linewidth=2, markersize=6)
    if len(values) > 1:
        ax.plot(epochs, smooth_curve(values, window=5), linestyle='--', color=color, alpha=0.75, label=f'{label} (smooth)')
    ax.set_title(title, fontsize=14, fontweight='semibold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False, fontsize=11)
    if len(epochs) <= 20:
        ax.set_xticks(epochs)
    save_figure(fig, path)


def plot_comparison_curve(epochs, train_values, val_values, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_values, marker='o', linestyle='-', color='#1f77b4', label='Train', linewidth=2, markersize=6)
    ax.plot(epochs, val_values, marker='o', linestyle='-', color='#ff7f0e', label='Validation', linewidth=2, markersize=6)
    if len(train_values) > 1:
        ax.plot(epochs, smooth_curve(train_values, window=5), linestyle='--', color='#1f77b4', alpha=0.75, label='Train (smooth)')
    if len(val_values) > 1:
        ax.plot(epochs, smooth_curve(val_values, window=5), linestyle='--', color='#ff7f0e', alpha=0.75, label='Validation (smooth)')
    ax.set_title(title, fontsize=14, fontweight='semibold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False, fontsize=11)
    if len(epochs) <= 20:
        ax.set_xticks(epochs)
    save_figure(fig, path)


def plot_confusion_matrix(cm, path, class_names=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.max() != 0 else 0.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(int(cm[i, j]), 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    save_figure(fig, path)


def read_metrics(metrics_path):
    metrics = {}
    if not os.path.exists(metrics_path):
        return metrics
    with open(metrics_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                metrics[key.strip()] = value.strip()
    return metrics


def load_history(history_path):
    if not os.path.exists(history_path):
        return {}
    data = np.load(history_path)
    history = {}
    for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']:
        if key in data:
            history[key] = data[key].tolist()
    return history


def plot_metrics_from_history(history, results_dir):
    if not history:
        print(f'History file not found or empty: {HISTORY_FILE}')
        return

    if 'loss' in history:
        plot_epoch_curve(list(range(1, len(history['loss']) + 1)), history['loss'], 'Training Loss', 'Loss', os.path.join(results_dir, 'training_loss.png'), 'Train Loss', '#2ca02c')
    if 'val_loss' in history:
        plot_epoch_curve(list(range(1, len(history['val_loss']) + 1)), history['val_loss'], 'Validation Loss', 'Loss', os.path.join(results_dir, 'validation_loss.png'), 'Validation Loss', '#d62728')
    if 'accuracy' in history:
        plot_epoch_curve(list(range(1, len(history['accuracy']) + 1)), history['accuracy'], 'Training Accuracy', 'Accuracy', os.path.join(results_dir, 'training_accuracy.png'), 'Train Accuracy', '#1f77b4')
    if 'val_accuracy' in history:
        plot_epoch_curve(list(range(1, len(history['val_accuracy']) + 1)), history['val_accuracy'], 'Validation Accuracy', 'Accuracy', os.path.join(results_dir, 'validation_accuracy.png'), 'Validation Accuracy', '#ff7f0e')
    if 'accuracy' in history and 'val_accuracy' in history:
        epochs = list(range(1, min(len(history['accuracy']), len(history['val_accuracy'])) + 1))
        train_values = history['accuracy'][:len(epochs)]
        val_values = history['val_accuracy'][:len(epochs)]
        plot_comparison_curve(epochs, train_values, val_values, 'Training vs Validation Accuracy', 'Accuracy', os.path.join(results_dir, 'accuracy_comparison.png'))
    if 'loss' in history and 'val_loss' in history:
        epochs = list(range(1, min(len(history['loss']), len(history['val_loss'])) + 1))
        train_values = history['loss'][:len(epochs)]
        val_values = history['val_loss'][:len(epochs)]
        plot_comparison_curve(epochs, train_values, val_values, 'Training vs Validation Loss', 'Loss', os.path.join(results_dir, 'loss_comparison.png'))
    if 'val_precision' in history:
        plot_epoch_curve(list(range(1, len(history['val_precision']) + 1)), history['val_precision'], 'Validation Precision', 'Precision', os.path.join(results_dir, 'precision_curve.png'), 'Precision', '#9467bd')
    if 'val_recall' in history:
        plot_epoch_curve(list(range(1, len(history['val_recall']) + 1)), history['val_recall'], 'Validation Recall', 'Recall', os.path.join(results_dir, 'recall_curve.png'), 'Recall', '#8c564b')
    if 'val_f1' in history:
        plot_epoch_curve(list(range(1, len(history['val_f1']) + 1)), history['val_f1'], 'Validation F1 Score', 'F1 Score', os.path.join(results_dir, 'f1_score_curve.png'), 'F1 Score', '#17becf')


def plot_metrics_summary(metrics, results_dir):
    if not metrics:
        return
    keys = []
    values = []
    for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        if metric_name in metrics:
            keys.append(metric_name)
            values.append(metrics[metric_name])
    if not keys:
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(keys, values, color=colors[:len(keys)], edgecolor='black', linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_title('Evaluation Metrics Summary', fontsize=14, fontweight='semibold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    save_figure(fig, os.path.join(results_dir, 'metrics_summary.png'))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(EVAL_DIR)
    history = load_history(HISTORY_FILE)
    metrics = read_metrics(METRICS_FILE)

    plot_metrics_from_history(history, EVAL_DIR)
    plot_metrics_summary(metrics, EVAL_DIR)

    if os.path.exists(CM_FILE):
        cm = np.load(CM_FILE)
        plot_confusion_matrix(cm, os.path.join(EVAL_DIR, 'confusion_matrix.png'))
    else:
        print(f'No confusion matrix file found: {CM_FILE}')

    print('Saved professional training graphs to', EVAL_DIR)


if __name__ == '__main__':
    main()
