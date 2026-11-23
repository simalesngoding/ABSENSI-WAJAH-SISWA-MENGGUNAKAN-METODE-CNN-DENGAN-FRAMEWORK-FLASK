"""Configuration settings for the Face Recognition system."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Collection Settings
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
NUM_SAMPLES = 300
FACE_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Model Settings
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.h5')
IMAGE_SIZE = (64, 64)
INPUT_SHAPE = (*IMAGE_SIZE, 1)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Face Detection Settings
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

# Training Settings
RANDOM_STATE = 0
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Recognition Settings
CONFIDENCE_THRESHOLD = 0.6

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'