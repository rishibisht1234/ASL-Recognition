
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

# Model paths
CLASSIFIER_PATH = MODEL_DIR / "asl_mlp_baseline.keras"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
HAND_LANDMARKER_PATH = BASE_DIR / "hand_landmarker.task"

# Webcam settings
CAMERA_INDEX = 0

# MediaPipe settings
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Prediction settings
CONFIDENCE_THRESHOLD = 60.0
PREDICTION_HISTORY_SIZE = 7
