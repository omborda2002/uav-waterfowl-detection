"""
Configuration file for UAV Waterfowl Detection Project
Contains all paths, hyperparameters, and settings
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
# Base project directory (adjust if needed)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "uav_raw"

# Raw data paths
THERMAL_DATASET_PATH = DATA_ROOT / "00_UAV-derived Thermal Waterfowl Dataset" / \
                       "00_UAV-derived Waterfowl Thermal Imagery Dataset" / \
                       "01_Thermal Images and Ground Truth (used for detector training and testing)"

POSITIVE_IMAGES_PATH = THERMAL_DATASET_PATH / "01_Posistive Image"
NEGATIVE_IMAGES_PATH = THERMAL_DATASET_PATH / "03_Negative Images"
ANNOTATIONS_PATH = THERMAL_DATASET_PATH / "02_Groundtruth Label for Positive Images" / "Bounding Box Label.csv"

RGB_IMAGES_PATH = DATA_ROOT / "01_RGB Images " / \
                  "01_RGB Images (used as visual reference for ground truth labeling)"

# Processed data paths
PROCESSED_DATA_ROOT = PROJECT_ROOT / "processed_data"
YOLO_DATASET_PATH = PROCESSED_DATA_ROOT / "yolo_format"
YOLO_IMAGES_PATH = YOLO_DATASET_PATH / "images"
YOLO_LABELS_PATH = YOLO_DATASET_PATH / "labels"

# Output paths
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
WEIGHTS_PATH = OUTPUTS_ROOT / "weights"
RESULTS_PATH = OUTPUTS_ROOT / "results"
VISUALIZATIONS_PATH = OUTPUTS_ROOT / "visualizations"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Dataset splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Class configuration
CLASS_NAMES = ['waterfowl']
NUM_CLASSES = len(CLASS_NAMES)

# Image properties (from dataset analysis)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512
IMAGE_CHANNELS = 1  # Grayscale thermal images

# ============================================================================
# YOLO CONFIGURATION
# ============================================================================
# YOLO model settings
YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
YOLO_IMG_SIZE = 640  # YOLO input size

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 16  # Adjust based on GPU memory
LEARNING_RATE = 0.001
PATIENCE = 20  # Early stopping patience
OPTIMIZER = "AdamW"

# Augmentation parameters
AUGMENTATION = {
    'hsv_h': 0.015,      # Hue augmentation (minimal for thermal)
    'hsv_s': 0.3,        # Saturation
    'hsv_v': 0.4,        # Value/brightness
    'degrees': 10,       # Rotation degrees
    'translate': 0.1,    # Translation
    'scale': 0.2,        # Scaling
    'shear': 0.0,        # Shear
    'perspective': 0.0,  # Perspective
    'flipud': 0.0,       # No vertical flip
    'fliplr': 0.5,       # Horizontal flip probability
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.0,        # Mixup augmentation
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# IoU thresholds
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25  # Confidence threshold for predictions

# Metrics to compute
METRICS = ['mAP50', 'mAP50-95', 'precision', 'recall', 'F1']

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
# Colors for visualization (BGR format for OpenCV)
BBOX_COLOR = (0, 255, 0)  # Green for predictions
GT_COLOR = (255, 0, 0)     # Blue for ground truth
FP_COLOR = (0, 0, 255)     # Red for false positives
FN_COLOR = (255, 255, 0)   # Cyan for false negatives

# Visualization settings
BBOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# Number of examples to visualize
NUM_VISUALIZATION_SAMPLES = 6  # 2-3 per category (TP, FP, FN)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        PROCESSED_DATA_ROOT,
        YOLO_DATASET_PATH,
        YOLO_IMAGES_PATH / "train",
        YOLO_IMAGES_PATH / "val",
        YOLO_IMAGES_PATH / "test",
        YOLO_LABELS_PATH / "train",
        YOLO_LABELS_PATH / "val",
        YOLO_LABELS_PATH / "test",
        OUTPUTS_ROOT,
        WEIGHTS_PATH,
        RESULTS_PATH,
        VISUALIZATIONS_PATH,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✓ All directories created successfully")

def get_yolo_data_yaml_path():
    """Get path for YOLO data.yaml configuration file"""
    return YOLO_DATASET_PATH / "data.yaml"

def print_config():
    """Print current configuration"""
    print("="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"\nDataset Splits:")
    print(f"  Train: {TRAIN_SPLIT*100}%")
    print(f"  Val: {VAL_SPLIT*100}%")
    print(f"  Test: {TEST_SPLIT*100}%")
    print(f"\nModel Configuration:")
    print(f"  Model: {YOLO_MODEL}")
    print(f"  Input Size: {YOLO_IMG_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"\nClasses: {CLASS_NAMES}")
    print("="*80)

if __name__ == "__main__":
    print_config()
    create_directories()