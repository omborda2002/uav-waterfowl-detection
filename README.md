# UAV Waterfowl Detection

**Automated waterfowl detection system using YOLOv8 on thermal UAV imagery**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)

## Overview

Real-time object detection system for identifying waterfowl in thermal imagery captured by UAVs (Unmanned Aerial Vehicles). This project implements a complete deep learning pipeline from data preprocessing to model evaluation, achieving **86.44% mAP@0.5** on thermal imagery.

## Key Features

- **High-Performance Detection**: YOLOv8-based architecture optimized for thermal imagery
- **Complete ML Pipeline**: End-to-end workflow including data preparation, training, and evaluation
- **Thermal Image Processing**: Custom preprocessing for grayscale thermal UAV imagery
- **Automated Evaluation**: Comprehensive metrics and visualization tools
- **Production-Ready Code**: Modular, well-documented, and reproducible

## Results

### Model Performance

| Metric | Score |
|--------|-------|
| **mAP@0.5** | **86.44%** |
| **mAP@0.5:0.95** | 51.78% |
| **Precision** | 93.21% |
| **Recall** | 77.82% |
| **F1 Score** | 84.82% |

### Detection Analysis

- **True Positives**: 1,172 detections
- **False Positives**: 82 (high precision)
- **False Negatives**: 239 (room for recall improvement)
- **Test Set**: 83 images with 1,411 ground truth boxes

### Key Insights

- **High Precision (93.21%)**: Model minimizes false alarms, crucial for wildlife monitoring
- **Strong Overall Performance (86.44% mAP@0.5)**: Exceeds typical detection thresholds
- **Optimized for Thermal Data**: Custom preprocessing and augmentation for UAV thermal imagery

## Tech Stack

- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV, PIL
- **Data Science**: NumPy, Pandas, Matplotlib
- **Development**: Jupyter Notebooks, Python 3.8+

## Project Structure

```
uav-waterfowl-detection/
├── data/                      # Data preprocessing scripts
│   ├── prepare_dataset.py     # Dataset preparation and splitting
│   ├── fix_grayscale.py       # Thermal image preprocessing
│   └── config.py              # Configuration and hyperparameters
├── models/                    # Training and evaluation
│   ├── train.py               # YOLOv8 training pipeline
│   └── evaluate.py            # Model evaluation and metrics
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── outputs/                   # Results and visualizations
│   ├── results/               # Evaluation reports and metrics
│   ├── visualizations/        # Detection visualizations
│   └── weights/               # Model checkpoints
└── best_model.pt             # Best trained model weights
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/uav-waterfowl-detection.git
cd uav-waterfowl-detection
```

### 2. Install Dependencies
```bash
pip install ultralytics torch torchvision opencv-python pandas numpy matplotlib
```

### 3. Run Inference (using pre-trained model)
```python
from ultralytics import YOLO

# Load model
model = YOLO('best_model.pt')

# Run inference
results = model.predict('path/to/thermal/image.tif', conf=0.25)
```

### 4. Train from Scratch
```bash
python models/train.py
```

### 5. Evaluate Model
```bash
python models/evaluate.py
```

## Dataset

This project uses the **UAV-derived Thermal Waterfowl Dataset**, containing:
- Thermal imagery captured from UAV platforms
- Ground truth bounding box annotations
- Train/Val/Test splits (70%/15%/15%)

## Model Architecture

- **Base Model**: YOLOv8-Nano (yolov8n)
- **Input Size**: 640x640 (resized from 640x512 thermal images)
- **Training**:
  - Optimizer: AdamW
  - Learning Rate: 0.001
  - Batch Size: 16
  - Epochs: 100 (with early stopping)
  - Augmentation: Custom thermal-optimized settings

## Use Cases

- **Wildlife Conservation**: Monitor waterfowl populations in wetlands
- **Ecological Research**: Track migration patterns and habitat usage
- **Agricultural Management**: Assess waterfowl impact on crops
- **Automated Surveillance**: Real-time detection for wildlife management

## Future Improvements

- [ ] Implement real-time video inference pipeline
- [ ] Add multi-class support for different bird species
- [ ] Optimize model for edge deployment (ONNX/TensorRT)
- [ ] Expand dataset with additional thermal imagery
- [ ] Integrate with GIS mapping systems

## License

This project is available for educational and research purposes.

## Author

**Your Name**
[LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourportfolio.com) | [Email](mailto:your.email@example.com)

---

*Developed as part of Computer Vision coursework - showcasing end-to-end ML project development skills*
