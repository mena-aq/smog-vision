# 🌫️ Smog Classification Pipeline

A modular, extensible desktop GUI application for smog detection and object detection using a trained CNN and multiple dehazing, and object detection techniques

## Pipeline

1. Select an image or video file for processing
2. Choose a dehazing method (DCP, CLAHE)
3. Choose an object detection method (HOG+SVM, YOLO, RCNN)
4. Run the pipeline to perform smog classification and object detection
5. View results in the Summary and Details tabs, with visualizations and metrics
6. Optionally view segmentation masks for detected objects


## Core Techniques

### Smog Classification
- **ResNet18**: Binary smog classification model trained on dataset ([SmogDetection](https://github.com/poojavinod100/SmogDetection)). Find training notebook in `train/smog-classify.ipynb`.

### Dehazing Methods
- **Dark Channel Prior (DCP)**: Traditional dehazing technique based on dark channel estimation
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for enhancing local contrast

### Object Detection Methods
- **Haar Cascade (Vehicle) and HOG+SVM (Pedestrian)**: Classical AdaBoost vehicle and pedestrian detectors
- **YOLOv8m**: Real-time multi-class object detection
- **R-CNN**: Instance-level region-based CNN for per-object segmentation

### Segmentation
- **Segmentation Masks**: Pixel-level semantic segmentation for object detction using morphological operations and contour detection 

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your model file exists at the configured path (default: `smog-classification/smog_classifier.pth`)

3. Setup haar
Download the Haar Cascade XML file from OpenCV's GitHub repository and save it as `haarcascade_car.xml` in the project root:

```
https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml#L18
```

Rename the downloaded file to `haarcascade_car.xml` and place it in this directory.

## Running the App

```bash
python app.py
```

The desktop GUI will launch as a standalone window.


## UI Overview

- **Left Panel**: Main processing interface with controls and results display
  - Upload Image/Video button for file selection
  - Dehazing Method selector (DCP, CLAHE)
  - Object Detection Method selector (HOG+SVM, YOLO, RCNN)
  - Process button to run the pipeline
  - Pipeline Output group: Original → Dehazed → Final Result (Dehazed)
  - Comparison group: Original → Final Result (Original)
  - View Masks button for segmentation visualization

- **Right Panel**: Model metrics and pipeline results
  - Classification results and confidence scores
  - Detection information
  - Processing statistics
  - Scrollable metrics display

<br>
<img width="2880" height="1704" alt="Screenshot 2026-05-24 131655" src="https://github.com/user-attachments/assets/0ca0bfac-1617-4dd9-ba6d-cef2e03a2e90" />


