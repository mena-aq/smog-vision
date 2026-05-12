# 🌫️ Smog Classification Pipeline

A modular, extensible desktop GUI application for smog detection and object detection using a trained CNN and multiple dehazing, and object detection dechniques

## Features

- **Professional Desktop UI** - PyQt6-based graphical interface
- **Simple Image Upload** - Select and preview images before classification
- **Real-time Predictions** - Fast inference with confidence scores
- **Extensible Pipeline** - Add new stages for image enhancement, segmentation, etc.
- **PyTorch-based** - Uses your trained ResNet18 model
- **Multi-threaded** - Non-blocking UI during model inference

## Core Techniques

- **ResNet18**: Binary smog classification with residual connections
- **YOLOv8m**: Real-time multi-class object detection
- **Haar Cascade**: Classical AdaBoost vehicle detector
- **Segmentation Masks**: Pixel-level semantic segmentation for object detction
- **R-CNN**: Instance-level region-based CNN for per-object segmentation
- **HOG + SVM**: Classical feature-based classification

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

## How It Works

### 1. **Desktop Interface** (`app.py`)
- Professional PyQt6 GUI with tabbed interface
- Image upload with drag-and-drop preview
- Model path configuration dialog
- Results displayed in Summary and Details tabs
- Real-time probability visualization with bar charts
- Multi-threaded inference to keep UI responsive
### 2. **Inference Module** (`inference.py`)
- Loads the ResNet18 model from checkpoint
- Handles image preprocessing (resize, normalize)
- Performs binary classification: Clear vs Smog
- Returns class, confidence, and probabilities

### 3. **Pipeline Architecture** (`pipeline.py`)
- Base `PipelineStage` class for creating new stages
- `SmogClassificationPipeline` manages stage execution
- Stages process data sequentially, passing output to next stage
- Graceful error handling

### 4. **Classification Stage** (`stages.py`)
- Implements `PipelineStage` for smog classification
- Wraps the inference module
- Can be extended with additional stages


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

## System Architecture

Multi-stage pipeline with parallel processing. ResNet18, YOLOv8, Haar Cascade, and HOG+SVM operate independently. Detection outputs guide region analysis; segmentation masks provide spatial context.