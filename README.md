# 🌫️ Smog Classification Pipeline

A modular, extensible desktop GUI application for smog detection using a trained ResNet18 CNN.

## Features

- **Professional Desktop UI** - PyQt6-based graphical interface
- **Simple Image Upload** - Select and preview images before classification
- **Real-time Predictions** - Fast inference with confidence scores
- **Probability Visualization** - Bar charts showing Clear vs Smog probabilities
- **Extensible Pipeline** - Add new stages for image enhancement, segmentation, etc.
- **PyTorch-based** - Uses your trained ResNet18 model
- **Multi-threaded** - Non-blocking UI during model inference

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your model file exists at the configured path (default: `smog-classification/smog_classifier.pth`)

## Running the App

```bash
python app.py
```

The desktop GUI will launch as a standalone window.

## Project Structure

```
dip-smog/
├── app.py              # PyQt6 desktop GUI
├── inference.py        # Model loading and inference
├── pipeline.py         # Pipeline architecture (extensible)
├── stages.py           # Individual pipeline stages
├── requirements.txt    # Dependencies
├── examples.py         # Example usage patterns
└── smog-classification/
    └── smog_classifier.pth  # Trained model
```

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

### Main Window
- **Left Panel**: Image upload, configuration, classify button
- **Right Panel**: Tabbed results view (Summary & Details)

### Summary Tab
- Large classification result (Clear/Smog)
- Confidence percentage
- Progress bar visualization
- Bar chart showing probability distribution

### Details Tab
- Full classification report
- All numerical values (logit, probabilities)
- Image and model information
- Useful for debugging and analysis

## Extending the Pipeline

To add a new pipeline stage (e.g., image enhancement, segmentation):

### Example: Add an Image Enhancement Stage

```python
# In stages.py, add:

class ImageEnhancementStage(PipelineStage):
    """Stage for image enhancement."""
    
    def __init__(self):
        super().__init__("Image Enhancement")
    
    def process(self, input_data: dict) -> PipelineResult:
        """
        Enhance image quality.
        
        Input: {"image_input": <PIL Image>, "classification": {...}}
        Output: {"enhanced_image": <PIL Image>, ...}
        """
        try:
            image = input_data.get("image_input")
            
            # Your enhancement logic here
            enhanced = self._enhance(image)
            
            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "enhanced_image": enhanced
                }
            )
        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e)
            )
    
    def _enhance(self, image):
        # Enhancement logic
        pass
```

### Then register the stage:

```python
# In stages.py, update create_default_pipeline():

def create_default_pipeline(model_path: str):
    from pipeline import SmogClassificationPipeline
    
    pipeline = SmogClassificationPipeline()
    pipeline.add_stage(SmogClassificationStage(model_path))
    pipeline.add_stage(ImageEnhancementStage())  # ← New stage
    
    return pipeline
```

## Model Details

- **Architecture**: ResNet18 (pretrained on ImageNet)
- **Input Size**: 224×224 pixels
- **Output**: Binary classification (1 logit → sigmoid)
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Classes**: 
  - 0: Clear
  - 1: Smog

## Usage Example

```python
from inference import SmogClassifier

classifier = SmogClassifier("smog-classification/smog_classifier.pth")

result = classifier.predict("path/to/image.jpg")
print(result)
# Output:
# {
#     "class": "Smog",
#     "confidence": 0.95,
#     "logit": 2.944,
#     "probability_clear": 0.05,
#     "probability_smog": 0.95
# }
```

## Pipeline Usage Example

```python
from stages import create_default_pipeline

pipeline = create_default_pipeline("smog-classification/smog_classifier.pth")
result = pipeline.run("path/to/image.jpg")

print(result)
# Output:
# {
#     "success": True,
#     "stages": [...],
#     "final_data": {...}
# }
```

## Troubleshooting

**Model not found**: Use the "Browse..." button in the app to locate your model file

**Image not loading**: Check that the image format is supported (JPG, PNG, BMP)

**CUDA errors**: The app will automatically fall back to CPU if CUDA is unavailable

**Slow inference**: First run loads the model; subsequent runs are faster. GPU (if available) speeds up inference significantly

## Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Drag-and-drop image upload
- [ ] Image segmentation visualization
- [ ] Confidence threshold alerts and filtering
- [ ] Prediction history and statistics dashboard
- [ ] Export results to CSV/JSON
- [ ] Support for video frame processing
- [ ] Webcam live feed classification
- [ ] Model performance metrics comparison
- [ ] Customizable color scheme and dark mode
