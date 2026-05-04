import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class SmogClassifier:
    """Smog detection classifier using ResNet18."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Load the smog classifier model.
        
        Args:
            model_path: Path to the .pth file
            device: 'cuda' or 'cpu' (auto-detects if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        self.class_names = ["Clear", "Smog"]
        
    def _load_model(self, model_path: str):
        """Load ResNet18 model from checkpoint (Sequential head, Sigmoid, weights=None)."""
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        # Load checkpoint
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Get preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path: str = None, image_array: np.ndarray = None) -> dict:
        """
        Classify an image.
        
        Args:
            image_path: Path to image file (or pass image_array instead)
            image_array: PIL Image or numpy array (or pass image_path instead)
            
        Returns:
            dict with keys:
                - class: "Clear" or "Smog"
                - confidence: float [0-1]
                - logit: raw model output
                - probabilities: dict with class probabilities
        """
        # Load image
        if image_path:
            image = Image.open(image_path).convert("RGB")
        elif image_array is not None:
            if isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array.astype('uint8')).convert("RGB")
            else:
                image = image_array.convert("RGB")
        else:
            raise ValueError("Provide either image_path or image_array")
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.inference_mode():
            prob = self.model(image_tensor).item()  # Already sigmoid

        predicted_class = "Smog" if prob > 0.5 else "Clear"
        confidence = prob if prob > 0.5 else 1.0 - prob

        return {
            "class": predicted_class,
            "confidence": float(confidence),
            "logit": None,
            "probability_clear": float(1.0 - prob),
            "probability_smog": float(prob),
            # Add your model's evaluation metrics here:
            "precision": 0.94, # Replace with your actual model's precision
            "recall": 0.91,    # Replace with your actual model's recall
            "f1_score": 0.92   # Replace with your actual model's F1 score
        }
