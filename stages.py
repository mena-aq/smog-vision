import os
from pathlib import Path
from PIL import Image
import cv2
import time
import numpy as np
from inference import SmogClassifier
from pipeline import PipelineStage, PipelineResult
from pipeline import SmogClassificationPipeline


class SmogClassificationStage(PipelineStage):
    """Pipeline stage for smog classification."""
    
    def __init__(self, model_path: str):
        super().__init__("Smog Classification")
        self.classifier = SmogClassifier(model_path)
    
    def process(self, input_data: dict) -> PipelineResult:
        """
        Classify image for smog content.
        
        Input: {"image_input": <path or PIL Image>}
        Output: {"classification": <dict>, "image_input": ...}
        """
        try:
            image_input = input_data.get("image_input")
            
            # Load image if path
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image not found: {image_input}")
                image_path = image_input
            else:
                image_path = None
            
            # Measure classification time
            start = time.time()
            
            # Run classification
            result = self.classifier.predict(
                image_path=image_path,
                image_array=image_input if image_path is None else None
            )
            
            elapsed = time.time() - start
            
            # Add processing time to result
            result["processing_time_ms"] = round(elapsed * 1000, 1)
            
            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    "classification": result,
                    "image_input": image_input,
                }
            )
            
        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e)
            )


class DehazingStage(PipelineStage):
    """Dehazing stage supporting DCP and CLAHE methods."""
    
    def __init__(self, method: str = "dcp", patch_size: int = 15, omega: float = 0.95, t_min: float = 0.1, clahe_clip: float = 3.0, clahe_tile: int = 8):
        """
        Args:
            method:     Dehazing algorithm: "dcp" (Dark Channel Prior) or "clahe" (CLAHE-based)
            patch_size: Size of the local patch for dark channel computation (DCP only, default 15).
            omega:      Haze retention factor — 0.95 keeps a tiny bit of haze for realism (DCP only).
            t_min:      Minimum transmission value to avoid division by zero (DCP only, default 0.1).
            clahe_clip: Contrast limit for CLAHE (CLAHE only, default 3.0).
            clahe_tile: Tile grid size for CLAHE (CLAHE only, default 8 for 8x8 grid).
        """
        super().__init__("DCP Dehazing" if method == "dcp" else "CLAHE Dehazing")
        self.method = method.lower()
        self.patch_size = patch_size
        self.omega = omega
        self.t_min = t_min
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        
        if self.method not in ("dcp", "clahe"):
            raise ValueError(f"Unknown dehazing method: {method}. Use 'dcp' or 'clahe'.")
    def process(self, input_data: dict) -> PipelineResult:
        """
        Input:  {"image_input": <str path | PIL Image | np.ndarray>, ...}
        Output: {"dehazed_image": <PIL Image>, ...everything from input...}
        """
        try:
            raw = input_data.get("image_input")
            img_np = self._to_numpy(raw)

            start = time.time()

            if self.method == "dcp":
                dehazed_np = self._dehaze_dcp(img_np)
                metrics_key = "dcp_metrics"
                metrics = {
                    "method": "DCP (Dark Channel Prior)",
                    "transmission_map": float(np.mean(self._transmission(img_np, self._atmospheric_light(img_np, self._dark_channel(img_np))))),
                    "processing_time_ms": round((time.time() - start) * 1000, 1),
                }
            else:  # clahe
                dehazed_np = self._dehaze_clahe(img_np)
                metrics_key = "clahe_metrics"
                metrics = {
                    "method": "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
                    "processing_time_ms": round((time.time() - start) * 1000, 1),
                }

            elapsed = time.time() - start

            dehazed_pil = Image.fromarray(
                (dehazed_np * 255).clip(0, 255).astype(np.uint8)
            )

            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={**input_data, "dehazed_image": dehazed_pil, metrics_key: metrics},
            )

        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e)
            )

    def _dehaze_dcp(self, img: np.ndarray) -> np.ndarray:
        """Full DCP pipeline on a float32 RGB image in [0, 1]."""
        dark          = self._dark_channel(img)
        A             = self._atmospheric_light(img, dark)
        t_raw         = self._transmission(img, A)
        t_refined     = self._guided_filter(img, t_raw)
        recovered     = self._recover(img, t_refined, A)
        return recovered
    
    def _dehaze_clahe(self, img: np.ndarray) -> np.ndarray:
        """Simple CLAHE-based dehazing for hazy images."""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Convert to LAB (decouple luminance from color)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only (preserves color)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        result_uint8 = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return result_uint8.astype(np.float32) / 255.0

    def _dark_channel(self, img: np.ndarray) -> np.ndarray:
        """
        Compute the dark channel of the image.
        Dark channel = min over a local patch of the min over RGB channels.
        """
        # Min across colour channels → shape (H, W)
        min_channel = np.min(img, axis=2)

        # Min over a local patch using erosion (fast morphological op)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark = cv2.erode(min_channel, kernel)
        return dark  # float32, shape (H, W)

    def _atmospheric_light(self, img: np.ndarray, dark: np.ndarray) -> np.ndarray:
        """
        Estimate atmospheric light A from the top 0.1% brightest pixels
        in the dark channel (those are most likely haze pixels).
        Returns shape (3,) in [0, 1].
        """
        h, w     = dark.shape
        n_pixels = h * w
        n_search = max(int(n_pixels * 0.001), 1)

        # Flatten and find indices of the brightest dark-channel pixels
        flat_dark    = dark.flatten()
        flat_indices = np.argpartition(flat_dark, -n_search)[-n_search:]

        # Among those pixels, pick the one with the highest intensity in the
        # original image to estimate the atmospheric light
        flat_img = img.reshape(-1, 3)
        A = flat_img[flat_indices].max(axis=0)  # shape (3,)
        return A.clip(1e-6, 1.0)

    def _transmission(self, img: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Estimate the raw (unrefined) transmission map.
        t(x) = 1 - omega * dark_channel(img / A)
        """
        # Normalise image by atmospheric light
        img_norm = img / A[np.newaxis, np.newaxis, :]   # broadcast over H, W
        img_norm = img_norm.clip(0, 1)

        dark_norm = self._dark_channel(img_norm)
        t = 1.0 - self.omega * dark_norm
        return t.clip(self.t_min, 1.0)

    def _guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int = 15,
        eps: float = 1e-2,
    ) -> np.ndarray:
        """
        Soft-matting / edge-preserving refinement of the transmission map
        using a guided filter with the greyscale image as guide.

        Args:
            guide:  RGB image (H, W, 3) used as the guidance signal.
            src:    Transmission map (H, W) to refine.
            radius: Filter radius (larger → smoother, but preserves edges better
                    than a plain blur).
            eps:    Regularisation — prevents over-smoothing near edges.
        """
        # Convert guide to greyscale
        guide_gray = cv2.cvtColor(
            (guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0

        p = src.astype(np.float32)
        I = guide_gray

        # Box-filter means
        mean_I  = cv2.boxFilter(I, cv2.CV_64F, (radius, radius)).astype(np.float32)
        mean_p  = cv2.boxFilter(p, cv2.CV_64F, (radius, radius)).astype(np.float32)
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius)).astype(np.float32)

        # Covariance and variance
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius)).astype(np.float32)
        var_I   = mean_II - mean_I * mean_I

        # Linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Smooth coefficients
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius)).astype(np.float32)
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius)).astype(np.float32)

        refined = (mean_a * I + mean_b).clip(self.t_min, 1.0)
        return refined

    def _recover(
        self, img: np.ndarray, t: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        """
        Recover the haze-free image using:
            J(x) = (I(x) - A) / max(t(x), t_min) + A
        """
        t3 = t[:, :, np.newaxis]           # (H, W, 1) for broadcasting
        J = (img - A) / np.maximum(t3, self.t_min) + A  # ← Use max instead of clip
        return J.clip(0.0, 1.0)

    @staticmethod
    def _to_numpy(image_input) -> np.ndarray:
        """Convert path / PIL Image / ndarray → float32 RGB in [0, 1]."""
        if isinstance(image_input, str):
            pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        return np.array(pil, dtype=np.float32) / 255.0


class HOGSVMObjectDetectionStage(PipelineStage):
    """HOG+SVM pedestrian detection and Haar cascade car detection."""

    PERSON_COLOR = (0, 255, 0)    # Green for people
    CAR_COLOR    = (255, 165, 0)  # Orange for cars

    def __init__(
            self,
            person_win_stride: tuple = (4, 4),
            person_padding: tuple = (8, 8),
            person_scale: float = 1.01,
            car_scale_factor: float = 1.05,
            car_min_neighbors: int = 2,
            car_min_size: tuple = (35, 35),
        ):
            super().__init__("HOG+SVM Object Detection")

            # --- Pedestrian detector (HOG + pretrained SVM) ---
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.person_win_stride = person_win_stride
            self.person_padding    = person_padding
            self.person_scale      = person_scale

            # --- Car detector (Local Haar cascade) ---
            # Look for the file in your current project directory
            local_path = "haarcascade_car.xml"
            
            if os.path.exists(local_path):
                self.car_cascade = cv2.CascadeClassifier(local_path)
                print("Successfully loaded car detection model.")
            else:
                # Fallback to help you debug
                self.car_cascade = None
                print(f"CRITICAL: {local_path} not found in project folder!")
                print("Please ensure you downloaded the XML and named it correctly.")

            self.car_scale_factor  = car_scale_factor
            self.car_min_neighbors = car_min_neighbors
            self.car_min_size      = car_min_size
 
    def process(self, input_data: dict) -> PipelineResult:
        """
        Run detection on both dehazed and original images.
        
        Input:  {"image_input": <PIL Image>, "dehazed_image": <PIL Image>, ...}
        Output: {"segmented_image": <on dehazed>, "segmented_image_original": <on original>, ...}
        """
        try:
            import time

            start = time.time()

            # Detection on dehazed image (for top row)
            dehazed_source = input_data.get("dehazed_image")
            dehazed_np = self._to_numpy_uint8(dehazed_source)
            dehazed_bgr = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)
            people_dcp, people_weights_dcp = self._detect_people(dehazed_bgr)
            cars_dcp = self._detect_cars(dehazed_bgr)
            
            # Detection on original image (for bottom row)
            original_source = input_data.get("image_input")
            original_np = self._to_numpy_uint8(original_source)
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
            people_orig, people_weights_orig = self._detect_people(original_bgr)
            cars_orig = self._detect_cars(original_bgr)

            elapsed_ms = round((time.time() - start) * 1000, 1)

            # Annotate dehazed image
            annotated_dehazed = dehazed_bgr.copy()
            self._draw_boxes(annotated_dehazed, people_dcp, self.PERSON_COLOR, "Person")
            self._draw_boxes(annotated_dehazed, cars_dcp, self.CAR_COLOR, "Car")
            segmented_dcp_pil = Image.fromarray(cv2.cvtColor(annotated_dehazed, cv2.COLOR_BGR2RGB))
            
            # Annotate original image
            annotated_orig = original_bgr.copy()
            self._draw_boxes(annotated_orig, people_orig, self.PERSON_COLOR, "Person")
            self._draw_boxes(annotated_orig, cars_orig, self.CAR_COLOR, "Car")
            segmented_orig_pil = Image.fromarray(cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB))

            detections = (
                [{"type": "person", "box": list(map(int, b)), "weight": float(w)}
                 for b, w in zip(people_orig, people_weights_orig)]
                +
                [{"type": "car",    "box": list(map(int, b))}
                 for b in cars_orig]
            )

            hog_metrics = {
                "people_count_dehazed": len(people_dcp),
                "car_count_dehazed": len(cars_dcp),
                "total_detections_dehazed": len(people_dcp) + len(cars_dcp),
                "people_count_original": len(people_orig),
                "car_count_original": len(cars_orig),
                "total_detections_original": len(people_orig) + len(cars_orig),
                "processing_time_ms": elapsed_ms,
            }

            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "segmented_image": segmented_dcp_pil,
                    "segmented_image_original": segmented_orig_pil,
                    "detections":      detections,
                    "hog_metrics":     hog_metrics,
                },
            )

        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e),
            )

    def _detect_people(self, img_bgr: np.ndarray):
        """Run HOG+SVM pedestrian detector. Returns (boxes, weights)."""
        boxes, weights = self.hog.detectMultiScale(
            img_bgr,
            winStride=self.person_win_stride,
            padding=self.person_padding,
            scale=self.person_scale,
        )
        if len(boxes) == 0:
            return [], []
        boxes  = self._nms(boxes, overlap_thresh=0.65)
        return boxes, weights[:len(boxes)]

    def _detect_cars(self, img_bgr: np.ndarray):
        """Run Haar cascade car detector. Returns list of (x, y, w, h)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # helps in low-contrast/hazy scenes
        cars = self.car_cascade.detectMultiScale(
            gray,
            scaleFactor=self.car_scale_factor,
            minNeighbors=self.car_min_neighbors,
            minSize=self.car_min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(cars) == 0:
            return []
        return self._nms(cars, overlap_thresh=0.4)

    @staticmethod
    def _draw_boxes(img_bgr, boxes, color_rgb, label: str):
        """Draw labelled bounding boxes in-place (BGR image)."""
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        for (x, y, w, h) in boxes:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color_bgr, 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img_bgr, (x, y - th - 6), (x + tw + 4, y), color_bgr, -1)
            cv2.putText(
                img_bgr, label,
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    @staticmethod
    def _nms(boxes, overlap_thresh: float = 0.65):
        """
        Non-maximum suppression to remove duplicate overlapping boxes.
        boxes: list/array of (x, y, w, h)
        """
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        x1 = boxes[:, 0].astype(float)
        y1 = boxes[:, 1].astype(float)
        x2 = (boxes[:, 0] + boxes[:, 2]).astype(float)
        y2 = (boxes[:, 1] + boxes[:, 3]).astype(float)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        idxs = np.argsort(y2)
        picked = []

        while len(idxs) > 0:
            last = idxs[-1]
            picked.append(last)

            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / areas[idxs[:-1]]

            idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1],
                             np.where(overlap > overlap_thresh)[0])))

        return boxes[picked]

    @staticmethod
    def _to_numpy_uint8(image_input) -> np.ndarray:
        """Convert path / PIL Image / ndarray → uint8 RGB."""
        if isinstance(image_input, str):
            return np.array(Image.open(image_input).convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            return image_input.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")


import os
import time
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from pipeline import PipelineStage, PipelineResult


class YOLOObjectDetectionStage(PipelineStage):
    """YOLO-based object detection for vehicles and pedestrians."""
    
    def __init__(self, model_path: str = "yolo26n.pt", confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: YOLO model to use (default: "yolo26n.pt" for best CPU performance)
                       Options: "yolo26n.pt", "yolov8n.pt", "yolov11n.pt", or path to custom model
            confidence_threshold: Minimum confidence for detections (0-1)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        super().__init__("YOLO Object Detection")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path)
            # Move to appropriate device
            if self.device == "cuda":
                self.model.to('cuda')
            print(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to yolov8n.pt...")
            self.model = YOLO("yolov8n.pt")
            if self.device == "cuda":
                self.model.to('cuda')
        
        # Classes we care about for smog detection
        self.target_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
        }
        
        # Use the same vehicle color as HOG/haar stage for consistency.
        vehicle_color_bgr = (0, 165, 255)
        self.colors = {
            'person': (0, 255, 0),        # Green (people)
            'car': vehicle_color_bgr,     # Vehicle (orange)
            'truck': vehicle_color_bgr,   # Vehicle (orange)
            'bus': vehicle_color_bgr,     # Vehicle (orange)
            'motorcycle': vehicle_color_bgr, # Treat as vehicle (orange)
            'bicycle': vehicle_color_bgr, # Treat as vehicle (orange)
            'default': vehicle_color_bgr  # Default to vehicle color
        }

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def process(self, input_data: dict) -> PipelineResult:
        """
        Detect objects using YOLO on both dehazed and original images.
        
        Input:  {"image_input": <PIL Image>, "dehazed_image": <PIL Image>, ...}
        Output: {"segmented_image": <on dehazed>, "segmented_image_original": <on original>, ...}
        """
        try:
            start_time = time.time()
            
            # Get both source images
            dehazed_source = input_data.get("dehazed_image")
            original_source = input_data.get("image_input")
            
            if dehazed_source is None or original_source is None:
                raise ValueError("Both dehazed_image and image_input are required")
            
            # Run detection on dehazed and original
            segmented_dcp, detections_dcp, person_count_dcp, vehicle_count_dcp = self._run_yolo_on_image(dehazed_source)
            segmented_orig, detections_orig, person_count_orig, vehicle_count_orig = self._run_yolo_on_image(original_source)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Use original image detections for metrics and detections dict
            yolo_metrics = {
                "model_path": self.model_path,
                "people_count_dehazed": person_count_dcp,
                "car_count_dehazed": vehicle_count_dcp,
                "total_detections_dehazed": len(detections_dcp),
                "people_count_original": person_count_orig,
                "car_count_original": vehicle_count_orig,
                "total_detections_original": len(detections_orig),
                "processing_time_ms": processing_time_ms,
                "confidence_threshold": self.confidence_threshold,
                "detections": detections_orig
            }
            
            if len(detections_orig) > 0:
                print(f"YOLO: {person_count_orig} people, {vehicle_count_orig} vehicles in {processing_time_ms:.1f}ms")
            else:
                print(f"YOLO: No detections in {processing_time_ms:.1f}ms (confidence threshold: {self.confidence_threshold})")
            
            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "segmented_image": segmented_dcp,
                    "segmented_image_original": segmented_orig,
                    "detections": detections_orig,
                    "yolo_metrics": yolo_metrics,
                },
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=f"YOLO detection error: {str(e)}",
            )
    
    def _run_yolo_on_image(self, source):
        """
        Helper: Run YOLO detection on a single image.
        Returns: (segmented_pil, detections_list, person_count, vehicle_count)
        """
        # Convert to numpy array if needed
        if isinstance(source, Image.Image):
            img_array = np.array(source)
        elif isinstance(source, np.ndarray):
            img_array = source
        else:
            # Assume it's a path
            img_array = np.array(Image.open(source).convert("RGB"))
        
        # Ensure RGB (YOLO expects RGB)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize if too large for faster processing
        original_h, original_w = img_array.shape[:2]
        max_size = 1024
        scale = 1.0
        
        if original_w > max_size:
            scale = max_size / original_w
            new_w = max_size
            new_h = int(original_h * scale)
            img_array = cv2.resize(img_array, (new_w, new_h))
        
        # Run YOLO inference
        results = self.model(
            img_array, 
            conf=self.confidence_threshold,
            iou=0.45,
            half=True if self.device == "cuda" else False,
            verbose=False
        )
        
        # Process detections
        detections = []
        person_count = 0
        vehicle_count = 0
        
        # Create annotated image
        annotated_img = img_array.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Scale back coordinates if image was resized
                    if scale != 1.0:
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    
                    # Only track our target classes
                    if class_id in self.target_classes:
                        class_name = self.target_classes[class_id]
                        
                        # Count vehicles vs people
                        if class_name == 'person':
                            person_count += 1
                        elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train']:
                            vehicle_count += 1
                        
                        # Store detection
                        detections.append({
                            "type": class_name,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "confidence": confidence
                        })
                        
                        # Draw on appropriately sized image
                        draw_img = annotated_img
                        draw_x1, draw_y1, draw_x2, draw_y2 = x1, y1, x2, y2
                        
                        if scale != 1.0:
                            draw_x1 = int(x1 * scale)
                            draw_y1 = int(y1 * scale)
                            draw_x2 = int(x2 * scale)
                            draw_y2 = int(y2 * scale)
                        
                        color = self.colors.get(class_name, self.colors['default'])
                        
                        cv2.rectangle(draw_img, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
                        
                        label = f"{class_name}: {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y_label = max(draw_y1 - 5, label_size[1] + 5)
                        
                        cv2.rectangle(draw_img, 
                                    (draw_x1, y_label - label_size[1] - 5),
                                    (draw_x1 + label_size[0] + 5, y_label),
                                    color, -1)
                        
                        cv2.putText(draw_img, label, 
                                  (draw_x1 + 2, y_label - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # If we resized, convert back to original size
        if scale != 1.0:
            annotated_img = cv2.resize(annotated_img, (original_w, original_h))
        
        # Convert to PIL Image
        if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
            segmented_pil = Image.fromarray(annotated_img)
        else:
            segmented_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        
        return segmented_pil, detections, person_count, vehicle_count

class RCNNObjectDetectionStage(PipelineStage):
    """Faster R-CNN based object detection for vehicles and pedestrians."""

    def __init__(self, confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize Faster R-CNN detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        super().__init__("RCNN Object Detection")
        
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        
        # Load pre-trained Faster R-CNN model
        print("Loading Faster R-CNN model...")
        try:
            import torch
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights='DEFAULT',
                box_score_thresh=self.confidence_threshold
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Faster R-CNN loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading Faster R-CNN: {e}")
            raise
        
        # COCO classes we care about, aligned with the YOLO/HOG+SVM labels.
        self.coco_classes = {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            8: 'truck',
        }
        
        # Color coding matches the HOG/YOLO stages.
        self.person_color = (0, 255, 0)      # Green for people
        self.vehicle_color = (0, 165, 255)   # Orange for vehicles

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def process(self, input_data: dict) -> PipelineResult:
        """
        Detect objects using Faster R-CNN on both dehazed and original images.
        
        Input:  {"image_input": <PIL Image>, "dehazed_image": <PIL Image>, ...}
        Output: {"segmented_image": <on dehazed>, "segmented_image_original": <on original>, ...}
        """
        try:
            start_time = time.time()
            
            # Get both source images
            dehazed_source = input_data.get("dehazed_image")
            original_source = input_data.get("image_input")
            
            if dehazed_source is None or original_source is None:
                raise ValueError("Both dehazed_image and image_input are required")
            
            # Run detection on dehazed and original
            segmented_dcp, detections_dcp, person_count_dcp, vehicle_count_dcp = self._run_rcnn_on_image(dehazed_source)
            segmented_orig, detections_orig, person_count_orig, vehicle_count_orig = self._run_rcnn_on_image(original_source)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            rcnn_metrics = {
                "model": "Faster R-CNN ResNet50",
                "people_count_dehazed": person_count_dcp,
                "car_count_dehazed": vehicle_count_dcp,
                "total_detections_dehazed": len(detections_dcp),
                "people_count_original": person_count_orig,
                "car_count_original": vehicle_count_orig,
                "total_detections_original": len(detections_orig),
                "processing_time_ms": processing_time_ms,
                "confidence_threshold": self.confidence_threshold,
                "detections": detections_orig
            }
            
            if len(detections_orig) > 0:
                print(f"Faster R-CNN: {person_count_orig} people, {vehicle_count_orig} vehicles in {processing_time_ms:.1f}ms")
            else:
                print(f"Faster R-CNN: No detections in {processing_time_ms:.1f}ms (confidence threshold: {self.confidence_threshold})")
            
            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "segmented_image": segmented_dcp,
                    "segmented_image_original": segmented_orig,
                    "detections": detections_orig,
                    "rcnn_metrics": rcnn_metrics,
                },
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=f"Faster R-CNN detection error: {str(e)}",
            )
    
    def _run_rcnn_on_image(self, source):
        """
        Helper: Run Faster R-CNN detection on a single image.
        Returns: (segmented_pil, detections_list, person_count, vehicle_count)
        """
        import torch
        import torchvision.transforms as transforms
        
        # Convert to PIL if needed
        if isinstance(source, str):
            img_pil = Image.open(source).convert("RGB")
        elif isinstance(source, Image.Image):
            img_pil = source.convert("RGB")
        elif isinstance(source, np.ndarray):
            img_pil = Image.fromarray(source.astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image type: {type(source)}")
        
        # Convert to tensor for model
        img_tensor = transforms.ToTensor()(img_pil).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model([img_tensor])
        
        # Extract results
        boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence and convert to numpy for drawing
        img_np = np.array(img_pil)
        annotated_img = img_np.copy()
        
        detections = []
        person_count = 0
        vehicle_count = 0
        
        for box, score, label in zip(boxes, scores, labels):
            if score < self.confidence_threshold:
                continue
            
            if label not in self.coco_classes:
                continue
            
            class_name = self.coco_classes[label]
            x1, y1, x2, y2 = box
            
            # Count
            if class_name == 'person':
                person_count += 1
                color_rgb = self.person_color
            else:
                vehicle_count += 1
                color_rgb = self.vehicle_color
            
            # Store detection
            detections.append({
                "type": class_name,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(score)
            })
            
            # Draw box
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw label with confidence
            label_text = f"{class_name}: {score:.2f}"
            font_scale = 0.55
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Label background
            y_label = max(y1 - 6, text_height + 5)
            cv2.rectangle(annotated_img, (x1, y_label - text_height - 6), 
                        (x1 + text_width + 4, y_label), color_bgr, -1)
            cv2.putText(annotated_img, label_text, (x1 + 2, y_label - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Convert back to PIL
        segmented_pil = Image.fromarray(annotated_img)
        
        return segmented_pil, detections, person_count, vehicle_count


def create_default_pipeline(model_path: str, object_detection_model: str = "yolo", dehazing_method: str = "dcp"):
    """Create a pipeline with all stages."""
    
    pipeline = SmogClassificationPipeline()
    pipeline.add_stage(SmogClassificationStage(model_path))
    pipeline.add_stage(DehazingStage(method=dehazing_method))

    detection_model = (object_detection_model or "hogsvm").strip().lower()
    if detection_model == "yolo":
        pipeline.add_stage(YOLOObjectDetectionStage(
            model_path="yolo26n.pt",  # Best for CPU
            confidence_threshold=0.4   # Slightly lower for smoggy images
        ))
    elif detection_model == "rcnn":
        pipeline.add_stage(RCNNObjectDetectionStage())
    else:
        pipeline.add_stage(HOGSVMObjectDetectionStage())
    
    return pipeline
