import os
from pathlib import Path
from PIL import Image
import cv2
import time
import numpy as np
from inference import SmogClassifier
from pipeline import PipelineStage, PipelineResult


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


class DCPDehazingStage(PipelineStage):
    """Placeholder for DCP Dehazing stage."""
    
    def __init__(self, patch_size: int = 15, omega: float = 0.95, t_min: float = 0.1):
        """
        Args:
            patch_size: Size of the local patch for dark channel computation (default 15).
            omega:      Haze retention factor — 0.95 keeps a tiny bit of haze for
                        realism; set to 1.0 to remove all estimated haze.
            t_min:      Minimum transmission value to avoid division by zero (default 0.1).
        """
        super().__init__("DCP Dehazing")
        self.patch_size = patch_size
        self.omega = omega
        self.t_min = t_min
    
    def process(self, input_data: dict) -> PipelineResult:
        """
        Input:  {"image_input": <str path | PIL Image | np.ndarray>, ...}
        Output: {"dehazed_image": <PIL Image>, ...everything from input...}
        """
        try:
            raw = input_data.get("image_input")
            img_np = self._to_numpy(raw)

            start = time.time()

            dark = self._dark_channel(img_np)
            A= self._atmospheric_light(img_np, dark)
            t_raw = self._transmission(img_np, A)
            t_refined= self._guided_filter(img_np, t_raw)
            dehazed_np= self._recover(img_np, t_refined, A)

            elapsed = time.time() - start

            dehazed_pil = Image.fromarray(
                (dehazed_np * 255).clip(0, 255).astype(np.uint8)
            )

            dcp_metrics = {
                "transmission_map": float(np.mean(t_refined)),
                "dark_channel": float(np.mean(dark)),
                "airlight_rgb": tuple(round(float(v), 3) for v in A),
                "processing_time_ms": round(elapsed * 1000, 1),
            }

            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={**input_data, "dehazed_image": dehazed_pil, "dcp_metrics": dcp_metrics},
            )

        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e)
            )

    def _dehaze(self, img: np.ndarray) -> np.ndarray:
        """Full DCP pipeline on a float32 RGB image in [0, 1]."""
        dark          = self._dark_channel(img)
        A             = self._atmospheric_light(img, dark)
        t_raw         = self._transmission(img, A)
        t_refined     = self._guided_filter(img, t_raw)
        recovered     = self._recover(img, t_refined, A)
        return recovered

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
        radius: int = 40,
        eps: float = 1e-3,
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
        J  = (img - A) / t3 + A
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
        Input:  {"dehazed_image": <PIL Image>, ...}
        Output: {"segmented_image": <PIL Image>, "detections": [...], ...}
        """
        try:
            import time

            # Prefer dehazed image, fall back to original
            source = input_data.get("dehazed_image", input_data.get("image_input"))
            img_np = self._to_numpy_uint8(source)   # uint8 RGB (H, W, 3)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            start = time.time()

            people, people_weights = self._detect_people(img_bgr)
            cars                   = self._detect_cars(img_bgr)

            elapsed_ms = round((time.time() - start) * 1000, 1)

            # Draw bounding boxes on a copy of the dehazed image
            annotated_bgr = img_bgr.copy()
            self._draw_boxes(annotated_bgr, people, self.PERSON_COLOR, "Person")
            self._draw_boxes(annotated_bgr, cars,   self.CAR_COLOR,    "Car")

            annotated_pil = Image.fromarray(
                cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            )

            detections = (
                [{"type": "person", "box": list(map(int, b)), "weight": float(w)}
                 for b, w in zip(people, people_weights)]
                +
                [{"type": "car",    "box": list(map(int, b))}
                 for b in cars]
            )

            hog_metrics = {
                "people_count":    len(people),
                "car_count":       len(cars),
                "total_detections": len(people) + len(cars),
                "processing_time_ms": elapsed_ms,
            }

            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "segmented_image": annotated_pil,
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


class YOLOObjectDetectionStage(PipelineStage):
    """Placeholder for YOLO-based object detection."""

    def __init__(self, model_path: str = None):
        super().__init__("YOLO Object Detection")
        self.model_path = model_path

    def process(self, input_data: dict) -> PipelineResult:
        """
        Placeholder YOLO stage.

        Input:  {"dehazed_image": <PIL Image>, ...}
        Output: {"segmented_image": <PIL Image>, "detections": [...], ...}
        """
        try:
            start = time.time()

            source = input_data.get("dehazed_image", input_data.get("image_input"))
            elapsed_ms = round((time.time() - start) * 1000, 1)

            yolo_metrics = {
                "model_path": self.model_path,
                "people_count": 0,
                "car_count": 0,
                "total_detections": 0,
                "processing_time_ms": elapsed_ms,
            }

            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "segmented_image": source,
                    "detections": [],
                    "yolo_metrics": yolo_metrics,
                },
            )
        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e),
            )


class RCNNObjectDetectionStage(PipelineStage):
    """Placeholder for RCNN-based object detection."""

    def __init__(self, model_path: str = None):
        super().__init__("RCNN Object Detection")
        self.model_path = model_path

    def process(self, input_data: dict) -> PipelineResult:
        """
        Placeholder RCNN stage.

        Input:  {"dehazed_image": <PIL Image>, ...}
        Output: {"segmented_image": <PIL Image>, "detections": [...], ...}
        """
        try:
            start = time.time()

            source = input_data.get("dehazed_image", input_data.get("image_input"))
            elapsed_ms = round((time.time() - start) * 1000, 1)

            rcnn_metrics = {
                "model_path": self.model_path,
                "people_count": 0,
                "car_count": 0,
                "total_detections": 0,
                "processing_time_ms": elapsed_ms,
            }

            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "segmented_image": source,
                    "detections": [],
                    "rcnn_metrics": rcnn_metrics,
                },
            )
        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e),
            )


def create_default_pipeline(model_path: str, object_detection_model: str = "hogsvm"):
    """Create a pipeline with all stages."""
    from pipeline import SmogClassificationPipeline
    
    pipeline = SmogClassificationPipeline()
    pipeline.add_stage(SmogClassificationStage(model_path))
    pipeline.add_stage(DCPDehazingStage())

    detection_model = (object_detection_model or "hogsvm").strip().lower()
    if detection_model == "yolo":
        pipeline.add_stage(YOLOObjectDetectionStage())
    elif detection_model == "rcnn":
        pipeline.add_stage(RCNNObjectDetectionStage())
    else:
        pipeline.add_stage(HOGSVMObjectDetectionStage())
    
    return pipeline
