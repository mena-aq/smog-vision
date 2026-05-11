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
            
            # Run classification
            result = self.classifier.predict(
                image_path=image_path,
                image_array=image_input if image_path is None else None
            )
            
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
    """Placeholder for HOG+SVM Object Detection stage."""
    
    def __init__(self):
        super().__init__("HOG+SVM Object Detection")
    
    def process(self, input_data: dict) -> PipelineResult:
        """
        TODO: Implement HOG+SVM object detection.
        
        Input: {"image_input": <PIL Image>, "dehazed_image": <PIL Image>, ...}
        Output: {"segmented_image": <PIL Image>, "detections": [...], ...}
        """
        try:
            # Use the dehazed image if available, otherwise original
            image_to_process = input_data.get("dehazed_image", input_data.get("image_input"))
            
            # Placeholder: return the same image
            segmented_image = image_to_process
            
            return PipelineResult(
                stage_name=self.name,
                success=True,
                data={
                    **input_data,
                    "segmented_image": segmented_image,
                    "detections": []  # Placeholder for detection results
                }
            )
        except Exception as e:
            return PipelineResult(
                stage_name=self.name,
                success=False,
                data={},
                error=str(e)
            )


def create_default_pipeline(model_path: str):
    """Create a pipeline with all stages."""
    from pipeline import SmogClassificationPipeline
    
    pipeline = SmogClassificationPipeline()
    pipeline.add_stage(SmogClassificationStage(model_path))
    pipeline.add_stage(DCPDehazingStage())
    pipeline.add_stage(HOGSVMObjectDetectionStage())
    
    return pipeline
