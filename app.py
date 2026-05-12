"""
SmogVision - Advanced Smog Detection and Analysis Pipeline
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox,
    QFrame, QScrollArea, QComboBox, QDialog
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from stages import create_default_pipeline


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ImageViewerDialog(QDialog):
    def __init__(self, image_data, title="Image Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 650)

        self.viewer_label = QLabel()
        self.viewer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_label.setScaledContents(False)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.viewer_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area)

        self._original_pixmap = self._to_pixmap(image_data)
        self._update_pixmap()

    def _to_pixmap(self, image_data):
        if isinstance(image_data, str):
            return QPixmap(image_data)
        if isinstance(image_data, np.ndarray):
            if image_data.ndim == 3 and image_data.shape[2] == 3:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            h, w = image_data.shape[:2]
            channels = image_data.shape[2] if image_data.ndim == 3 else 1
            bytes_per_line = channels * w
            q_img = QImage(image_data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(q_img)
        if isinstance(image_data, Image.Image):
            image = image_data.convert("RGBA")
            data = image.tobytes("raw", "RGBA")
            q_img = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
            return QPixmap.fromImage(q_img)
        return QPixmap()

    def _update_pixmap(self):
        if self._original_pixmap.isNull():
            self.viewer_label.setText("No image available")
            return

        viewport_size = self.scroll_area.viewport().size()
        scaled_pixmap = self._original_pixmap.scaled(
            viewport_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.viewer_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()


class MaskViewerDialog(QDialog):
    """Dialog for viewing and downloading segmentation masks."""
    
    def __init__(self, mask_dehazed, mask_original, parent=None):
        super().__init__(parent)
        self.setWindowTitle("View & Download Masks")
        self.setMinimumSize(1000, 700)
        self.mask_dehazed = mask_dehazed
        self.mask_original = mask_original
        
        # Light theme stylesheet
        self.setStyleSheet("""
            MaskViewerDialog {
                background-color: #FFFFFF;
            }
            QGroupBox {
                background-color: #F8F9FA;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                padding-top: 10px;
                margin-top: 10px;
                font-weight: bold;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 3px 0 3px;
            }
            QLabel {
                color: #333333;
                background-color: transparent;
            }
            QScrollArea {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #007AFF;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #0051D5;
            }
            QPushButton:pressed {
                background-color: #003DA8;
            }
        """)
        
        try:
            print(f"MaskViewerDialog init - dehazed type: {type(mask_dehazed)}, original type: {type(mask_original)}")
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            # Title
            title = QLabel("Segmentation Masks")
            title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #1F2937;")
            layout.addWidget(title)
            
            # Masks display area
            masks_layout = QHBoxLayout()
            
            # Dehazed mask
            dehazed_group = QGroupBox("Mask - Dehazed")
            dehazed_layout = QVBoxLayout()
            self.dehazed_label = QLabel()
            self.dehazed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.dehazed_label.setScaledContents(False)
            dehazed_scroll = QScrollArea()
            dehazed_scroll.setWidgetResizable(True)
            dehazed_scroll.setWidget(self.dehazed_label)
            dehazed_layout.addWidget(dehazed_scroll)
            dehazed_group.setLayout(dehazed_layout)
            masks_layout.addWidget(dehazed_group)
            
            # Arrow separator
            arrow_label = QLabel("→")
            arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            arrow_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #CCCCCC;")
            masks_layout.addWidget(arrow_label)
            
            # Original mask
            original_group = QGroupBox("Mask - Original")
            original_layout = QVBoxLayout()
            self.original_label = QLabel()
            self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.original_label.setScaledContents(False)
            original_scroll = QScrollArea()
            original_scroll.setWidgetResizable(True)
            original_scroll.setWidget(self.original_label)
            original_layout.addWidget(original_scroll)
            original_group.setLayout(original_layout)
            masks_layout.addWidget(original_group)
            
            layout.addLayout(masks_layout, 1)
            
            # Download buttons
            download_layout = QHBoxLayout()
            download_layout.addStretch()
            
            self.download_dehazed_btn = QPushButton("↓ Download Mask (Dehazed)")
            self.download_dehazed_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.download_dehazed_btn.clicked.connect(self.download_dehazed)
            download_layout.addWidget(self.download_dehazed_btn)
            
            self.download_original_btn = QPushButton("↓ Download Mask (Original)")
            self.download_original_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.download_original_btn.clicked.connect(self.download_original)
            download_layout.addWidget(self.download_original_btn)
            
            download_layout.addStretch()
            layout.addLayout(download_layout)
            
            # Display masks with error handling
            self._display_masks()
        except Exception as e:
            print(f"ERROR in MaskViewerDialog.__init__: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _display_masks(self):
        """Display both masks in the dialog."""
        try:
            print("_display_masks called")
            if self.mask_dehazed:
                print(f"Converting dehazed mask: type={type(self.mask_dehazed)}")
                dehazed_pixmap = self._to_pixmap(self.mask_dehazed)
                if not dehazed_pixmap.isNull():
                    scaled = dehazed_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.dehazed_label.setPixmap(scaled)
                else:
                    self.dehazed_label.setText("Failed to load dehazed mask")
            
            if self.mask_original:
                print(f"Converting original mask: type={type(self.mask_original)}")
                original_pixmap = self._to_pixmap(self.mask_original)
                if not original_pixmap.isNull():
                    scaled = original_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.original_label.setPixmap(scaled)
                else:
                    self.original_label.setText("Failed to load original mask")
        except Exception as e:
            print(f"ERROR in _display_masks: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def _to_pixmap(image_data, max_size=1024):
        """Convert image to QPixmap with automatic downscaling for large images."""
        try:
            if image_data is None:
                print("Image data is None")
                return QPixmap()
            
            if isinstance(image_data, Image.Image):
                print(f"Converting PIL Image: size={image_data.size}, mode={image_data.mode}")
                image = image_data.convert("RGB")
                
                # Downscale large images to prevent memory issues
                w, h = image.size
                if w > max_size or h > max_size:
                    scale = max_size / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    print(f"Downscaling from {image.size} to {new_size}")
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                data = image.tobytes("raw", "RGB")
                q_img = QImage(data, image.width, image.height, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                print(f"Successfully converted to QPixmap: {pixmap.size()}")
                return pixmap
            
            if isinstance(image_data, np.ndarray):
                print(f"Converting numpy array: shape={image_data.shape}, dtype={image_data.dtype}")
                if image_data.ndim == 3 and image_data.shape[2] == 3:
                    # Make a contiguous copy for QImage
                    img_copy = np.ascontiguousarray(image_data)
                    h, w, ch = img_copy.shape
                    bytes_per_line = ch * w
                    q_img = QImage(img_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    return QPixmap.fromImage(q_img)
                else:
                    print(f"Unexpected numpy array shape: {image_data.shape}")
            
            print(f"Unsupported image data type: {type(image_data)}")
            return QPixmap()
        except Exception as e:
            print(f"ERROR in _to_pixmap: {e}")
            import traceback
            traceback.print_exc()
            return QPixmap()
    
    def download_dehazed(self):
        """Download dehazed mask."""
        if self.mask_dehazed is None:
            QMessageBox.warning(self, "Warning", "No mask available for download.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask (Dehazed)", "", "PNG Images (*.png);;All Files (*)"
        )
        if file_path:
            try:
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                self.mask_dehazed.save(file_path)
                QMessageBox.information(self, "Success", f"Mask saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mask: {str(e)}")
    
    def download_original(self):
        """Download original mask."""
        if self.mask_original is None:
            QMessageBox.warning(self, "Warning", "No mask available for download.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask (Original)", "", "PNG Images (*.png);;All Files (*)"
        )
        if file_path:
            try:
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                self.mask_original.save(file_path)
                QMessageBox.information(self, "Success", f"Mask saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mask: {str(e)}")


class VideoWorker(QThread):
    """Worker thread for processing video frames."""
    
    frame_processed = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model_path: str, video_path: str):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self._is_running = True
    
    def run(self):
        try:
            pipeline = create_default_pipeline(self.model_path)
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                self.error.emit("Could not open video file.")
                return
            
            while self._is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Run pipeline
                result = pipeline.run(pil_image)
                self.frame_processed.emit(result)
                
                # Allow GUI to update
                self.msleep(30)
            
            cap.release()
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Video processing error: {str(e)}")
    
    def stop(self):
        self._is_running = False


class InferenceWorker(QThread):
    """Worker thread for single image inference."""
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, model_path: str, image_path: str, detector: str = "hogsvm", dehazing_method: str = "dcp"):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.detector = detector
        self.dehazing_method = dehazing_method
    
    def run(self):
        try:
            if not os.path.exists(self.model_path):
                self.error.emit(f"Model file not found: {self.model_path}")
                return
            
            pipeline = create_default_pipeline(self.model_path, object_detection_model=self.detector, dehazing_method=self.dehazing_method)
            result = pipeline.run(self.image_path)
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Inference error: {str(e)}")


class SmogVisionGUI(QMainWindow):
    """Main desktop GUI for the SmogVision pipeline."""
    
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.model_path = "smog-classification/smog_classifier.pth"
        self.video_worker = None
        self.inference_worker = None
        self.detector_dropdown = None
        self.dehazing_method_dropdown = None
        self.selected_dehazing_method = "dcp"  # Default dehazing method
        self.segmented_image_dehazed = None  # Top row - Pipeline Output
        self.segmented_image_original = None  # Bottom row - Comparison
        self.mask_dehazed = None  # Colored segmentation mask - dehazed
        self.mask_original = None  # Colored segmentation mask - original
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SmogVision")
        self.setGeometry(50, 50, 1400, 850)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left panel with horizontal scroll
        left_panel_widget = self.create_left_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel_widget)
        left_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
            QScrollBar:horizontal {
                background-color: #f3f4f6;
                height: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background-color: #d1d5db;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #9ca3af;
            }
            QScrollBar:vertical {
                background-color: #f3f4f6;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #d1d5db;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #9ca3af;
            }
        """)
        main_layout.addWidget(left_scroll, 1)  # Takes remaining space

        # Right panel
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 0)  # Fixed size

        self.apply_stylesheet()
    
    def create_left_panel(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(12)

        # Title with Clouds
        title_layout = QHBoxLayout()
        title = QLabel("SmogVision ☁️☁️☁️")
        title.setObjectName("h1")
        title_layout.addWidget(title)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Upload Button
        upload_row = QHBoxLayout()
        self.upload_button = QPushButton("Upload Image or Video")
        self.upload_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_button.clicked.connect(self.upload_file)
        upload_row.addWidget(self.upload_button)

        # Dehazing Method Selection Dropdown
        self.dehazing_method_dropdown = QComboBox()
        self.dehazing_method_dropdown.addItems(["Select Dehazing Method", "DCP", "CLAHE"])
        self.dehazing_method_dropdown.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.dehazing_method_dropdown.currentTextChanged.connect(self.on_dehazing_method_changed)
        upload_row.addWidget(self.dehazing_method_dropdown)

        # Detector Selection Dropdown
        self.detector_dropdown = QComboBox()
        self.detector_dropdown.addItems(["Select Detection Method", "HOG + SVM", "YOLO", "RCNN"])
        self.detector_dropdown.setEnabled(True)
        self.detector_dropdown.setVisible(True)
        self.detector_dropdown.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.detector_dropdown.currentTextChanged.connect(self.on_detector_changed)
        upload_row.addWidget(self.detector_dropdown)


        # Process Button
        self.process_button = QPushButton("Process")
        self.process_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_button.setEnabled(False)  # Disabled until a file is uploaded
        self.process_button.clicked.connect(self.process_file) # Connect to the process method
        upload_row.addWidget(self.process_button)

        upload_row.addStretch()
        layout.addLayout(upload_row)

        # Results area with scroll support
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setStyleSheet("QScrollArea { border: none; background-color: #F9FAFB; }")
        results_scroll.setVisible(False)
        
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(30)

        # Pipeline Output Group
        pipeline_group = QGroupBox("Pipeline Output")
        pipeline_layout = QHBoxLayout()
        self.original_view = self.create_image_view("Original")
        self.dehazed_view = self.create_image_view("Dehazed (DCP)")
        self.segmented_view = self.create_image_view("Final Ressult - Dehazed")

        # Make segmented view clickable (top row - Pipeline Output)
        segmented_label = self.segmented_view.property("image_label")
        segmented_label.setCursor(Qt.CursorShape.PointingHandCursor)
        segmented_label.clicked.connect(lambda: self.select_image_for_stage1("dehazed"))
        
        # Change this in create_left_panel
        pipeline_layout.addWidget(self.original_view, 1) # Added stretch factor 1
        pipeline_layout.addWidget(self.create_arrow())
        pipeline_layout.addWidget(self.dehazed_view, 1)   # Added stretch factor 1
        pipeline_layout.addWidget(self.create_arrow())
        pipeline_layout.addWidget(self.segmented_view, 1) # Added stretch factor 1
        pipeline_group.setLayout(pipeline_layout)
        results_layout.addWidget(pipeline_group)

        # Comparison Group
        comparison_group = QGroupBox("Comparison")
        comparison_layout = QHBoxLayout()
        self.final_original_view = self.create_image_view("Original")
        self.final_result_view = self.create_image_view("Final Result - Original")
        final_result_label = self.final_result_view.property("image_label")
        final_result_label.setCursor(Qt.CursorShape.PointingHandCursor)
        final_result_label.clicked.connect(lambda: self.select_image_for_stage1("original"))
        
        comparison_layout.addWidget(self.final_original_view)
        comparison_layout.addWidget(self.create_arrow(large=True))
        comparison_layout.addWidget(self.final_result_view)
        comparison_group.setLayout(comparison_layout)
        results_layout.addWidget(comparison_group)

        # View Masks Button
        masks_button_layout = QHBoxLayout()
        masks_button_layout.addStretch()
        self.view_masks_btn = QPushButton("🎭 View Masks")
        self.view_masks_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_masks_btn.setEnabled(False)
        self.view_masks_btn.clicked.connect(self.open_mask_viewer)
        self.view_masks_btn.setMinimumHeight(40)
        self.view_masks_btn.setStyleSheet("""
            QPushButton {
                font-size: 13px;
                font-weight: bold;
                padding: 10px 20px;
                background-color: #8B5CF6;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover:!pressed {
                background-color: #7C3AED;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #999999;
            }
        """)
        masks_button_layout.addWidget(self.view_masks_btn)
        masks_button_layout.addStretch()
        results_layout.addLayout(masks_button_layout)

        layout.addWidget(self.results_widget)
        layout.addStretch()

        return widget
    
    def create_image_view(self, title: str):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        label = ClickableLabel()
        label.setObjectName("imageView")
        label.setFixedSize(320, 180)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setScaledContents(False)
        layout.addWidget(label)
        
        title_label = QLabel(title)
        title_label.setObjectName("imageSubTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        widget.setProperty("image_label", label)
        return widget

    def open_segmented_viewer(self, view_type: str):
        """Open viewer for segmented results (dehazed or original)."""
        if view_type == "dehazed" and self.segmented_image_dehazed is None:
            return
        if view_type == "original" and self.segmented_image_original is None:
            return

        image = self.segmented_image_dehazed if view_type == "dehazed" else self.segmented_image_original
        title = "Detection Result - Dehazed" if view_type == "dehazed" else "Detection Result - Original"
        viewer = ImageViewerDialog(image, title, self)
        viewer.exec()

    def select_image_for_stage1(self, view_type: str):
        """When a thumbnail is clicked, keep pipeline visible and fill Stage 1 with that image."""
        try:
            if view_type == "dehazed":
                img = self.segmented_image_dehazed
            else:
                img = self.segmented_image_original or self.segmented_image_dehazed

            if img is None:
                return

            # Ensure results panel remains visible and update Stage 1 (original_view)
            self.results_widget.setVisible(True)
            self.display_image(img, self.original_view.property("image_label"))

            # Also update CNN stage display to reflect the selected image
            self.display_image(img, self.dehazed_view.property("image_label"))

        except Exception as e:
            print(f"ERROR in select_image_for_stage1: {e}")

    def open_mask_viewer(self):
        """Open mask viewer dialog."""
        try:
            if self.mask_dehazed is None or self.mask_original is None:
                QMessageBox.warning(self, "Warning", "Masks are not available yet.")
                return
            
            viewer = MaskViewerDialog(self.mask_dehazed, self.mask_original, self)
            viewer.exec()
        except Exception as e:
            print(f"ERROR in open_mask_viewer: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to open mask viewer:\n{str(e)}")

    def create_right_panel(self):
        # Create scroll area for metrics
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(300)
        scroll_area.setMaximumWidth(450)  # Set reasonable max width for metrics panel
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #f3f4f6;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #d1d5db;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #9ca3af;
            }
        """)
        
        widget = QWidget()
        widget.setObjectName("metricsPanel")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(25, 30, 25, 30)
        
        title = QLabel("Model Metrics")
        title.setObjectName("h2")
        layout.addWidget(title)

        # Updated Color Schemes to match Screenshot
        self.cnn_metrics = self.create_metric_box(
            "CNN Classifier", 
            "Stage 1", 
            "#EF4444", 
            "#FFF1F1", 
            ["Predicted Class", "Confidence", "Probability Smog", "Probability Clear", "Processing Time"]
        )
        layout.addWidget(self.cnn_metrics)

        self.dcp_metrics = self.create_metric_box("DCP Dehazing", "Stage 2", "#F59E0B", "#FFFBEB", 
                                               ["Transmission Map", "Dark Channel", "Airlight RGB", "Processing Time"])
        layout.addWidget(self.dcp_metrics)

        self.clahe_metrics = self.create_metric_box("CLAHE Dehazing", "Stage 2", "#F59E0B", "#FFFBEB", 
                                                ["Processing Time"])
        self.clahe_metrics.setVisible(False)
        layout.addWidget(self.clahe_metrics)

        self.hog_svm_metrics = self.create_comparison_metric_box("HOG + SVM", "Stage 3", "#10B981", "#F0FDF4",
                                               ["People Detected", "Vehicles Detected", "Total Detections"])
        layout.addWidget(self.hog_svm_metrics)

        # Placeholders for other detectors (hidden until used)
        self.yolo_metrics = self.create_comparison_metric_box("YOLO Detector", "Stage 3", "#10B981", "#F0FDF4",
                              ["People Detected", "Vehicles Detected", "Total Detections"])
        self.yolo_metrics.setVisible(False)
        layout.addWidget(self.yolo_metrics)

        self.rcnn_metrics = self.create_comparison_metric_box("RCNN Detector", "Stage 3", "#10B981", "#F0FDF4",
                               ["People Detected", "Vehicles Detected", "Total Detections"])
        self.rcnn_metrics.setVisible(False)
        layout.addWidget(self.rcnn_metrics)

        layout.addStretch()
        
        self.total_time_label = self.create_metric_row("Total Pipeline Time", "0ms", highlight=True)
        layout.addWidget(self.total_time_label)

        self.update_detection_metrics_title(self.detector_dropdown.currentText() if self.detector_dropdown else "Select Detection Method")
        
        # Set widget in scroll area and return scroll area
        scroll_area.setWidget(widget)
        return scroll_area
    
    def create_arrow(self, large=False):
        """Helper to create an arrow label."""
        arrow = QLabel("→")
        font_size = 24 if large else 18
        arrow.setStyleSheet(f"font-size: {font_size}px; color: #999; margin-top: -24px;")
        return arrow
    
    def create_metric_box(self, title, stage, accent_color, bg_color, labels):
        box = QGroupBox()
        box.setStyleSheet(f"""
            QGroupBox {{ 
                background-color: {bg_color}; 
                border-radius: 12px; 
                border: none; 
                margin-top: 10px;
            }}
        """)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(15, 15, 15, 15)

        header = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {accent_color}; font-size: 18px; border: none;") # Explicitly remove border here too
        t_label = QLabel(title)
        t_label.setObjectName("h3") # This uses the style from your apply_stylesheet
        header.addWidget(t_label)
        
        header.addWidget(dot)
        header.addStretch()
        
        s_label = QLabel(stage)
        s_label.setObjectName("stageLabel")
        header.addWidget(s_label)
        layout.addLayout(header)

        metric_widgets = {}
        for text in labels:
            row = self.create_metric_row(text, "N/A")
            layout.addWidget(row)
            metric_widgets[text] = row
            
        box.setProperty("metric_widgets", metric_widgets)
        box.setProperty("title_label", t_label)
        return box

    def create_metric_row(self, label_text, value_text, highlight=False):
        widget = QWidget()
        if highlight: widget.setObjectName("totalBox")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        
        l = QLabel(label_text)
        l.setStyleSheet("color: #6B7280; font-size: 13px;")
        v = QLabel(value_text)
        v.setStyleSheet("color: #111827; font-weight: 600; font-size: 13px;")
        
        layout.addWidget(l)
        layout.addStretch()
        layout.addWidget(v)
        widget.setProperty("value_label", v)
        return widget
    
    def create_comparison_metric_box(self, title, stage, accent_color, bg_color, labels):
        """Create a metric box with 2-column comparison (Dehazed vs Original)."""
        box = QGroupBox()
        box.setStyleSheet(f"""
            QGroupBox {{ 
                background-color: {bg_color}; 
                border-radius: 12px; 
                border: none; 
                margin-top: 10px;
            }}
        """)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {accent_color}; font-size: 18px; border: none;")
        t_label = QLabel(title)
        t_label.setObjectName("h3")
        header.addWidget(t_label)
        header.addWidget(dot)
        header.addStretch()
        
        s_label = QLabel(stage)
        s_label.setObjectName("stageLabel")
        header.addWidget(s_label)
        layout.addLayout(header)

        # Column headers
        col_header = QHBoxLayout()
        col_header.addWidget(QLabel(""), 1)  # Empty space for label column
        dehazed_header = QLabel("Dehazed")
        dehazed_header.setObjectName("h3")
        dehazed_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col_header.addWidget(dehazed_header, 1)
        original_header = QLabel("Original")
        original_header.setObjectName("h3")
        original_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col_header.addWidget(original_header, 1)
        layout.addLayout(col_header)

        # Metric rows with 2 columns
        metric_widgets = {}
        for text in labels:
            row_layout = QHBoxLayout()
            
            # Label
            l = QLabel(text)
            l.setStyleSheet("color: #6B7280; font-size: 13px;")
            row_layout.addWidget(l, 1)
            
            # Dehazed value
            v_dehazed = QLabel("N/A")
            v_dehazed.setStyleSheet("color: #111827; font-weight: 600; font-size: 13px;")
            v_dehazed.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row_layout.addWidget(v_dehazed, 1)
            
            # Original value
            v_original = QLabel("N/A")
            v_original.setStyleSheet("color: #111827; font-weight: 600; font-size: 13px;")
            v_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row_layout.addWidget(v_original, 1)
            
            row_widget = QWidget()
            row_widget.setLayout(row_layout)
            layout.addWidget(row_widget)
            
            metric_widgets[text] = {
                "dehazed": v_dehazed,
                "original": v_original
            }
        
        box.setProperty("metric_widgets", metric_widgets)
        box.setProperty("title_label", t_label)
        return box

    def on_detector_changed(self, selected_method: str):
        """Update the object-detection metrics title as soon as the dropdown changes."""
        self.update_detection_metrics_title(selected_method)

    def on_dehazing_method_changed(self, selected_method: str):
        """Update the selected dehazing method."""
        self.selected_dehazing_method = selected_method.lower()

        # Immediately toggle visible metric box based on selection
        sel = selected_method.lower() if isinstance(selected_method, str) else ""
        if "clahe" in sel:
            if hasattr(self, 'clahe_metrics'):
                self.clahe_metrics.setVisible(True)
            if hasattr(self, 'dcp_metrics'):
                self.dcp_metrics.setVisible(False)
        elif "dcp" in sel:
            if hasattr(self, 'dcp_metrics'):
                self.dcp_metrics.setVisible(True)
            if hasattr(self, 'clahe_metrics'):
                self.clahe_metrics.setVisible(False)
        else:
            # Hide both if no valid selection
            if hasattr(self, 'dcp_metrics'):
                self.dcp_metrics.setVisible(False)
            if hasattr(self, 'clahe_metrics'):
                self.clahe_metrics.setVisible(False)

    def update_detection_metrics_title(self, selected_method: str):
        """Set the active object-detection metrics title from the selected detector."""
        method_titles = {
            "HOG + SVM": "HOG + SVM",
            "YOLO": "YOLO Detector",
            "RCNN": "RCNN Detector",
        }
        title_text = method_titles.get(selected_method, "Object Detection")

        for box in (self.hog_svm_metrics, self.yolo_metrics, self.rcnn_metrics):
            if box is not None:
                box.property("title_label").setText(title_text)
    
    def create_divider(self):
        """Helper to create a horizontal divider."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #ddd;")
        return line
    
    def upload_file(self):
        """Handle file upload."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image or Video", "",
            "Media Files (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            self.upload_button.setText(f" {os.path.basename(file_path)}")
            self.process_button.setEnabled(True)
            self.results_widget.setVisible(True)  # Keep pipeline UI visible
            
            # Display first frame if video
            if file_path.lower().endswith(('.mp4', '.avi')):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    self.display_image(frame, self.original_view.property("image_label"))
                    self.display_image(frame, self.final_original_view.property("image_label"))
                cap.release()
            else:
                self.display_image(file_path, self.original_view.property("image_label"))
                self.display_image(file_path, self.final_original_view.property("image_label"))
    
    def process_file(self):
        """Start processing the selected file."""
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "Please select a file first.")
            return

        if self.detector_dropdown.currentIndex() == 0:
            QMessageBox.warning(self, "Warning", "Please select a detection method first.")
            self.process_button.setEnabled(True)
            self.process_button.setText("Process")
            return
        
        self.results_widget.setVisible(True)
        self.process_button.setEnabled(False)
        self.process_button.setText(" Processing...")
        self.detector_dropdown.setVisible(False)        
                                                            # ADD
        if self.file_path.lower().endswith(('.mp4', '.avi')):
            self.video_worker = VideoWorker(self.model_path, self.file_path)
            self.video_worker.frame_processed.connect(self.update_ui_with_results)
            self.video_worker.finished.connect(self.on_processing_finished)
            self.video_worker.error.connect(self.on_processing_error)
            self.video_worker.start()
        else:
            # Get selected detector from dropdown
            detector_map = {"HOG + SVM": "hogsvm", "YOLO": "yolo", "RCNN": "rcnn"}
            selected_detector = detector_map.get(self.detector_dropdown.currentText(), "hogsvm")
            self.worker = InferenceWorker(self.model_path, self.file_path, detector=selected_detector, dehazing_method=self.selected_dehazing_method)
            self.worker.finished.connect(self.update_ui_with_results)
            self.worker.error.connect(self.on_processing_error)
            self.worker.start()
    
    def update_ui_with_results(self, result: dict):
        """Update the GUI with processing results."""
        if not result["success"]:
            self.on_processing_error(result.get("stages", [{}])[-1].get("error", "Unknown pipeline error"))
            return
        
        total_time = 0
        
        # Update images
        self.display_image(result["final_data"].get("image_input"), self.original_view.property("image_label"))
        self.display_image(result["final_data"].get("dehazed_image"), self.dehazed_view.property("image_label"))
        self.display_image(result["final_data"].get("segmented_image"), self.segmented_view.property("image_label"))
        self.display_image(result["final_data"].get("image_input"), self.final_original_view.property("image_label"))
        # Bottom row shows detection on ORIGINAL image (not dehazed)
        self.display_image(result["final_data"].get("segmented_image_original"), self.final_result_view.property("image_label"))
        # Store both detection results for viewers
        self.segmented_image_dehazed = result["final_data"].get("segmented_image")
        self.segmented_image_original = result["final_data"].get("segmented_image_original") or result["final_data"].get("image_input")
        
        # Update metrics
        for stage_result in result["stages"]:
            stage_name = stage_result["stage_name"]
            data = stage_result["data"]
            
            if stage_name == "Smog Classification":
                classification = data.get("classification", {})
                
                # Get real processing time from classification
                proc_time = classification.get("processing_time_ms", 0)
                total_time += proc_time
                
                confidence = classification.get('confidence', 0)
                prob_smog = classification.get('probability_smog', 0)
                prob_clear = classification.get('probability_clear', 0)
                
                self.update_metric_row(self.cnn_metrics, "Confidence", f"{confidence*100:.1f}%")
                self.update_metric_row(self.cnn_metrics, "Probability Smog", f"{prob_smog*100:.1f}%")
                self.update_metric_row(self.cnn_metrics, "Probability Clear", f"{prob_clear*100:.1f}%")
                self.update_metric_row(self.cnn_metrics, "Processing Time", f"{proc_time}ms")
                
                # Predicted Class Logic
                pred_class = classification.get('class', 'Unknown').upper()
                self.update_metric_row(self.cnn_metrics, "Predicted Class", pred_class)
                
                # Color code the Predicted Class label
                class_row_label = self.cnn_metrics.property("metric_widgets")["Predicted Class"].property("value_label")
                if "SMOG" in pred_class:
                    class_row_label.setStyleSheet("color: #EF4444; font-weight: bold; font-size: 13px;")
                else:
                    class_row_label.setStyleSheet("color: #10B981; font-weight: bold; font-size: 13px;")
            
            elif stage_name in ("DCP Dehazing", "CLAHE Dehazing"):
                # Handle DCP metrics
                dcp = data.get("dcp_metrics", {})
                if dcp:
                    # Get real processing time from DCP metrics
                    proc_time = dcp.get("processing_time_ms", 0)
                    total_time += proc_time
                    
                    self.dcp_metrics.setVisible(True)
                    self.clahe_metrics.setVisible(False)
                    airlight = dcp.get("airlight_rgb", (0, 0, 0))
                    self.update_metric_row(self.dcp_metrics, "Transmission Map", f"{dcp.get('transmission_map', 0):.3f}")
                    self.update_metric_row(self.dcp_metrics, "Dark Channel",     f"{dcp.get('dark_channel', 0):.3f}")
                    self.update_metric_row(self.dcp_metrics, "Airlight RGB",     f"({airlight[0]}, {airlight[1]}, {airlight[2]})")
                    self.update_metric_row(self.dcp_metrics, "Processing Time",  f"{proc_time}ms")
                
                # Handle CLAHE metrics
                clahe = data.get("clahe_metrics", {})
                if clahe:
                    # Get real processing time from CLAHE metrics
                    proc_time = clahe.get("processing_time_ms", 0)
                    total_time += proc_time
                    
                    self.dcp_metrics.setVisible(False)
                    self.clahe_metrics.setVisible(True)
                    self.update_metric_row(self.clahe_metrics, "Processing Time",  f"{proc_time}ms")
            
            elif stage_name == "HOG+SVM Object Detection":
                hog = data.get("hog_metrics", {})
                if hog:
                    # Get real processing time from HOG metrics
                    proc_time = hog.get("processing_time_ms", 0)
                    total_time += proc_time

                    self.update_detection_metrics_title("HOG + SVM")
                    self.hog_svm_metrics.setVisible(True)
                    self.yolo_metrics.setVisible(False)
                    self.rcnn_metrics.setVisible(False)
                    
                    self.update_comparison_metric_row(self.hog_svm_metrics, "People Detected", 
                                                     str(hog.get("people_count_dehazed", 0)),
                                                     str(hog.get("people_count_original", 0)))
                    self.update_comparison_metric_row(self.hog_svm_metrics, "Vehicles Detected",
                                                     str(hog.get("car_count_dehazed", 0)),
                                                     str(hog.get("car_count_original", 0)))
                    self.update_comparison_metric_row(self.hog_svm_metrics, "Total Detections",
                                                     str(hog.get("total_detections_dehazed", 0)),
                                                     str(hog.get("total_detections_original", 0)))

            elif stage_name == "YOLO Object Detection":
                yolo = data.get("yolo_metrics", {})
                if yolo:
                    proc_time = yolo.get("processing_time_ms", 0)
                    total_time += proc_time

                    # Ensure the YOLO metrics box is visible
                    self.update_detection_metrics_title("YOLO")
                    self.yolo_metrics.setVisible(True)
                    self.hog_svm_metrics.setVisible(False)
                    self.rcnn_metrics.setVisible(False)

                    self.update_comparison_metric_row(self.yolo_metrics, "People Detected",
                                                     str(yolo.get("people_count_dehazed", 0)),
                                                     str(yolo.get("people_count_original", 0)))
                    self.update_comparison_metric_row(self.yolo_metrics, "Vehicles Detected",
                                                     str(yolo.get("car_count_dehazed", 0)),
                                                     str(yolo.get("car_count_original", 0)))
                    self.update_comparison_metric_row(self.yolo_metrics, "Total Detections",
                                                     str(yolo.get("total_detections_dehazed", 0)),
                                                     str(yolo.get("total_detections_original", 0)))
                else:
                    # Hide YOLO box if no metrics
                    self.yolo_metrics.setVisible(False)

            elif stage_name == "RCNN Object Detection":
                rcnn = data.get("rcnn_metrics", {})
                if rcnn:
                    proc_time = rcnn.get("processing_time_ms", 0)
                    total_time += proc_time

                    # Ensure RCNN metrics box is visible
                    self.update_detection_metrics_title("RCNN")
                    self.rcnn_metrics.setVisible(True)
                    self.hog_svm_metrics.setVisible(False)
                    self.yolo_metrics.setVisible(False)

                    self.update_comparison_metric_row(self.rcnn_metrics, "People Detected",
                                                     str(rcnn.get("people_count_dehazed", 0)),
                                                     str(rcnn.get("people_count_original", 0)))
                    self.update_comparison_metric_row(self.rcnn_metrics, "Vehicles Detected",
                                                     str(rcnn.get("car_count_dehazed", 0)),
                                                     str(rcnn.get("car_count_original", 0)))
                    self.update_comparison_metric_row(self.rcnn_metrics, "Total Detections",
                                                     str(rcnn.get("total_detections_dehazed", 0)),
                                                     str(rcnn.get("total_detections_original", 0)))
                else:
                    self.rcnn_metrics.setVisible(False)
            
            elif stage_name == "Mask Generation":
                mask_data = data.get("mask_metrics", {})
                if mask_data:
                    proc_time = mask_data.get("processing_time_ms", 0)
                    total_time += proc_time
                    
                    # Store masks for viewing/download
                    self.mask_dehazed = data.get("mask_dehazed")
                    self.mask_original = data.get("mask_original")
                    
                    # Enable View Masks button
                    self.view_masks_btn.setEnabled(True)
        
        self.total_time_label.property("value_label").setText(f"{total_time}ms")
        self.total_time_label.setVisible(True)
        
        if not isinstance(self.sender(), VideoWorker):
            self.on_processing_finished()
    
    def update_metric_row(self, box, label, value):
        """Helper to update a specific metric row."""
        metric_widgets = box.property("metric_widgets")
        if label in metric_widgets:
            metric_widgets[label].property("value_label").setText(value)
    
    def update_comparison_metric_row(self, box, label, dehazed_value, original_value):
        """Helper to update a comparison metric row with both dehazed and original values."""
        metric_widgets = box.property("metric_widgets")
        if label in metric_widgets:
            metric_widgets[label]["dehazed"].setText(dehazed_value)
            metric_widgets[label]["original"].setText(original_value)
    
    def on_processing_finished(self):
        """Reset button state after processing."""
        self.process_button.setEnabled(True)
        self.process_button.setText("Process")
        self.detector_dropdown.setVisible(True)           
    
    def on_processing_error(self, error_msg: str):
        """Show error message."""
        QMessageBox.critical(self, "Error", error_msg)
        self.on_processing_finished()
    
    def display_image(self, image_data, label_widget):
        """Display image data without permanently downscaling it."""
        if isinstance(image_data, str):
            pixmap = QPixmap(image_data)
        elif isinstance(image_data, np.ndarray):
            h, w, ch = image_data.shape
            bytes_per_line = ch * w
            # Note: Ensure it is RGB for QImage
            q_img = QImage(image_data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        elif isinstance(image_data, Image.Image):
            # Efficient PIL to QPixmap conversion
            im2 = image_data.convert("RGBA")
            data = im2.tobytes("raw", "RGBA")
            q_img = QImage(data, image_data.width, image_data.height, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_img)
        else:
            return

        target_size = label_widget.size()
        if not target_size.isValid() or target_size.width() <= 0 or target_size.height() <= 0:
            target_size = label_widget.minimumSize()

        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label_widget.setPixmap(scaled_pixmap)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def closeEvent(self, event):
        """Ensure worker threads are stopped on exit."""
        if self.video_worker:
            self.video_worker.stop()
        event.accept()
    
    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F9FAFB; }
            QWidget#metricsPanel { background-color: #FFFFFF; border-left: 1px solid #E5E7EB; }
            
            QLabel#h1 { font-size: 28px; font-weight: 800; color: #111827; }
            QLabel#h2 { font-size: 20px; font-weight: 700; color: #111827; margin-bottom: 5px; }
            QLabel#h3 { font-size: 15px; font-weight: 700; color: #111827; }
            QMainWindow { background-color: #F9FAFB; }
            QWidget#metricsPanel { background-color: #FFFFFF; border-left: 1px solid #E5E7EB; }
            
            /* Target all GroupBoxes to ensure no borders leak through */
            QGroupBox { 
                border: none; 
                font-weight: 700; 
                color: #4B5563; 
                padding-top: 20px; 
            }

            /* Ensure internal metric rows don't have borders */
            QWidget#totalBox { 
                background-color: #F3F4F6; 
                border-radius:8px; 
                border: none;
                padding: 10px;
                margin-top: 20px;
            }

            QLabel { border: none; }
                           
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 8px 20px;
                font-weight: 600;
                color: #374151;
            }
            QPushButton:hover { background-color: #F3F4F6; }
            
            QGroupBox { font-weight: 700; color: #4B5563; border: none; padding-top: 20px; }
            
            QLabel#imageView { 
                background-color: #E5E7EB; 
                border-radius: 12px; 
                border: 1px solid #D1D5DB;
            }
            
            QLabel#imageSubTitle { color: #6B7280; font-size: 12px; font-weight: 500; margin-top: 5px; }
            
            QLabel#stageLabel { color: #9CA3AF; font-size: 10px; font-weight: 600; }
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 6px 20px;
                font-weight: 600;
                color: #374151;
                min-width: 160px;
            }
            QComboBox:hover { background-color: #F3F4F6; }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                selection-background-color: #F3F4F6;
                color: #374151;
            }
                        
        """)


def main():
    app = QApplication(sys.argv)
    window = SmogVisionGUI()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
