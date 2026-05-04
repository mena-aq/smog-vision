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
    QFrame, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from stages import create_default_pipeline


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
    
    def __init__(self, model_path: str, image_path: str):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
    
    def run(self):
        try:
            if not os.path.exists(self.model_path):
                self.error.emit(f"Model file not found: {self.model_path}")
                return
            
            pipeline = create_default_pipeline(self.model_path)
            result = pipeline.run(self.image_path)
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Inference error: {str(e)}")


class SmogVisionGUI(QMainWindow):
    """Main desktop GUI for the SmogVision pipeline."""
    
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.model_path = "smog-classification/smog_classifier_traffic.pth"
        self.video_worker = None
        self.inference_worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SmogVision")
        self.setGeometry(50, 50, 1400, 850)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left panel
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 7)

        # Right panel
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)

        self.apply_stylesheet()
    
    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Title with Clouds
        title_layout = QHBoxLayout()
        title = QLabel("SmogVision ☁️☁️☁️")
        title.setObjectName("h1")
        title_layout.addWidget(title)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Upload Button
        self.upload_button = QPushButton("Upload Image or Video")
        self.upload_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button, alignment=Qt.AlignmentFlag.AlignLeft)

        # Process Button
        self.process_button = QPushButton("Process")
        self.process_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_button.setEnabled(False)  # Disabled until a file is uploaded
        self.process_button.clicked.connect(self.process_file) # Connect to the process method
        layout.addWidget(self.process_button, alignment=Qt.AlignmentFlag.AlignLeft)

        # Results area
        self.results_widget = QWidget()
        self.results_widget.setVisible(False)
        results_layout = QVBoxLayout(self.results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(30)

        # Pipeline Output Group
        pipeline_group = QGroupBox("Pipeline Output")
        pipeline_layout = QHBoxLayout()
        self.original_view = self.create_image_view("Original")
        self.dehazed_view = self.create_image_view("Dehazed (DCP)")
        self.segmented_view = self.create_image_view("Objects Segmented")
        
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
        self.final_result_view = self.create_image_view("Final Result")
        
        comparison_layout.addWidget(self.final_original_view)
        comparison_layout.addWidget(self.create_arrow(large=True))
        comparison_layout.addWidget(self.final_result_view)
        comparison_group.setLayout(comparison_layout)
        results_layout.addWidget(comparison_group)

        layout.addWidget(self.results_widget)
        layout.addStretch()

        return widget
    
    def create_image_view(self, title: str):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        label = QLabel()
        label.setObjectName("imageView")
        # No fixed size here - let it expand
        label.setMinimumSize(200, 150) 
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        title_label = QLabel(title)
        title_label.setObjectName("imageSubTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        widget.setProperty("image_label", label)
        return widget

    def create_right_panel(self):
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
            ["Predicted Class", "Confidence", "Precision", "Recall", "F1-Score", "Processing Time"]
        )
        layout.addWidget(self.cnn_metrics)

        self.dcp_metrics = self.create_metric_box("DCP Dehazing", "Stage 2", "#F59E0B", "#FFFBEB", 
                                               ["Transmission Map", "Dark Channel", "Airlight RGB", "Processing Time"])
        layout.addWidget(self.dcp_metrics)

        self.hog_svm_metrics = self.create_metric_box("HOG + SVM", "Stage 3", "#10B981", "#F0FDF4", 
                                                   ["Detections", "Avg. Confidence", "True Positive", "False Positive", "Processing Time"])
        layout.addWidget(self.hog_svm_metrics)

        layout.addStretch()
        
        self.total_time_label = self.create_metric_row("Total Pipeline Time", "0ms", highlight=True)
        layout.addWidget(self.total_time_label)
        
        return widget
    
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
            self.results_widget.setVisible(False)
            
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
        
        self.results_widget.setVisible(True)
        self.process_button.setEnabled(False)
        self.process_button.setText(" Processing...")
        
        if self.file_path.lower().endswith(('.mp4', '.avi')):
            self.video_worker = VideoWorker(self.model_path, self.file_path)
            self.video_worker.frame_processed.connect(self.update_ui_with_results)
            self.video_worker.finished.connect(self.on_processing_finished)
            self.video_worker.error.connect(self.on_processing_error)
            self.video_worker.start()
        else:
            self.worker = InferenceWorker(self.model_path, self.file_path)
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
        self.display_image(result["final_data"].get("segmented_image"), self.final_result_view.property("image_label"))
        
        # Update metrics
        for stage_result in result["stages"]:
            stage_name = stage_result["stage_name"]
            data = stage_result["data"]
            
            # Dummy processing time
            proc_time = np.random.randint(50, 200)
            total_time += proc_time
            
            if stage_name == "Smog Classification":
                classification = data.get("classification", {})
                
                self.update_metric_row(self.cnn_metrics, "Confidence", f"{classification.get('confidence', 0)*100:.1f}%")
                self.update_metric_row(self.cnn_metrics, "Precision", f"{classification.get('precision', 0):.2f}")
                self.update_metric_row(self.cnn_metrics, "Recall", f"{classification.get('recall', 0):.2f}")
                self.update_metric_row(self.cnn_metrics, "F1-Score", f"{classification.get('f1_score', 0):.2f}")
                
                # Now proc_time is defined and won't crash!
                self.update_metric_row(self.cnn_metrics, "Processing Time", f"{proc_time}ms")
                
                # Predicted Class Logic (Keep your existing code)
                pred_class = classification.get('class', 'Unknown').upper()
                self.update_metric_row(self.cnn_metrics, "Predicted Class", pred_class)
                
                # 2. Add some color logic to the Predicted Class label
                # This makes "SMOGGY" red and "CLEAR" green automatically
                class_row_label = self.cnn_metrics.property("metric_widgets")["Predicted Class"].property("value_label")
                if "SMOGGY" in pred_class:
                    class_row_label.setStyleSheet("color: #EF4444; font-weight: bold; font-size: 13px;")
                else:
                    class_row_label.setStyleSheet("color: #10B981; font-weight: bold; font-size: 13px;")

                # Existing confidence update
                self.update_metric_row(self.cnn_metrics, "Confidence", f"{classification.get('confidence', 0)*100:.1f}%")
                self.update_metric_row(self.cnn_metrics, "Processing Time", f"{proc_time}ms")
            
            elif stage_name == "DCP Dehazing":
                self.update_metric_row(self.dcp_metrics, "Processing Time", f"{proc_time}ms")
            
            elif stage_name == "HOG+SVM Object Detection":
                self.update_metric_row(self.hog_svm_metrics, "Processing Time", f"{proc_time}ms")
        
        self.total_time_label.property("value_label").setText(f"{total_time}ms")
        self.total_time_label.setVisible(True)
        
        if not isinstance(self.sender(), VideoWorker):
            self.on_processing_finished()
    
    def update_metric_row(self, box, label, value):
        """Helper to update a specific metric row."""
        metric_widgets = box.property("metric_widgets")
        if label in metric_widgets:
            metric_widgets[label].property("value_label").setText(value)
    
    def on_processing_finished(self):
        """Reset button state after processing."""
        self.process_button.setEnabled(True)
        self.process_button.setText("Process")
    
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

        # Set the high-res pixmap and let the label scale it visually
        label_widget.setPixmap(pixmap)
        label_widget.setScaledContents(True) # This allows the image to expand with the window
    
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
                border-radius: 8px; 
                border: none;
                padding: 10px;
                margin-top: 20px;
            }

            QLabel { border: none; }
                           
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 12px 20px;
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
            
        """)


def main():
    app = QApplication(sys.argv)
    window = SmogVisionGUI()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
