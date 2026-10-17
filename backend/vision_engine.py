"""
Vision Engine — OpenCV 5 based object detection and image processing.

Uses OpenCV 5's DNN module to run YOLOv8n (ONNX) inference on CPU.
Provides image processing tools: crop, enhance, resize, histogram EQ.
"""

import cv2
import numpy as np
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from coco_classes import COCO_CLASSES

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")
INPUT_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_NMS_THRESHOLD = 0.45


@dataclass
class Detection:
    """A single detected object with bounding box and confidence."""
    class_id: int
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    """Result of running object detection on an image."""
    detections: list[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    model_name: str = "YOLOv8n (COCO 80-class)"
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
            "conf_threshold": self.conf_threshold,
            "detection_count": len(self.detections),
            "error": self.error,
        }


class VisionEngine:
    """OpenCV 5 DNN-based object detection engine."""

    def __init__(self):
        self._net = None
        self._model_loaded = False

    def load_model(self) -> bool:
        """Load the YOLOv8n ONNX model using OpenCV 5 DNN."""
        if self._model_loaded:
            return True

        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at {MODEL_PATH}")
            return False

        try:
            self._net = cv2.dnn.readNetFromONNX(MODEL_PATH)
            # Force CPU backend for compatibility (no GPU required)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._model_loaded = True
            logger.info(f"Model loaded from {MODEL_PATH}")
            logger.info(f"OpenCV version: {cv2.__version__}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def detect_objects(
        self,
        image: np.ndarray,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ) -> DetectionResult:
        """
        Run YOLOv8n object detection on an image using OpenCV 5 DNN.

        Args:
            image: BGR numpy array from cv2.imread
            conf_threshold: minimum confidence to keep a detection
            nms_threshold: IoU threshold for non-maximum suppression

        Returns:
            DetectionResult with list of detections
        """
        if not self._model_loaded:
            return DetectionResult(error="Model not loaded")

        h, w = image.shape[:2]
        result = DetectionResult(
            image_width=w,
            image_height=h,
            conf_threshold=conf_threshold,
        )

        try:
            # Preprocess: create blob for YOLOv8 (640x640, normalized)
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=1.0 / 255.0,
                size=(INPUT_SIZE, INPUT_SIZE),
                swapRB=True,
                crop=False,
            )
            self._net.setInput(blob)

            # Forward pass
            outputs = self._net.forward()

            # Parse YOLOv8 output: shape (1, 84, N) -> (N, 84)
            # 84 = 4 box coords + 80 class scores
            outputs = outputs.squeeze(0).T

            x_factor = w / INPUT_SIZE
            y_factor = h / INPUT_SIZE

            boxes = []
            confidences = []
            class_ids = []

            for row in outputs:
                class_scores = row[4:]
                max_score = float(np.max(class_scores))

                if max_score >= conf_threshold:
                    class_id = int(np.argmax(class_scores))
                    cx, cy, pw, ph = row[0], row[1], row[2], row[3]

                    # Convert center format to top-left corner format
                    left = int((cx - 0.5 * pw) * x_factor)
                    top = int((cy - 0.5 * ph) * y_factor)
                    box_w = int(pw * x_factor)
                    box_h = int(ph * y_factor)

                    # Clamp to image bounds
                    left = max(0, left)
                    top = max(0, top)
                    box_w = min(box_w, w - left)
                    box_h = min(box_h, h - top)

                    boxes.append([left, top, box_w, box_h])
                    confidences.append(max_score)
                    class_ids.append(class_id)

            # Non-maximum suppression
            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes, confidences, conf_threshold, nms_threshold
                )
                if len(indices) > 0:
                    for i in indices.flatten():
                        label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else f"class_{class_ids[i]}"
                        det = Detection(
                            class_id=class_ids[i],
                            label=label,
                            confidence=round(confidences[i], 4),
                            x=boxes[i][0],
                            y=boxes[i][1],
                            width=boxes[i][2],
                            height=boxes[i][3],
                        )
                        result.detections.append(det)

            # Sort by confidence descending
            result.detections.sort(key=lambda d: d.confidence, reverse=True)

        except Exception as e:
            logger.exception("Detection failed")
            result.error = str(e)

        return result

    def draw_detections(
        self, image: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        annotated = image.copy()
        colors = self._generate_colors(len(COCO_CLASSES))

        for det in detections:
            color = colors[det.class_id % len(colors)]
            # Draw box
            cv2.rectangle(
                annotated,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color,
                2,
            )
            # Draw label background
            label_text = f"{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (det.x, det.y - th - 8),
                (det.x + tw + 4, det.y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (det.x + 2, det.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated

    @staticmethod
    def _generate_colors(n: int) -> list[tuple[int, int, int]]:
        """Generate n visually distinct colors using HSV."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color_hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color_bgr))
        return colors


# --- Image processing tools for the agent ---

def crop_region(image: np.ndarray, x: int, y: int, w: int, h: int,
                padding: float = 0.15) -> np.ndarray:
    """
    Crop a region from the image with padding around it.
    Padding is a fraction of the region size added on each side.
    """
    img_h, img_w = image.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return image[y1:y2, x1:x2]


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image using OpenCV 5:
    - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Apply sharpening kernel
    Both are substantive OpenCV operations.
    """
    # Convert to LAB color space for CLAHE on luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # CLAHE on luminance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    # Merge and convert back
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def upscale_region(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Upscale a small cropped region so it's large enough for the detector.
    Uses INTER_CUBIC for better quality than simple nearest-neighbor.
    """
    h, w = image.shape[:2]
    if h >= target_size and w >= target_size:
        return image

    scale = max(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def compute_image_quality_metrics(image: np.ndarray) -> dict:
    """
    Compute simple image quality metrics using OpenCV.
    Used by the agent to decide if enhancement might help.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness: mean pixel value
    brightness = float(np.mean(gray))

    # Contrast: standard deviation of pixel values
    contrast = float(np.std(gray))

    # Sharpness: variance of Laplacian (blur detection)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(laplacian_var, 2),
        "is_dark": brightness < 80,
        "is_low_contrast": contrast < 40,
        "is_blurry": laplacian_var < 100,
    }
