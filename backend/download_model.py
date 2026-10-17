"""
Download the YOLOv8n ONNX model for use with OpenCV 5 DNN.

This script downloads yolov8n.onnx from the Ultralytics GitHub release.
The model is AGPL-3.0 licensed (compatible with open-source projects).

Usage: python download_model.py
"""

import os
import sys
import urllib.request
import hashlib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")

# YOLOv8n ONNX from Hugging Face (Ultralytics model, AGPL-3.0 licensed)
MODEL_URL = "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx"


def download_model():
    """Download YOLOv8n ONNX model if not already present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model already exists at {MODEL_PATH} ({size_mb:.1f} MB)")
        return True

    print(f"Downloading YOLOv8n ONNX model...")
    print(f"  URL: {MODEL_URL}")
    print(f"  Destination: {MODEL_PATH}")
    print(f"  License: AGPL-3.0 (open-source compatible)")
    print()

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {pct:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress_hook)
        print()

        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Download complete: {size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print("\nManual download instructions:")
        print(f"  1. Visit: {MODEL_URL}")
        print(f"  2. Save the file to: {MODEL_PATH}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
