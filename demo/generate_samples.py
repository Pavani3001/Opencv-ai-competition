"""
Generate synthetic benchmark and sample demo images for Agentic Vision.

Saves demo images into demo/samples/ for immediate demonstration and testing.
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_indoor_workspace_sample():
    """Generates a workspace/room environment with identifiable objects."""
    img = np.full((640, 640, 3), (240, 240, 240), dtype=np.uint8)

    # Floor and desk surface
    cv2.rectangle(img, (0, 380), (640, 640), (140, 110, 80), -1)   # Wooden desk
    cv2.rectangle(img, (0, 370), (640, 385), (100, 80, 50), -1)    # Desk edge

    # Wall with poster / painting
    cv2.rectangle(img, (180, 50), (460, 220), (200, 180, 150), -1)
    cv2.rectangle(img, (190, 60), (450, 210), (90, 140, 190), -1)
    cv2.circle(img, (320, 130), 40, (230, 210, 80), -1)

    # Monitor / Screen
    cv2.rectangle(img, (140, 180), (420, 370), (30, 30, 30), -1)     # Bezel
    cv2.rectangle(img, (155, 195), (405, 355), (180, 220, 240), -1)   # Display screen
    cv2.rectangle(img, (260, 370), (300, 430), (50, 50, 50), -1)     # Stand neck
    cv2.rectangle(img, (220, 425), (340, 440), (40, 40, 40), -1)     # Stand base

    # Keyboard & Mouse
    cv2.rectangle(img, (160, 460), (380, 510), (50, 50, 55), -1)
    cv2.ellipse(img, (440, 485), (25, 40), 0, 0, 360, (60, 60, 65), -1)

    # Cup / bottle on desk
    cv2.rectangle(img, (80, 420), (125, 520), (70, 70, 220), -1)     # Bottle
    cv2.rectangle(img, (90, 395), (115, 420), (200, 200, 200), -1)   # Cap

    # Potted plant on side
    cv2.rectangle(img, (510, 410), (580, 490), (60, 80, 150), -1)    # Pot
    cv2.circle(img, (545, 370), 50, (50, 140, 60), -1)               # Leaves
    cv2.circle(img, (515, 340), 35, (40, 120, 50), -1)
    cv2.circle(img, (575, 350), 38, (60, 160, 70), -1)

    path = os.path.join(OUTPUT_DIR, "workspace_scene.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Generated: {path}")


def generate_low_light_sample():
    """Generates a dim, underexposed environment that tests enhancement."""
    img = np.full((640, 640, 3), (30, 25, 25), dtype=np.uint8)

    # Dim furniture contours
    cv2.rectangle(img, (50, 300), (350, 580), (45, 40, 40), -1)
    cv2.circle(img, (480, 320), 100, (55, 50, 50), -1)
    cv2.rectangle(img, (200, 120), (440, 280), (60, 55, 50), -1)

    # Subtle dim light source in corner
    cv2.circle(img, (580, 60), 40, (90, 85, 75), -1)

    path = os.path.join(OUTPUT_DIR, "low_light_scene.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Generated: {path}")


def generate_blurred_sample():
    """Generates a motion-blurred sample that triggers blur assessment."""
    img = np.full((640, 640, 3), (220, 220, 220), dtype=np.uint8)
    cv2.rectangle(img, (100, 150), (300, 450), (50, 100, 200), -1)
    cv2.circle(img, (450, 300), 120, (180, 60, 60), -1)
    cv2.putText(img, "OpenCV 5 Agent", (120, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)

    # Apply horizontal motion blur
    kernel_size = 35
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred = cv2.filter2D(img, -1, kernel)

    path = os.path.join(OUTPUT_DIR, "blurred_scene.jpg")
    cv2.imwrite(path, blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Generated: {path}")


def generate_empty_sample():
    """Generates a uniform frame that triggers human review fallback."""
    img = np.full((640, 640, 3), (205, 210, 215), dtype=np.uint8)
    path = os.path.join(OUTPUT_DIR, "empty_scene.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Generated: {path}")


if __name__ == "__main__":
    print("[*] Generating sample benchmark images...")
    generate_indoor_workspace_sample()
    generate_low_light_sample()
    generate_blurred_sample()
    generate_empty_sample()
    print("[+] All sample images generated in demo/samples/")
