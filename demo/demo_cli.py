"""
Command-line Demonstration for Agentic Vision.

Runs the perception-decision-action loop on an image and prints
the agent's step-by-step reasoning trace in the console.

Usage:
    python demo_cli.py [path_to_image]
    python demo_cli.py --all
"""

import os
import sys
import argparse
import cv2

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from vision_engine import VisionEngine, Detection
from agent_loop import AgentLoop


def format_step_icon(action: str) -> str:
    icons = {
        "initial_detection": "[DETECT] ",
        "quality_assessment": "[ASSESS] ",
        "enhance_and_redetect": "[ENHANCE]",
        "crop_and_inspect": "[INSPECT]",
        "accept_result": "[ACCEPT] ",
        "request_human_review": "[REVIEW] ",
    }
    return icons.get(action, "[STEP]   ")


def run_demo(image_path: str):
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return

    print("=" * 75)
    print(f"  AGENTIC VISION — DEMONSTRATION RUN: {os.path.basename(image_path)}")
    print("=" * 75)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Could not decode image at {image_path}")
        return

    h, w = img.shape[:2]
    print(f"[*] Input Resolution: {w}x{h} px")

    engine = VisionEngine()
    if not engine.load_model():
        print("[!] Failed to load YOLOv8n ONNX model.")
        return

    agent = AgentLoop(engine)
    print("[*] Running Agentic Perception-Decision-Action loop...")
    print("-" * 75)

    result = agent.run(img)

    # Print execution trace
    for s in result.trace:
        icon = format_step_icon(s.action)
        print(f"{icon} Step {s.step_number}: {s.action.upper()}")
        print(f"        Reason : {s.reason}")
        if s.tool_input:
            print(f"        Input  : {s.tool_input}")
        print(f"        Result : {s.result_summary} ({s.duration_ms:.1f}ms)")
        print()

    print("-" * 75)
    print(f"[*] Agent Decision Summary: {result.stop_reason}")
    if result.needs_human_review:
        print(f"[!] Human Review Needed   : {result.human_review_reason}")

    print(f"[*] Total Execution Time  : {result.total_duration_ms:.1f}ms")
    print(f"[*] Detections Found      : {len(result.final_detections)}")

    if result.final_detections:
        print("\n    Detected Entities:")
        for idx, d in enumerate(result.final_detections, 1):
            print(f"    {idx}. {d['label'].ljust(15)} Confidence: {d['confidence']:.1%} | Bounding Box: ({d['x']}, {d['y']}, {d['width']}x{d['height']})")

    # Generate and save annotated output
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"annotated_{os.path.basename(image_path)}"
    out_path = os.path.join(output_dir, out_name)

    dets_objs = [
        Detection(
            class_id=d.get("class_id", 0),
            label=d["label"],
            confidence=d["confidence"],
            x=d["x"],
            y=d["y"],
            width=d["width"],
            height=d["height"],
        )
        for d in result.final_detections
    ]
    annotated = engine.draw_detections(img, dets_objs)
    cv2.imwrite(out_path, annotated)
    print(f"\n[+] Annotated output saved to: {out_path}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Agentic Vision CLI Demonstration")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image")
    parser.add_argument("--all", action="store_true", help="Run demo on all samples in demo/samples/")
    args = parser.parse_args()

    samples_dir = os.path.join(os.path.dirname(__file__), "samples")

    if args.all or args.image is None:
        sample_files = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith(".jpg")]
        if not sample_files:
            print("[!] No sample images found. Running generate_samples.py first...")
            import generate_samples
            generate_samples.generate_indoor_workspace_sample()
            sample_files = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith(".jpg")]

        for sample_path in sample_files:
            run_demo(sample_path)
    else:
        run_demo(args.image)


if __name__ == "__main__":
    main()
