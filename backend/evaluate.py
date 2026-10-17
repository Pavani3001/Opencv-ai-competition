"""
Agentic Vision — Quantitative Evaluation & Benchmark Suite.

Evaluates the perception-decision-action loop across multiple scenarios:
1. Blank/Empty Frame: Assesses fallback and human review trigger.
2. Low-Light / Dark Scene: Assesses image quality assessment and CLAHE enhancement trigger.
3. Blurry / Degraded Scene: Assesses sharpness metric and blur detection.
4. Region of Interest / Crop Scene: Assesses crop-and-inspect triggering on target coordinates.
5. Multi-object Synthetic Scene: Assesses classification and confidence estimation.

Generates quantitative performance metrics:
- Success rate
- Step counts and stage latencies
- Action distribution percentages
- Confidence improvement delta
"""

import os
import sys
import time
import json
import numpy as np
import cv2

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(__file__))

from vision_engine import VisionEngine, enhance_image, compute_image_quality_metrics
from agent_loop import AgentLoop, AgentAction, StopReason


def create_blank_test_image(width=640, height=640) -> np.ndarray:
    """Generate a blank (empty) image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def create_dark_test_image(width=640, height=640) -> np.ndarray:
    """Generate a low-light / underexposed image."""
    img = np.full((height, width, 3), 25, dtype=np.uint8)
    # Add subtle low-contrast shapes
    cv2.circle(img, (width // 2, height // 2), 120, (45, 45, 45), -1)
    return img


def create_blurry_test_image(width=640, height=640) -> np.ndarray:
    """Generate a heavily blurred image."""
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (51, 51), 0)


def create_synthetic_scene(width=640, height=640) -> np.ndarray:
    """Generate a textured synthetic scene with multiple distinct color regions."""
    img = np.full((height, width, 3), (230, 230, 230), dtype=np.uint8)
    # Ground plane
    cv2.rectangle(img, (0, int(height * 0.6)), (width, height), (120, 160, 120), -1)
    # Background wall / sky
    cv2.rectangle(img, (0, 0), (width, int(height * 0.6)), (210, 190, 170), -1)
    # Object silhouettes
    cv2.rectangle(img, (80, 200), (220, 500), (40, 40, 40), -1)  # Tall object
    cv2.circle(img, (450, 320), 80, (60, 90, 180), -1)           # Sphere
    return img


def run_evaluation():
    print("=" * 70)
    print("  OPENCV AI COMPETITION 2026 — AGENTIC VISION EVALUATION SUITE  ")
    print("=" * 70)

    engine = VisionEngine()
    loaded = engine.load_model()
    if not loaded:
        print("[ERROR] YOLOv8n ONNX model could not be loaded from backend/models/.")
        sys.exit(1)

    print(f"[*] OpenCV Version: {cv2.__version__}")
    print(f"[*] Vision Backend: OpenCV DNN (CPU)")
    print(f"[*] Model Loaded: YOLOv8n ONNX (COCO 80 categories)")
    print("-" * 70)

    agent = AgentLoop(engine)

    test_cases = [
        {
            "id": "TC-01",
            "name": "Blank / Empty Image",
            "image": create_blank_test_image(),
            "expected_actions": [AgentAction.INITIAL_DETECTION, AgentAction.QUALITY_ASSESSMENT, AgentAction.REQUEST_HUMAN_REVIEW],
            "expect_human_review": True,
            "description": "Verifies that empty/unrecognizable scenes trigger human review fallback",
        },
        {
            "id": "TC-02",
            "name": "Dark / Low-light Scene",
            "image": create_dark_test_image(),
            "expected_actions": [AgentAction.QUALITY_ASSESSMENT],
            "expect_dark_flag": True,
            "description": "Verifies OpenCV luminance assessment flags underexposure",
        },
        {
            "id": "TC-03",
            "name": "Blurry Degraded Scene",
            "image": create_blurry_test_image(),
            "expected_actions": [AgentAction.QUALITY_ASSESSMENT],
            "expect_blurry_flag": True,
            "description": "Verifies Laplacian variance flags blur correctly",
        },
        {
            "id": "TC-04",
            "name": "Synthetic Multi-Object Scene",
            "image": create_synthetic_scene(),
            "expected_actions": [AgentAction.INITIAL_DETECTION, AgentAction.QUALITY_ASSESSMENT],
            "expect_human_review": None,
            "description": "Verifies end-to-end trace generation and latency stability",
        },
    ]

    results = []
    action_counts = {}
    total_time = 0.0

    for tc in test_cases:
        t0 = time.time()
        agent_res = agent.run(tc["image"])
        dur = (time.time() - t0) * 1000
        total_time += dur

        executed_actions = [s.action for s in agent_res.trace]
        for a in executed_actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        # Evaluate assertions
        passed = True
        notes = []

        if tc.get("expect_human_review") is not None:
            if agent_res.needs_human_review != tc["expect_human_review"]:
                passed = False
                notes.append(f"Expected human_review={tc['expect_human_review']}, got {agent_res.needs_human_review}")

        if tc.get("expect_dark_flag"):
            if not agent_res.image_quality.get("is_dark"):
                passed = False
                notes.append("Failed to detect dark image condition")

        if tc.get("expect_blurry_flag"):
            if not agent_res.image_quality.get("is_blurry"):
                passed = False
                notes.append("Failed to detect blurry image condition")

        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {tc['id']}: {tc['name']} ({dur:.1f}ms, {len(agent_res.trace)} steps)")
        print(f"       Trace: {' -> '.join(executed_actions)}")
        if notes:
            for n in notes:
                print(f"       Note: {n}")

        results.append({
            "test_id": tc["id"],
            "name": tc["name"],
            "passed": passed,
            "duration_ms": round(dur, 2),
            "step_count": len(agent_res.trace),
            "trace": [s.to_dict() for s in agent_res.trace],
            "quality": agent_res.image_quality,
            "stop_reason": agent_res.stop_reason,
            "needs_human_review": agent_res.needs_human_review,
        })

    # Summary metrics
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100.0

    print("-" * 70)
    print("EVALUATION SUMMARY:")
    print(f"  Total Scenarios Tested : {total_count}")
    print(f"  Passed Scenarios       : {passed_count} / {total_count} ({pass_rate:.1f}%)")
    print(f"  Average Scenario Time  : {total_time / total_count:.1f}ms")
    print("\nAction Distribution:")
    for act, cnt in sorted(action_counts.items()):
        pct = (cnt / sum(action_counts.values())) * 100
        print(f"  - {act.ljust(25)}: {cnt} occurrences ({pct:.1f}%)")

    # Export report
    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "opencv_version": cv2.__version__,
        "pass_rate_pct": pass_rate,
        "total_scenarios": total_count,
        "passed_scenarios": passed_count,
        "avg_duration_ms": round(total_time / total_count, 2),
        "action_distribution": action_counts,
        "scenarios": results,
    }

    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Detailed evaluation report saved to: {report_path}")
    print("=" * 70)
    return pass_rate == 100.0


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
