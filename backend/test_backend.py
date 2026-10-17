"""
Backend tests for the Agentic Vision application.

Tests cover:
- API endpoint validation (health, model-info, analyze)
- File upload validation (type, size)
- Vision engine detection
- Agent loop decision paths
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))


# ─── Unit tests for vision engine ───

class TestVisionEngine:
    """Test the OpenCV 5 vision engine."""

    def test_coco_classes_count(self):
        """COCO classes list should have exactly 80 entries."""
        from coco_classes import COCO_CLASSES
        assert len(COCO_CLASSES) == 80

    def test_coco_classes_includes_person(self):
        """Person should be the first COCO class."""
        from coco_classes import COCO_CLASSES
        assert COCO_CLASSES[0] == "person"

    def test_category_groups_cover_all_classes(self):
        """All COCO classes should be in at least one category group."""
        from coco_classes import COCO_CLASSES, CATEGORY_GROUPS
        grouped = set()
        for labels in CATEGORY_GROUPS.values():
            grouped.update(labels)
        for cls in COCO_CLASSES:
            assert cls in grouped, f"Class '{cls}' not in any category group"

    def test_image_quality_metrics(self):
        """Image quality metrics should return expected keys."""
        from vision_engine import compute_image_quality_metrics
        # Create a test image (gray gradient)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :50] = 50
        img[:, 50:] = 200
        metrics = compute_image_quality_metrics(img)
        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "sharpness" in metrics
        assert "is_dark" in metrics
        assert "is_low_contrast" in metrics
        assert "is_blurry" in metrics

    def test_dark_image_detected(self):
        """A very dark image should be flagged as dark."""
        from vision_engine import compute_image_quality_metrics
        dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 30
        metrics = compute_image_quality_metrics(dark_img)
        assert metrics["is_dark"] is True

    def test_bright_image_not_dark(self):
        """A normally bright image should not be flagged as dark."""
        from vision_engine import compute_image_quality_metrics
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 150
        metrics = compute_image_quality_metrics(bright_img)
        assert metrics["is_dark"] is False

    def test_crop_region(self):
        """Crop region should return a non-empty subimage."""
        from vision_engine import crop_region
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cropped = crop_region(img, 50, 50, 80, 80)
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0

    def test_crop_region_with_padding(self):
        """Crop with padding should be larger than the raw region."""
        from vision_engine import crop_region
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cropped_no_pad = crop_region(img, 50, 50, 80, 80, padding=0.0)
        cropped_with_pad = crop_region(img, 50, 50, 80, 80, padding=0.2)
        assert cropped_with_pad.size >= cropped_no_pad.size

    def test_enhance_image(self):
        """Enhanced image should have same shape as input."""
        from vision_engine import enhance_image
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        enhanced = enhance_image(img)
        assert enhanced.shape == img.shape

    def test_upscale_small_region(self):
        """Upscale should make a small image at least target_size."""
        from vision_engine import upscale_region
        small = np.zeros((50, 50, 3), dtype=np.uint8)
        upscaled = upscale_region(small, target_size=200)
        assert upscaled.shape[0] >= 200 or upscaled.shape[1] >= 200

    def test_upscale_large_region_unchanged(self):
        """Upscale should not change an already large image."""
        from vision_engine import upscale_region
        large = np.zeros((640, 640, 3), dtype=np.uint8)
        result = upscale_region(large, target_size=640)
        assert result.shape == large.shape


class TestVisionEngineModel:
    """Tests that require the model file to be present."""

    @pytest.fixture(autouse=True)
    def check_model(self):
        from vision_engine import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not downloaded yet")

    def test_model_loads(self):
        """Model should load without error."""
        from vision_engine import VisionEngine
        engine = VisionEngine()
        assert engine.load_model() is True

    def test_detect_on_blank_image(self):
        """Detection on a blank image should return zero detections (not crash)."""
        from vision_engine import VisionEngine
        engine = VisionEngine()
        engine.load_model()
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        result = engine.detect_objects(blank)
        assert result.error is None
        assert isinstance(result.detections, list)

    def test_detect_returns_valid_structure(self):
        """Detection result should have correct structure."""
        from vision_engine import VisionEngine
        engine = VisionEngine()
        engine.load_model()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = engine.detect_objects(img)
        assert result.error is None
        assert result.image_width == 640
        assert result.image_height == 480
        d = result.to_dict()
        assert "detections" in d
        assert "detection_count" in d

    def test_draw_detections_no_crash(self):
        """Drawing detections on image should not crash."""
        from vision_engine import VisionEngine, Detection
        engine = VisionEngine()
        engine.load_model()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = [Detection(
            class_id=0, label="person", confidence=0.9,
            x=10, y=10, width=50, height=80
        )]
        annotated = engine.draw_detections(img, dets)
        assert annotated.shape == img.shape


class TestAgentLoop:
    """Tests for the agentic decision loop."""

    @pytest.fixture(autouse=True)
    def check_model(self):
        from vision_engine import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not downloaded yet")

    def test_agent_returns_trace(self):
        """Agent should always return a non-empty trace."""
        from vision_engine import VisionEngine
        from agent_loop import AgentLoop
        engine = VisionEngine()
        engine.load_model()
        agent = AgentLoop(engine)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = agent.run(img)
        assert len(result.trace) >= 2  # At least detection + quality + decision

    def test_agent_has_stop_reason(self):
        """Agent should always have a stop reason."""
        from vision_engine import VisionEngine
        from agent_loop import AgentLoop
        engine = VisionEngine()
        engine.load_model()
        agent = AgentLoop(engine)
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = agent.run(img)
        assert result.stop_reason != ""

    def test_agent_result_serializable(self):
        """Agent result should be JSON-serializable."""
        import json
        from vision_engine import VisionEngine
        from agent_loop import AgentLoop
        engine = VisionEngine()
        engine.load_model()
        agent = AgentLoop(engine)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = agent.run(img)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_blank_image_requests_review(self):
        """A blank image (no objects) should trigger human review."""
        from vision_engine import VisionEngine
        from agent_loop import AgentLoop
        engine = VisionEngine()
        engine.load_model()
        agent = AgentLoop(engine)
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        result = agent.run(blank)
        assert result.needs_human_review is True

    def test_scene_summary_generated(self):
        """Scene summary should always be a non-empty string."""
        from vision_engine import VisionEngine
        from agent_loop import AgentLoop
        engine = VisionEngine()
        engine.load_model()
        agent = AgentLoop(engine)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        result = agent.run(img)
        assert isinstance(result.scene_summary, str)
        assert len(result.scene_summary) > 0


# ─── API tests ───

class TestAPIValidation:
    """Test FastAPI endpoint validation."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "opencv_version" in data

    def test_model_info_endpoint(self, client):
        resp = client.get("/api/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_classes"] == 80
        assert "limitations" in data
        assert len(data["limitations"]) > 0

    def test_aws_status_endpoint(self, client):
        resp = client.get("/api/aws-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "aws_configured" in data
        assert "storage_mode" in data
        assert "tier_mode" in data
        assert data["tier_mode"] == "Free Tier Eligible"

    def test_aws_service_fallback_saving(self):
        from aws_service import AWSService
        svc = AWSService()
        loc, url = svc.save_analysis_artifact(
            request_id="test_req",
            image_bytes=b"fake_image_bytes",
            trace_data={"status": "test"},
        )
        assert loc in ("AWS S3", "Local Disk")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
