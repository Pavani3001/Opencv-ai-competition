"""
FastAPI backend for Agentic Vision application.

Endpoints:
  POST /api/analyze    — Upload an image for agentic analysis
  GET  /api/health     — Health check and model status
  GET  /api/model-info — Model capabilities and limitations
"""

import os
import io
import logging
import base64
import uuid
import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vision_engine import VisionEngine
from agent_loop import AgentLoop
from coco_classes import COCO_CLASSES, CATEGORY_GROUPS
from aws_service import AWSService

# ─── Logging setup ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Constants ───
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── App setup ───
app = FastAPI(
    title="Agentic Vision API",
    description="OpenCV 5 powered agentic scene understanding system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Engine initialization ───
engine = VisionEngine()
agent = AgentLoop(engine)
aws_service = AWSService()

MODEL_LOADED = engine.load_model()
if MODEL_LOADED:
    logger.info("Vision engine ready — model loaded successfully.")
else:
    logger.warning(
        "Model not loaded! Download yolov8n.onnx to backend/models/ first."
    )


# ─── Endpoints ───

@app.get("/api/aws-status")
async def aws_status():
    """Return AWS cloud integration status and telemetry."""
    return aws_service.get_status()

@app.get("/api/health")
async def health_check():
    """Health check endpoint with model status."""
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "opencv_version": cv2.__version__,
        "model_name": "YOLOv8n (COCO 80-class)" if MODEL_LOADED else None,
    }


@app.get("/api/model-info")
async def model_info():
    """Return model capabilities and limitations."""
    return {
        "model_name": "YOLOv8n (nano)",
        "dataset": "COCO (Common Objects in Context)",
        "num_classes": len(COCO_CLASSES),
        "classes": COCO_CLASSES,
        "category_groups": CATEGORY_GROUPS,
        "license": "AGPL-3.0 (compatible with open-source projects)",
        "input_size": "640x640",
        "backend": f"OpenCV {cv2.__version__} DNN module",
        "limitations": [
            "Can only detect the 80 COCO categories listed above.",
            "Cannot identify specific individuals or read text.",
            "Cannot recognize plant species — only 'potted plant' category.",
            "Small or heavily occluded objects may be missed.",
            "Performance varies with image quality, lighting, and angle.",
            "Runs on CPU — inference takes ~1-3 seconds per image.",
        ],
    }


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image for full agentic analysis.

    The agent will:
    1. Detect objects using YOLOv8n via OpenCV 5 DNN
    2. Assess image quality
    3. Decide whether to enhance or crop-inspect based on results
    4. Compare initial and follow-up results
    5. Return detections, scene summary, and full execution trace
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Analyze request: {file.filename}")

    # ─── Validate ───
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place yolov8n.onnx in backend/models/",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # Read file
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f} MB). "
                   f"Maximum: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.",
        )

    # Decode image
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    logger.info(
        f"[{request_id}] Image loaded: {image.shape[1]}x{image.shape[0]}"
    )

    # ─── Run agent ───
    try:
        result = agent.run(image)
    except Exception as e:
        logger.exception(f"[{request_id}] Agent error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # ─── Generate annotated image ───
    from vision_engine import Detection as Det
    final_dets = [
        Det(**{k: v for k, v in d.items() if k != "class_id" and k in Det.__dataclass_fields__},
            class_id=d.get("class_id", 0))
        for d in result.final_detections
    ]
    annotated = engine.draw_detections(image, final_dets)

    # Encode annotated image as base64 JPEG
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    # Encode original image as base64 JPEG
    _, orig_buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    original_b64 = base64.b64encode(orig_buffer).decode("utf-8")

    # Persist artifact via AWS service (S3 or local fallback)
    storage_loc, cloud_url = aws_service.save_analysis_artifact(
        request_id=request_id,
        image_bytes=buffer.tobytes(),
        trace_data=result.to_dict(),
    )

    logger.info(
        f"[{request_id}] Analysis complete: {len(result.final_detections)} detections, "
        f"{len(result.trace)} steps, {result.total_duration_ms:.0f}ms, stored to {storage_loc}"
    )

    return JSONResponse({
        "request_id": request_id,
        "filename": file.filename,
        "image_original": f"data:image/jpeg;base64,{original_b64}",
        "image_annotated": f"data:image/jpeg;base64,{annotated_b64}",
        "analysis": result.to_dict(),
        "opencv_version": cv2.__version__,
        "storage": {
            "location": storage_loc,
            "cloud_url": cloud_url,
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
