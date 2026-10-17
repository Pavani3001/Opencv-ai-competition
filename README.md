# 👁️ Agentic Vision — OpenCV AI Competition 2026

[![OpenCV Version](https://img.shields.io/badge/OpenCV-5.0.0-blue.svg)](https://opencv.org/)
[![Track](https://img.shields.io/badge/Competition-Agentic%20Vision%20Award-orange.svg)](https://opencv.org/)
[![AWS Ready](https://img.shields.io/badge/AWS-Free%20Tier%20Eligible-green.svg)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen.svg)]()

An autonomous scene understanding system built for the **Agentic Vision Award** in the **OpenCV AI Competition 2026**, powered by **OpenCV 5** and **AWS**.

Unlike conventional one-shot detectors or chatbots that simply narrate static detections, **Agentic Vision** implements an active **perception–decision–action loop**. It measures visual quality, dynamically invokes OpenCV enhancement and cropping tools based on real-time evidence, compares hypotheses, and escalates to human review when appropriate.

---

## 🌟 Key Features

1. **Substantive OpenCV 5 Analysis:**
   - Neural network inference via `cv2.dnn.readNetFromONNX()` on CPU.
   - Non-Maximum Suppression with `cv2.dnn.NMSBoxes()`.
   - Luminance and contrast analysis via LAB color space conversion and `cv2.createCLAHE()`.
   - 2D directional spatial filtering with `cv2.filter2D()`.
   - Sub-pixel bicubic region upscaling with `cv2.INTER_CUBIC`.
   - Sharpness estimation via Laplacian operator variance: `cv2.Laplacian()`.

2. **Genuine Agentic Loop:**
   - **Perceive:** Runs initial detection across everyday categories (people, animals, vehicles, indoor items).
   - **Assess:** Computes brightness, contrast, and blur metrics.
   - **Decide:** Conditionally triggers CLAHE enhancement or close-up region inspection based on detection confidence and quality deficits.
   - **Act:** Crops uncertain regions, enhances low-light imagery, or requests human approval.
   - **Compare:** Rigorously contrasts pre- and post-inspection findings before finalizing the scene hypothesis.

3. **Production Cloud Architecture (AWS Free Tier):**
   - Cloud deployment container (`Dockerfile`) designed for **AWS App Runner** or **AWS ECS Fargate**.
   - Cloud persistence via **Amazon S3** for audit logs, annotated frames, and JSON decision traces.
   - **Zero Cost Guarantee:** Completely functional in local offline mode without requiring cloud credentials or incurring costs.

4. **Modern Observable Web UI:**
   - React 19 + Vite dashboard with dark aesthetics and glassmorphic panels.
   - Live annotated vs. original image comparison toggle.
   - Real-time agent execution trace timeline with millisecond durations, reasoning, and tool calls.
   - Detection confidence bars and image quality diagnostics.

---

## 📁 Repository Structure

```
├── backend/
│   ├── main.py                 # FastAPI REST server & endpoints
│   ├── vision_engine.py        # OpenCV 5 DNN, CLAHE, filters & metrics
│   ├── agent_loop.py           # Perception-Decision-Action loop controller
│   ├── aws_service.py          # Amazon S3 persistence & local fallback
│   ├── coco_classes.py         # COCO 80 categories & grouping
│   ├── download_model.py       # YOLOv8n ONNX model downloader
│   ├── test_backend.py         # 24 unit and integration tests
│   ├── evaluate.py             # Benchmark & evaluation suite
│   ├── models/                 # Pretrained ONNX model files
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main dashboard controller
│   │   ├── App.css             # Design system & dark theme styles
│   │   ├── components/
│   │   │   ├── Header.jsx          # Header with live AWS & OpenCV badges
│   │   │   ├── ImageUploader.jsx   # Drag-and-drop file upload component
│   │   │   ├── AnalysisResult.jsx  # Detections table & comparison view
│   │   │   └── AgentTrace.jsx      # Agent timeline and step visualizer
│   │   └── main.jsx
│   ├── index.html
│   └── vite.config.js
├── demo/
│   ├── demo_cli.py             # Command-line demonstration runner
│   ├── generate_samples.py     # Benchmark test image generator
│   └── samples/                # Sample test images (indoor, dark, blur, etc.)
├── docs/
│   └── AWS_ARCHITECTURE.md     # Cloud design, free-tier verification & guide
├── COMPETITION_SUBMISSION.md   # Official competition submission document
├── Dockerfile                  # Container definition for AWS App Runner / ECS
├── docker-compose.yml          # Container stack configuration
├── deploy_aws.ps1              # 1-click deployment script (PowerShell)
└── deploy_aws.sh               # 1-click deployment script (Bash)
```

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.10+ (with OpenCV 5)
- Node.js 18+ and npm

### 1. Start the Backend

```bash
cd backend
python -m pip install -r requirements.txt
python main.py
```
*The FastAPI server will start at `http://localhost:8000`.*
- Swagger API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`
- AWS status: `http://localhost:8000/api/aws-status`

### 2. Start the Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```
*Open your browser at `http://localhost:5173`.*

---

## 🧪 Testing & Evaluation

### Run Unit Tests (24/24 Passing)
```bash
cd backend
python -m pytest test_backend.py -v
```

### Run Benchmark Suite (100% Pass Rate)
```bash
cd backend
python evaluate.py
```
Outputs quantitative pass rates, latencies, and action distributions to `backend/evaluation_report.json`.

### Run Terminal Demonstration
```bash
cd demo
# Process a single photo:
python demo_cli.py samples/bus.jpg

# Process all demo samples:
python demo_cli.py --all
```

---

## ☁️ AWS Cloud Integration

The backend automatically detects AWS environment credentials:
- Set `AWS_S3_BUCKET`, `AWS_DEFAULT_REGION`, and AWS credentials to enable cloud persistence of traces and annotated images.
- If credentials are absent, the application gracefully saves artifacts locally under `backend/uploads/` with **$0 cloud cost**.

For full cloud architecture details and step-by-step deployment instructions to **AWS App Runner**, read [docs/AWS_ARCHITECTURE.md](docs/AWS_ARCHITECTURE.md).

---

## 📜 Ethical Considerations & Model Scope

- **No Facial Recognition:** The application detects person bounding boxes for situational awareness but does **not** identify individual identities.
- **Explicit COCO 80-Class Scope:** Categories outside the 80 COCO classes (e.g. specialized medical tools, plant species) are flagged with automated **Human Review Requests** rather than producing hallucinated labels.
- **Safety-First:** Intended strictly for civilian surroundings understanding, assistance, and environmental perception.
