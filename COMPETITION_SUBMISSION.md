# OpenCV AI Competition 2026 — Official Submission Document
## Track: Agentic Vision Award

**Project Title:** Agentic Vision: Dynamic Surroundings Understanding via Autonomous Perception-Decision-Action Loops  
**OpenCV Version:** 5.0.0  
**Cloud Infrastructure:** Amazon Web Services (AWS)  
**License:** AGPL-3.0 / Open Source  

---

### Executive Summary

Most computer vision applications operate as one-shot forward pipelines: an image is ingested, a neural network inference executes once, and a static output or chatbot explanation is produced. If an object is small, occluded, or captured under low-light conditions, one-shot systems either miss the object or return unreliable, low-confidence predictions.

**Agentic Vision** is an autonomous perception-decision-action system that actively interrogates visual evidence. Instead of blindly accepting initial detections, an agentic decision controller evaluates image quality metrics and detection confidence, dynamically decides whether follow-up vision tools are needed (such as adaptive CLAHE contrast enhancement, region-of-interest cropping with bicubic super-sampling, or human review escalation), executes those tools via OpenCV 5, compares the outcomes, and iteratively refines its understanding of everyday surroundings.

---

### Criterion 1: Substantive OpenCV 5 Image & Video Analysis

OpenCV 5 is the computational core of the entire application. The vision engine does not delegate image analysis to third-party proprietary APIs; all perceptual operations are performed locally using OpenCV 5:

1. **Deep Neural Network Inference (`cv2.dnn`):**
   - Model compilation via `cv2.dnn.readNetFromONNX()` targeting the CPU graph engine.
   - Blob normalization and tensor generation via `cv2.dnn.blobFromImage()` (640x640, RGB channel swap, 1/255 scaling).
   - Post-processing Non-Maximum Suppression via `cv2.dnn.NMSBoxes()` with dynamic confidence and intersection-over-union (IoU) thresholds.
2. **Quality Perception & Blur Estimation:**
   - Sharpness estimation via discrete Laplacian operator variance: `cv2.Laplacian(gray, cv2.CV_64F).var()`.
   - Luminance and contrast analysis across pixel distributions to diagnose underexposure (`brightness < 80`) and dynamic range compression (`contrast < 40`).
3. **Adaptive Histogram Equalization & Filtering:**
   - LAB color space conversion (`cv2.cvtColor(img, cv2.COLOR_BGR2LAB)`).
   - Contrast Limited Adaptive Histogram Equalization on luminance channel: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`.
   - Directional high-pass spatial filtering via 2D convolution kernel with `cv2.filter2D()`.
4. **Sub-Pixel Region Super-Sampling:**
   - Autonomous bounding box cropping with contextual padding.
   - High-fidelity bicubic interpolation upscaling via `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`.

---

### Criterion 2: Genuine Agentic Perception–Decision–Action Loop

The system strictly adheres to the agentic cycle:

```
                  ┌──────────────────────┐
                  │ 1. PERCEIVE (Global) │
                  │ Full-image detector  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ 2. ASSESS & MEASURE  │
                  │ Laplacian & Contrast │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ 3. DYNAMIC DECISION  │
                  │ Evaluate Evidence    │
                  └──────────┬───────────┘
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   [Quality Deficit] [Uncertain Target] [High Confidence]
            │                │                │
            ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ 4a. ENHANCE  │ │ 4b. CROP &   │ │ 4c. ACCEPT   │
     │ CLAHE+Filter │ │ SUPER-SAMPLE │ │ No follow-up │
     └──────┬───────┘ └──────┬───────┘ └──────────────┘
            │                │
            └───────┬────────┘
                    ▼
          ┌────────────────────┐
          │ 5. COMPARE & VERIFY│
          │ Initial vs Followup│
          └─────────┬──────────┘
                    │
            ┌───────┴────────┐
            ▼                ▼
     [Resolved]       [Ambiguous / Out of Domain]
            │                │
            ▼                ▼
     ┌──────────────┐ ┌───────────────────────────┐
     │ Accept Best  │ │ 6. ESCALATE HUMAN REVIEW  │
     └──────────────┘ └───────────────────────────┘
```

The system does **NOT** run a hardcoded sequence. Depending on visual input:
- A pristine image with high confidence immediately concludes in 3 steps (`StopReason.HIGH_CONFIDENCE`).
- A dark image with marginal detections triggers enhancement and updates confidence (`StopReason.IMPROVED_AFTER_INSPECTION`).
- An ambiguous small bounding box triggers targeted region cropping and bicubic super-sampling.
- An empty or non-COCO scene triggers human escalation (`StopReason.NO_DETECTIONS`).

---

### Criterion 3: Visual Results Influence Later Decisions & Tools

The agent's decisions are strictly conditional on mathematical visual outcomes:
1. **Decision Point A (Enhancement Trigger):** Triggered if and only if `is_dark` or `is_low_contrast` or `is_blurry` is true **and** detections are either missing or have confidence below the low threshold (`< 0.45`).
2. **Decision Point B (Region-of-Interest Crop):** The agent identifies the single detection with the highest uncertainty (`min(uncertain, key=confidence)`), computes padded coordinates, crops the region, upscales it, and re-runs localized inference.
3. **Decision Point C (Evidence Comparison):** The agent compares `enhanced_avg_confidence` and `crop_detection_confidence` against original detections. It only updates the scene hypothesis if the follow-up inspection measurably improved detection confidence.
4. **Decision Point D (Human Review Escalation):** If visual evidence remains unclassifiable after tool execution, the agent proactively escalates to a human operator with diagnostic reasoning.

---

### Criterion 4: Distinction from Fixed Chatbots

This application is **not a chatbot**:
- It does not pass image captions into a generic large language model to produce conversational text.
- The agent controller is a deterministic, observable Python decision state machine whose actions correspond to OpenCV tool executions.
- The complete trace of perception, tool invocations, inputs, execution latencies, and state transitions is exposed in the UI as an auditable timeline.

---

### Criterion 5: AWS Architecture & Cloud Integration

The solution incorporates Amazon Web Services (AWS) using a production-ready, Free-Tier-eligible architecture:
- **AWS App Runner / AWS ECS Fargate:** Hosts the containerized FastAPI and OpenCV 5 backend with health checks and auto-scaling.
- **Amazon S3 Storage:** Persists annotated frames and full JSON decision traces for auditability and compliance.
- **Amazon CloudWatch:** Collects telemetry, step-by-step latency, and agent decision distributions.
- **Zero Cost Guarantee:** All AWS features are 100% Free Tier compliant, and the application includes automatic local fallback when credentials are not present.

---

### Criterion 6: Testing, Failure Handling & Demonstration

- **Test Suite:** 24 unit and integration tests passing (`pytest test_backend.py`).
- **Benchmark Suite:** Automated quantitative evaluation suite (`evaluate.py`) with 100% pass rate across degradation scenarios.
- **Observable Dashboard:** Modern React 19 + Vite dashboard with image comparison toggle, confidence bars, quality assessment cards, and interactive execution trace timeline.
- **CLI Demo:** Terminal demonstration tool (`demo_cli.py`) with sample dataset for offline and automated verification.
