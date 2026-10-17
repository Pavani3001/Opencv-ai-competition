# AWS Cloud Architecture & Integration Guide

**Project:** Agentic Vision — Intelligent Surroundings Understanding  
**Track:** Agentic Vision Award (OpenCV AI Competition 2026)  
**Cloud Provider:** Amazon Web Services (AWS)  
**Cost Model:** 100% Free Tier Eligible ($0 / month)

---

## 1. Architectural Overview

The Agentic Vision application implements a hybrid cloud-edge architecture. The perception and agent loop can run both on local compute or in a containerized AWS cloud service, with cloud persistence and telemetry powered by AWS services.

```
┌─────────────────────────────────────────────────────────────┐
│                       CLIENT TIER                           │
│  React 19 + Vite Dashboard (Desktop / Mobile / Tablet)      │
│  Hosted via AWS CloudFront + Amazon S3 (Static Web)         │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTPS / REST API
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      COMPUTE TIER                           │
│  AWS App Runner / AWS ECS Fargate (Containerized FastAPI)   │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Vision Engine (OpenCV 5 DNN + YOLOv8n ONNX)          │  │
│  │ - Laplacian Sharpness & CLAHE Enhancement             │  │
│  │ - Region Cropping & Bicubic Upscaling                 │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              │ Perception Feed              │
│  ┌───────────────────────────▼───────────────────────────┐  │
│  │ Agent Loop (Perception-Decision-Action Engine)        │  │
│  │ - Quality Assessment & Enhancement Trigger            │  │
│  │ - Low-Confidence Region Inspection                    │  │
│  │ - Dynamic Acceptance or Human Review Escalation       │  │
│  └───────────────────────────┬───────────────────────────┘  │
└──────────────────────────────┼──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────────────┐   ┌───────────────────────────┐
│        PERSISTENCE            │   │        TELEMETRY          │
│  Amazon S3                    │   │  Amazon CloudWatch        │
│  - Analysis frames (JPEG)     │   │  - Agent decision logs    │
│  - Decision traces (JSON)     │   │  - Latency & SLA metrics  │
└───────────────────────────────┘   └───────────────────────────┘
```

---

## 2. AWS Services Used & Free-Tier Verification

All services selected are within the standard **AWS Free Tier**:

| AWS Service | Architectural Role | Free Tier Allowance | Our Usage |
| :--- | :--- | :--- | :--- |
| **AWS App Runner / ECS** | Hosts containerized FastAPI backend with OpenCV 5 | 1 active service / Fargate trial | ~0.5 vCPU, 1 GB RAM |
| **Amazon S3** | Persists annotated frames and audit traces | 5 GB Standard Storage, 20,000 GET, 2,000 PUT | < 100 MB per 1,000 runs |
| **Amazon CloudWatch** | Ingests real-time agent execution telemetry | 5 GB Log ingestion / month | < 50 MB / month |
| **AWS CloudFront** | Low-latency CDN distribution for React UI | 1 TB Data Transfer Out, 10M HTTP requests | Minimal |

> **Zero Unexpected Charges Guarantee:** The backend features a transparent local fallback. If no AWS credentials or S3 bucket are configured, all operations run on local disk without error or interruption.

---

## 3. Storage & Audit Persistence (Amazon S3)

When an image is analyzed, the agent generates:
1. An annotated image highlighting detected entities with confidence bars and labels.
2. A structured JSON execution trace containing step-by-step reasoning:
   - Observation
   - Quality metrics (brightness, contrast, blur)
   - Decision rationale
   - Tool calls made (enhancement, crop, upscale)
   - Pre- and post-inspection comparison metrics

In AWS mode, these artifacts are uploaded to:
```
s3://<AWS_S3_BUCKET>/analyses/run_<TIMESTAMP>_<REQUEST_ID>.jpg
s3://<AWS_S3_BUCKET>/analyses/run_<TIMESTAMP>_<REQUEST_ID>_trace.json
```

---

## 4. Step-by-Step Deployment Walkthrough

### Option A: AWS App Runner (Recommended, Simplest)

AWS App Runner builds and deploys containers directly from a source repository or Amazon ECR image with automatic HTTPS, health checks, and zero server maintenance.

1. **Build and push container to Amazon ECR:**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t agentic-vision-backend .
   docker tag agentic-vision-backend:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/agentic-vision:latest
   docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/agentic-vision:latest
   ```

2. **Create App Runner Service:**
   - In AWS Console -> App Runner -> **Create Service**.
   - Source: Container registry -> select your ECR image.
   - Port: `8000`
   - CPU: `1 vCPU`, Memory: `2 GB`
   - Environment Variables:
     - `AWS_S3_BUCKET`: your-bucket-name
     - `AWS_DEFAULT_REGION`: us-east-1

### Option B: Local Development with Cloud Storage (Hybrid)

You can run the application on your computer while saving all traces to your AWS S3 bucket:

1. Configure AWS CLI credentials:
   ```bash
   aws configure
   ```
2. Set the environment variable:
   ```powershell
   $env:AWS_S3_BUCKET="my-agentic-vision-bucket"
   $env:AWS_DEFAULT_REGION="us-east-1"
   ```
3. Run the backend:
   ```powershell
   python main.py
   ```
4. The dashboard header will display: **☁️ AWS S3 Connected**!
