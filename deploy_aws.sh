#!/bin/bash
# Deployment helper script for AWS App Runner / ECR

set -e

REGION="${1:-us-east-1}"
REPO_NAME="agentic-vision-backend"

echo "=== Agentic Vision: AWS Deployment Helper ==="

if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or running."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "[1/4] Deploying with Account: ${ACCOUNT_ID} in Region: ${REGION}"

echo "[2/4] Ensuring ECR Repository exists..."
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION}

echo "[3/4] Authenticating Docker with ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "[4/4] Building and Pushing Container Image..."
IMAGE_TAG="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"
docker build -t ${IMAGE_TAG} .
docker push ${IMAGE_TAG}

echo -e "\n[SUCCESS] Image pushed successfully to: ${IMAGE_TAG}"
echo "You can now select this image in AWS App Runner to deploy your service!"
