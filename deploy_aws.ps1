# Deployment helper script for AWS App Runner / ECR
param(
    [string]$Region = "us-east-1",
    [string]$RepositoryName = "agentic-vision-backend"
)

Write-Host "=== Agentic Vision: AWS Deployment Helper ===" -ForegroundColor Cyan

# Check AWS CLI
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Warning "AWS CLI is not installed or not in PATH. Please install AWS CLI from https://aws.amazon.com/cli/"
    exit 1
}

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Warning "Docker is not running. Please start Docker Desktop before deploying."
    exit 1
}

Write-Host "[1/4] Verifying AWS Identity..." -ForegroundColor Yellow
$caller = aws sts get-caller-identity --output json | ConvertFrom-Json
$accountId = $caller.Account
Write-Host "Deploying with Account: $accountId in Region: $Region" -ForegroundColor Green

Write-Host "[2/4] Ensuring ECR Repository exists..." -ForegroundColor Yellow
aws ecr describe-repositories --repository-names $RepositoryName --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --repository-name $RepositoryName --region $Region
}

Write-Host "[3/4] Authenticating Docker with ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "$accountId.dkr.ecr.$Region.amazonaws.com"

Write-Host "[4/4] Building and Pushing Container Image..." -ForegroundColor Yellow
$imageTag = "$accountId.dkr.ecr.$Region.amazonaws.com/$RepositoryName`:latest"
docker build -t $imageTag .
docker push $imageTag

Write-Host "`n[SUCCESS] Image pushed successfully to: $imageTag" -ForegroundColor Green
Write-Host "You can now select this image in AWS App Runner to deploy your service with zero server management!" -ForegroundColor Cyan
