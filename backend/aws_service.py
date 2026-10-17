"""
AWS Integration Service for Agentic Vision.

Provides optional AWS cloud capabilities with zero mandatory cloud charges:
1. S3 Audit & Storage: Uploads analyzed images and agent decision traces to S3
   when AWS credentials are provided.
2. Graceful Local Fallback: If no AWS credentials or bucket are configured,
   operates locally without error or degraded functionality.
3. Telemetry & Health: Reports AWS connection status and deployment tier.
"""

import os
import json
import logging
from typing import Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Environment variable configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", None)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)


class AWSService:
    """Manages cloud persistence and metrics with transparent local fallback."""

    def __init__(self):
        self._s3_client = None
        self._is_connected = False
        self._bucket_name = AWS_S3_BUCKET
        self._region = AWS_REGION
        self._init_client()

    def _init_client(self):
        """Attempts to initialize boto3 S3 client if credentials exist."""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError

            if self._bucket_name:
                self._s3_client = boto3.client(
                    "s3",
                    region_name=self._region,
                    aws_access_key_id=AWS_ACCESS_KEY_ID or None,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
                )
                # Verify bucket connectivity or availability
                try:
                    self._s3_client.head_bucket(Bucket=self._bucket_name)
                    self._is_connected = True
                    logger.info(f"[AWS] S3 client connected to bucket: {self._bucket_name}")
                except Exception as b_err:
                    logger.warning(
                        f"[AWS] S3 bucket '{self._bucket_name}' not reachable ({b_err}). Running in local mode."
                    )
                    self._is_connected = False
            else:
                logger.info("[AWS] No AWS_S3_BUCKET configured. Running with local storage fallback.")
                self._is_connected = False
        except ImportError:
            logger.info("[AWS] boto3 not installed. Running with local storage fallback.")
            self._is_connected = False
        except Exception as e:
            logger.warning(f"[AWS] Initialization error ({e}). Running in local fallback mode.")
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_status(self) -> dict:
        """Returns cloud integration telemetry."""
        return {
            "aws_configured": self._is_connected,
            "storage_mode": "AWS S3" if self._is_connected else "Local Filesystem",
            "bucket_name": self._bucket_name if self._is_connected else None,
            "region": self._region,
            "cloud_provider": "Amazon Web Services (AWS)",
            "tier_mode": "Free Tier Eligible",
        }

    def save_analysis_artifact(
        self, request_id: str, image_bytes: bytes, trace_data: dict
    ) -> Tuple[str, Optional[str]]:
        """
        Saves the analysis image and decision trace.
        If AWS S3 is active, uploads to s3://{bucket}/runs/{request_id}/.
        Otherwise, saves to local disk under backend/uploads/.

        Returns:
            Tuple of (storage_location, s3_url_or_none)
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename_img = f"run_{timestamp}_{request_id}.jpg"
        filename_trace = f"run_{timestamp}_{request_id}_trace.json"

        if self._is_connected and self._s3_client:
            try:
                # Upload image
                s3_key_img = f"analyses/{filename_img}"
                self._s3_client.put_object(
                    Bucket=self._bucket_name,
                    Key=s3_key_img,
                    Body=image_bytes,
                    ContentType="image/jpeg",
                )

                # Upload JSON trace
                s3_key_trace = f"analyses/{filename_trace}"
                self._s3_client.put_object(
                    Bucket=self._bucket_name,
                    Key=s3_key_trace,
                    Body=json.dumps(trace_data, indent=2).encode("utf-8"),
                    ContentType="application/json",
                )

                s3_url = f"https://{self._bucket_name}.s3.{self._region}.amazonaws.com/{s3_key_img}"
                logger.info(f"[AWS] Successfully stored analysis to S3: {s3_url}")
                return "AWS S3", s3_url
            except Exception as e:
                logger.warning(f"[AWS] S3 upload failed ({e}); falling back to local storage.")

        # Local fallback
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        img_path = os.path.join(uploads_dir, filename_img)
        trace_path = os.path.join(uploads_dir, filename_trace)

        with open(img_path, "wb") as f:
            f.write(image_bytes)

        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)

        return "Local Disk", None
