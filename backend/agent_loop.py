"""
Agentic Vision Loop — The decision-making core.

This module implements a genuine perception-decision-action loop:
1. PERCEIVE: Run initial object detection on the uploaded image
2. ASSESS: Evaluate detection quality (confidence, image quality)
3. DECIDE: Choose next action based on actual visual results
4. ACT: Execute the chosen tool (crop, enhance, re-detect, etc.)
5. COMPARE: Compare new results with previous results
6. RESOLVE: Accept, escalate to human review, or iterate

The agent does NOT follow a fixed sequence. Its path depends on
the actual detection results and image quality metrics.
"""

import logging
import time
import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from vision_engine import (
    VisionEngine,
    DetectionResult,
    Detection,
    crop_region,
    enhance_image,
    upscale_region,
    compute_image_quality_metrics,
)

logger = logging.getLogger(__name__)


class AgentAction(str, Enum):
    """Actions the agent can take."""
    INITIAL_DETECTION = "initial_detection"
    CROP_AND_INSPECT = "crop_and_inspect"
    ENHANCE_AND_REDETECT = "enhance_and_redetect"
    QUALITY_ASSESSMENT = "quality_assessment"
    ACCEPT_RESULT = "accept_result"
    REQUEST_HUMAN_REVIEW = "request_human_review"


class StopReason(str, Enum):
    """Why the agent stopped its loop."""
    HIGH_CONFIDENCE = "All detections have high confidence — result accepted."
    IMPROVED_AFTER_INSPECTION = "Follow-up inspection improved or confirmed the result."
    NO_IMPROVEMENT = "Follow-up inspection did not improve results — best result kept."
    NO_DETECTIONS = "No objects detected even after enhancement — requesting human review."
    UNCERTAIN = "Detections remain uncertain after inspection — requesting human review."
    MAX_ITERATIONS = "Maximum iterations reached — returning best result."
    ERROR = "An error occurred during processing."


@dataclass
class TraceStep:
    """A single step in the agent's execution trace."""
    step_number: int
    action: str
    reason: str
    tool_input: Optional[str] = None
    result_summary: str = ""
    detections_count: int = 0
    avg_confidence: float = 0.0
    duration_ms: float = 0.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    """Complete result from the agentic analysis."""
    trace: list[TraceStep] = field(default_factory=list)
    initial_detections: list[dict] = field(default_factory=list)
    final_detections: list[dict] = field(default_factory=list)
    scene_summary: str = ""
    stop_reason: str = ""
    needs_human_review: bool = False
    human_review_reason: str = ""
    total_duration_ms: float = 0.0
    image_quality: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    model_limitations: str = (
        "This model (YOLOv8n) can detect 80 COCO object categories including "
        "people, common animals, vehicles, furniture, electronics, food items, "
        "and sports equipment. It cannot identify specific individuals, read text, "
        "recognize plants by species, or detect objects outside the COCO categories."
    )

    def to_dict(self) -> dict:
        return {
            "trace": [s.to_dict() for s in self.trace],
            "initial_detections": self.initial_detections,
            "final_detections": self.final_detections,
            "scene_summary": self.scene_summary,
            "stop_reason": self.stop_reason,
            "needs_human_review": self.needs_human_review,
            "human_review_reason": self.human_review_reason,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "image_quality": self.image_quality,
            "comparison": self.comparison,
            "model_limitations": self.model_limitations,
        }


# --- Confidence thresholds for agent decisions ---
HIGH_CONFIDENCE = 0.65
LOW_CONFIDENCE = 0.45
MIN_REGION_SIZE = 40  # minimum px for a region worth re-inspecting
MAX_ITERATIONS = 4


class AgentLoop:
    """
    The agentic perception-decision-action loop.

    Execution flow (NOT fixed — depends on results):

    1. Run initial detection
    2. Check image quality metrics
    3. IF image is dark/blurry/low-contrast AND has low-confidence detections:
       -> Enhance the full image and re-detect
    4. IF any detections have low confidence:
       -> Crop the most uncertain region, upscale, and re-detect
       -> Compare with original detection
    5. IF still uncertain:
       -> Request human review
    6. ELSE:
       -> Accept the best result
    """

    def __init__(self, engine: VisionEngine):
        self.engine = engine

    def run(self, image: np.ndarray) -> AgentResult:
        """Execute the full agentic analysis loop on an image."""
        overall_start = time.time()
        result = AgentResult()
        step_count = 0

        try:
            # ─── Step 1: Initial detection ───
            step_count += 1
            step_start = time.time()
            initial = self.engine.detect_objects(image)
            step_duration = (time.time() - step_start) * 1000

            result.initial_detections = [d.to_dict() for d in initial.detections]
            avg_conf = self._avg_confidence(initial.detections)

            result.trace.append(TraceStep(
                step_number=step_count,
                action=AgentAction.INITIAL_DETECTION,
                reason="Starting analysis: run object detection on the full uploaded image.",
                result_summary=f"Detected {len(initial.detections)} object(s) with average confidence {avg_conf:.0%}.",
                detections_count=len(initial.detections),
                avg_confidence=round(avg_conf, 4),
                duration_ms=round(step_duration, 1),
            ))

            if initial.error:
                result.stop_reason = StopReason.ERROR
                result.final_detections = result.initial_detections
                result.total_duration_ms = (time.time() - overall_start) * 1000
                return result

            # ─── Step 2: Assess image quality ───
            step_count += 1
            step_start = time.time()
            quality = compute_image_quality_metrics(image)
            result.image_quality = quality
            step_duration = (time.time() - step_start) * 1000

            quality_issues = []
            if quality["is_dark"]:
                quality_issues.append("dark")
            if quality["is_low_contrast"]:
                quality_issues.append("low contrast")
            if quality["is_blurry"]:
                quality_issues.append("blurry")

            quality_summary = (
                f"Image quality issues: {', '.join(quality_issues)}."
                if quality_issues
                else "Image quality is acceptable."
            )

            result.trace.append(TraceStep(
                step_number=step_count,
                action=AgentAction.QUALITY_ASSESSMENT,
                reason="Assess image quality to decide if enhancement could improve results.",
                result_summary=quality_summary,
                duration_ms=round(step_duration, 1),
                details=quality,
            ))

            best_detections = initial.detections
            current_image = image

            # ─── Decision Point A: Should we enhance the image? ───
            # Enhance ONLY if quality is poor AND detections are weak
            has_quality_issues = len(quality_issues) > 0
            has_low_confidence = any(
                d.confidence < LOW_CONFIDENCE for d in initial.detections
            )
            has_few_detections = len(initial.detections) == 0

            if has_quality_issues and (has_low_confidence or has_few_detections):
                step_count += 1
                step_start = time.time()

                enhanced = enhance_image(image)
                enhanced_result = self.engine.detect_objects(enhanced)
                step_duration = (time.time() - step_start) * 1000

                enhanced_avg = self._avg_confidence(enhanced_result.detections)
                improvement = enhanced_avg - avg_conf
                improved = (
                    enhanced_avg > avg_conf
                    or len(enhanced_result.detections) > len(initial.detections)
                )

                reason = (
                    f"Image has quality issues ({', '.join(quality_issues)}) "
                    f"and {'no detections' if has_few_detections else 'low-confidence detections'}. "
                    f"Enhancing with CLAHE and sharpening to see if detection improves."
                )

                if improved:
                    result_text = (
                        f"Enhancement helped: {len(enhanced_result.detections)} detections "
                        f"(was {len(initial.detections)}) with avg confidence "
                        f"{enhanced_avg:.0%} (was {avg_conf:.0%})."
                    )
                    best_detections = enhanced_result.detections
                    current_image = enhanced
                    avg_conf = enhanced_avg
                else:
                    result_text = (
                        f"Enhancement did NOT improve results. "
                        f"Keeping original: {len(initial.detections)} detections at {avg_conf:.0%}."
                    )

                result.trace.append(TraceStep(
                    step_number=step_count,
                    action=AgentAction.ENHANCE_AND_REDETECT,
                    reason=reason,
                    result_summary=result_text,
                    detections_count=len(enhanced_result.detections) if improved else len(initial.detections),
                    avg_confidence=round(enhanced_avg if improved else avg_conf, 4),
                    duration_ms=round(step_duration, 1),
                    details={
                        "enhancement_improved": improved,
                        "original_count": len(initial.detections),
                        "enhanced_count": len(enhanced_result.detections),
                        "confidence_change": round(improvement, 4),
                    },
                ))

            # ─── Decision Point B: Should we inspect uncertain regions? ───
            uncertain = [
                d for d in best_detections
                if d.confidence < HIGH_CONFIDENCE
                and d.width >= MIN_REGION_SIZE
                and d.height >= MIN_REGION_SIZE
            ]

            if uncertain and step_count < MAX_ITERATIONS:
                # Pick the most uncertain detection to inspect
                target = min(uncertain, key=lambda d: d.confidence)

                step_count += 1
                step_start = time.time()

                # Crop the region around the uncertain detection
                cropped = crop_region(
                    current_image, target.x, target.y, target.width, target.height
                )
                # Upscale if too small for the detector
                cropped = upscale_region(cropped)
                # Run detection on the cropped region
                crop_result = self.engine.detect_objects(cropped, conf_threshold=0.30)
                step_duration = (time.time() - step_start) * 1000

                crop_avg = self._avg_confidence(crop_result.detections)
                improved = crop_avg > target.confidence

                reason = (
                    f"Detection '{target.label}' has low confidence ({target.confidence:.0%}). "
                    f"Cropping and inspecting this region to verify."
                )

                if improved and crop_result.detections:
                    top_crop_det = max(crop_result.detections, key=lambda d: d.confidence)
                    if top_crop_det.confidence > target.confidence:
                        result_text = (
                            f"Close-up inspection found '{top_crop_det.label}' at "
                            f"{top_crop_det.confidence:.0%} (was {target.confidence:.0%}). "
                            f"Confidence improved — updating result."
                        )
                        # Update the detection in best_detections
                        for i, d in enumerate(best_detections):
                            if d is target:
                                best_detections[i] = Detection(
                                    class_id=top_crop_det.class_id,
                                    label=top_crop_det.label,
                                    confidence=top_crop_det.confidence,
                                    x=target.x,
                                    y=target.y,
                                    width=target.width,
                                    height=target.height,
                                )
                                break
                    else:
                        result_text = (
                            f"Close-up inspection did not improve confidence for '{target.label}'. "
                            f"Best crop result: '{top_crop_det.label}' at {top_crop_det.confidence:.0%}."
                        )
                else:
                    result_text = (
                        f"Close-up inspection of '{target.label}' region "
                        f"{'found no objects' if not crop_result.detections else 'did not improve confidence'}. "
                        f"Keeping original detection."
                    )

                result.trace.append(TraceStep(
                    step_number=step_count,
                    action=AgentAction.CROP_AND_INSPECT,
                    reason=reason,
                    tool_input=f"Region: ({target.x}, {target.y}, {target.width}x{target.height})",
                    result_summary=result_text,
                    detections_count=len(crop_result.detections),
                    avg_confidence=round(crop_avg, 4),
                    duration_ms=round(step_duration, 1),
                    details={
                        "target_label": target.label,
                        "target_confidence": target.confidence,
                        "crop_improved": improved,
                        "crop_detections": [d.to_dict() for d in crop_result.detections[:5]],
                    },
                ))

            # ─── Final decision ───
            final_avg = self._avg_confidence(best_detections)

            # Build comparison
            result.comparison = self._build_comparison(
                initial.detections, best_detections
            )

            if len(best_detections) == 0:
                # No objects found at all
                step_count += 1
                result.trace.append(TraceStep(
                    step_number=step_count,
                    action=AgentAction.REQUEST_HUMAN_REVIEW,
                    reason="No objects were detected even after enhancement attempts.",
                    result_summary="Requesting human review — the image may contain objects outside the model's 80 COCO categories, or the image may be too unusual for the model.",
                ))
                result.stop_reason = StopReason.NO_DETECTIONS
                result.needs_human_review = True
                result.human_review_reason = (
                    "No objects detected. The image may contain objects not in "
                    "the COCO 80-class set, or the image quality may be insufficient."
                )

            elif final_avg < LOW_CONFIDENCE:
                # Detections exist but are all low confidence
                step_count += 1
                result.trace.append(TraceStep(
                    step_number=step_count,
                    action=AgentAction.REQUEST_HUMAN_REVIEW,
                    reason=f"Average confidence is {final_avg:.0%}, below the {LOW_CONFIDENCE:.0%} threshold.",
                    result_summary=f"Results are uncertain (avg {final_avg:.0%}). Requesting human review to verify.",
                ))
                result.stop_reason = StopReason.UNCERTAIN
                result.needs_human_review = True
                result.human_review_reason = (
                    f"Detections have low average confidence ({final_avg:.0%}). "
                    f"Please verify the results visually."
                )

            elif len(result.trace) > 2:
                # We did follow-up work — report the outcome
                step_count += 1
                changed = result.comparison.get("labels_changed", False)
                result.trace.append(TraceStep(
                    step_number=step_count,
                    action=AgentAction.ACCEPT_RESULT,
                    reason="Follow-up inspection completed. Accepting best result.",
                    result_summary=(
                        f"Accepted {len(best_detections)} detection(s) at {final_avg:.0%} avg confidence. "
                        f"{'Additional inspection changed some results.' if changed else 'Additional inspection confirmed initial results.'}"
                    ),
                    detections_count=len(best_detections),
                    avg_confidence=round(final_avg, 4),
                ))
                result.stop_reason = (
                    StopReason.IMPROVED_AFTER_INSPECTION
                    if changed
                    else StopReason.NO_IMPROVEMENT
                )
            else:
                # High confidence from the start — accept directly
                step_count += 1
                result.trace.append(TraceStep(
                    step_number=step_count,
                    action=AgentAction.ACCEPT_RESULT,
                    reason=f"All detections have sufficient confidence (avg {final_avg:.0%}).",
                    result_summary=f"Accepted {len(best_detections)} detection(s) — no follow-up needed.",
                    detections_count=len(best_detections),
                    avg_confidence=round(final_avg, 4),
                ))
                result.stop_reason = StopReason.HIGH_CONFIDENCE

            result.final_detections = [d.to_dict() for d in best_detections]
            result.scene_summary = self._build_scene_summary(best_detections)

        except Exception as e:
            logger.exception("Agent loop error")
            result.stop_reason = StopReason.ERROR
            result.trace.append(TraceStep(
                step_number=step_count + 1,
                action=AgentAction.REQUEST_HUMAN_REVIEW,
                reason=f"Error during analysis: {str(e)}",
                result_summary="An error occurred. Requesting human review.",
            ))
            result.needs_human_review = True
            result.human_review_reason = f"Processing error: {str(e)}"

        result.total_duration_ms = (time.time() - overall_start) * 1000
        return result

    @staticmethod
    def _avg_confidence(detections: list[Detection]) -> float:
        if not detections:
            return 0.0
        return sum(d.confidence for d in detections) / len(detections)

    @staticmethod
    def _build_comparison(
        initial: list[Detection], final: list[Detection]
    ) -> dict:
        """Compare initial and final detections to report changes."""
        initial_labels = sorted([d.label for d in initial])
        final_labels = sorted([d.label for d in final])

        initial_avg = (
            sum(d.confidence for d in initial) / len(initial) if initial else 0
        )
        final_avg = (
            sum(d.confidence for d in final) / len(final) if final else 0
        )

        return {
            "initial_count": len(initial),
            "final_count": len(final),
            "initial_avg_confidence": round(initial_avg, 4),
            "final_avg_confidence": round(final_avg, 4),
            "confidence_change": round(final_avg - initial_avg, 4),
            "labels_changed": initial_labels != final_labels,
            "initial_labels": initial_labels,
            "final_labels": final_labels,
        }

    @staticmethod
    def _build_scene_summary(detections: list[Detection]) -> str:
        """Build a human-readable scene summary from detections."""
        if not detections:
            return (
                "No recognizable objects were detected in this image. "
                "The scene may contain objects outside the model's 80 COCO categories."
            )

        from coco_classes import CATEGORY_GROUPS

        # Count objects by label
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.label] = counts.get(d.label, 0) + 1

        # Group by category
        grouped: dict[str, list[str]] = {}
        ungrouped: list[str] = []

        for label, count in counts.items():
            found = False
            for group_name, group_labels in CATEGORY_GROUPS.items():
                if label in group_labels:
                    if group_name not in grouped:
                        grouped[group_name] = []
                    item = f"{count} {label}(s)" if count > 1 else f"1 {label}"
                    grouped[group_name].append(item)
                    found = True
                    break
            if not found:
                item = f"{count} {label}(s)" if count > 1 else f"1 {label}"
                ungrouped.append(item)

        # Build summary text
        parts = []
        for group_name, items in grouped.items():
            parts.append(f"{group_name}: {', '.join(items)}")
        if ungrouped:
            parts.append(f"Other: {', '.join(ungrouped)}")

        summary = "Scene contains: " + "; ".join(parts) + "."
        summary += (
            f" Total: {len(detections)} object(s) from {len(counts)} "
            f"distinct category/categories."
        )

        return summary
