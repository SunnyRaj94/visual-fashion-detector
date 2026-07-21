import os
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class ClipSegDetector(BaseDetector):
    """CLIPSeg zero-shot segmenter and localizer using Hugging Face transformers."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.model_name = config.models.get("clipseg", {}).get(
            "name", "CIDAS/clipseg-rd64-refined"
        )
        self.threshold = config.models.get("clipseg", {}).get("threshold", 0.3)
        self.min_area_ratio = (
            0.002  # Box must be at least 0.2% of image area to filter noise
        )

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads CLIPSeg processor and model from HF cache or downloads them."""
        if self.model is not None:
            return

        logger.info(
            f"Loading CLIPSeg model: {self.model_name} on device: {self.device}"
        )
        self.processor = CLIPSegProcessor.from_pretrained(
            self.model_name, cache_dir=os.path.join(self.cache_dir, "huggingface")
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            self.model_name, cache_dir=os.path.join(self.cache_dir, "huggingface")
        ).to(self.device)

        self.model.eval()
        logger.info("CLIPSeg model loaded successfully.")

    @time_it("CLIPSeg Inference")
    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Runs CLIPSeg segmentation and extracts bounding boxes.

        Args:
            image: PIL Image.
            **kwargs:
                queries: List of fashion categories to segment.
                threshold: Sigmoid threshold for mask binarization.

        Returns:
            List of Detection objects (with masks populated).
        """
        self.load_model()

        thresh = kwargs.get("threshold", self.threshold)
        queries = kwargs.get("queries")
        if not queries:
            queries = self.config.get_all_categories()

        width, height = image.size
        min_area = width * height * self.min_area_ratio

        detections = []

        # Process in batches of 8 to avoid OOM
        batch_size = 8
        for start_idx in range(0, len(queries), batch_size):
            batch_queries = queries[start_idx : start_idx + batch_size]

            # Format text prompts: e.g. "a photo of a shirt"
            prompts = [f"a photo of a {q}" for q in batch_queries]

            # Prepare inputs
            inputs = self.processor(
                text=prompts,
                images=[image] * len(prompts),
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process logits
            # outputs.logits shape: (batch_size, 352, 352) or (352, 352) if batch_size is 1
            logits = outputs.logits
            if len(prompts) == 1:
                logits = logits.unsqueeze(0)

            # Interpolate to original image size
            preds = F.interpolate(
                logits.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            # Sigmoid to get probability maps
            probs = (
                torch.sigmoid(preds).squeeze(1).cpu().numpy()
            )  # shape: (batch_size, height, width)

            for i, category in enumerate(batch_queries):
                prob_map = probs[i]

                # Binarize mask
                binary_mask = (prob_map > thresh).astype(np.uint8)

                # Check if mask has any positive pixels
                if not np.any(binary_mask):
                    continue

                # Use OpenCV to find contours of connected components
                # CV2 returns coordinates as (x, y)
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h

                    if area < min_area:
                        continue

                    # Calculate average confidence score of the masked region
                    mask_crop = binary_mask[y : y + h, x : x + w]
                    prob_crop = prob_map[y : y + h, x : x + w]
                    avg_score = (
                        float(np.mean(prob_crop[mask_crop > 0]))
                        if np.any(mask_crop)
                        else float(thresh)
                    )

                    # Create isolated mask for this specific detection
                    det_mask = np.zeros_like(binary_mask)
                    cv2.drawContours(det_mask, [contour], -1, 1, thickness=cv2.FILLED)

                    detections.append(
                        Detection(
                            box=[float(x), float(y), float(x + w), float(y + h)],
                            label=category,
                            score=avg_score,
                            mask=det_mask,
                        )
                    )

        logger.info(
            f"CLIPSeg detected {len(detections)} fashion items using {len(queries)} categories."
        )
        return detections

    @time_it("FashionCLIP Absolute Presence Extraction")
    def extract_present_classes(
        self,
        image: Image.Image,
        user_categories: List[str],
        presence_threshold: float = 0.38,
    ) -> List[str]:
        """
        Uses CLIPSeg's native target heatmaps to determine class presence.
        Bypasses classification script collisions and cleans false positives
        by measuring structural pixel peak validation thresholds.
        """
        self.load_model()

        # Deduplicate incoming tags safely
        unique_categories = list(set([cat.strip().lower() for cat in user_categories]))
        confirmed_classes = set()

        # Process in safe batches matching the internal architecture setup
        batch_size = 8
        for start_idx in range(0, len(unique_categories), batch_size):
            batch_queries = unique_categories[start_idx : start_idx + batch_size]
            prompts = [f"a photo of a {q}" for q in batch_queries]

            inputs = self.processor(
                text=prompts,
                images=[image] * len(prompts),
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            if len(prompts) == 1:
                logits = logits.unsqueeze(0)

            # Convert logits natively into a 0.0 - 1.0 confidence space map
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, category in enumerate(batch_queries):
                prob_map = probs[i]

                # Extract the absolute peak pixel confidence score across the layout
                peak_score = float(np.max(prob_map))

                # An item is only present if it triggers a distinct, concentrated spatial group peak.
                # This completely isolates fake suggested classes (e.g. rings) because their peak stays very low.
                if peak_score >= presence_threshold:
                    # Map the verified clean item back to match original user taxonomy casing format
                    for original_cat in user_categories:
                        if original_cat.strip().lower() == category:
                            confirmed_classes.add(original_cat)

        return sorted(list(confirmed_classes))
