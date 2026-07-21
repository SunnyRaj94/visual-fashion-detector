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

    @time_it("CLIPSeg Pure Class Presence Extraction")
    def extract_present_classes(
        self,
        image: Image.Image,
        user_categories: List[str],
        presence_threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Runs the optimized batch segmentation pipeline across all categories
        and extracts a unique list of classes physically present in the image.

        Bypasses string splitting hallucinations and filters out noise natively.
        """
        # 1. Force use of custom threshold if provided, else fall back to class default
        thresh = (
            presence_threshold if presence_threshold is not None else self.threshold
        )

        # 2. Call your existing, highly optimized detect function.
        # This leverages your batch size splitting (8) to protect from GPU OOM crashes,
        # runs the bilinear interpolation sizing, and applies the min_area_ratio filter.
        detections = self.detect(image, queries=user_categories, threshold=thresh)

        # 3. Aggregate a clean, unique python set of matching string labels
        # If an item didn't generate positive pixels or failed the area constraint,
        # it won't exist in the 'detections' array list.
        present_classes = {det.label for det in detections}

        # 4. Return as a clean, alphabetically sorted list matching your taxonomy casing
        return sorted(list(present_classes))
