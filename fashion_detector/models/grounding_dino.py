import os
from typing import Any, List, Optional
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class GroundingDinoDetector(BaseDetector):
    """Grounding DINO zero-shot object detector using Hugging Face transformers."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.model_name = config.models.get("grounding_dino", {}).get(
            "name", "IDEA-Research/grounding-dino-tiny"
        )
        self.box_threshold = config.models.get("grounding_dino", {}).get(
            "box_threshold", 0.25
        )
        self.text_threshold = config.models.get("grounding_dino", {}).get(
            "text_threshold", 0.25
        )

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads Grounding DINO processor and model from HF cache or downloads them."""
        if self.model is not None:
            return

        logger.info(
            f"Loading Grounding DINO model: {self.model_name} on device: {self.device}"
        )

        # Hugging Face environment variables are already set by the Config class
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=os.path.join(self.cache_dir, "huggingface")
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name, cache_dir=os.path.join(self.cache_dir, "huggingface")
        ).to(self.device)

        self.model.eval()
        logger.info("Grounding DINO model loaded successfully.")

    @time_it("Grounding DINO Inference")
    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Runs Grounding DINO detection on the image.

        Args:
            image: PIL Image.
            **kwargs: Can override 'box_threshold', 'text_threshold', or provide 'queries' list.

        Returns:
            List of Detection objects.
        """
        self.load_model()

        box_thresh = kwargs.get("box_threshold", self.box_threshold)
        text_thresh = kwargs.get("text_threshold", self.text_threshold)

        # Use provided queries list or fall back to config's flat categories
        queries = kwargs.get("queries")
        if not queries:
            queries = self.config.get_all_categories()

        # Format the queries for Grounding DINO. It requires a string like "shirt . pants . shoes ."
        query_str = " . ".join(queries) + " ."

        # Prepare inputs
        inputs = self.processor(images=image, text=query_str, return_tensors="pt").to(
            self.device
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)

        # Perform post-processing
        # Note: target_sizes must be on CPU for post_process if it runs there, but matching device is safer
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=target_sizes.cpu(),
        )[0]

        # Configuration for small box removal
        remove_small_boxes = kwargs.get("remove_small_boxes", True)
        img_w, img_h = image.size
        img_area = img_w * img_h
        # Default threshold: 150 pixels or 0.15% of total image area, whichever is larger
        min_area = kwargs.get("min_area_threshold", max(150.0, 0.0015 * img_area))

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        # Future-proof: use "text_labels" if available (HuggingFace v4.51+), else fallback to "labels"
        labels = results.get("text_labels", results.get("labels", []))

        for box, score, label in zip(boxes, scores, labels):
            # Ensure label is a string
            if not isinstance(label, str):
                label = str(label)

            cleaned_label = label.strip().lower()

            # Filter out empty, punctuation, or subword fragment labels
            if (
                not cleaned_label
                or cleaned_label.startswith("##")
                or cleaned_label in [".", ",", ";", ":", "-", ""]
            ):
                continue

            # Grounding DINO returns bounding box as [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box.tolist()

            # Clip coordinates to image boundary
            xmin = max(0.0, min(xmin, float(img_w)))
            ymin = max(0.0, min(ymin, float(img_h)))
            xmax = max(0.0, min(xmax, float(img_w)))
            ymax = max(0.0, min(ymax, float(img_h)))

            # Filter out very small boxes to remove noise/false positives
            if remove_small_boxes:
                box_area = (xmax - xmin) * (ymax - ymin)
                if box_area < min_area:
                    logger.info(
                        f"Filtering out small box '{cleaned_label}' with area {box_area:.1f} pixels (min threshold: {min_area:.1f})"
                    )
                    continue

            detections.append(
                Detection(
                    box=[xmin, ymin, xmax, ymax],
                    label=cleaned_label,
                    score=float(score),
                    metadata={"raw_label": label},
                )
            )

        logger.info(f"Grounding DINO detected {len(detections)} items.")
        return detections
