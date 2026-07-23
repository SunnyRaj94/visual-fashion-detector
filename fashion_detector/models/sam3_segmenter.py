import os
from typing import Any, List, Dict
import torch
import numpy as np
from PIL import Image

from transformers import AutoProcessor, AutoModelForMaskGeneration

from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class Sam2Detector(BaseDetector):
    """SAM 2 / SAM 2.1 specific segmenter utilizing geometric bounding box prompts."""

    def __init__(self, config: Any):
        super().__init__(config)
        sam_cfg = config.models.get("sam", {})
        self.model_name = sam_cfg.get("name", "facebook/sam2.1-hiera-small")
        self.box_threshold = sam_cfg.get("box_threshold", 0.5)
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads SAM 2 dependencies using Auto Classes framework."""
        if self.model is not None:
            return

        logger.info(f"Loading SAM 2 model: {self.model_name} on device: {self.device}")
        hf_cache = os.path.join(self.cache_dir, "huggingface")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=hf_cache
        )
        self.model = AutoModelForMaskGeneration.from_pretrained(
            self.model_name, cache_dir=hf_cache
        ).to(self.device)

        self.model.eval()
        logger.info("SAM 2 model loaded successfully.")

    @time_it("sam2_box_segmentation")
    def segment_with_boxes(
        self, image: Image.Image, box_inputs: List[Dict[str, Any]]
    ) -> Image.Image:
        """Segments image using geometric box tracking configurations."""
        if self.model is None or self.processor is None:
            self.load_model()

        orig_image = image.convert("RGB")
        target_size = orig_image.size[::-1]  # (Height, Width)
        converted_boxes = [b["box"] for b in box_inputs]

        inputs = self.processor(
            images=orig_image, input_boxes=[converted_boxes], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        raw_masks = self.processor.image_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=[target_size],
            target_sizes=[target_size],
        )

        masks_tensor = raw_masks[0] if isinstance(raw_masks, list) else raw_masks
        iou_tensor = (
            outputs.iou_scores[0]
            if outputs.iou_scores.ndim == 3
            else outputs.iou_scores
        )

        if masks_tensor.ndim == 5:
            masks_tensor = masks_tensor.squeeze(0)
            iou_tensor = iou_tensor.squeeze(0)

        masks_np = []
        for box_idx in range(len(converted_boxes)):
            best_mask_idx = torch.argmax(iou_tensor[box_idx]).item()
            binary_mask = (masks_tensor[box_idx, best_mask_idx] > 0.0).cpu().numpy()
            masks_np.append(binary_mask)

        img_np = np.array(orig_image)
        color_layer = np.zeros_like(img_np)
        for idx, mask in enumerate(masks_np):
            color = np.array(
                [
                    (idx * 75 + 60) % 255,
                    (idx * 145 + 30) % 255,
                    (255 - (idx * 40)) % 255,
                ],
                dtype=np.uint8,
            )
            color_layer[mask] = color

        blended_np = np.where(
            color_layer > 0, (img_np * 0.4 + color_layer * 0.6).astype(np.uint8), img_np
        )
        return Image.fromarray(blended_np)

    @time_it("sam2_box_extraction")
    def extract_segmented_parts(
        self, image: Image.Image, box_inputs: List[Dict[str, Any]]
    ) -> List[Image.Image]:
        """Extracts segmented instances as clean transparent alpha PNGs."""
        if self.model is None or self.processor is None:
            self.load_model()

        orig_image = image.convert("RGB")
        target_size = orig_image.size[::-1]
        converted_boxes = [b["box"] for b in box_inputs]

        inputs = self.processor(
            images=orig_image, input_boxes=[converted_boxes], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        raw_masks = self.processor.image_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=[target_size],
            target_sizes=[target_size],
        )

        masks_tensor = raw_masks[0] if isinstance(raw_masks, list) else raw_masks
        iou_tensor = (
            outputs.iou_scores[0]
            if outputs.iou_scores.ndim == 3
            else outputs.iou_scores
        )

        if masks_tensor.ndim == 5:
            masks_tensor = masks_tensor.squeeze(0)
            iou_tensor = iou_tensor.squeeze(0)

        extracted_cutouts = []
        img_np = np.array(image.convert("RGBA"))

        for box_idx in range(len(converted_boxes)):
            best_mask_idx = torch.argmax(iou_tensor[box_idx]).item()
            binary_mask = (masks_tensor[box_idx, best_mask_idx] > 0.0).cpu().numpy()

            cutout_np = np.zeros_like(img_np)
            cutout_np[binary_mask] = img_np[binary_mask]
            extracted_cutouts.append(Image.fromarray(cutout_np))

        return extracted_cutouts

    def detect(self, image: Image.Image, queries: List[str]) -> List[Detection]:
        return []


class Sam3Detector(BaseDetector):
    """SAM 3 specific segmenter optimized natively for Promptable Concept Segmentation (PCS)."""

    def __init__(self, config: Any):
        super().__init__(config)
        sam3_cfg = config.models.get("sam3", config.models.get("sam", {}))
        self.model_name = sam3_cfg.get("name", "facebook/sam3")
        self.box_threshold = sam3_cfg.get("box_threshold", 0.5)
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads SAM 3 model using absolute submodule architecture imports."""
        if self.model is not None:
            return

        logger.info(f"Loading SAM 3 model: {self.model_name} on device: {self.device}")
        hf_cache = os.path.join(self.cache_dir, "huggingface")

        # Explicitly importing backend submodule to handle Dev build configurations
        # from transformers.models.sam3.modeling_sam3 import Sam3Model
        from transformers import Sam3Processor, Sam3Model

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=hf_cache
        )
        self.model = Sam3Model.from_pretrained(self.model_name, cache_dir=hf_cache).to(
            self.device
        )

        self.model.eval()
        logger.info("SAM 3 model loaded successfully.")

    @time_it("sam3_concept_segmentation")
    def segment_with_concepts(
        self, image: Image.Image, text_queries: List[str]
    ) -> Image.Image:
        """Segments textual label phrases using SAM 3 native text features."""
        if self.model is None or self.processor is None:
            self.load_model()

        orig_image = image.convert("RGB")
        target_size = orig_image.size[::-1]

        # SAM 3 native execution layer via text embeddings prompts
        inputs = self.processor(
            images=orig_image, text=text_queries, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        post_processed = self.processor.post_process_instance_segmentation(
            outputs, threshold=self.box_threshold, target_sizes=[target_size]
        )
        masks_np = post_processed["masks"].cpu().numpy()

        img_np = np.array(orig_image)
        color_layer = np.zeros_like(img_np)
        for idx, mask in enumerate(masks_np):
            color = np.array(
                [
                    (idx * 80 + 40) % 255,
                    (idx * 120 + 60) % 255,
                    (255 - (idx * 30)) % 255,
                ],
                dtype=np.uint8,
            )
            color_layer[mask] = color

        blended_np = np.where(
            color_layer > 0, (img_np * 0.4 + color_layer * 0.6).astype(np.uint8), img_np
        )
        return Image.fromarray(blended_np)

    def detect(self, image: Image.Image, queries: List[str]) -> List[Detection]:
        return []


class SamSegmenter:
    """Factory Router class distributing active context tasks to proper SAM submodules."""

    def __new__(cls, config: Any) -> BaseDetector:
        sam_cfg = config.models.get("sam3.1", config.models.get("sam", {}))
        model_name = sam_cfg.get("name", "facebook/sam2.1-hiera-small").lower()

        if "sam3" in model_name:
            logger.info(
                "Factory routing execution architecture -> Sam3Detector Subclass"
            )
            return Sam3Detector(config)
        else:
            logger.info(
                "Factory routing execution architecture -> Sam2Detector Subclass"
            )
            return Sam2Detector(config)
