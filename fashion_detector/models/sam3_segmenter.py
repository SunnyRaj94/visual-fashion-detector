import os
from typing import Any, List, Dict, Optional
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
        converted_boxes = [
            (
                b["box"]
                if isinstance(b, dict) and "box" in b
                else (b.box if hasattr(b, "box") else b)
            )
            for b in box_inputs
        ]

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
        converted_boxes = [
            (
                b["box"]
                if isinstance(b, dict) and "box" in b
                else (b.box if hasattr(b, "box") else b)
            )
            for b in box_inputs
        ]

        inputs = self.processor(
            images=orig_image, input_boxes=[converted_boxes], return_tensors="pt"
        ).to(self.device)

        device_str = str(self.device).lower()
        with torch.inference_mode():
            if any(d in device_str for d in ["cuda", "mps"]):
                device_type = "cuda" if "cuda" in device_str else "mps"
                dtype = torch.bfloat16 if device_type == "mps" else torch.float16
                with torch.autocast(device_type=device_type, dtype=dtype):
                    outputs = self.model(**inputs)
            else:
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
        name = sam3_cfg.get("name", "facebook/sam3")
        self.model_name = name
        self.box_threshold = sam3_cfg.get("box_threshold", 0.1)
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads SAM 3 model using absolute submodule architecture imports."""
        if self.model is not None:
            return

        logger.info(f"Loading SAM 3 model: {self.model_name} on device: {self.device}")
        hf_cache = os.path.join(self.cache_dir, "huggingface")

        # Explicitly importing backend submodule to handle Dev build configurations
        from transformers import Sam3Processor, Sam3Model

        try:
            self.processor = Sam3Processor.from_pretrained(
                self.model_name, cache_dir=hf_cache
            )
            logger.info("SAM 3 processor loaded successfully.")
        except Exception:
            logger.warning(
                "Failed to load SAM 3 processor from HuggingFace cache. Attempting local files only load."
            )
            self.processor = Sam3Processor.from_pretrained(
                self.model_name, cache_dir=hf_cache, local_files_only=True
            )

        try:
            self.model = Sam3Model.from_pretrained(
                self.model_name, cache_dir=hf_cache
            ).to(self.device)
            logger.info("SAM 3 model loaded successfully.")
        except Exception:
            logger.warning(
                "Failed to load SAM 3 model from HuggingFace cache. Attempting local files only load."
            )
            self.model = Sam3Model.from_pretrained(
                self.model_name, cache_dir=hf_cache, local_files_only=True
            ).to(self.device)

        self.model.eval()
        logger.info("SAM 3 model is now ready for inference.")

    def _match_masks_to_boxes(
        self,
        pred_masks: torch.Tensor,
        converted_boxes: List[List[float]],
        orig_h: int,
        orig_w: int,
    ) -> List[np.ndarray]:
        """Matches SAM 3 candidate prediction masks 1-to-1 to each input box based on spatial overlap."""
        masks_interpolated = torch.nn.functional.interpolate(
            pred_masks,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0]

        extracted_masks = []
        for box in converted_boxes:
            ymin, xmin, ymax, xmax = map(int, box)
            ymin_cl = max(0, min(orig_h, ymin))
            ymax_cl = max(0, min(orig_h, ymax))
            xmin_cl = max(0, min(orig_w, xmin))
            xmax_cl = max(0, min(orig_w, xmax))

            box_crop = masks_interpolated[:, ymin_cl:ymax_cl, xmin_cl:xmax_cl]
            if box_crop.numel() > 0:
                scores = box_crop.mean(dim=(-2, -1))
            else:
                scores = masks_interpolated.mean(dim=(-2, -1))

            best_idx = torch.argmax(scores).item()
            binary_mask = (masks_interpolated[best_idx] > 0.0).cpu().numpy()
            extracted_masks.append(binary_mask)

        return extracted_masks

    @time_it("sam3_box_segmentation")
    def segment_with_boxes(
        self,
        image: Image.Image,
        box_inputs: List[Dict[str, Any]],
        labels: Optional[List[str]] = None,
    ) -> Image.Image:
        """Segments image using geometric box tracking configurations and concept labels for each box area."""
        if self.model is None or self.processor is None:
            self.load_model()

        orig_image = image.convert("RGB")
        orig_w, orig_h = orig_image.size
        img_np = np.array(orig_image)
        color_layer = np.zeros_like(img_np)

        for idx, item in enumerate(box_inputs):
            box = (
                item["box"]
                if isinstance(item, dict) and "box" in item
                else (
                    item.box
                    if hasattr(item, "box")
                    else (
                        item[0]
                        if isinstance(item, (list, tuple)) and len(item) == 2
                        else item
                    )
                )
            )
            label = (
                item["label"]
                if isinstance(item, dict) and "label" in item
                else (
                    item.label
                    if hasattr(item, "label")
                    else (
                        item[1]
                        if isinstance(item, (list, tuple)) and len(item) == 2
                        else (
                            labels[idx]
                            if labels and idx < len(labels)
                            else "fashion item"
                        )
                    )
                )
            )

            ymin, xmin, ymax, xmax = map(int, box)
            ymin_cl, ymax_cl = max(0, min(orig_h, ymin)), max(0, min(orig_h, ymax))
            xmin_cl, xmax_cl = max(0, min(orig_w, xmin)), max(0, min(orig_w, xmax))

            if ymax_cl > ymin_cl and xmax_cl > xmin_cl:
                crop_img = orig_image.crop((xmin_cl, ymin_cl, xmax_cl, ymax_cl))
                concept_cutouts = self.extract_concept_parts(crop_img, [label])
                if concept_cutouts:
                    cutout_np = np.array(concept_cutouts[0])
                    alpha_mask = cutout_np[:, :, 3] > 0
                else:
                    alpha_mask = np.ones(
                        (ymax_cl - ymin_cl, xmax_cl - xmin_cl), dtype=bool
                    )

                color = np.array(
                    [
                        (idx * 75 + 60) % 255,
                        (idx * 145 + 30) % 255,
                        (255 - (idx * 40)) % 255,
                    ],
                    dtype=np.uint8,
                )
                color_layer[ymin_cl:ymax_cl, xmin_cl:xmax_cl][alpha_mask] = color

        blended_np = np.where(
            color_layer > 0, (img_np * 0.4 + color_layer * 0.6).astype(np.uint8), img_np
        )
        return Image.fromarray(blended_np)

    @time_it("sam3_box_extraction")
    def extract_segmented_parts(
        self,
        image: Image.Image,
        box_inputs: List[Dict[str, Any]],
        labels: Optional[List[str]] = None,
    ) -> List[Image.Image]:
        """Extracts segmented instances as clean transparent alpha PNGs for each of the N input box+label pairs."""
        if self.model is None or self.processor is None:
            self.load_model()

        orig_image = image.convert("RGB")
        orig_w, orig_h = orig_image.size
        extracted_cutouts = []

        for idx, item in enumerate(box_inputs):
            box = (
                item["box"]
                if isinstance(item, dict) and "box" in item
                else (
                    item.box
                    if hasattr(item, "box")
                    else (
                        item[0]
                        if isinstance(item, (list, tuple)) and len(item) == 2
                        else item
                    )
                )
            )
            label = (
                item["label"]
                if isinstance(item, dict) and "label" in item
                else (
                    item.label
                    if hasattr(item, "label")
                    else (
                        item[1]
                        if isinstance(item, (list, tuple)) and len(item) == 2
                        else (
                            labels[idx]
                            if labels and idx < len(labels)
                            else "fashion item"
                        )
                    )
                )
            )

            ymin, xmin, ymax, xmax = map(int, box)
            ymin_cl, ymax_cl = max(0, min(orig_h, ymin)), max(0, min(orig_h, ymax))
            xmin_cl, xmax_cl = max(0, min(orig_w, xmin)), max(0, min(orig_w, xmax))

            if ymax_cl > ymin_cl and xmax_cl > xmin_cl:
                crop_img = orig_image.crop((xmin_cl, ymin_cl, xmax_cl, ymax_cl))
                concept_cutouts = self.extract_concept_parts(crop_img, [label])

                full_cutout = Image.new("RGBA", (orig_w, orig_h))
                if concept_cutouts:
                    full_cutout.paste(concept_cutouts[0], (xmin_cl, ymin_cl))
                else:
                    full_cutout.paste(crop_img.convert("RGBA"), (xmin_cl, ymin_cl))
                extracted_cutouts.append(full_cutout)
            else:
                extracted_cutouts.append(Image.new("RGBA", (orig_w, orig_h)))

        return extracted_cutouts

    @time_it("sam3_concept_segmentation")
    def segment_with_concepts(
        self,
        image: Image.Image,
        text_queries: List[str],
        threshold: Optional[float] = None,
        return_mode: str = "cutout",
    ) -> Image.Image:
        """
        Segments textual label phrases using SAM 3 native text features.

        Args:
            image: Input PIL Image.
            text_queries: List of textual concept prompts.
            threshold: Confidence threshold for post-processing filtering.
            return_mode: 'overlay' for semi-transparent color overlay,
                         'cutout' for isolated segmented parts on transparent RGBA background.

        Returns:
            A new PIL Image object containing the segmented result.
        """
        if self.model is None or self.processor is None:
            self.load_model()

        active_thresh = threshold if threshold is not None else self.box_threshold
        orig_image = image.convert("RGB")

        # Format text queries properly for Sam3Processor
        if isinstance(text_queries, str):
            text_prompt_list = [text_queries]
            query_labels = [text_queries]
        elif isinstance(text_queries, list):
            query_labels = [str(q) for q in text_queries]
            text_prompt_list = [", ".join(query_labels)]
        else:
            text_prompt_list = [str(text_queries)]
            query_labels = [str(text_queries)]

        # 1. Generate text embeddings
        inputs = self.processor(
            images=orig_image, text=text_prompt_list, return_tensors="pt"
        ).to(self.device)

        # 2. Run model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Post-process instance segmentation with candidate threshold (0.05) to capture raw scores
        eval_thresh = min(active_thresh, 0.05)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=eval_thresh,
            mask_threshold=eval_thresh,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        scores_np = results["scores"].cpu().numpy()
        masks_np = results["masks"].cpu().numpy()

        # Log confidence scores as requested
        logger.info(
            "For SAM 3 zero-shot promptable concept queries on this image, the confidence scores were:"
        )
        if len(scores_np) > 0:
            for idx in range(min(len(query_labels), len(scores_np))):
                label = (
                    query_labels[idx] if idx < len(query_labels) else f"concept_{idx}"
                )
                logger.info(f"  • {label}: {scores_np[idx]:.4f}")
        else:
            logger.info("  • No candidate concept masks detected above score 0.05.")

        valid_indices = [i for i, s in enumerate(scores_np) if s >= active_thresh]
        if len(valid_indices) == 0 and len(scores_np) > 0:
            logger.info(
                f"No masks exceeded active_thresh={active_thresh:.2f}. Using top candidate detection (score={scores_np[0]:.4f})."
            )
            valid_indices = list(range(min(len(query_labels), len(scores_np))))

        filtered_masks = masks_np[valid_indices] if len(valid_indices) > 0 else masks_np

        if return_mode == "cutout":
            # Extract ONLY the segmented parts as a single RGBA PIL Image with transparent background
            img_rgba = image.convert("RGBA")
            img_np = np.array(img_rgba)
            transparent_np = np.zeros_like(img_np)

            for idx, mask in enumerate(filtered_masks):
                bool_mask = mask.astype(bool)
                transparent_np[bool_mask] = img_np[bool_mask]

            return Image.fromarray(transparent_np, mode="RGBA")
        else:
            # Render clean color layers onto the image matrix
            img_np = np.array(orig_image)
            color_layer = np.zeros_like(img_np)

            for idx, mask in enumerate(filtered_masks):
                color = np.array(
                    [
                        (idx * 80 + 40) % 255,
                        (idx * 120 + 60) % 255,
                        (255 - (idx * 30)) % 255,
                    ],
                    dtype=np.uint8,
                )
                bool_mask = mask.astype(bool)
                color_layer[bool_mask] = color

            blended_np = np.where(
                color_layer > 0,
                (img_np * 0.4 + color_layer * 0.6).astype(np.uint8),
                img_np,
            )
            return Image.fromarray(blended_np)

    @time_it("sam3_concept_extraction")
    def extract_concept_parts(
        self,
        image: Image.Image,
        text_queries: List[str],
        threshold: Optional[float] = None,
    ) -> List[Image.Image]:
        """
        Extracts each segmented concept into individual transparent RGBA PIL images in memory.

        Returns:
            List of in-memory PIL Image objects (RGBA cutouts).
        """
        if self.model is None or self.processor is None:
            self.load_model()

        active_thresh = threshold if threshold is not None else self.box_threshold
        orig_rgba = image.convert("RGBA")

        if isinstance(text_queries, str):
            text_prompt_list = [text_queries]
            query_labels = [text_queries]
        elif isinstance(text_queries, list):
            query_labels = [str(q) for q in text_queries]
            text_prompt_list = [", ".join(query_labels)]
        else:
            text_prompt_list = [str(text_queries)]
            query_labels = [str(text_queries)]

        inputs = self.processor(
            images=image.convert("RGB"), text=text_prompt_list, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        eval_thresh = min(active_thresh, 0.05)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=eval_thresh,
            mask_threshold=eval_thresh,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        scores_np = results["scores"].cpu().numpy()
        masks_np = results["masks"].cpu().numpy()

        logger.info(
            "For SAM 3 zero-shot promptable concept queries on this image, the confidence scores were:"
        )
        if len(scores_np) > 0:
            for idx in range(min(len(text_queries), len(scores_np))):
                label = (
                    text_queries[idx] if idx < len(text_queries) else f"concept_{idx}"
                )
                logger.info(f"  • {label}: {scores_np[idx]:.4f}")
        else:
            logger.info("  • No candidate concept masks detected above score 0.05.")

        valid_indices = [i for i, s in enumerate(scores_np) if s >= active_thresh]
        if len(valid_indices) == 0 and len(scores_np) > 0:
            logger.info(
                f"No masks exceeded active_thresh={active_thresh:.2f}. Using top candidate detection (score={scores_np[0]:.4f})."
            )
            valid_indices = list(range(min(len(text_queries), len(scores_np))))

        filtered_masks = masks_np[valid_indices] if len(valid_indices) > 0 else masks_np

        img_np = np.array(orig_rgba)
        extracted_cutouts = []

        for idx, mask in enumerate(filtered_masks):
            bool_mask = mask.astype(bool)
            cutout_np = np.zeros_like(img_np)
            cutout_np[bool_mask] = img_np[bool_mask]
            extracted_cutouts.append(Image.fromarray(cutout_np, mode="RGBA"))

        return extracted_cutouts

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
