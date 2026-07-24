import os
from typing import Any, List, Optional, Dict, Union, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForMaskGeneration,
    SamModel,
    SamProcessor,
    Sam2Model,
    Sam2Processor,
)

try:
    from transformers import (
        Sam2VideoModel,
        Sam2VideoProcessor,
        Sam3Model,
        Sam3Processor,
        Sam3VideoModel,
        Sam3VideoProcessor,
    )
except ImportError:
    Sam2VideoModel = None
    Sam2VideoProcessor = None
    Sam3Model = None
    Sam3Processor = None
    Sam3VideoModel = None
    Sam3VideoProcessor = None

from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class SamDetector(BaseDetector):
    """SAM (Segment Anything Model 3.1 / SAM 2 / SAM) object detector and segmenter using Hugging Face transformers."""

    def __init__(self, config: Any):
        super().__init__(config)
        # Check config for 'sam3.1' first, then fallback to 'sam'
        sam_cfg = config.models.get("sam3.1", config.models.get("sam", {}))

        self.model_name = sam_cfg.get("name", "facebook/sam2.1-hiera-small")
        self.points_per_side = sam_cfg.get("points_per_side", 16)
        self.pred_iou_thresh = sam_cfg.get("pred_iou_thresh", 0.8)
        self.stability_score_thresh = sam_cfg.get("stability_score_thresh", 0.85)
        self.box_threshold = sam_cfg.get("box_threshold", 0.25)

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads SAM processor and model from HF cache or downloads them."""
        if self.model is not None:
            return

        logger.info(f"Loading SAM model: {self.model_name} on device: {self.device}")

        hf_cache = os.path.join(self.cache_dir, "huggingface")

        # Custom loader for SAM 3.1 multiplex checkpoint (sam3.1_multiplex.pt)
        if "sam3" in self.model_name.lower():
            try:
                from huggingface_hub import hf_hub_download
                from transformers import AutoConfig

                logger.info(
                    f"Attempting to download custom SAM 3.1 checkpoint 'sam3.1_multiplex.pt' for {self.model_name}..."
                )
                ckpt_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="sam3.1_multiplex.pt",
                    cache_dir=hf_cache,
                )
                config = AutoConfig.from_pretrained(self.model_name, cache_dir=hf_cache)

                if Sam3Model is not None:
                    self.model = Sam3Model(config)
                elif Sam3VideoModel is not None:
                    self.model = Sam3VideoModel(config)
                elif Sam2VideoModel is not None:
                    self.model = Sam2VideoModel(config)
                else:
                    self.model = Sam2Model(config)

                state_dict = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]

                det_state_dict = {}
                if isinstance(state_dict, dict):
                    for k, v in state_dict.items():
                        if k.startswith("detector."):
                            det_state_dict[k[len("detector.") :]] = v
                        else:
                            det_state_dict[k] = v
                else:
                    det_state_dict = state_dict

                self.model.load_state_dict(det_state_dict, strict=False)
                self.model.to(self.device)

                if Sam3Processor is not None:
                    try:
                        self.processor = Sam3Processor.from_pretrained(
                            self.model_name, cache_dir=hf_cache
                        )
                    except Exception:
                        self.processor = AutoProcessor.from_pretrained(
                            self.model_name, cache_dir=hf_cache
                        )
                elif Sam3VideoProcessor is not None:
                    self.processor = Sam3VideoProcessor.from_pretrained(
                        self.model_name, cache_dir=hf_cache
                    )
                else:
                    self.processor = Sam2Processor.from_pretrained(
                        self.model_name, cache_dir=hf_cache
                    )

                self.model.eval()
                logger.info("SAM 3.1 custom checkpoint loaded successfully.")
                return
            except Exception as e_sam3:
                logger.warning(
                    f"SAM 3.1 custom weight loading raised: {e_sam3}. Falling back to standard model loaders."
                )

        # Load processor and model dynamically
        kwargs_model = {"cache_dir": hf_cache}
        if "sam3" in self.model_name.lower():
            kwargs_model["weight_name"] = "sam3.1_multiplex.pt"

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, cache_dir=hf_cache
            )
            try:
                self.model = AutoModelForMaskGeneration.from_pretrained(
                    self.model_name, **kwargs_model
                ).to(self.device)
            except Exception:
                self.model = AutoModelForMaskGeneration.from_pretrained(
                    self.model_name, cache_dir=hf_cache
                ).to(self.device)
        except Exception as e1:
            logger.warning(
                f"AutoModelForMaskGeneration attempt for {self.model_name} raised: {e1}. Trying SAM2/SAM3 video fallbacks."
            )
            loaded = False
            if Sam2VideoProcessor is not None and Sam2VideoModel is not None:
                try:
                    self.processor = Sam2VideoProcessor.from_pretrained(
                        self.model_name, cache_dir=hf_cache
                    )
                    try:
                        self.model = Sam2VideoModel.from_pretrained(
                            self.model_name, **kwargs_model
                        ).to(self.device)
                    except Exception:
                        self.model = Sam2VideoModel.from_pretrained(
                            self.model_name, cache_dir=hf_cache
                        ).to(self.device)
                    loaded = True
                except Exception as e2:
                    logger.warning(
                        f"Sam2VideoProcessor attempt for {self.model_name} raised: {e2}. Trying Sam2Processor."
                    )

            if not loaded:
                try:
                    self.processor = Sam2Processor.from_pretrained(
                        self.model_name, cache_dir=hf_cache
                    )
                    try:
                        self.model = Sam2Model.from_pretrained(
                            self.model_name, **kwargs_model
                        ).to(self.device)
                    except Exception:
                        self.model = Sam2Model.from_pretrained(
                            self.model_name, cache_dir=hf_cache
                        ).to(self.device)
                except Exception as e3:
                    logger.warning(
                        f"Sam2Processor attempt for {self.model_name} raised: {e3}. Falling back to SamProcessor/SamModel."
                    )
                    self.processor = SamProcessor.from_pretrained(
                        self.model_name, cache_dir=hf_cache
                    )
                    try:
                        self.model = SamModel.from_pretrained(
                            self.model_name, **kwargs_model
                        ).to(self.device)
                    except Exception:
                        self.model = SamModel.from_pretrained(
                            self.model_name, cache_dir=hf_cache
                        ).to(self.device)

        self.model.eval()
        logger.info("SAM model loaded successfully.")

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures input tensors are cast from float64 to float32 before moving to device for MPS/CPU compatibility."""
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float64:
                    v = v.to(torch.float32)
                prepared[k] = v.to(self.device)
            else:
                prepared[k] = v
        return prepared

    def _post_process_masks(
        self, outputs: Any, inputs: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """Post-processes raw SAM / SAM2 / SAM3 model masks into original image coordinates."""
        pred_masks = outputs.pred_masks.cpu()
        orig_sizes = inputs["original_sizes"].cpu()

        # Reshape pred_masks to 4D tensor (batch_size, num_masks, height, width) if spatial dims are 1D
        if pred_masks.ndim == 3:
            B, N, L = pred_masks.shape
            side = int(round(L**0.5))
            if side * side == L:
                pred_masks = pred_masks.view(B, N, side, side)
            else:
                pred_masks = pred_masks.unsqueeze(1)
        elif pred_masks.ndim == 2:
            N, L = pred_masks.shape
            side = int(round(L**0.5))
            if side * side == L:
                pred_masks = pred_masks.view(1, N, side, side)

        # Try processor post_process_masks
        if hasattr(self.processor, "post_process_masks"):
            try:
                return self.processor.post_process_masks(
                    pred_masks, orig_sizes, binarize=True
                )
            except Exception:
                try:
                    return self.processor.post_process_masks(pred_masks, orig_sizes)
                except Exception:
                    pass

        if hasattr(self.processor, "image_processor"):
            img_proc = self.processor.image_processor
            if hasattr(img_proc, "post_process_masks"):
                try:
                    reshaped_sizes = inputs.get("reshaped_input_sizes")
                    if reshaped_sizes is not None:
                        return img_proc.post_process_masks(
                            pred_masks, orig_sizes, reshaped_sizes.cpu(), binarize=True
                        )
                except Exception:
                    pass

                try:
                    return img_proc.post_process_masks(
                        pred_masks, orig_sizes, binarize=True
                    )
                except Exception:
                    try:
                        return img_proc.post_process_masks(pred_masks, orig_sizes)
                    except Exception:
                        pass

        # Fallback: Direct PyTorch bilinear interpolation to original image dimensions
        results = []
        for i, orig_size in enumerate(orig_sizes):
            h_orig, w_orig = int(orig_size[0]), int(orig_size[1])
            m = pred_masks[i] if pred_masks.ndim >= 3 else pred_masks
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(1)

            interpolated = (
                torch.nn.functional.interpolate(
                    m.float(),
                    size=(h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False,
                )
                > 0
            )
            results.append(interpolated.squeeze(1))

        return results

    def _get_iou_scores(self, outputs: Any) -> np.ndarray:
        """Safely extracts iou_scores or confidence scores array from SAM/SAM2/SAM3 model outputs."""
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            return outputs.iou_scores.cpu().numpy()
        elif hasattr(outputs, "scores") and outputs.scores is not None:
            return outputs.scores.cpu().numpy()
        elif hasattr(outputs, "pred_logits") and outputs.pred_logits is not None:
            logits = outputs.pred_logits.detach().cpu()
            return torch.sigmoid(logits).numpy()
        elif (
            hasattr(outputs, "presence_logits") and outputs.presence_logits is not None
        ):
            logits = outputs.presence_logits.detach().cpu()
            return torch.sigmoid(logits).numpy()

        # Fallback to default ones array matching mask batch shape
        if hasattr(outputs, "pred_masks"):
            masks = outputs.pred_masks
            shape = masks.shape[:-2] if masks.ndim >= 3 else (1, 1)
            return np.ones(shape, dtype=np.float32)
        return np.ones((1, 1), dtype=np.float32)

    def _extract_box_from_mask(self, mask: np.ndarray) -> Optional[List[float]]:
        """Computes [xmin, ymin, xmax, ymax] bounding box from a binary mask array."""
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return None
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        return [float(xmin), float(ymin), float(xmax + 1), float(ymax + 1)]

    def _format_points_for_processor(self, points: List[Any]) -> List[Any]:
        """Formats points to the nesting depth expected by the active SAM / SAM2 processor."""
        is_sam2 = "sam2" in type(self.processor).__name__.lower()
        if len(points) > 0 and isinstance(points[0], (int, float)):
            pts_2d = [points]
        else:
            pts_2d = points

        return [[pts_2d]] if is_sam2 else [pts_2d]

    @time_it("SAM Inference")
    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Runs SAM segmentation and object detection on the image.

        Args:
            image: PIL Image.
            **kwargs:
                - input_boxes / boxes: List of bounding boxes [[xmin, ymin, xmax, ymax], ...] for prompt-guided segmentation.
                - input_points / points: List of point prompts [[x, y], ...] for point-guided segmentation.
                - labels / queries: Target class labels to associate with detected regions.
                - points_per_side: Number of grid points per side for automatic mask generation.
                - pred_iou_thresh: Minimum predicted IoU threshold (0.0 to 1.0).
                - remove_small_boxes: Boolean, whether to filter out small boxes.
                - min_area_threshold: Area threshold for filtering small boxes.

        Returns:
            List of Detection objects containing bounding boxes, scores, labels, and binary segmentation masks.
        """
        self.load_model()

        img_w, img_h = image.size
        img_area = img_w * img_h

        # Extract parameters
        pred_iou_thresh = kwargs.get("pred_iou_thresh", self.pred_iou_thresh)
        remove_small_boxes = kwargs.get("remove_small_boxes", True)
        min_area = kwargs.get("min_area_threshold", max(150.0, 0.0015 * img_area))

        prompt_boxes = kwargs.get("input_boxes", kwargs.get("boxes"))
        prompt_points = kwargs.get("input_points", kwargs.get("points"))
        queries = kwargs.get("queries", kwargs.get("labels"))

        detections: List[Detection] = []

        if prompt_boxes is not None and len(prompt_boxes) > 0:
            # --- Mode 1: Box-Prompted Segmentation ---
            formatted_boxes = [[box for box in prompt_boxes]]
            raw_inputs = self.processor(
                image, input_boxes=formatted_boxes, return_tensors="pt"
            )
            inputs = self._prepare_inputs(raw_inputs)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process masks to original image dimensions
            masks = self._post_process_masks(outputs, inputs)[0]

            iou_scores = self._get_iou_scores(outputs)
            if iou_scores.ndim > 2:
                iou_scores = iou_scores[0]  # shape: (num_boxes, num_masks)

            for idx, box in enumerate(prompt_boxes):
                box_masks = masks[idx].numpy() if idx < len(masks) else masks[0].numpy()
                if box_masks.ndim == 2:
                    box_masks = np.expand_dims(box_masks, axis=0)

                box_scores = iou_scores[idx] if idx < len(iou_scores) else iou_scores[0]
                if np.isscalar(box_scores) or box_scores.ndim == 0:
                    box_scores = np.array([box_scores])

                # Select best mask for this box
                best_mask_idx = int(np.argmax(box_scores))
                best_score = float(box_scores[best_mask_idx])
                raw_best_mask = (
                    box_masks[best_mask_idx]
                    if best_mask_idx < len(box_masks)
                    else box_masks[0]
                )
                if raw_best_mask.dtype == bool:
                    best_mask = raw_best_mask
                else:
                    best_mask = raw_best_mask > 0.0

                # Determine label
                if queries and idx < len(queries):
                    label = str(queries[idx]).strip().lower()
                elif isinstance(queries, str):
                    label = queries.strip().lower()
                else:
                    label = "object"

                # Use original box or refine from mask
                refined_box = self._extract_box_from_mask(best_mask) or [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ]

                # Clip coordinates
                xmin = max(0.0, min(refined_box[0], float(img_w)))
                ymin = max(0.0, min(refined_box[1], float(img_h)))
                xmax = max(0.0, min(refined_box[2], float(img_w)))
                ymax = max(0.0, min(refined_box[3], float(img_h)))

                if remove_small_boxes:
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area < min_area:
                        continue

                detections.append(
                    Detection(
                        box=[xmin, ymin, xmax, ymax],
                        label=label,
                        score=best_score,
                        mask=best_mask,
                        metadata={
                            "iou_score": best_score,
                            "prompt_box": box,
                            "raw_label": label,
                        },
                    )
                )

        elif prompt_points is not None and len(prompt_points) > 0:
            # --- Mode 2: Point-Prompted Segmentation ---
            is_sam3 = (
                Sam3Processor is not None and isinstance(self.processor, Sam3Processor)
            ) or (type(self.processor).__name__ == "Sam3Processor")

            if is_sam3:
                input_boxes = []
                for pt in prompt_points:
                    if isinstance(pt[0], (int, float)):
                        input_boxes.append(
                            [
                                float(pt[0]),
                                float(pt[1]),
                                float(pt[0]) + 1.0,
                                float(pt[1]) + 1.0,
                            ]
                        )
                    else:
                        input_boxes.append(
                            [
                                [
                                    float(p[0]),
                                    float(p[1]),
                                    float(p[0]) + 1.0,
                                    float(p[1]) + 1.0,
                                ]
                                for p in pt
                            ]
                        )
                formatted_boxes = [[box for box in input_boxes]]
                raw_inputs = self.processor(
                    image, input_boxes=formatted_boxes, return_tensors="pt"
                )
            else:
                formatted_points = self._format_points_for_processor(prompt_points)
                raw_inputs = self.processor(
                    image, input_points=formatted_points, return_tensors="pt"
                )
            inputs = self._prepare_inputs(raw_inputs)

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self._post_process_masks(outputs, inputs)[0]

            iou_scores = self._get_iou_scores(outputs)
            if iou_scores.ndim > 1:
                iou_scores = iou_scores.reshape(-1)

            num_masks = min(
                masks.shape[1] if masks.ndim > 1 else len(masks), len(iou_scores)
            )
            for mask_idx in range(num_masks):
                score = float(iou_scores[mask_idx])
                if score < pred_iou_thresh:
                    continue

                raw_mask = (
                    masks[0, mask_idx].numpy()
                    if masks.ndim >= 3
                    else masks[mask_idx].numpy()
                )
                mask_np = raw_mask if raw_mask.dtype == bool else (raw_mask > 0.0)
                extracted_box = self._extract_box_from_mask(mask_np)
                if not extracted_box:
                    continue

                xmin, ymin, xmax, ymax = extracted_box
                xmin = max(0.0, min(xmin, float(img_w)))
                ymin = max(0.0, min(ymin, float(img_h)))
                xmax = max(0.0, min(xmax, float(img_w)))
                ymax = max(0.0, min(ymax, float(img_h)))

                if remove_small_boxes:
                    if (xmax - xmin) * (ymax - ymin) < min_area:
                        continue

                detections.append(
                    Detection(
                        box=[xmin, ymin, xmax, ymax],
                        label="object",
                        score=score,
                        mask=mask_np,
                        metadata={"iou_score": score},
                    )
                )

        else:
            # --- Mode 3: Automatic Grid-based Instance Segmentation ---
            is_sam3 = (
                Sam3Processor is not None and isinstance(self.processor, Sam3Processor)
            ) or (type(self.processor).__name__ == "Sam3Processor")

            pts_side = kwargs.get("points_per_side", self.points_per_side)
            x_pts = np.linspace(15, img_w - 15, pts_side)
            y_pts = np.linspace(15, img_h - 15, pts_side)
            xx, yy = np.meshgrid(x_pts, y_pts)
            grid_points = np.stack([xx.flatten(), yy.flatten()], axis=-1).tolist()

            # Process grid points in batches to manage memory
            batch_size = 16
            for i in range(0, len(grid_points), batch_size):
                batch_pts = grid_points[i : i + batch_size]
                if is_sam3:
                    input_boxes = [
                        [
                            float(pt[0]),
                            float(pt[1]),
                            float(pt[0]) + 1.0,
                            float(pt[1]) + 1.0,
                        ]
                        for pt in batch_pts
                    ]
                    formatted_boxes = [[box for box in input_boxes]]
                    raw_inputs = self.processor(
                        image, input_boxes=formatted_boxes, return_tensors="pt"
                    )
                else:
                    formatted_points = self._format_points_for_processor(batch_pts)
                    raw_inputs = self.processor(
                        image, input_points=formatted_points, return_tensors="pt"
                    )
                inputs = self._prepare_inputs(raw_inputs)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                masks = self._post_process_masks(outputs, inputs)[0]

                iou_scores = self._get_iou_scores(outputs)
                if iou_scores.ndim > 1:
                    iou_scores = iou_scores.reshape(-1)

                num_masks = min(
                    masks.shape[1] if masks.ndim > 1 else len(masks), len(iou_scores)
                )
                for mask_idx in range(num_masks):
                    score = float(iou_scores[mask_idx])
                    if score < pred_iou_thresh:
                        continue

                    raw_mask = (
                        masks[0, mask_idx].numpy()
                        if masks.ndim >= 3
                        else masks[mask_idx].numpy()
                    )
                    mask_np = raw_mask if raw_mask.dtype == bool else (raw_mask > 0.0)
                    extracted_box = self._extract_box_from_mask(mask_np)
                    if not extracted_box:
                        continue

                    xmin, ymin, xmax, ymax = extracted_box
                    xmin = max(0.0, min(xmin, float(img_w)))
                    ymin = max(0.0, min(ymin, float(img_h)))
                    xmax = max(0.0, min(xmax, float(img_w)))
                    ymax = max(0.0, min(ymax, float(img_h)))

                    if remove_small_boxes:
                        if (xmax - xmin) * (ymax - ymin) < min_area:
                            continue

                    detections.append(
                        Detection(
                            box=[xmin, ymin, xmax, ymax],
                            label="object",
                            score=score,
                            mask=mask_np,
                            metadata={"iou_score": score},
                        )
                    )

        logger.info(f"SAM detected/segmented {len(detections)} items.")
        return detections

    @time_it("SAM Segmented Objects Extraction")
    def extract_segmented_objects(
        self,
        image: Image.Image,
        detections: Optional[List[Detection]] = None,
        crop_to_box: bool = True,
        transparent_bg: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Extracts individual detected fashion objects into isolated PIL Images (RGBA/transparent).

        Args:
            image: Original PIL Image.
            detections: Optional list of Detection objects. If None, calls detect(image, **kwargs).
            crop_to_box: Whether to crop each isolated object to its bounding box.
            transparent_bg: If True, makes non-object pixels transparent (RGBA mode).

        Returns:
            List of dictionaries containing:
                - "label": Class label string
                - "score": Detection confidence score
                - "box": Bounding box coordinates [xmin, ymin, xmax, ymax]
                - "segmented_image": Isolated object PIL Image (RGBA)
                - "mask": Binary mask numpy array
        """
        if detections is None:
            detections = self.detect(image, **kwargs)

        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape

        results = []
        for det in detections:
            if det.mask is None:
                mask_np = np.zeros((h, w), dtype=bool)
                xmin, ymin, xmax, ymax = map(int, det.box)
                mask_np[max(0, ymin) : min(h, ymax), max(0, xmin) : min(w, xmax)] = True
            else:
                mask_np = det.mask

            if mask_np.ndim > 2:
                mask_np = np.squeeze(mask_np)
                if mask_np.ndim > 2:
                    mask_np = mask_np[0]

            if mask_np.shape != (h, w):
                mask_pil = Image.fromarray((mask_np > 0).astype(np.uint8) * 255).resize(
                    (w, h), Image.NEAREST
                )
                mask_np = np.array(mask_pil) > 0
            else:
                mask_np = mask_np.astype(bool)

            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_np
            rgba[:, :, 3] = np.where(mask_np, 255, 0).astype(np.uint8)

            segmented_pil = Image.fromarray(rgba, mode="RGBA")

            if crop_to_box:
                crop_box = self._extract_box_from_mask(mask_np) or det.box
                xmin, ymin, xmax, ymax = crop_box
                pad = kwargs.get("padding", 5)
                crop_xmin = max(0, int(xmin) - pad)
                crop_ymin = max(0, int(ymin) - pad)
                crop_xmax = min(w, int(xmax) + pad)
                crop_ymax = min(h, int(ymax) + pad)

                if crop_xmax > crop_xmin and crop_ymax > crop_ymin:
                    segmented_pil = segmented_pil.crop(
                        (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
                    )

            results.append(
                {
                    "label": det.label,
                    "score": det.score,
                    "box": det.box,
                    "segmented_image": segmented_pil,
                    "mask": mask_np,
                    "metadata": det.metadata,
                }
            )

        logger.info(f"SAM extracted {len(results)} isolated segmented object images.")
        return results

    def segment_classes(
        self,
        image: Image.Image,
        user_categories: Optional[List[str]] = None,
        boxes: Optional[List[List[float]]] = None,
        points: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Image.Image]]:
        """Extracts segmented objects grouped natively into a category dictionary.

        Returns:
            Dict mapping category labels -> list of isolated PIL Images (RGBA).
        """
        if user_categories is not None:
            kwargs["queries"] = user_categories
        if boxes is not None:
            kwargs["input_boxes"] = boxes
        if points is not None:
            kwargs["input_points"] = points

        extracted = self.extract_segmented_objects(image, **kwargs)
        cat_map: Dict[str, List[Image.Image]] = {}
        for item in extracted:
            cat = item["label"]
            cat_map.setdefault(cat, []).append(item["segmented_image"])
        return cat_map


# Class aliases for flexible instantiation
SAMDetector = SamDetector
Sam3Detector = SamDetector
Sam31Detector = SamDetector
