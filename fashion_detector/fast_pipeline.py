import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field, ConfigDict

from fashion_detector.config import Config
from fashion_detector.logging import logger, time_it
from fashion_detector.models.base import Detection
from fashion_detector.models.grounding_dino import GroundingDinoDetector
from fashion_detector.models.sam3_segmenter import Sam2Detector, SamSegmenter
from fashion_detector.models.fashion_clip import FashionClipDetector
from fashion_detector.utils import (
    CATEGORY_HIERARCHY,
    get_broad_categories,
    get_fine_categories_for_broad,
    get_parent_taxonomy_for_fine,
    draw_bounding_boxes,
    generate_interactive_html,
)


class DetectedFashionObject(BaseModel):
    """Pydantic v2 schema representing a single detected fashion item with its in-memory RGBA crop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str = Field(description="Verified fine-grained fashion category label")
    broad_category: str = Field(
        description="Top-level category (Clothing, Footwear, Accessories, Bags)"
    )
    subcategory: str = Field(
        description="Subcategory grouping (e.g. Dresses, Tops, Sneakers, Tote)"
    )
    score: float = Field(
        description="Final confidence score from FashionCLIP verification"
    )
    box: List[float] = Field(
        description="Bounding box [xmin, ymin, xmax, ymax] in pixels"
    )
    mask: Optional[np.ndarray] = Field(
        default=None, description="Binary segmentation mask numpy array"
    )
    image: Image.Image = Field(
        description="Isolated object transparent PIL Image cutout (RGBA format)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Model metadata and intermediate scores"
    )


class FastFashionPipelineResult(BaseModel):
    """Pydantic v2 schema containing pipeline detection results and latency metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    objects: List[DetectedFashionObject] = Field(
        description="List of detected fashion objects with masks and transparent RGBA images"
    )
    total_objects: int = Field(description="Total count of detected objects")
    processing_time_ms: float = Field(
        description="Total pipeline execution latency in ms"
    )
    image_size: Tuple[int, int] = Field(
        description="Original image dimensions (width, height)"
    )
    processed_image: Optional[Image.Image] = Field(
        default=None,
        description="The loaded RGB PIL Image object processed by the pipeline",
    )
    annotated_image: Optional[Image.Image] = Field(
        default=None,
        description="PIL Image object with bounding boxes and labels drawn on top",
    )
    interactive_html: Optional[str] = Field(
        default=None,
        description="Interactive HTML visualization snippet ready for Jupyter/web",
    )

    def visualize(self, mode: str = "interactive"):
        """Displays visualization directly inside Jupyter notebooks.

        Args:
            mode: 'interactive' (HTML) or 'annotated' (PIL Image).
        """
        from IPython.display import HTML, display

        if mode == "interactive" and self.interactive_html:
            display(HTML(self.interactive_html))
        elif self.annotated_image:
            display(self.annotated_image)
        elif self.processed_image:
            display(self.processed_image)


class FastFashionPipeline:
    """High-speed zero-shot fashion segmentation & classification pipeline (<1s target latency).

    Pipeline Architecture:
    1. Grounding DINO: Detects candidate bounding boxes using batched broad category prompts.
    2. NMS & Filtering: Merges overlapping region proposals.
    3. SAM 2 (Sam2Detector / sam3_segmenter.py): Batch box-prompted instance segmentation in a single forward pass.
    4. FashionCLIP: Fast batch crop verification and fine-grained classification.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        sam_model_name: str = "facebook/sam2.1-hiera-small",
        use_broad_category_batches: bool = True,
    ):
        self.config = config or Config()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_broad_category_batches = use_broad_category_batches

        # Update config model settings for SAM2 if needed
        if "sam" not in self.config.models:
            self.config.models["sam"] = {}
        self.config.models["sam"]["name"] = sam_model_name

        # Initialize detector components using sam3_segmenter.py classes
        self.dino = GroundingDinoDetector(self.config)
        self.sam2 = Sam2Detector(self.config)
        self.fashion_clip = FashionClipDetector(self.config)

    def load_models(self) -> None:
        """Preloads all pipeline models to warm up memory and GPU caches."""
        logger.info(
            "Warming up fast pipeline models (Grounding DINO, SAM2, FashionCLIP)..."
        )
        self.dino.load_model()
        self.sam2.load_model()
        self.fashion_clip.load_model()
        logger.info("Fast pipeline models warm-up complete.")

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return float(inter / union) if union > 0 else 0.0

    def _apply_nms(
        self, proposals: List[Detection], iou_threshold: float = 0.5
    ) -> List[Detection]:
        """Applies Non-Maximum Suppression (NMS) to filter duplicate proposals across batches."""
        if not proposals:
            return []

        # Sort by confidence score descending
        sorted_props = sorted(proposals, key=lambda d: d.score, reverse=True)
        keep: List[Detection] = []

        for prop in sorted_props:
            overlap = False
            for kept in keep:
                if self._compute_iou(prop.box, kept.box) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(prop)

        return keep

    @time_it("Grounding DINO Batched Detection")
    def _detect_boxes_batched(
        self, image: Image.Image, batch_size: int = 4
    ) -> List[Detection]:
        """Runs Grounding DINO detection using category batches to improve latency and accuracy."""
        all_proposals: List[Detection] = []

        if self.use_broad_category_batches:
            # Broad stage queries: Clothing, Footwear, Accessories, Bags
            broad_queries = ["clothing", "footwear", "accessories", "bags"]
            proposals = self.dino.detect(
                image,
                queries=broad_queries,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            all_proposals.extend(proposals)
        else:
            # Fine categories split into query batches
            all_fine = []
            for broad, subcats in CATEGORY_HIERARCHY.items():
                for subcat, fine_list in subcats.items():
                    all_fine.extend(fine_list)
            all_fine = list(set(all_fine))

            # Query in batches
            for i in range(0, len(all_fine), batch_size):
                query_batch = all_fine[i : i + batch_size]
                proposals = self.dino.detect(
                    image,
                    queries=query_batch,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
                all_proposals.extend(proposals)

        # Merge and NMS deduplicate candidate boxes
        filtered_proposals = self._apply_nms(all_proposals, iou_threshold=0.5)
        logger.info(
            f"Batched Grounding DINO detected {len(all_proposals)} raw proposals, reduced to {len(filtered_proposals)} after NMS."
        )
        return filtered_proposals

    @time_it("SAM 2 Batch Box Segmentation")
    def _segment_boxes_batch(
        self, image: Image.Image, proposals: List[Detection]
    ) -> List[Dict[str, Any]]:
        """Passes all candidate boxes to Sam2Detector (from sam3_segmenter.py) for instant batch segmentation."""
        if not proposals:
            return []

        # Call extract_segmented_parts directly from Sam2Detector in sam3_segmenter.py
        cutouts = self.sam2.extract_segmented_parts(image, box_inputs=proposals)

        segmented_objects = []
        for prop, cutout in zip(proposals, cutouts):
            # Extract binary mask array from alpha channel of transparent RGBA cutout
            cutout_np = np.array(cutout)
            alpha_channel = (
                cutout_np[:, :, 3]
                if cutout_np.ndim == 3 and cutout_np.shape[2] == 4
                else cutout_np > 0
            )
            mask_np = alpha_channel > 0

            # Crop transparent cutout image to candidate bounding box with padding
            xmin, ymin, xmax, ymax = map(int, prop.box)
            w, h = cutout.size
            crop_xmin = max(0, xmin - 5)
            crop_ymin = max(0, ymin - 5)
            crop_xmax = min(w, xmax + 5)
            crop_ymax = min(h, ymax + 5)

            cropped_cutout = cutout
            if crop_xmax > crop_xmin and crop_ymax > crop_ymin:
                cropped_cutout = cutout.crop(
                    (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
                )

            segmented_objects.append(
                {
                    "label": prop.label,
                    "score": prop.score,
                    "box": prop.box,
                    "segmented_image": cropped_cutout,
                    "mask": mask_np,
                }
            )

        return segmented_objects

    @time_it("FashionCLIP Batch Crop Verification")
    def _verify_labels_fashion_clip(
        self, image: Image.Image, segmented_objects: List[Dict[str, Any]]
    ) -> List[DetectedFashionObject]:
        """Classifies each segmented cutout crop using FashionCLIP to verify fine-grained labels."""
        if not segmented_objects:
            return []

        # Build mock proposals for FashionCLIP classify_crops
        proposals_to_classify = []
        for obj in segmented_objects:
            proposals_to_classify.append(
                Detection(
                    box=obj["box"],
                    label=obj["label"],
                    score=obj["score"],
                    mask=obj["mask"],
                )
            )

        # Gather target fine categories
        all_fine_categories = []
        for broad, subcats in CATEGORY_HIERARCHY.items():
            for subcat, fine_list in subcats.items():
                all_fine_categories.extend(fine_list)
        all_fine_categories = list(set(all_fine_categories))

        # Classify all crops in batch with FashionCLIP
        classified_detections = self.fashion_clip.classify_crops(
            image=image,
            proposals=proposals_to_classify,
            categories=all_fine_categories,
        )

        final_objects: List[DetectedFashionObject] = []
        for idx, (obj, det) in enumerate(zip(segmented_objects, classified_detections)):
            verified_label = det.label
            broad_cat, subcat = get_parent_taxonomy_for_fine(verified_label)

            final_objects.append(
                DetectedFashionObject(
                    label=verified_label,
                    broad_category=broad_cat,
                    subcategory=subcat,
                    score=float(det.score),
                    box=obj["box"],
                    mask=obj["mask"],
                    image=obj["segmented_image"],  # Transparent PIL RGBA Image object
                    metadata={
                        "proposal_label": obj["label"],
                        "proposal_score": float(obj["score"]),
                        "fashion_clip_score": float(det.score),
                    },
                )
            )

        return final_objects

    def process(
        self, image_input: Union[str, Image.Image]
    ) -> FastFashionPipelineResult:
        """Executes the complete high-speed fashion detection and segmentation pipeline.

        Args:
            image_input: PIL Image or filepath/URL string.

        Returns:
            FastFashionPipelineResult: Pydantic v2 result model containing DetectedFashionObjects with RGBA images.
        """
        start_time = time.perf_counter()

        # 1. Load Image
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            from fashion_detector.utils import load_image

            image = load_image(image_input)

        img_w, img_h = image.size

        # 2. Stage 1: Batched Grounding DINO Detection
        proposals = self._detect_boxes_batched(image)

        if not proposals:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return FastFashionPipelineResult(
                objects=[],
                total_objects=0,
                processing_time_ms=round(elapsed_ms, 2),
                image_size=(img_w, img_h),
                processed_image=image,
                annotated_image=image.copy(),
                interactive_html=generate_interactive_html(image, []),
            )

        # 3. Stage 2: SAM 2 Single Batch Box Segmentation
        segmented_objects = self._segment_boxes_batch(image, proposals)

        # 4. Stage 3: FashionCLIP Fine-grained Crop Verification
        verified_objects = self._verify_labels_fashion_clip(image, segmented_objects)

        # 5. Generate Annotated Image & Interactive HTML
        detections_list = [
            Detection(
                box=obj.box,
                label=obj.label,
                score=obj.score,
                mask=obj.mask,
            )
            for obj in verified_objects
        ]

        annotated_img = draw_bounding_boxes(image, detections_list)
        html_str = generate_interactive_html(image, detections_list)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"FastFashionPipeline completed: {len(verified_objects)} objects detected in {elapsed_ms:.1f} ms."
        )

        return FastFashionPipelineResult(
            objects=verified_objects,
            total_objects=len(verified_objects),
            processing_time_ms=round(elapsed_ms, 2),
            image_size=(img_w, img_h),
            processed_image=image,
            annotated_image=annotated_img,
            interactive_html=html_str,
        )
