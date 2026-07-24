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
from fashion_detector.models.sam3_segmenter import Sam2Detector
from fashion_detector.models.fashion_clip import FashionClipDetector
from fashion_detector.utils import (
    CATEGORY_HIERARCHY,
    get_broad_categories,
    get_fine_categories_for_broad,
    get_parent_taxonomy_for_fine,
    draw_bounding_boxes,
    generate_interactive_html,
)

NEGATIVE_CLASSES = ["human face", "skin", "hair", "background", "nothing"]


class DetectedBoxObject(BaseModel):
    """Pydantic v2 schema representing a single detected fashion item (bounding box focus)."""

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
        default=None, description="Optional binary segmentation mask numpy array"
    )
    crop_image: Image.Image = Field(
        description="Bounding box cropped PIL Image cutout (RGB or RGBA format)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Model metadata and intermediate scores"
    )


class BoxFashionPipelineResult(BaseModel):
    """Pydantic v2 schema containing box detection results and ultra-low latency metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    objects: List[DetectedBoxObject] = Field(
        description="List of detected fashion objects with bounding boxes and crop images"
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

    def to_json_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dictionary excluding raw PIL images."""
        return {
            "total_objects": self.total_objects,
            "processing_time_ms": self.processing_time_ms,
            "image_size": self.image_size,
            "objects": [
                {
                    "label": obj.label,
                    "broad_category": obj.broad_category,
                    "subcategory": obj.subcategory,
                    "score": obj.score,
                    "box": obj.box,
                    "metadata": obj.metadata,
                }
                for obj in self.objects
            ],
        }


class FastBoxFashionPipeline:
    """High-speed bounding-box-first fashion detection & classification pipeline (~200ms latency).

    Pipeline Architecture:
    1. Grounding DINO: High-precision zero-shot bounding box detection using broad category prompts.
    2. NMS & Filtering: Merges overlapping region proposals.
    3. Direct Box Cropping: Instant region extraction directly from bounding box coordinates (No SAM 2).
    4. FashionCLIP: Fast batch crop verification, scoped category classification, and negative class suppression.
    5. Optional SAM 2: Lazily invoked only when include_masks=True.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        min_score_threshold: float = 0.20,
        max_detection_size: int = 640,
        use_broad_category_batches: bool = False,
    ):
        self.config = config or Config()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_score_threshold = min_score_threshold
        self.max_detection_size = max_detection_size
        self.use_broad_category_batches = use_broad_category_batches

        # Initialize core detectors (SAM 2 is initialized lazily if requested)
        self.dino = GroundingDinoDetector(self.config)
        self.fashion_clip = FashionClipDetector(self.config)
        self.sam2 = None

    def load_models(self, include_sam2: bool = False) -> None:
        """Preloads pipeline models and pre-caches FashionCLIP text embeddings on GPU/MPS."""
        logger.info(
            "Warming up FastBoxFashionPipeline models (Grounding DINO, FashionCLIP)..."
        )
        self.dino.load_model()
        self.fashion_clip.load_model()

        if include_sam2:
            if self.sam2 is None:
                self.sam2 = Sam2Detector(self.config)
            self.sam2.load_model()

        # Pre-cache FashionCLIP text features for broad candidate sets + negative classes
        logger.info("Pre-caching vectorized FashionCLIP text embeddings...")
        for broad_cat in get_broad_categories():
            fine_cats = get_fine_categories_for_broad(broad_cat) + NEGATIVE_CLASSES
            self.fashion_clip.get_text_features(fine_cats)

        logger.info("FastBoxFashionPipeline warm-up & text embeddings cache complete.")

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
        """Runs Grounding DINO detection using broad category prompts."""
        all_proposals: List[Detection] = []

        if self.use_broad_category_batches:
            broad_queries = ["clothing", "footwear", "accessories", "bags"]
            proposals = self.dino.detect(
                image,
                queries=broad_queries,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                max_detection_size=self.max_detection_size,
            )
            for p in proposals:
                label_lower = p.label.lower()
                if "footwear" in label_lower or "shoe" in label_lower:
                    p.metadata["broad_category"] = "Footwear"
                elif "bag" in label_lower:
                    p.metadata["broad_category"] = "Bags"
                elif "accessor" in label_lower or "watch" in label_lower or "hat" in label_lower or "belt" in label_lower:
                    p.metadata["broad_category"] = "Accessories"
                else:
                    p.metadata["broad_category"] = "Clothing"

            all_proposals.extend(proposals)
        else:
            all_fine = []
            for broad, subcats in CATEGORY_HIERARCHY.items():
                for subcat, fine_list in subcats.items():
                    all_fine.extend(fine_list)
            all_fine = list(set(all_fine))

            for i in range(0, len(all_fine), batch_size):
                query_batch = all_fine[i : i + batch_size]
                proposals = self.dino.detect(
                    image,
                    queries=query_batch,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    max_detection_size=self.max_detection_size,
                )
                all_proposals.extend(proposals)

        filtered_proposals = self._apply_nms(all_proposals, iou_threshold=0.5)
        logger.info(
            f"Batched Grounding DINO detected {len(all_proposals)} raw proposals, reduced to {len(filtered_proposals)} after NMS."
        )
        return filtered_proposals

    @time_it("Direct Box Crop Extraction")
    def _extract_box_crops_direct(
        self, image: Image.Image, proposals: List[Detection]
    ) -> List[Dict[str, Any]]:
        """Instantly crops region proposals directly from bounding box coordinates (0ms latency, No SAM 2)."""
        if not proposals:
            return []

        width, height = image.size
        cropped_objects = []

        for prop in proposals:
            xmin, ymin, xmax, ymax = map(int, prop.box)
            crop_xmin = max(0, xmin)
            crop_ymin = max(0, ymin)
            crop_xmax = min(width, xmax)
            crop_ymax = min(height, ymax)

            if crop_xmax > crop_xmin and crop_ymax > crop_ymin:
                crop = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
            else:
                crop = image.copy()

            cropped_objects.append(
                {
                    "label": prop.label,
                    "score": prop.score,
                    "box": prop.box,
                    "crop_image": crop,
                    "mask": None,
                    "broad_category": prop.metadata.get("broad_category", "Clothing"),
                }
            )

        return cropped_objects

    @time_it("FashionCLIP Scoped Verification")
    def _verify_labels_fashion_clip(
        self, image: Image.Image, candidate_objects: List[Dict[str, Any]]
    ) -> List[DetectedBoxObject]:
        """Classifies each candidate box crop using FashionCLIP with broad category candidate scoping and negative class suppression."""
        if not candidate_objects:
            return []

        final_objects: List[DetectedBoxObject] = []

        # Group proposals by broad category for optimal scoped batch classification
        grouped_proposals: Dict[str, List[Dict[str, Any]]] = {}
        for obj in candidate_objects:
            broad_cat = obj.get("broad_category", "Clothing")
            grouped_proposals.setdefault(broad_cat, []).append(obj)

        for broad_cat, group_objs in grouped_proposals.items():
            candidate_cats = get_fine_categories_for_broad(broad_cat) + NEGATIVE_CLASSES

            proposals_to_classify = [
                Detection(
                    box=obj["box"],
                    label=obj["label"],
                    score=obj["score"],
                    mask=obj["mask"],
                )
                for obj in group_objs
            ]

            classified_detections = self.fashion_clip.classify_crops(
                image=image,
                proposals=proposals_to_classify,
                categories=candidate_cats,
            )

            for obj, det in zip(group_objs, classified_detections):
                verified_label = det.label

                # Filter out negative class detections (face, hair, skin, background) and low-confidence items
                if (
                    verified_label.lower() in NEGATIVE_CLASSES
                    or det.score < self.min_score_threshold
                ):
                    logger.info(
                        f"Filtering false positive / low score object: '{verified_label}' (score={det.score:.2f})"
                    )
                    continue

                derived_broad, subcat = get_parent_taxonomy_for_fine(verified_label)
                final_broad = derived_broad if derived_broad != "Clothing" or broad_cat == "Clothing" else broad_cat

                final_objects.append(
                    DetectedBoxObject(
                        label=verified_label,
                        broad_category=final_broad,
                        subcategory=subcat,
                        score=float(det.score),
                        box=obj["box"],
                        mask=obj["mask"],
                        crop_image=obj["crop_image"],
                        metadata={
                            "proposal_label": obj["label"],
                            "proposal_score": float(obj["score"]),
                            "fashion_clip_score": float(det.score),
                        },
                    )
                )

        # Apply post-classification NMS & Containment Filtering to eliminate outer group boxes
        return self._apply_containment_nms(
            final_objects,
            image_size=image.size,
            iou_threshold=0.45,
            ioa_threshold=0.75,
        )

    @staticmethod
    def _apply_containment_nms(
        objects: List[DetectedBoxObject],
        image_size: Tuple[int, int] = (800, 600),
        iou_threshold: float = 0.45,
        ioa_threshold: float = 0.75,
    ) -> List[DetectedBoxObject]:
        """Applies IoU, IoA containment, and image coverage filtering to eliminate giant outer group boxes."""
        if not objects:
            return []

        img_area = float(image_size[0] * image_size[1])

        # Sort objects by score descending
        sorted_objs = sorted(objects, key=lambda o: o.score, reverse=True)
        kept_objs: List[DetectedBoxObject] = []

        for obj in sorted_objs:
            box_a = obj.box
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            cov_a = area_a / img_area if img_area > 0 else 0.0
            drop = False

            # If a box covers > 60% of the total image area AND other smaller item boxes exist, drop the giant container box
            if cov_a > 0.60 and len(sorted_objs) > 1:
                logger.info(f"Filtering giant container box covering {cov_a*100:.1f}% of image: label='{obj.label}'")
                continue

            for kept in kept_objs:
                box_b = kept.box
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

                x1 = max(box_a[0], box_b[0])
                y1 = max(box_a[1], box_b[1])
                x2 = min(box_a[2], box_b[2])
                y2 = min(box_a[3], box_b[3])

                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0

                min_area = min(area_a, area_b)
                ioa = inter / min_area if min_area > 0 else 0.0

                # Drop if high IoU or high containment overlap
                if iou > iou_threshold or ioa > ioa_threshold:
                    drop = True
                    break

            if not drop:
                kept_objs.append(obj)

        return kept_objs

    def process(
        self,
        image_input: Union[str, Image.Image],
        include_masks: bool = False,
    ) -> BoxFashionPipelineResult:
        """Executes the high-speed bounding-box fashion detection & classification pipeline (~200ms latency).

        Args:
            image_input: PIL Image or filepath/URL string.
            include_masks: Set to True to lazily run SAM 2 segmentation masks (~139ms extra time). Default False.

        Returns:
            BoxFashionPipelineResult: Pydantic v2 result model containing DetectedBoxObjects.
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
            return BoxFashionPipelineResult(
                objects=[],
                total_objects=0,
                processing_time_ms=round(elapsed_ms, 2),
                image_size=(img_w, img_h),
                processed_image=image,
                annotated_image=image.copy(),
                interactive_html=generate_interactive_html(image, []),
            )

        # 3. Stage 2: Crop Extraction (Fast Direct Box Crop vs Optional SAM 2 Segmentation)
        if include_masks:
            if self.sam2 is None:
                self.sam2 = Sam2Detector(self.config)
                self.sam2.load_model()
            cutouts = self.sam2.extract_segmented_parts(image, box_inputs=proposals)
            candidate_objects = []
            for prop, cutout in zip(proposals, cutouts):
                cutout_np = np.array(cutout)
                alpha_channel = cutout_np[:, :, 3] if cutout_np.ndim == 3 and cutout_np.shape[2] == 4 else cutout_np > 0
                mask_np = alpha_channel > 0
                candidate_objects.append(
                    {
                        "label": prop.label,
                        "score": prop.score,
                        "box": prop.box,
                        "crop_image": cutout,
                        "mask": mask_np,
                        "broad_category": prop.metadata.get("broad_category", "Clothing"),
                    }
                )
        else:
            candidate_objects = self._extract_box_crops_direct(image, proposals)

        # 4. Stage 3: FashionCLIP Fine-grained Crop Verification with Scoped Filtering
        verified_objects = self._verify_labels_fashion_clip(image, candidate_objects)

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
            f"FastBoxFashionPipeline completed: {len(verified_objects)} objects detected in {elapsed_ms:.1f} ms."
        )

        return BoxFashionPipelineResult(
            objects=verified_objects,
            total_objects=len(verified_objects),
            processing_time_ms=round(elapsed_ms, 2),
            image_size=(img_w, img_h),
            processed_image=image,
            annotated_image=annotated_img,
            interactive_html=html_str,
        )
