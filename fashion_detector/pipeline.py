from typing import Any, List, Optional, Dict
import numpy as np
import torch
from PIL import Image
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.models.fashion_clip import FashionClipDetector
from fashion_detector.logging import logger, time_it

# Refined CATEGORY_GROUPS with comprehensive child categories and common aliases
CATEGORY_GROUPS = {
    "footwear": [
        "sneakers",
        "flats",
        "loafers",
        "mules slides",
        "heels",
        "sandals",
        "boots",
        "dress shoes",
        "shoes",
    ],
    "bag": [
        "tote bags",
        "shoulder bags",
        "crossbody bags",
        "handle bags",
        "backpacks",
        "belt bags",
        "clutches",
        "briefcases",
        "duffel bags",
        "messenger bags",
        "handbags",
        "handbag",
        "bags",
        "bag",
    ],
    "upper body": [
        "tops",
        "sweaters",
        "shirts",
        "t shirts",
        "jackets blazers",
        "coats",
        "jackets",
        "blazers",
        "hoodies",
        "hoodie",
        "polo",
        "t-shirt",
        "shirt",
    ],
    "lower body": ["pants", "jeans", "skirts", "shorts", "jeans", "skirt"],
    "one-piece dress, suit, jumpsuit": [
        "dresses",
        "suits sets",
        "jumpsuits",
        "suits",
        "dress",
        "suit",
        "jumpsuit",
    ],
    "jewelry": [
        "jewelry",
        "earrings",
        "necklaces",
        "bracelets",
        "rings",
        "brooches",
        "necklace",
        "bracelet",
        "ring",
    ],
    "headwear": ["hats", "hat", "cap", "caps"],
    "eyewear": ["sunglasses"],
    "belt": ["belts", "belt"],
    "scarf": ["scarves shawls", "scarves", "scarf"],
    "wallet": ["wallets", "wallet"],
    "watch": ["watches", "watch"],
}

# General negative/neutral classes to absorb false positive crops in background, skin, face, hair, and nothing
NEGATIVE_CLASSES = ["human face", "hair", "skin", "background", "nothing"]


def normalize_string(s: str) -> str:
    """Helper to normalize text categories (lowercases, strips spaces, hyphens, and plurals)."""
    s = s.lower().replace("-", "").replace(" ", "").strip()
    if s.endswith("s") and not s.endswith("ss"):  # avoid converting dress -> dres
        s = s[:-1]
    return s


class HybridPipeline:
    """A two-stage hybrid pipeline for fashion item detection and fine-grained classification.

    Uses Composition over Inheritance:
    Stage 1: Locates regions of interest using a detector (e.g., Grounding DINO or YOLO).
             If using Grounding DINO, parent queries are run in small batches to prevent
             token competition, garbage subwords (like ##wear) are filtered out, and
             overlapping proposals are merged using overlap-aware NMS.
    Stage 2: Refines the labels and scores of these regions using FashionCLIP, restricting
             the classification pool to the subcategories of the detected parent groups,
             filtering against negative classes, and validating semantic garment containment
             to eliminate false part-garment duplicates (e.g. skirt inside dress).
    """

    @staticmethod
    def compute_iou(box1, box2) -> float:
        """Computes Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union else 0.0

    @staticmethod
    def build_reverse_mapping() -> Dict[str, str]:
        """Builds a mapping from fine-grained child categories to parent queries."""
        mapping = {}
        for parent, children in CATEGORY_GROUPS.items():
            for child in children:
                norm_child = normalize_string(child)
                mapping[norm_child] = parent
        return mapping

    def __init__(
        self,
        detector: BaseDetector,
        classifier: FashionClipDetector,
        min_box_confidence: float = 0.20,
        min_crop_size: int = 16,
    ):
        """Initializes the hybrid pipeline with a detector and classifier.

        Args:
            detector: Stage 1 object detector (e.g., GroundingDinoDetector, YoloDetector).
            classifier: Stage 2 region classifier (e.g., FashionClipDetector).
            min_box_confidence: Minimum score a Stage 1 proposal must have to be kept.
            min_crop_size: Minimum width/height in pixels for a proposal to be kept.
        """
        self.detector = detector
        self.classifier = classifier
        self.min_box_confidence = min_box_confidence
        self.min_crop_size = min_crop_size

    def _filter_contained_garments(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """Performs semantic garment containment validation.

        Discards nested part-body items (like tops, skirts, pants) if they are contained
        inside one-piece garments (like dresses, suits, jumpsuits) to prevent duplicate
        detections of the same garment.
        """
        one_piece_cats = {
            "dresses",
            "suits sets",
            "jumpsuits",
            "suits",
            "dress",
            "suit",
            "jumpsuit",
        }

        # Only discard nested part-body items (from upper/lower body)
        nestable_cats = {
            normalize_string(c)
            for c in (CATEGORY_GROUPS["upper body"] + CATEGORY_GROUPS["lower body"])
        }

        # Sort detections by area descending so larger garments are processed first
        sorted_dets = sorted(
            detections,
            key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
            reverse=True,
        )

        keep = []
        for det in sorted_dets:
            is_contained_in_one_piece = False
            det_area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
            if det_area == 0:
                continue

            for kept in keep:
                norm_det = normalize_string(det.label)
                if (
                    kept.label.lower().strip() in one_piece_cats
                    and det.label.lower().strip() not in one_piece_cats
                    and norm_det in nestable_cats
                ):
                    # Compute intersection box
                    x1 = max(det.box[0], kept.box[0])
                    y1 = max(det.box[1], kept.box[1])
                    x2 = min(det.box[2], kept.box[2])
                    y2 = min(det.box[3], kept.box[3])

                    inter_w = max(0, x2 - x1)
                    inter_h = max(0, y2 - y1)
                    inter_area = inter_w * inter_h

                    # If containment ratio is high (> 70%), filter out the sub-garment
                    containment_ratio = inter_area / det_area
                    if containment_ratio > 0.70:
                        logger.info(
                            f"Filtering out contained garment '{det.label}' inside one-piece '{kept.label}' (containment: {containment_ratio:.2f})"
                        )
                        is_contained_in_one_piece = True
                        break

            if not is_contained_in_one_piece:
                keep.append(det)

        return keep

    @time_it("Hybrid Pipeline Total Execution")
    def detect(
        self, image: Image.Image, categories: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Detection]:
        """Runs the hybrid two-stage detection pipeline on an image.

        Args:
            image: PIL Image to detect items in.
            categories: List of target categories for Stage 2. If None, uses config.
            **kwargs: Extra parameters to pass to Stage 1.

        Returns:
            List of refined and filtered Detection objects.
        """
        # Load default categories if not specified
        if not categories:
            categories = self.detector.config.get_all_categories()

        # Deduplicate and normalize categories
        categories = list(set(categories))

        # --- Stage 1: Coarse Proposal ---
        logger.info("Executing Stage 1: Region Proposal...")

        from fashion_detector.models.grounding_dino import GroundingDinoDetector

        is_dino = isinstance(self.detector, GroundingDinoDetector)

        raw_proposals = []

        if is_dino:
            # Map target categories to parent queries
            reverse_map = self.build_reverse_mapping()
            parent_queries = set()
            for cat in categories:
                norm_cat = normalize_string(cat)
                parent = reverse_map.get(norm_cat)
                if parent:
                    parent_queries.add(parent)
                else:
                    parent_queries.add(cat.lower().strip())

            parent_queries = list(parent_queries)
            logger.info(
                f"Hierarchical Mode: Map target categories to {len(parent_queries)} parent queries."
            )

            # Run Grounding DINO in batches of 4 to prevent prompt token competition
            batch_size = 4
            batches = [
                parent_queries[i : i + batch_size]
                for i in range(0, len(parent_queries), batch_size)
            ]
            for idx, batch in enumerate(batches):
                logger.info(
                    f"Running Grounding DINO Batch {idx+1}/{len(batches)}: {batch}"
                )
                dino_kwargs = kwargs.copy()
                dino_kwargs["queries"] = batch
                dino_kwargs["box_threshold"] = dino_kwargs.get(
                    "box_threshold", self.min_box_confidence
                )

                batch_proposals = self.detector.detect(image, **dino_kwargs)

                # Filter out garbage proposals (like empty labels or subwords starting with ##)
                for prop in batch_proposals:
                    clean_label = prop.label.strip().lower()
                    if clean_label and not clean_label.startswith("##"):
                        raw_proposals.append(prop)
        else:
            # For non-DINO detectors (e.g. YOLO), run once
            raw_proposals = self.detector.detect(image, **kwargs)

        # Sort proposals by score descending
        sorted_props = sorted(raw_proposals, key=lambda d: d.score, reverse=True)

        # Overlap-Aware NMS: keeps all parent labels that matched overlapping regions
        filtered_proposals = []
        for prop in sorted_props:
            if prop.score < self.min_box_confidence:
                continue

            xmin, ymin, xmax, ymax = prop.box
            w = xmax - xmin
            h = ymax - ymin

            if w < self.min_crop_size or h < self.min_crop_size:
                logger.debug(
                    f"Filtering proposal {prop.label} due to size: {w:.1f}x{h:.1f}"
                )
                continue

            overlap = False
            for kept in filtered_proposals:
                iou = self.compute_iou(prop.box, kept["box"])
                if iou > 0.45:
                    kept["labels"].add(prop.label)
                    overlap = True
                    break

            if not overlap:
                filtered_proposals.append(
                    {
                        "box": prop.box,
                        "labels": {prop.label},
                        "score": prop.score,
                        "mask": prop.mask,
                    }
                )

        logger.info(
            f"Stage 1 yielded {len(raw_proposals)} proposals, filtered down to {len(filtered_proposals)} unique boxes."
        )

        if not filtered_proposals:
            return []

        # --- Stage 2: Fine-grained FashionCLIP Crop Classification ---
        logger.info("Executing Stage 2: FashionCLIP Crop Classification...")
        self.classifier.load_model()

        final_detections = []
        width, height = image.size
        reverse_map = self.build_reverse_mapping()

        # Process each crop
        for prop in filtered_proposals:
            # Build union of subcategories for all matched parent labels (exact, normalized, or partial)
            valid_candidates = []
            matched_groups = []
            for l in prop["labels"]:
                parent = l.lower().strip()
                norm_parent = normalize_string(parent)

                # Check exact match first
                if parent in CATEGORY_GROUPS:
                    matched_groups.append(parent)
                elif norm_parent in reverse_map:
                    matched_groups.append(reverse_map[norm_parent])
                else:
                    # Check normalized key matching
                    for k in CATEGORY_GROUPS.keys():
                        norm_key = normalize_string(k)
                        if (
                            norm_parent == norm_key
                            or norm_parent in norm_key
                            or norm_key in norm_parent
                        ):
                            matched_groups.append(k)

            matched_groups = list(set(matched_groups))
            for g in matched_groups:
                candidate_children = CATEGORY_GROUPS[g]
                valid_candidates.extend(
                    [
                        c
                        for c in categories
                        if normalize_string(c)
                        in [normalize_string(x) for x in candidate_children]
                    ]
                )

            # If the proposal doesn't match any category group, discard it entirely as noise
            if not valid_candidates:
                logger.info(
                    f"Discarded unrecognized proposal at {list(map(int, prop['box']))} with labels {list(prop['labels'])}"
                )
                continue

            valid_candidates = list(set(valid_candidates))
            classification_candidates = valid_candidates + NEGATIVE_CLASSES

            # Crop region
            xmin, ymin, xmax, ymax = prop["box"]
            crop = image.crop(
                (
                    max(0, int(xmin)),
                    max(0, int(ymin)),
                    min(width, int(xmax)),
                    min(height, int(ymax)),
                )
            )

            # Encode features and evaluate similarity
            text_features = self.classifier.get_text_features(classification_candidates)
            inputs = self.classifier.processor(images=crop, return_tensors="pt").to(
                self.classifier.device
            )

            with torch.no_grad():
                image_features = self.classifier.model.get_image_features(**inputs)
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif not isinstance(image_features, torch.Tensor):
                    image_features = image_features[0]
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                similarity = (
                    image_features @ text_features.T
                ) * self.classifier.model.logit_scale.exp()
                probs = similarity.softmax(dim=-1).cpu().numpy()[0]

            best_idx = np.argmax(probs)
            best_label = classification_candidates[best_idx]
            best_score = float(probs[best_idx])

            # Discard detection if it matched to a negative/background class
            if best_label in NEGATIVE_CLASSES:
                logger.info(
                    f"Discarded false positive '{list(prop['labels'])}' matched to negative class '{best_label}' with confidence {best_score:.2f}"
                )
                continue

            final_detections.append(
                Detection(
                    box=prop["box"],
                    label=best_label,
                    score=best_score,
                    mask=prop["mask"],
                    metadata={
                        "proposal_label": (
                            list(prop["labels"])[0] if prop["labels"] else None
                        ),
                        "proposal_labels": list(prop["labels"]),
                        "proposal_score": prop["score"],
                        "confidence": best_score,
                    },
                )
            )

        # Apply semantic containment validation round to eliminate part-of-garment duplicates
        final_detections = self._filter_contained_garments(final_detections)

        logger.info(
            f"Hybrid pipeline successfully completed. Yielded {len(final_detections)} refined fashion items."
        )
        return final_detections


class ClassificationFirstPipeline:
    """A classification-first hybrid pipeline for fashion item detection.

    Stage 1: Uses a classifier (e.g., FashionCLIP) to extract categories present in
             the image from a flat target category list.
    Stage 2: Runs a precision object detector (e.g., Grounding DINO or Florence-2)
             to locate ONLY the extracted active categories, preventing prompt competition
             and false positive boxes from unrelated classes.
    """

    def __init__(
        self,
        classifier: FashionClipDetector,
        detector: BaseDetector,
        presence_threshold: float = 0.15,
        min_box_confidence: float = 0.20,
    ):
        """Initializes the classification-first pipeline.

        Args:
            classifier: Stage 1 classifier used to extract present classes.
            detector: Stage 2 precision object detector.
            presence_threshold: Category presence alignment threshold for Stage 1.
            min_box_confidence: Minimum box score to keep in Stage 2.
        """
        self.classifier = classifier
        self.detector = detector
        self.presence_threshold = presence_threshold
        self.min_box_confidence = min_box_confidence

    @time_it("Classification-First Pipeline Total Execution")
    def detect(
        self, image: Image.Image, categories: Optional[List[str]] = None, **kwargs: Any
    ) -> List[Detection]:
        """Runs the classification-first pipeline on an image.

        Args:
            image: PIL Image.
            categories: Target categories to consider. If None, uses config categories.
            **kwargs: Extra parameters to pass to Stage 2 detector.

        Returns:
            List of validated and localized Detection objects.
        """
        if not categories:
            categories = self.detector.config.get_all_categories()

        # Deduplicate categories
        categories = list(set(categories))

        # --- Stage 1: Active Category Extraction ---
        logger.info("Executing Stage 1: FashionCLIP Active Category Extraction...")
        self.classifier.load_model()
        
        presence_thresh = kwargs.get("presence_threshold", self.presence_threshold)
        active_categories = self.classifier.extract_present_classes(
            image=image,
            user_categories=categories,
            presence_threshold=presence_thresh,
        )
        
        logger.info(f"Stage 1 completed. Active categories found: {active_categories}")
        if not active_categories:
            logger.info("No active categories extracted. Returning empty list.")
            return []

        # --- Stage 2: Precision Object Detection ---
        logger.info("Executing Stage 2: Precision Object Detection...")
        self.detector.load_model()

        det_kwargs = kwargs.copy()
        det_kwargs["queries"] = active_categories
        
        # Determine if using Florence-2 or Grounding DINO to set correct tasks/parameters
        from fashion_detector.models.florence2 import Florence2Detector
        if isinstance(self.detector, Florence2Detector):
            det_kwargs["task"] = det_kwargs.get("task", "<CAPTION_TO_PHRASE_GROUNDING>")
            det_kwargs["conf_threshold"] = det_kwargs.get("conf_threshold", self.min_box_confidence)
        else:
            det_kwargs["box_threshold"] = det_kwargs.get("box_threshold", self.min_box_confidence)

        # Run precision detection on active categories
        raw_detections = self.detector.detect(image, **det_kwargs)

        # Filter detections by minimum confidence
        min_conf = det_kwargs.get("box_threshold", det_kwargs.get("conf_threshold", self.min_box_confidence))
        final_detections = []
        for det in raw_detections:
            if det.score >= min_conf:
                final_detections.append(det)

        logger.info(
            f"Classification-first pipeline completed. Localized {len(final_detections)} items."
        )
        return final_detections
