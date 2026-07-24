import os
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class FashionClipDetector(BaseDetector):
    """FashionCLIP domain-specific classifier and zero-shot tagger.

    Acts as a Stage 2 classifier in a hybrid pipeline: takes region proposals
    (from YOLO or Grounding DINO), crops them, and classifies them using FashionCLIP.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.model_name = config.models.get("fashion_clip", {}).get(
            "name", "patrickjohncyh/fashion-clip"
        )

        self.processor = None
        self.model = None
        self.text_features_cache = {}

    def load_model(self) -> None:
        """Loads FashionCLIP processor and model from HF cache or downloads them."""
        if self.model is not None:
            return

        logger.info(
            f"Loading FashionCLIP model: {self.model_name} on device: {self.device}"
        )

        # Hugging Face CLIPModel can load FashionCLIP since it uses standard CLIP architecture
        self.processor = CLIPProcessor.from_pretrained(
            self.model_name, cache_dir=os.path.join(self.cache_dir, "huggingface")
        )
        self.model = CLIPModel.from_pretrained(
            self.model_name, cache_dir=os.path.join(self.cache_dir, "huggingface")
        ).to(self.device)

        self.model.eval()
        logger.info("FashionCLIP model loaded successfully.")

    @time_it("FashionCLIP Text Embeddings")
    def get_text_features(self, categories: List[str]) -> torch.Tensor:
        """Encodes candidate categories into normalized text embeddings using prompt ensembling.
        Vectorized into a single batch forward pass for maximum performance (<10ms).
        """
        cache_key = tuple(sorted(categories))
        if cache_key in self.text_features_cache:
            return self.text_features_cache[cache_key]

        self.load_model()

        # Ensembled templates for fashion domain
        templates = [
            "a photo of a {}",
            "a fashion photo of a {}",
            "a person wearing {}",
            "close-up photo of a {}",
            "a model wearing a {}",
            "a product photo of a {}",
        ]

        # Flatten all category x template prompt combinations into a single batch
        all_prompts = [tpl.format(cat) for cat in categories for tpl in templates]
        inputs = self.processor(text=all_prompts, padding=True, return_tensors="pt").to(
            self.device
        )

        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "text_embeds"):
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            elif not isinstance(text_features, torch.Tensor):
                text_features = text_features[0]

            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Reshape tensor to (num_categories, num_templates, embedding_dim)
            num_cats = len(categories)
            num_tpls = len(templates)
            embed_dim = text_features.shape[-1]
            text_features = text_features.view(num_cats, num_tpls, embed_dim)

            # Average across templates and normalize final vector per category
            mean_embeddings = text_features.mean(dim=1)
            mean_embeddings = mean_embeddings / mean_embeddings.norm(
                dim=-1, keepdim=True
            )

        self.text_features_cache[cache_key] = mean_embeddings
        return mean_embeddings

    @time_it("FashionCLIP Crop Classification")
    def classify_crops(
        self,
        image: Image.Image,
        proposals: List[Detection],
        categories: Optional[List[str]] = None,
    ) -> List[Detection]:
        """Classifies each region proposal using FashionCLIP.

        Args:
            image: Original PIL Image.
            proposals: List of Detection objects (proposals).
            categories: List of target categories. If None, uses config categories.

        Returns:
            List of updated Detection objects with refined labels and scores.
        """
        if not proposals:
            return []

        self.load_model()

        if not categories:
            categories = self.config.get_all_categories()

        # Get text embeddings
        text_features = self.get_text_features(categories)  # Shape: (num_categories, D)

        refined_detections = []
        width, height = image.size

        # Classify crops in batches of 16 to optimize GPU usage
        batch_size = 16
        for start_idx in range(0, len(proposals), batch_size):
            batch_props = proposals[start_idx : start_idx + batch_size]
            crops = []

            for prop in batch_props:
                xmin, ymin, xmax, ymax = prop.box
                # Crop and pad to make sure it's valid
                crop = image.crop(
                    (
                        max(0, int(xmin)),
                        max(0, int(ymin)),
                        min(width, int(xmax)),
                        min(height, int(ymax)),
                    )
                )
                crops.append(crop)

            # Prepare image inputs
            inputs = self.processor(images=crops, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                image_features = self.model.get_image_features(**inputs)
                # Safe extraction if transformers returns a BaseModelOutputWithPooling or other ModelOutput object
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif not isinstance(image_features, torch.Tensor):
                    image_features = image_features[0]
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )  # Shape: (batch_crops, D)

                # Compute similarity matrix (batch_crops, num_categories)
                similarity = (
                    image_features @ text_features.T
                ) * self.model.logit_scale.exp()
                probs = similarity.softmax(dim=-1).cpu().numpy()

            for idx, prop in enumerate(batch_props):
                crop_probs = probs[idx]
                best_class_idx = np.argmax(crop_probs)
                best_score = float(crop_probs[best_class_idx])
                best_label = categories[best_class_idx]

                # Keep original mask if available
                refined_detections.append(
                    Detection(
                        box=prop.box,
                        label=best_label,
                        score=best_score,
                        mask=prop.mask,
                        metadata={
                            "proposal_label": prop.label,
                            "proposal_score": prop.score,
                            "fashion_clip_similarities": crop_probs.tolist(),
                        },
                    )
                )

        return refined_detections

    @time_it("FashionCLIP Global Presence Extraction")
    def extract_present_classes(
        self,
        image: Image.Image,
        user_categories: List[str],
        presence_threshold: float = 0.15,
    ) -> List[str]:
        """
        Adapts the existing batch classification infrastructure to run global
        and spatial-quadrant visual-text alignment over the whole image.

        Args:
            image: Original PIL Image.
            user_categories: Flat taxonomy string list to search for.
            presence_threshold: Probability margin above which a category is confirmed.

        Returns:
            A clean list of unique string categories present in the image matrix.
        """
        width, height = image.size

        # 1. Mock proposals representing the whole image and key zones (left, right, center)
        # This keeps group photos from drowning out local items (e.g. shoes, small accessories)
        mock_proposals = [
            Detection(box=[0, 0, width, height], label="global", score=1.0),
            Detection(box=[0, 0, width // 2, height], label="left_half", score=1.0),
            Detection(
                box=[width // 2, 0, width, height], label="right_half", score=1.0
            ),
            Detection(
                box=[0, height // 3, width, 2 * height // 3],
                label="center_belt",
                score=1.0,
            ),
            Detection(
                box=[0, 2 * height // 3, width, height],
                label="bottom_footwear",
                score=1.0,
            ),
        ]

        # 2. Reuse your existing, highly optimized batch processing & ensembled templates logic
        # This protects you from text parsing logic errors (like matching "ring" in "wearing")
        refined_detections = self.classify_crops(
            image=image, proposals=mock_proposals, categories=user_categories
        )

        # 3. Aggregate only the labels that cross your threshold bar
        confirmed_classes = set()
        for det in refined_detections:
            if det.score >= presence_threshold:
                confirmed_classes.add(det.label)

        return sorted(list(confirmed_classes))

    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Runs the complete detection pipeline.

        If proposals are provided in kwargs, classifies them.
        Otherwise, raises an error as FashionCLIP requires candidate regions.

        Args:
            image: PIL Image.
            **kwargs: Must contain 'proposals' (List[Detection]).

        Returns:
            List of Detection objects.
        """
        proposals = kwargs.get("proposals")
        if proposals is None:
            raise ValueError(
                "FashionCLIP requires a list of candidate region proposals to classify. Please provide 'proposals=...' in kwargs."
            )

        queries = kwargs.get("queries")
        return self.classify_crops(image, proposals, queries)
