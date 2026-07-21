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
        """Encodes candidate categories into normalized text embeddings using prompt ensembling."""
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

        ensemble_embeddings = []
        for cat in categories:
            # Generate prompts using all templates for this category
            prompts = [tpl.format(cat) for tpl in templates]
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                if hasattr(text_features, "text_embeds"):
                    text_features = text_features.text_embeds
                elif hasattr(text_features, "pooler_output"):
                    text_features = text_features.pooler_output
                elif not isinstance(text_features, torch.Tensor):
                    text_features = text_features[0]

                # Normalize and average embeddings across templates
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_embedding = text_features.mean(dim=0)
                mean_embedding = mean_embedding / mean_embedding.norm(dim=-1)

            ensemble_embeddings.append(mean_embedding)

        stacked_features = torch.stack(ensemble_embeddings, dim=0)
        self.text_features_cache[cache_key] = stacked_features
        return stacked_features

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

            with torch.no_grad():
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
