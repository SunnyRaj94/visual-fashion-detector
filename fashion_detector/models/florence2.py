import os
from typing import Any, List, Optional, Union, Dict
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class Florence2Detector(BaseDetector):
    """Florence-2 vision-language object detector by Microsoft."""

    ####################################################################
    # Helpers
    ####################################################################

    @staticmethod
    def compute_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter

        return inter / union if union else 0

    def __init__(self, config: Any):
        super().__init__(config)
        self.model_name = config.models.get("florence2", {}).get(
            "name", "microsoft/Florence-2-base"
        )
        self.conf_threshold = config.models.get("florence2", {}).get(
            "conf_threshold", 0.3
        )
        self.DEFAULT_LIFESTYLE_CATEGORIES = self.config.get_all_categories()
        logger.info(
            f"Florence-2 Detector initialized with model: {self.model_name}, confidence threshold: {self.conf_threshold}"
        )
        logger.info(
            f"Florence-2 Detector default lifestyle categories: {self.DEFAULT_LIFESTYLE_CATEGORIES}"
        )

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads Florence-2 processor and model from HF cache or downloads them."""
        if self.model is not None:
            return

        logger.info(
            f"Loading Florence-2 model: {self.model_name} on device: {self.device}"
        )

        # Apply runtime patches for transformers v5 compatibility with Florence-2
        from transformers import PretrainedConfig
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # 1. Patch config for forced_bos_token_id (removed/renamed in transformers v5)
        PretrainedConfig.forced_bos_token_id = None

        # 2. Patch tokenizer for additional_special_tokens (removed in transformers v5)
        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):

            @property
            def additional_special_tokens(self):
                standard_tokens = {
                    getattr(self, "bos_token", None),
                    getattr(self, "eos_token", None),
                    getattr(self, "pad_token", None),
                    getattr(self, "unk_token", None),
                    getattr(self, "mask_token", None),
                    getattr(self, "cls_token", None),
                    getattr(self, "sep_token", None),
                }
                return [
                    t
                    for t in self.all_special_tokens
                    if t not in standard_tokens and t is not None
                ]

            PreTrainedTokenizerBase.additional_special_tokens = (
                additional_special_tokens
            )

        # 3. Patch model class to resolve the _supports_sdpa initialization race condition
        try:
            model_class = get_class_from_dynamic_module(
                "modeling_florence2.Florence2ForConditionalGeneration", self.model_name
            )

            @property
            def safe_supports_sdpa(self):
                if not hasattr(self, "language_model") or self.language_model is None:
                    return False
                return self.language_model._supports_sdpa

            model_class._supports_sdpa = safe_supports_sdpa
        except Exception as e:
            logger.warning(
                f"Failed to patch Florence-2 model class: {e}. If loading fails, verify library compatibility."
            )

        # We need to trust remote code since Microsoft's Florence-2 uses custom model code.
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=os.path.join(self.cache_dir, "huggingface"),
        )

        # Note: on CPU or MPS we load FP32 or float16 if supported. Let's load float32 by default
        # and support float16 on CUDA/MPS for speed/memory efficiency.
        torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir=os.path.join(self.cache_dir, "huggingface"),
        ).to(self.device)

        self.model.eval()
        self.templates = [
            "a photo of a {}",
            "a fashion photo of a {}",
            "a person wearing {}",
            "close-up photo of a {}",
            "a model wearing a {}",
            "a product photo of a {}",
        ]

        # In-Memory Cache Mapping: { "t shirts": torch.Tensor([1, 512]) }
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        logger.info("Florence-2 model loaded successfully.")

    @time_it("Florence-2 Inference")
    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Runs Florence-2 detection.

        Args:
            image: PIL Image.
            **kwargs:
                task: Either '<OD>' (default) or '<CAPTION_TO_PHRASE_GROUNDING>'.
                queries: List of custom queries (for grounding task).

        Returns:
            List of Detection objects.
        """
        self.load_model()

        task = kwargs.get("task", "<OD>")
        queries = kwargs.get("queries")

        # Select prompt based on task
        if task == "<CAPTION_TO_PHRASE_GROUNDING>":
            if not queries:
                queries = self.config.get_all_categories()
            # Combine queries into a comma-separated list
            query_str = ", ".join(queries)
            prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{query_str}"
        else:
            task = "<OD>"
            prompt = "<OD>"

        logger.info(f"Florence-2 running task: {task}")

        # Format inputs
        # Florence-2 expects PIL image and prompt.
        # We manually resize to a square shape (e.g. 768x768) to prevent AssertionError in dynamic model code.
        orig_w, orig_h = image.size
        resized_image = image.resize((768, 768))

        torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        inputs = self.processor(text=prompt, images=resized_image, return_tensors="pt")
        inputs = {
            k: (
                v.to(self.device).to(torch_dtype)
                if v.dtype == torch.float
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }

        # Run generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                use_cache=False,  # Disable cache to prevent NoneType shape AssertionError/TypeErrors during decoding
            )

        generated_text = self.processor.batch_decode(
            outputs, skip_special_tokens=False
        )[0]

        # Parse answers
        # We scale bounding boxes back to the original image dimensions.
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task, image_size=(orig_w, orig_h)
        )

        detections = []

        if task == "<OD>":
            od_data = parsed_answer.get("<OD>", {})
            bboxes = od_data.get("bboxes", [])
            labels = od_data.get("labels", [])

            # Florence-2 doesn't always provide confidence scores directly in OD task output.
            # We assign a default score or use confidence values if available.
            for box, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax = box
                detections.append(
                    Detection(
                        box=[float(xmin), float(ymin), float(xmax), float(ymax)],
                        label=label.lower().strip(),
                        score=1.0,  # Florence-2 OD is binary detection, confidence is not returned by HF post_process
                        metadata={"raw_label": label},
                    )
                )
        elif task == "<CAPTION_TO_PHRASE_GROUNDING>":
            grounding_data = parsed_answer.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
            bboxes = grounding_data.get("bboxes", [])
            labels = grounding_data.get("labels", [])

            for box, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax = box
                detections.append(
                    Detection(
                        box=[float(xmin), float(ymin), float(xmax), float(ymax)],
                        label=label.lower().strip(),
                        score=1.0,
                        metadata={"raw_label": label},
                    )
                )

        # Synonym mapping to bridge foundation model vocabulary with our database schema
        synonyms = {
            "footwear": "shoes",
            "trousers": "pants",
            "purse": "handbag",
            "clutch": "handbag",
            "satchel": "handbag",
            "spectacles": "sunglasses",
            "glasses": "sunglasses",
            "eyewear": "sunglasses",
            "outerwear": "jacket",
            "headwear": "hat",
            "neckwear": "scarf",
        }

        # Optional confidence thresholding or filtering to fashion categories
        valid_cats = set(self.config.get_all_categories())
        filtered_detections = []
        for det in detections:
            # Map synonym if present
            mapped_label = synonyms.get(det.label, det.label)

            # Match label to our fashion categories (partial match or exact match)
            matched = False
            for cat in valid_cats:
                if cat in mapped_label or mapped_label in cat:
                    det.label = cat
                    matched = True
                    break

            if matched:
                filtered_detections.append(det)

        logger.info(
            f"Florence-2 detected {len(filtered_detections)} fashion items (filtered from {len(detections)} total)."
        )
        return filtered_detections

    def get_text_embedding_with_cache(self, category: str) -> torch.Tensor:
        """Generates an ensembled vector embedding, fetching from cache if hit."""
        cleaned_cat = category.strip().lower()

        # 1. Cache Hit Check
        if cleaned_cat in self.embedding_cache:
            return self.embedding_cache[cleaned_cat]

        # 2. Cache Miss: Fill templates and encode text-only features via FashionCLIP
        prompt_variations = [tmpl.format(cleaned_cat) for tmpl in self.templates]

        # FashionCLIP's processor allows pure text tokens without needing an accompanying image matrix
        inputs = self.processor(
            text=prompt_variations, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Aggregate variants together
            ensembled_embedding = text_features.mean(dim=0, keepdim=True)
            ensembled_embedding = ensembled_embedding / ensembled_embedding.norm(
                dim=-1, keepdim=True
            )

        # 3. Commit vector footprint to internal cache memory
        self.embedding_cache[cleaned_cat] = ensembled_embedding
        return ensembled_embedding

    @time_it("Florence-2 Pure Class Presence Extraction")
    def extract_present_classes(
        self, image: Image.Image, user_categories: List[str]
    ) -> List[str]:
        """
        Extracts verified active classes by intersecting a non-suggestive detailed
        caption canvas with your target fashion taxonomy.
        """
        self.load_model()

        # Phase 1: Generate a Neutral Image Caption (Zero Suggestion Prompts)
        # Using <DETAILED_CAPTION> keeps the model from hallucinating a provided list
        prompt = "<DETAILED_CAPTION>"

        orig_w, orig_h = image.size
        resized_image = image.resize((768, 768))
        torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        inputs = self.processor(text=prompt, images=resized_image, return_tensors="pt")
        inputs = {
            k: (
                v.to(self.device).to(torch_dtype)
                if v.dtype == torch.float
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
                use_cache=False,
            )

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[
            0
        ]
        clean_canvas_description = generated_text.lower().strip()
        logger.info(f"Verified Image Canvas Text Footprint: {clean_canvas_description}")

        # Phase 2: Run Strict Linguistic Correlation Checks
        verified_classes = set()

        for category in user_categories:
            cat_lower = category.strip().lower()

            # 1. Base keyword checks (e.g., "blazer" or "sunglasses")
            if cat_lower in clean_canvas_description:
                verified_classes.add(category)
                continue

            # 2. De-pluralization logic for common fashion words
            if cat_lower.endswith("s"):
                singular = cat_lower[:-1]
                # Avoid turning words like "dress" into "dres"
                if len(singular) > 3 and singular in clean_canvas_description:
                    verified_classes.add(category)
                    continue

            # 3. Component word alignment (e.g., matching "jackets blazers" to "blazer")
            words = cat_lower.split()
            if len(words) > 1:
                for word in words:
                    if len(word) > 3 and word in clean_canvas_description:
                        verified_classes.add(category)
                        break

        return sorted(list(verified_classes))
