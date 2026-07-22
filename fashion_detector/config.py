import os
import yaml
from typing import Any, Dict, List, Optional
from fashion_detector.logging import logger

DEFAULT_CATEGORIES = {
    "clothing": [
        "shirt",
        "t-shirt",
        "polo",
        "sweater",
        "hoodie",
        "jacket",
        "blazer",
        "coat",
        "dress",
        "top",
        "jeans",
        "pants",
        "shorts",
        "skirt",
        "suit",
        "jumpsuit",
        "scarf",
    ],
    "accessories": [
        "handbag",
        "backpack",
        "wallet",
        "belt",
        "watch",
        "bracelet",
        "necklace",
        "earrings",
        "ring",
        "hat",
        "cap",
        "sunglasses",
        "shoes",
        "boots",
        "sneakers",
    ],
}


class ModelConfig:
    def __init__(
        self, name: str, threshold: float = 0.25, extra: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.threshold = threshold
        self.extra = extra or {}


class Config:
    """Configuration class that loads settings from a YAML file."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.raw_config: Dict[str, Any] = {}

        # Default settings
        self.device = "cpu"
        self.cache_dir = "./cache"
        self.log_level = "INFO"
        self.log_file = "logs/fashion_detector.log"

        self.categories: Dict[str, List[str]] = DEFAULT_CATEGORIES
        self.models: Dict[str, Dict[str, Any]] = {}

        if config_path:
            self.load_from_file(config_path)
        else:
            self._set_defaults()

    def _set_defaults(self) -> None:
        """Set up standard default values when no config file is loaded."""
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.cache_dir = os.path.abspath("./cache")
        self.models = {
            "grounding_dino": {
                "name": "IDEA-Research/grounding-dino-tiny",
                "box_threshold": 0.25,
                "text_threshold": 0.25,
            },
            "sam": {
                "name": "facebook/sam3.1",
                "points_per_side": 16,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.85,
                "box_threshold": 0.25,
            },
            "yolo": {
                "name": "yolov8m.pt",
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
            },
            "florence2": {"name": "microsoft/Florence-2-base", "conf_threshold": 0.3},
            "clipseg": {"name": "CIDAS/clipseg-rd64-refined", "threshold": 0.3},
            "fashion_clip": {"name": "patrickjohncyh/fashion-clip"},
            "vision_llm": {
                "name": "gemini/gemini-1.5-flash",
                "temperature": 0.0,
                "max_tokens": 1000,
            },
        }

    def load_from_file(self, path: str) -> None:
        """Loads configuration from a YAML file and overrides default settings."""
        if not os.path.exists(path):
            logger.warning(f"Config file not found at {path}. Using default settings.")
            self._set_defaults()
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.raw_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(
                f"Error parsing configuration YAML from {path}: {e}. Using defaults."
            )
            self._set_defaults()
            return

        # Parse logging settings first
        logging_cfg = self.raw_config.get("logging", {})
        self.log_level = logging_cfg.get("level", "INFO")
        self.log_file = logging_cfg.get("file", "logs/fashion_detector.log")

        # Parse device
        import torch

        raw_device = self.raw_config.get("device", "auto")
        if raw_device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = raw_device

        # Parse cache
        self.cache_dir = os.path.abspath(self.raw_config.get("cache_dir", "./cache"))
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set cache directories in environment for Hugging Face
        os.environ["HF_HOME"] = os.path.join(self.cache_dir, "huggingface")
        os.environ["TORCH_HOME"] = os.path.join(self.cache_dir, "torch")

        # Parse categories
        self.categories = self.raw_config.get("categories", DEFAULT_CATEGORIES)

        # Parse models config
        self.models = self.raw_config.get("models", {})

        logger.info(f"Configuration loaded from {path}. Device set to: {self.device}")

    def get_all_categories(self) -> List[str]:
        """Flattens clothing and accessories categories into a single list."""
        all_cats = []
        for cat_list in self.categories.values():
            all_cats.extend(cat_list)
        return all_cats
