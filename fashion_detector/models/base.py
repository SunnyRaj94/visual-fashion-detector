from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image


@dataclass
class Detection:
    """Dataclass representing a single fashion item detection."""

    box: List[float]  # [xmin, ymin, xmax, ymax] in absolute pixel coordinates
    label: str  # Predicted class name
    score: float  # Confidence score between 0.0 and 1.0
    mask: Optional[np.ndarray] = (
        None  # Optional binary mask for segmentation (same shape as image)
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Any model-specific extra metadata

    def to_dict(self) -> Dict[str, Any]:
        """Converts detection to a serializable dictionary."""
        return {
            "box": self.box,
            "label": self.label,
            "score": self.score,
            "has_mask": self.mask is not None,
            "metadata": self.metadata,
        }


class BaseDetector(ABC):
    """Abstract base class for all fashion item detectors."""

    def __init__(self, config: Any):
        self.config = config
        self.device = config.device
        self.cache_dir = config.cache_dir

    @abstractmethod
    def load_model(self) -> None:
        """Loads the model weights and pushes to the configured device.

        Should automatically download weights from Hugging Face or another source
        if not already present in the cache.
        """
        pass

    @abstractmethod
    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Runs object detection on the input image.

        Args:
            image: A PIL Image to detect objects in.
            **kwargs: Extra parameters (e.g., confidence thresholds, text queries).

        Returns:
            A list of Detection objects.
        """
        pass

    def _to_dict(self, detections: List[Detection]) -> List[Dict[str, Any]]:
        return [det.to_dict() for det in detections]
