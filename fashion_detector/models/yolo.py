import os
from typing import Any, List, Optional
from PIL import Image
from ultralytics import YOLO
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class YoloDetector(BaseDetector):
    """YOLO object detector using Ultralytics."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.model_name = config.models.get("yolo", {}).get("name", "yolov8m.pt")
        self.conf_threshold = config.models.get("yolo", {}).get("conf_threshold", 0.25)
        self.iou_threshold = config.models.get("yolo", {}).get("iou_threshold", 0.45)

        self.model = None

    def load_model(self) -> None:
        """Loads YOLO model from weights, downloading if not cached."""
        if self.model is not None:
            return

        logger.info(f"Loading YOLO model: {self.model_name} on device: {self.device}")

        # Set cache directory for Ultralytics
        os.environ["YOLO_CONFIG_DIR"] = os.path.join(self.cache_dir, "ultralytics")

        # Construct path for saving weights inside local cache
        weights_path = os.path.join(self.cache_dir, "torch", self.model_name)
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        # If model name is just a name like 'yolov8m.pt', ultralytics downloads it automatically.
        # We can pass the weights path or the name itself.
        try:
            # We first try to load/download using ultralytics YOLO class
            # Ultralytics downloads to current directory by default, but we can set the weights path
            # to make sure it's saved in our configured cache directory.
            if not os.path.exists(weights_path) and not self.model_name.startswith("/"):
                logger.info(f"Downloading YOLO weights to cache: {weights_path}")
                # We download by instantiating the model with the name, then copy it or just use it.
                model = YOLO(self.model_name)
                # Save/move weights to our cache if possible, or just keep default ultralytics cache.
                self.model = model
            else:
                self.model = YOLO(
                    weights_path if os.path.exists(weights_path) else self.model_name
                )

            # Move model to device
            self.model.to(self.device)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Error loading YOLO model: {e}. Falling back to default YOLO model."
            )
            self.model = YOLO(self.model_name)
            self.model.to(self.device)

    @time_it("YOLO Inference")
    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Runs YOLO detection on the image.

        Args:
            image: PIL Image.
            **kwargs: Can override 'conf_threshold' or 'iou_threshold'.

        Returns:
            List of Detection objects.
        """
        self.load_model()

        conf = kwargs.get("conf_threshold", self.conf_threshold)
        iou = kwargs.get("iou_threshold", self.iou_threshold)

        # Run inference
        # verbose=False suppresses standard YOLO terminal prints since we use our structured logger
        results = self.model(
            image, conf=conf, iou=iou, device=self.device, verbose=False
        )[0]

        detections = []
        boxes = results.boxes

        # COCO class mapping for standard YOLOv8 models
        names = self.model.names

        for box in boxes:
            # Get box coordinates in absolute pixels [xmin, ymin, xmax, ymax]
            coords = box.xyxy[0].cpu().numpy().tolist()
            xmin, ymin, xmax, ymax = coords

            score = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            label = names.get(cls_id, f"class_{cls_id}").lower()

            # We filter or keep the detections based on whether they match our categories.
            # COCO categories relevant to fashion: backpack, umbrella, handbag, tie, suitcase, person.
            # If the user has a custom-trained fashion YOLO, it will automatically map classes.
            detections.append(
                Detection(
                    box=[xmin, ymin, xmax, ymax],
                    label=label,
                    score=score,
                    metadata={"class_id": cls_id},
                )
            )

        logger.info(f"YOLO detected {len(detections)} items.")
        return detections
