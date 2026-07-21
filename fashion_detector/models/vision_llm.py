import io
import os
import json
import base64
from typing import Any, List, Optional
from PIL import Image
import litellm
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it


class VisionLlmDetector(BaseDetector):
    """Vision LLM object detector using LiteLLM (Gemini, Claude, GPT, etc.)."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.model_name = config.models.get("vision_llm", {}).get(
            "name", "gemini/gemini-1.5-flash"
        )
        self.temperature = config.models.get("vision_llm", {}).get("temperature", 0.0)
        self.max_tokens = config.models.get("vision_llm", {}).get("max_tokens", 1000)
        self.api_key = config.models.get("vision_llm", {}).get("api_key", None)
        self.api_base = config.models.get("vision_llm", {}).get("api_base", None)

    def load_model(self) -> None:
        """Vision LLMs are queried via API, so no weights are loaded locally."""
        logger.info(f"Vision LLM API model configured: {self.model_name}")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Converts PIL image to base64 JPEG string."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @time_it("Vision LLM Inference")
    def detect(self, image: Image.Image, **kwargs: Any) -> List[Detection]:
        """Queries the Vision LLM to locate and label fashion items.

        Args:
            image: PIL Image.
            **kwargs:
                queries: List of candidate fashion categories.
                model_name: Can override default model name.

        Returns:
            List of Detection objects.
        """
        model_to_use = kwargs.get("model_name", self.model_name)
        categories = kwargs.get("queries")
        if not categories:
            categories = self.config.get_all_categories()

        width, height = image.size
        base64_image = self._encode_image_to_base64(image)

        # Build prompt instructing the model to output a structured JSON list of boxes
        prompt = f"""
You are an expert fashion AI visual search system.
Detect all fashion items (clothing, accessories, shoes) worn by people or present in this image.
Do not detect people, only detect the fashion items themselves.

For each detected fashion item, provide:
1. "label": The specific category name, which MUST be selected from the Allowed Categories list below.
2. "box_2d": The bounding box coordinates as [ymin, xmin, ymax, xmax] normalized on a 0 to 1000 scale.
   (0,0 is the top-left and 1000,1000 is the bottom-right corner of the image. For example, [ymin, xmin, ymax, xmax] coordinates. Scale 0 is the top/left edge, 1000 is the bottom/right edge).
3. "score": Estimate a confidence score from 0.0 to 1.0.

Allowed Categories:
{categories}

Your output MUST be a valid JSON array of objects and NOTHING ELSE. Do not include markdown formatting like ```json or ```. Return only the raw JSON string.

Example Output format:
[
  {{"label": "jacket", "box_2d": [100, 200, 800, 900], "score": 0.95}},
  {{"label": "sneakers", "box_2d": [850, 400, 990, 600], "score": 0.88}}
]
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        # Gather completion parameters
        completion_kwargs = {
            "model": model_to_use,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        api_key_to_use = kwargs.get("api_key", self.api_key)
        api_base_to_use = kwargs.get("api_base", self.api_base)

        if api_key_to_use:
            completion_kwargs["api_key"] = api_key_to_use
        if api_base_to_use:
            completion_kwargs["api_base"] = api_base_to_use

        logger.info(f"Querying Vision LLM {model_to_use} via LiteLLM...")

        try:
            response = litellm.completion(**completion_kwargs)

            response_text = response.choices[0].message.content.strip()

            # Clean up potential markdown formatting wrapping the JSON
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                # Remove first line if it starts with ```
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove last line if it starts with ```
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            logger.debug(f"Raw response from Vision LLM: {response_text}")

            raw_detections = json.loads(response_text)

        except Exception as e:
            logger.error(f"Error calling or parsing Vision LLM response: {e}")
            # Return empty list on failure
            return []

        detections = []
        for raw_det in raw_detections:
            try:
                label = raw_det.get("label", "").lower().strip()
                box_2d = raw_det.get("box_2d", [])
                score = float(raw_det.get("score", 0.8))

                if len(box_2d) != 4:
                    continue

                ymin_1000, xmin_1000, ymax_1000, xmax_1000 = box_2d

                # Convert 0-1000 scale back to absolute pixel coordinates
                xmin = (xmin_1000 / 1000.0) * width
                ymin = (ymin_1000 / 1000.0) * height
                xmax = (xmax_1000 / 1000.0) * width
                ymax = (ymax_1000 / 1000.0) * height

                # Clip box coordinates
                xmin = max(0.0, min(xmin, float(width)))
                ymin = max(0.0, min(ymin, float(height)))
                xmax = max(0.0, min(xmax, float(width)))
                ymax = max(0.0, min(ymax, float(height)))

                # Double-check label is in allowed categories
                if label in categories:
                    detections.append(
                        Detection(
                            box=[xmin, ymin, xmax, ymax],
                            label=label,
                            score=score,
                            metadata={"raw_llm_box": box_2d},
                        )
                    )
            except Exception as item_error:
                logger.warning(
                    f"Failed to parse item detection: {raw_det}. Error: {item_error}"
                )

        logger.info(f"Vision LLM detected {len(detections)} items.")
        return detections
