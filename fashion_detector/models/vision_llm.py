import io
import os
import json
import base64
from typing import Any, Dict, List, Optional
from PIL import Image
import litellm
from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel, Field, field_validator
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger, time_it

DETECTION_PROMPT_TEMPLATE = """
You are an expert fashion AI visual search system.
# Task: Detect all fashion items (clothing, accessories, shoes) worn by people or present in the image.

# Constraints:
1. Detect ONLY fashion products. Never detect people, faces, body parts, hair, hands, or any non-fashion objects.
2. Detect only fashion items that are clearly visible and at least 50% visible. Ignore heavily occluded, blurry, or unrecognizable items.
3. The closest and most visually prominent foreground fashion items are mandatory detections. Never omit a large, clearly visible fashion item that appears closest to the camera.
4. Detect every visible instance separately. If multiple people are wearing the same product category, return a separate detection for each physical item.
5. Return one tight bounding box for each detected fashion item. The bounding box should tightly enclose only the visible portion of the product and should not include unnecessary background or neighboring items.
6. Do NOT return duplicate detections for the same physical fashion item.
7. If you are uncertain about a product category or the item is not clearly visible, do NOT return a detection.

# For each detected fashion item, provide an object with exactly these keys:
1. "label": The specific category name. It MUST be selected strictly from the 'Allowed Categories' list below. Do not hallucinate categories not in the list.
2. "box_2d": An array of 4 numbers [ymin, xmin, ymax, xmax] normalized on a 0 to 1000 scale.
   - Scale: (0,0) is top-left, (1000,1000) is bottom-right.
   - Format: Ensure numbers are integers or floats. Do not include coordinate names like "y1".
3. "score": A float between 0.0 and 1.0 representing confidence.

# Confidence Score
Assign a floating-point score between 0.0 and 1.0 based on how visually prominent the fashion item is in the image.

The score should consider:
- Visibility (how much of the item is visible)
- Size relative to the image
- Distance to the camera (foreground vs. background)
- Occlusion
- Image clarity

Scoring Guidelines:
1.00
- Largest and most visually dominant fashion item in the image.
- Nearly or completely visible (>90%).
- Closest to the camera.

0.80 - 0.99
- Large and clearly visible.
- Mostly unobstructed (>70% visible).
- One of the primary fashion items in the image.

0.50 - 0.79
- Medium-sized item.
- Partially visible (50–70%).
- Background or moderately occluded.

0.30 - 0.49
- Small or distant item.
- Barely satisfies the 50% visibility requirement.
- Not visually prominent.

0.00 - 0.29
- Do NOT use this range.
- Items with less than 50% visibility or extremely small, blurry, or heavily occluded should NOT be returned.


# Allowed Categories:
<<categories>>

# Critical Formatting Rules:
- Output MUST be a single block of raw text starting with '[' and ending with ']').
- Ensure valid JSON syntax (escape quotes if necessary, use commas between items, no trailing commas).
- If no items are detected, return an empty array: []

Example Output (Raw JSON):
[{
    "label": "jacket",
    "box_2d": [100, 200, 800, 900],
    "score": 0.95
},
{
    "label": "sneakers",
    "box_2d": [850, 400, 990, 600],
    "score": 0.88
}]

Return your Json output now.
"""


CLASSIFICATION_PROMPT_TEMPLATE = """You are an expert fashion AI visual search classifier.
Analyze the image and identify all fashion items (clothing, accessories, shoes) worn by people or present in this image.

Your task is to identify which of the Allowed Categories are present in the image.
For each category present in the image, provide:
1. "label": The specific category name, which MUST be selected from the Allowed Categories list below.
2. "score": Estimate a presence confidence score from 0.0 to 1.0.

Allowed Categories:
{categories}

Your output MUST be a valid JSON array of objects and NOTHING ELSE. Do not include markdown formatting like ```json or ```. Return only the raw JSON string.

Example Output format:
[
  {{"label": "jacket", "score": 0.95}},
  {{"label": "sneakers", "score": 0.88}}
]"""


class LlmDetectionItem(BaseModel):
    label: str
    box_2d: List[float] = Field(
        ..., description="[ymin, xmin, ymax, xmax] normalized coordinates"
    )
    score: float = Field(default=0.8)

    @field_validator("box_2d")
    @classmethod
    def validate_box_2d(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("box_2d must contain exactly 4 coordinates")
        for val in v:
            if not (0.0 <= val <= 1000.0):
                raise ValueError("Coordinates must be between 0.0 and 1000.0")
        if v[0] > v[2]:
            raise ValueError("ymin cannot be greater than ymax")
        if v[1] > v[3]:
            raise ValueError("xmin cannot be greater than xmax")
        return v

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("score must be between 0.0 and 1.0")
        return v


class VisionLlmDetector(BaseDetector):
    """Vision LLM object detector using LiteLLM (Gemini, Claude, GPT, etc.)."""

    def __init__(self, config: Any):
        super().__init__(config)
        vision_cfg = config.models.get("vision_llm", {})
        self.provider = vision_cfg.get("provider", None)
        raw_model_name = (
            vision_cfg.get("model")
            or vision_cfg.get("name")
            or "gemini/gemini-1.5-flash"
        )
        if (
            self.provider
            and not raw_model_name.startswith(f"{self.provider}/")
            and not raw_model_name.startswith("openai/")
        ):
            self.model_name = f"{self.provider}/{raw_model_name}"
        else:
            self.model_name = raw_model_name

        self.temperature = vision_cfg.get("temperature", 0.0)
        self.max_tokens = vision_cfg.get("max_tokens", 1000)
        self.api_key_env = vision_cfg.get("api_key_env", None)
        self.api_key = vision_cfg.get("api_key", None)
        self.api_base = vision_cfg.get("api_base", None)

    def _get_api_key(self, kwargs: Optional[Dict[str, Any]] = None) -> Optional[str]:
        kwargs = kwargs or {}
        if kwargs.get("api_key"):
            return kwargs["api_key"]

        if self.api_key_env:
            env_val = os.getenv(self.api_key_env)
            if env_val:
                return env_val

        if self.api_key:
            if (
                isinstance(self.api_key, str)
                and self.api_key.startswith("${")
                and self.api_key.endswith("}")
            ):
                env_name = self.api_key[2:-1].strip()
                env_val = os.getenv(env_name)
                if env_val:
                    return env_val
            return (
                os.path.expandvars(self.api_key)
                if isinstance(self.api_key, str)
                else self.api_key
            )

        return None

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

        prompt = DETECTION_PROMPT_TEMPLATE.replace("<<categories>>", str(categories))

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

        api_key_to_use = self._get_api_key(kwargs)
        api_base_to_use = kwargs.get("api_base", self.api_base)

        if (
            api_base_to_use
            and not model_to_use.startswith("openai/")
            and not model_to_use.startswith("openrouter/")
            and self.provider is None
        ):
            model_to_use = f"openai/{model_to_use}"

        # Gather completion parameters
        completion_kwargs = {
            "model": model_to_use,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if api_key_to_use:
            completion_kwargs["api_key"] = api_key_to_use
        if api_base_to_use:
            completion_kwargs["api_base"] = api_base_to_use

        logger.info(f"Querying Vision LLM {model_to_use} via LiteLLM...")

        try:
            response = litellm.completion(**completion_kwargs)
            response_text = response.choices[0].message.content.strip()

            # Always log raw_response for debugging
            logger.debug(f"Raw response from Vision LLM: \n{response_text}\n")

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

            raw_detections = json.loads(response_text)
            if not isinstance(raw_detections, list):
                raise ValueError("Expected raw response to parse to a list")

        except Exception as e:
            logger.error(f"Error calling or parsing Vision LLM response: {e}")
            if "response_text" in locals():
                logger.debug(f"Failed response text was: {response_text}")
            return []

        detections = []
        for raw_det in raw_detections:
            try:
                # Do output validation using Pydantic v2
                validated = LlmDetectionItem.model_validate(raw_det)

                label = validated.label.lower().strip()
                ymin_1000, xmin_1000, ymax_1000, xmax_1000 = validated.box_2d
                score = validated.score

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
                if label in [c.lower().strip() for c in categories]:
                    detections.append(
                        Detection(
                            box=[xmin, ymin, xmax, ymax],
                            label=label,
                            score=score,
                            metadata={"raw_llm_box": validated.box_2d},
                        )
                    )
            except Exception as item_error:
                logger.warning(
                    f"Failed to parse or validate item detection: {raw_det}. Error: {item_error}"
                )

        logger.info(f"Vision LLM detected {len(detections)} items.")
        return detections

    @time_it("Vision LLM Class Presence Extraction")
    def extract_present_classes(
        self,
        image: Image.Image,
        user_categories: List[str],
        presence_threshold: float = 0.15,
    ) -> List[str]:
        """Extracts active fashion categories present in the input image using Vision LLM classification prompt.

        Args:
            image: PIL Image.
            user_categories: List of candidate category names to search for.
            presence_threshold: Minimum confidence score threshold (0.0 to 1.0).

        Returns:
            List of unique category names present in the image with score >= presence_threshold.
        """
        if not user_categories:
            return []

        base64_image = self._encode_image_to_base64(image)
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(categories=user_categories)

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

        model_to_use = self.model_name
        api_key_to_use = self._get_api_key()
        api_base_to_use = self.api_base

        completion_kwargs = {
            "model": model_to_use,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if api_key_to_use:
            completion_kwargs["api_key"] = api_key_to_use
        if api_base_to_use:
            completion_kwargs["api_base"] = api_base_to_use

        logger.info(
            f"Querying Vision LLM {model_to_use} for class presence extraction..."
        )

        confirmed_classes = set()
        norm_user_cats = {c.strip().lower(): c for c in user_categories}

        try:
            response = litellm.completion(**completion_kwargs)
            response_text = response.choices[0].message.content.strip()

            # Clean up potential markdown formatting wrapping the JSON
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            raw_classes = json.loads(response_text)
            if not isinstance(raw_classes, list):
                raise ValueError("Expected raw response to parse to a list")

            for item in raw_classes:
                if isinstance(item, dict):
                    label = str(item.get("label", "")).strip().lower()
                    score = float(item.get("score", 1.0))
                    if score >= presence_threshold and label in norm_user_cats:
                        confirmed_classes.add(norm_user_cats[label])

        except Exception as e:
            logger.error(
                f"Error calling or parsing Vision LLM classification response: {e}"
            )
            if "response_text" in locals():
                logger.debug(f"Failed response text was: {response_text}")

        logger.info(
            f"Vision LLM extracted {len(confirmed_classes)} active present classes."
        )
        return sorted(list(confirmed_classes))
