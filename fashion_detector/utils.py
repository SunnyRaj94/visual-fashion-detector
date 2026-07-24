import base64
import io
import os
import requests
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fashion_detector.models.base import Detection
from fashion_detector.logging import logger
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import math

CATEGORY_MAPPING = {
    # Clothing Overlaps
    "dresses": "dresses",
    "jumpsuits": "jumpsuits",
    "skirts": "skirts",
    "shorts": "shorts",
    "tops": "tops_shirts",
    "shirts": "tops_shirts",
    "t shirts": "tops_shirts",
    "sweaters": "sweaters",
    "jackets": "jackets_blazers",
    "blazers": "jackets_blazers",
    "jackets blazers": "jackets_blazers",
    "coats": "coats",
    "pants": "pants_jeans",
    "jeans": "pants_jeans",
    "suits": "suits_sets",
    "suits sets": "suits_sets",
    # Footwear Overlaps
    "sneakers": "sneakers",
    "boots": "boots",
    "sandals": "sandals",
    "heels": "heels",
    "flats": "flats_loafers",
    "loafers": "flats_loafers",
    "mules slides": "mules_slides",
    "dress shoes": "dress_shoes",
    # Bags Overlaps
    "tote bags": "tote_bags",
    "backpacks": "backpacks",
    "belt bags": "belt_bags",
    "briefcases": "briefcases",
    "duffel bags": "travel_duffel_bags",
    "shoulder bags": "shoulder_crossbody_bags",
    "crossbody bags": "shoulder_crossbody_bags",
    "messenger bags": "shoulder_crossbody_bags",
    "handle bags": "hand_handle_bags",
    "clutches": "hand_handle_bags",
    # Accessories & Jewelry Overlaps
    "sunglasses": "sunglasses",
    "belts": "belts",
    "wallets": "wallets",
    "hats": "hats",
    "watches": "watches",
    "scarves": "scarves_shawls_ties",
    "scarves shawls": "scarves_shawls_ties",
    "ties": "scarves_shawls_ties",
    "jewelry": "jewelry",
    "earrings": "jewelry",
    "necklaces": "jewelry",
    "bracelets": "jewelry",
    "rings": "jewelry",
    "brooches": "jewelry",
}
CATEGORY_HIERARCHY = {
    "Clothing": {
        "Dresses": ["dresses", "jumpsuits", "skirts"],
        "Tops": ["tops", "shirts", "t shirts", "sweaters"],
        "Bottoms": ["shorts", "pants", "jeans"],
        "Outerwear": [
            "jackets",
            "blazers",
            "jackets blazers",
            "coats",
            "suits",
            "suits sets",
        ],
    },
    "Footwear": {
        "Sneakers": ["sneakers"],
        "Boots": ["boots"],
        "Sandals": [
            "sandals",
            "heels",
            "flats",
            "loafers",
            "mules slides",
            "dress shoes",
        ],
    },
    "Accessories": {
        "Hats": ["hats"],
        "Watches": ["watches"],
        "Belts": [
            "belts",
            "sunglasses",
            "wallets",
            "scarves",
            "scarves shawls",
            "ties",
            "jewelry",
            "earrings",
            "necklaces",
            "bracelets",
            "rings",
            "brooches",
        ],
    },
    "Bags": {
        "Tote": ["tote bags", "handle bags", "clutches"],
        "Backpack": ["backpacks"],
        "Crossbody": [
            "crossbody bags",
            "shoulder bags",
            "messenger bags",
            "belt bags",
            "briefcases",
            "duffel bags",
        ],
    },
}

user_categories = list(set(CATEGORY_MAPPING.keys()))
mapped_user_categories = list(set(CATEGORY_MAPPING.values()))


def get_broad_categories() -> List[str]:
    """Returns top-level broad categories from taxonomy hierarchy."""
    return list(CATEGORY_HIERARCHY.keys())


def get_fine_categories_for_broad(broad_cat: str) -> List[str]:
    """Returns all fine-grained category keys matching a broad category."""
    if broad_cat not in CATEGORY_HIERARCHY:
        return user_categories
    fine_cats = []
    for subcat_list in CATEGORY_HIERARCHY[broad_cat].values():
        fine_cats.extend(subcat_list)
    return list(set(fine_cats))


def get_parent_taxonomy_for_fine(fine_cat: str) -> Tuple[str, str]:
    """Finds broad category and subcategory for a given fine category key."""
    fine_clean = fine_cat.strip().lower()
    for broad, subcats in CATEGORY_HIERARCHY.items():
        for subcat, fine_list in subcats.items():
            if fine_clean in [f.strip().lower() for f in fine_list]:
                return broad, subcat
    # Default fallback mapping
    return "Clothing", "Other"


def clean_categories(raw_detected_categories: List[str]) -> List[str]:
    """Cleans and maps raw detected categories to a standardized set."""
    # Example Cleanup Sequence:
    cleaned_unique_categories = list(
        set(CATEGORY_MAPPING.get(cat, cat) for cat in raw_detected_categories)
    )
    logger.info(f"Cleaned and mapped categories: {cleaned_unique_categories}")
    return cleaned_unique_categories


def execute_detection(
    image_path,
    detector,
    visualize=False,
    categories=user_categories,
):
    image = load_image(image_path)
    detections = detector.detect(image, queries=categories)
    logger.info(f"Hybrid pipeline detected {len(detections)} fashion items:")
    for d in detections:
        logger.info(
            f"- {d.label.capitalize()} (refined from {d.metadata.get('proposal_label')}): score={d.score:.2f}"
        )
    if visualize:
        visualize_detections(image, detector._to_dict(detections))
    return detections


def load_image(image_input: Union[str, Image.Image]) -> Image.Image:
    """Loads an image from a local path, a URL, or returns the image if it is already a PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if not isinstance(image_input, str):
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if image_input.startswith("http://") or image_input.startswith("https://"):
        logger.info(f"Downloading image from URL: {image_input}")
        try:
            response = requests.get(image_input, timeout=15)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to download image from URL {image_input}: {e}")
            raise
    else:
        logger.info(f"Loading image from local path: {image_input}")
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Local image file not found at: {image_input}")
        try:
            image = Image.open(image_input)
            return image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to read local image file at {image_input}: {e}")
            raise


def get_color_map(categories: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Generates a consistent random color map for a list of categories."""
    random.seed(42)  # For consistent coloring
    color_map = {}
    for cat in categories:
        color_map[cat] = (
            random.randint(50, 220),
            random.randint(50, 220),
            random.randint(50, 220),
        )
    return color_map


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[Detection],
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Image.Image:
    """Draws bounding boxes and labels onto the image.

    Args:
        image: Original PIL Image.
        detections: List of Detection objects.
        color_map: Dictionary mapping class name to RGB tuple.

    Returns:
        Annotated PIL Image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load a font, fallback to default
    try:
        # Load a standard font if possible
        font = ImageFont.load_default()
    except IOError:
        font = None

    if color_map is None:
        categories = list(set(d.label for d in detections))
        color_map = get_color_map(categories)

    for i, det in enumerate(detections):
        box = det.box
        xmin, ymin, xmax, ymax = box
        label = det.label
        score = det.score

        color = color_map.get(label, (255, 0, 0))

        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

        # Draw text background and text
        text = f"{label} {score:.2f}"

        # Handle mask overlay if present
        if det.mask is not None:
            mask_img = Image.fromarray(
                (det.mask * 100).astype(np.uint8)
            )  # Semi-transparent mask
            # Color the mask
            colored_mask = Image.new("RGB", annotated.size, color=color)
            annotated.paste(colored_mask, mask=mask_img)
            # Re-draw the rect on top of the mask
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

        # Draw label text box
        try:
            # For older and newer Pillow compatibility
            if hasattr(draw, "textbbox"):
                text_box = draw.textbbox((xmin, ymin), text, font=font)
                text_w = text_box[2] - text_box[0]
                text_h = text_box[3] - text_box[1]
            else:
                text_w, text_h = draw.textsize(text, font=font)
        except Exception:
            text_w, text_h = len(text) * 6, 12

        draw.rectangle([xmin, ymin - text_h - 4, xmin + text_w + 4, ymin], fill=color)
        draw.text((xmin + 2, ymin - text_h - 2), text, fill=(255, 255, 255), font=font)

    return annotated


def normalize_box_to_pixels(
    box: List[float], img_width: int, img_height: int, is_1000_scale: bool = False
) -> Tuple[float, float, float, float]:
    """Robustly normalizes bounding box coordinates to absolute pixel [xmin, ymin, xmax, ymax].

    Handles:
    - Absolute pixel coordinates: [xmin, ymin, xmax, ymax]
    - Normalized 0..1 coordinates: [xmin, ymin, xmax, ymax]
    - Normalized 0..1000 coordinates (LLM/Gemini style): [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
    """
    b = [float(x) for x in box]

    # Detect if box is 0-1000 scale (e.g. Gemini Vision LLM output)
    # Check 1: Explicit flag
    # Check 2: Index 0 > Index 3 (e.g., 232 > 214 -> ymin > xmax, impossible for xmin, ymin, xmax, ymax)
    # Check 3: Max value <= 1000 and max value > 1.0 while image dimensions are smaller or different
    is_1000 = (
        is_1000_scale
        or (b[0] > b[3] and max(b) <= 1000.0)
        or (b[1] > b[2] and max(b) <= 1000.0)
        or (
            max(b) > 1.0
            and max(b) <= 1000.0
            and (
                b[0] > img_width
                or b[2] > img_width
                or b[1] > img_height
                or b[3] > img_height
            )
        )
    )

    if is_1000:
        ymin_1000, xmin_1000, ymax_1000, xmax_1000 = b
        # Swap if needed assuming [ymin, xmin, ymax, xmax]
        if xmin_1000 > xmax_1000 or ymin_1000 > ymax_1000:
            xmin_1000, ymin_1000, xmax_1000, ymax_1000 = b
        xmin = (xmin_1000 / 1000.0) * img_width
        ymin = (ymin_1000 / 1000.0) * img_height
        xmax = (xmax_1000 / 1000.0) * img_width
        ymax = (ymax_1000 / 1000.0) * img_height
    elif max(b) <= 1.0:
        # Case 2: Normalized 0..1 coordinates
        xmin = b[0] * img_width
        ymin = b[1] * img_height
        xmax = b[2] * img_width
        ymax = b[3] * img_height
    else:
        # Case 3: Already absolute pixel coordinates [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]

    # Clip within image bounds
    xmin = max(0.0, min(xmin, float(img_width)))
    ymin = max(0.0, min(ymin, float(img_height)))
    xmax = max(0.0, min(xmax, float(img_width)))
    ymax = max(0.0, min(ymax, float(img_height)))

    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    return xmin, ymin, xmax, ymax


def generate_interactive_html(
    image: Image.Image,
    detections: List[Detection],
    title: str = "Fashion Item Detections",
) -> str:
    """Generates a self-contained HTML/CSS/JS snippet to display the image
    with interactive, clickable bounding box overlays.

    Args:
        image: The PIL Image.
        detections: List of Detection objects.
        title: Title of the visualization.

    Returns:
        HTML string.
    """
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    width, height = image.size

    # Generate unique ID for containment to prevent CSS collision
    container_id = f"fashion-container-{random.randint(1000, 9999)}"

    # Gather unique classes for color mapping
    categories = list(set(d.label for d in detections))
    color_map = get_color_map(categories)

    html_out = []

    # Stylesheet
    html_out.append(f"""
    <style>
        #{container_id} {{
            position: relative;
            display: inline-block;
            max-width: 100%;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            background: #111;
            margin: 10px 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        #{container_id} img {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
        #{container_id} .bbox-overlay {{
            position: absolute;
            border: 2px solid;
            box-sizing: border-box;
            transition: all 0.2s ease-in-out;
            cursor: pointer;
        }}
        #{container_id} .bbox-overlay:hover {{
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
            z-index: 10;
        }}
        #{container_id} .tooltip {{
            visibility: hidden;
            position: absolute;
            background-color: rgba(0, 0, 0, 0.85);
            color: #fff;
            text-align: center;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            z-index: 100;
            opacity: 0;
            transition: opacity 0.2s;
            pointer-events: none;
            white-space: nowrap;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }}
        #{container_id} .bbox-overlay:hover .tooltip {{
            visibility: visible;
            opacity: 1;
        }}
        #{container_id} .info-panel {{
            background: #1a1a1a;
            color: #eee;
            padding: 12px;
            font-size: 14px;
            border-top: 1px solid #333;
            min-height: 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        #{container_id} .active-det-info {{
            font-weight: 600;
            color: #4fc3f7;
        }}
    </style>
    
    <div id="{container_id}">
        <div style="position: relative;">
            <img src="data:image/jpeg;base64,{img_str}" width="{width}" height="{height}" alt="Source Image" />
    """)

    # Draw interactive divs (sorted by area descending so smaller boxes stack on top of larger ones)
    sorted_dets = sorted(
        detections,
        key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
        reverse=True,
    )
    for idx, det in enumerate(sorted_dets):
        xmin, ymin, xmax, ymax = normalize_box_to_pixels(det.box, width, height)
        # xmin, ymin, xmax, ymax = det.box

        # Calculate percentage coordinates for responsiveness
        left = (xmin / width) * 100
        top = (ymin / height) * 100
        w = ((xmax - xmin) / width) * 100
        h = ((ymax - ymin) / height) * 100

        rgb = color_map.get(det.label, (255, 0, 0))
        color_hex = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        bg_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]}, 0.15)"

        tooltip_dir = "bottom: 105%; left: 50%; transform: translateX(-50%);"
        if top < 15:  # if too close to top, show tooltip below
            tooltip_dir = "top: 105%; left: 50%; transform: translateX(-50%);"

        onclick_js = f"document.getElementById('{container_id}-info').innerHTML = 'Selected: <span class=\\\"active-det-info\\\">{det.label.capitalize()}</span> | Confidence: <b>{det.score:.2f}</b> | Box: [{int(xmin)}, {int(ymin)}, {int(xmax)}, {int(ymax)}]';"

        html_out.append(f"""
            <div class="bbox-overlay" 
                 style="left: {left}%; top: {top}%; width: {w}%; height: {h}%; border-color: {color_hex}; background-color: {bg_rgba};"
                 onclick="{onclick_js}">
                 <span class="tooltip" style="{tooltip_dir}">{det.label} ({det.score:.2f})</span>
            </div>
        """)

    html_out.append(f"""
        </div>
        <div class="info-panel">
            <div id="{container_id}-info">Click on any bounding box to inspect the fashion item details.</div>
            <div style="font-size: 11px; color: #888;">{title} ({len(detections)} detected)</div>
        </div>
    </div>
    """)

    return "\n".join(html_out)


def display_img(image: Image.Image, figsize=(6, 6)):
    """Display a PIL image inside Jupyter."""
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def visualize_detections(
    image: Union[str, Image.Image],
    detections: List[Dict],
    max_width: int = 700,
    show_score: bool = True,
):
    """
    Display clickable detections in a Jupyter notebook.

    Supports:
        - image path
        - PIL Image

    Detection format:

    {
        "label": "...",
        "score": 0.91,
        "box": [xmin, ymin, xmax, ymax]
    }
    """

    # -------------------------------------------------------------
    # Load image
    # -------------------------------------------------------------

    if isinstance(image, str):

        img = Image.open(image).convert("RGB")

        with open(image, "rb") as f:
            img_bytes = f.read()

    elif isinstance(image, Image.Image):

        img = image.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

    else:
        raise TypeError("image must be a file path or PIL.Image")

    img_width, img_height = img.size

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    html = f"""
    <div style="
        position:relative;
        display:inline-block;
        max-width:{max_width}px;
    ">
        <img
            src="data:image/jpeg;base64,{img_base64}"
            style="
                width:100%;
                display:block;
                height:auto;
            "
        />
    """

    # -------------------------------------------------------------
    # Draw detections
    # -------------------------------------------------------------

    COLORS = [
        "#00E5FF",
        "#00C853",
        "#FF5252",
        "#FFAB00",
        "#AA00FF",
        "#2979FF",
        "#EC407A",
        "#7CB342",
    ]

    def get_value(obj, key, default=None):
        """Get a value from a dict or object."""

        if isinstance(obj, dict):
            return obj.get(key, default)

        if hasattr(obj, key):
            return getattr(obj, key)

        return default

    def get_box(det):
        return get_value(det, "box") or get_value(det, "bbox") or [0, 0, 0, 0]

    sorted_dets = sorted(
        detections,
        key=lambda d: (
            (get_box(d)[2] - get_box(d)[0]) * (get_box(d)[3] - get_box(d)[1])
        ),
        reverse=True,
    )
    for idx, det in enumerate(sorted_dets):

        # box = det.get("box") or det.get("bbox")
        box = get_box(det)

        if box is None:
            continue

        xmin, ymin, xmax, ymax = box

        label = get_value(det, "label")
        score = get_value(det, "score")

        area = (xmax - xmin) * (ymax - ymin)

        label_text = (
            f"{label.upper()} ({score:.2f})"
            if show_score and score is not None
            else label.upper()
        )

        color = COLORS[idx % len(COLORS)]

        html += f"""
        <div
            title="{label}"
            onclick="alert(
                'Label : {label}\\n'
                + 'Score : {score}\\n'
                + 'Area : {area}px²\\n'
                + 'Box : {box}'
            )"
            style="
                position:absolute;
                left:{xmin/img_width*100:.3f}%;
                top:{ymin/img_height*100:.3f}%;
                width:{(xmax-xmin)/img_width*100:.3f}%;
                height:{(ymax-ymin)/img_height*100:.3f}%;

                border:3px solid {color};
                background:rgba(255,255,255,.05);
                cursor:pointer;
                transition:.2s;
                box-sizing:border-box;
            "

            onmouseover="
                this.style.background='rgba(255,255,255,.25)';
            "

            onmouseout="
                this.style.background='rgba(255,255,255,.05)';
            "
        >

            <div
                style="
                    position:absolute;
                    left:0;
                    top:-24px;

                    background:{color};
                    color:white;

                    font-size:12px;
                    font-weight:bold;

                    padding:2px 6px;
                    white-space:nowrap;
                "
            >
                {label_text}
            </div>

        </div>
        """

    html += "</div>"

    display(HTML(html))


def display_imageGrid(
    images: List[Image.Image],
    imgs_per_row: int = 3,
    max_width: int = 15,
    border_color: str = "#cccccc",
    border_width: float = 1.5,
) -> None:
    """
    Displays a list of PIL Images in a grid layout inside a Jupyter Notebook cell.

    Args:
        images: List of PIL.Image.Image instances to display.
        imgs_per_row: Number of images per row (e.g., 2 or 3).
        max_width: Maximum width of the entire figure layout in inches.
        border_color: Color of the bounding border around each image (default: '#cccccc').
        border_width: Line width of the bounding border in points (default: 1.5).

    Raises:
        TypeError: If input validation fails for types or structures.
        ValueError: If imgs_per_row is less than 1 or images list is empty.
    """
    # 1. Input Type and Value Validations
    if not isinstance(images, list):
        raise TypeError(
            f"Expected a list for 'images', but got {type(images).__name__}."
        )

    if not images:
        raise ValueError("The 'images' list cannot be empty.")

    for idx, img in enumerate(images):
        if not isinstance(img, Image.Image):
            raise TypeError(
                f"Element at index {idx} is not a valid PIL Image. Got {type(img).__name__}."
            )

    if not isinstance(imgs_per_row, int) or isinstance(imgs_per_row, bool):
        raise TypeError(
            f"Expected an integer for 'imgs_per_row', but got {type(imgs_per_row).__name__}."
        )

    if imgs_per_row < 1:
        raise ValueError(
            f"Value of 'imgs_per_row' must be 1 or greater. Got {imgs_per_row}."
        )

    # 2. Grid Dimensions Calculations
    num_images = len(images)
    num_rows = math.ceil(num_images / imgs_per_row)

    # Dynamically scale height proportionally to maintain reasonable image aspect ratios
    fig_width = max_width
    fig_height = (fig_width / imgs_per_row) * num_rows

    # 3. Render Canvas
    fig, axes = plt.subplots(num_rows, imgs_per_row, figsize=(fig_width, fig_height))

    # Flatten axes matrix to a simple 1D array for easier iteration
    # Handle edge case where a 1x1 subplots call returns a single axis object rather than an array
    if num_images == 1 and imgs_per_row == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # 4. Populate Grid Subplots
    for i in range(len(axes)):
        if i < num_images:
            # Display image data arrays safely
            axes[i].imshow(images[i])
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            if border_width > 0:
                for spine in axes[i].spines.values():
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(border_width)
            else:
                axes[i].axis("off")
        else:
            # Hide leftover empty subplot containers in the final grid row
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
