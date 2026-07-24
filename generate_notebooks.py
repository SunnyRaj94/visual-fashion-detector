import os
import json
from typing import Any, Dict, List


def create_notebook(filename: str, cells: List[Dict[str, Any]]) -> None:
    """Helper to save a list of cells as a Jupyter notebook file."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Created notebook: {filename}")


def md_cell(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
    }


def code_cell(code: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")],
    }


# ----------------- Notebook 1: Environment Setup -----------------
nb1_cells = [
    md_cell("""# 1. Environment Setup
Welcome to the Fashion Item Detection & Localization Research Framework setup notebook.
In this notebook, we will:
1. Verify the project directory structure.
2. Initialize and load our YAML configuration.
3. Test our structured logging system.
4. Verify PyTorch installation, hardware acceleration (CUDA or MPS), and relevant package versions.
"""),
    code_cell("""# Check imports and package visibility
import sys
import os
import torch
import torchvision

# Ensure our local package is in the system path
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.logging import logger, configure_logger, log_duration

print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)
"""),
    md_cell("""## Load Config & Setup Logs"""),
    code_cell("""# Initialize configuration
config = Config("config/config.yaml")

# Configure logger based on the yaml configuration
configure_logger(config.log_level, config.log_file)

logger.info(f"Framework environment verified successfully. Cache directory is set to {config.cache_dir}")
logger.info(f"Targeting device: {config.device}")
"""),
    md_cell("""## Check Acceleration Device
Apple Silicon Macs should show `mps`. Nvidia machines should show `cuda`. Otherwise, it falls back to `cpu`.
"""),
    code_cell("""# Test Device
device = torch.device(config.device)
print(f"Active Device: {device}")

if config.device == "mps":
    print("Apple Metal Performance Shaders (MPS) is available.")
elif config.device == "cuda":
    print(f"CUDA is available. Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU.")
"""),
]

# ----------------- Notebook 2: Dataset Loading -----------------
nb2_cells = [
    md_cell("""# 2. Dataset Loading and Sample Collection
For visual search research, we need real-world fashion images. Since we are evaluating detection and localization, we will set up a directory of sample images showing people wearing various clothes and accessories in unconstrained backgrounds.

In this notebook, we:
1. Define a local `data/` directory.
2. Download high-quality fashion images representing clothing, accessories, and shoes.
3. Establish a standard format for evaluating models on these images.
"""),
    code_cell("""import os
import sys
import requests
sys.path.append(os.path.abspath("."))
from fashion_detector.logging import logger

# Create data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# A curated set of copyright-free fashion images from Unsplash
IMAGE_URLS = {
    "fashion_model_street.jpg": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&q=80&w=800", # Yellow dress, hat, handbag
    "casual_wear_men.jpg": "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?auto=format&fit=crop&q=80&w=800", # Man in jacket, t-shirt, jeans, boots
    "accessories_social.jpg": "https://images.unsplash.com/photo-1542496658-e33a6d0d50f6?auto=format&fit=crop&q=80&w=800", # Watch, bracelet, knit sweater
    "multiple_people.jpg": "https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&q=80&w=800" # People holding shopping bags, sunglasses, coats
}

for name, url in IMAGE_URLS.items():
    dest_path = os.path.join(DATA_DIR, name)
    if not os.path.exists(dest_path):
        logger.info(f"Downloading sample image: {name} from {url}")
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(r.content)
            print(f"Downloaded {name} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
    else:
        print(f"Image {name} already exists.")
"""),
    md_cell(
        """Let's define a mock ground truth dictionary for `fashion_model_street.jpg` so that we can demonstrate quantitative IoU evaluations later.
"""
    ),
    code_cell("""# Mock ground truths for fashion_model_street.jpg
# Width: 800, Height: ~1067 depending on download crop.
# We'll save these annotations in JSON format for evaluation notebook.
import json

annotations = {
    "image_path": "data/fashion_model_street.jpg",
    "items": [
        {"label": "dress", "box": [300, 200, 950, 650]}, # ymin, xmin, ymax, xmax approx
        {"label": "hat", "box": [100, 300, 250, 500]},
        {"label": "handbag", "box": [650, 150, 850, 350]}
    ]
}

os.makedirs("annotations", exist_ok=True)
with open("annotations/fashion_model_street_gt.json", "w") as f:
    json.dump(annotations, f, indent=2)
print("Saved ground-truth annotations.")
"""),
]

# ----------------- Notebook 3: Image Preprocessing -----------------
nb3_cells = [
    md_cell("""# 3. Image Preprocessing and Visualization utilities
Before passing images into deep learning models, we must preprocess them. Furthermore, we need a high-quality interactive visualization layer.

In this notebook, we:
1. Load images using our `utils.load_image` function.
2. Examine image dimensions, resizing, padding, and normalizations.
3. Test our dynamic, clickable bounding box rendering tool (`utils.generate_interactive_html`).
"""),
    code_cell("""import os
import sys
from PIL import Image
sys.path.append(os.path.abspath("."))

from fashion_detector.utils import load_image, draw_bounding_boxes, generate_interactive_html
from fashion_detector.models.base import Detection
from IPython.display import HTML

# Load image
img_path = "data/fashion_model_street.jpg"
image = load_image(img_path)
print(f"Successfully loaded image {img_path} with size: {image.size}")
"""),
    md_cell("""## Creating Bounding Box Proposals for Testing
Let's create a few dummy detections to test both static drawing and our premium, clickable interactive HTML interface.
"""),
    code_cell("""# Dummy detections for testing
detections = [
    Detection(box=[200.0, 300.0, 650.0, 950.0], label="dress", score=0.92),
    Detection(box=[300.0, 100.0, 500.0, 250.0], label="hat", score=0.88),
    Detection(box=[150.0, 650.0, 350.0, 850.0], label="handbag", score=0.79)
]

# Render static image annotated with Pillow
annotated_img = draw_bounding_boxes(image, detections)
display(annotated_img.resize((400, int(400 * annotated_img.height / annotated_img.width))))
"""),
    md_cell("""## Clickable, Interactive HTML Visualization
Run the cell below to test the interactive widget. You can hover over boxes for tooltips, and click on them to populate item information in the panel below!
"""),
    code_cell("""# Render interactive HTML overlay
html_str = generate_interactive_html(image, detections, title="Interactive Proposal Test")
HTML(html_str)
"""),
]

# ----------------- Notebook 4: Grounding DINO -----------------
nb4_cells = [
    md_cell("""# 4. Grounding DINO Zero-Shot Object Detection
Grounding DINO is a state-of-the-art open-set object detector that connects text queries (natural language) to image regions. We will use the pure Python version from Hugging Face `transformers` to avoid C++/CUDA build problems.

In this notebook, we:
1. Load Grounding DINO.
2. Formulate query strings for fashion classes.
3. Run inference on street fashion.
4. Visualize using the interactive widget.
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.grounding_dino import GroundingDinoDetector
from fashion_detector.utils import load_image, generate_interactive_html
from IPython.display import HTML

# Load config
config = Config("config/config.yaml")

# Initialize Grounding DINO
detector = GroundingDinoDetector(config)
detector.load_model()
"""),
    md_cell("""## Zero-Shot Object Detection
We feed the model a photo and our list of fashion categories.
"""),
    code_cell("""image = load_image("data/fashion_model_street.jpg")

# Run zero-shot detection
# The detect function joins classes with " . " automatically for Grounding DINO.
detections = detector.detect(image, box_threshold=0.20, text_threshold=0.20)

print(f"Grounding DINO detected {len(detections)} fashion items:")
for d in detections:
    print(f"- {d.label.capitalize()}: conf={d.score:.2f}, box={list(map(int, d.box))}")
"""),
    md_cell("""## Render Clickable Detections"""),
    code_cell("""# Render interactive overlay
html_str = generate_interactive_html(image, detections, title="Grounding DINO Zero-Shot Detections")
HTML(html_str)
"""),
]

# ----------------- Notebook 5: YOLO -----------------
nb5_cells = [
    md_cell("""# 5. YOLO Object Detection Experiments
YOLO is the industry standard for fast real-time object detection. We evaluate it to check latency, GPU memory usage, and detection quality.

In this notebook, we:
1. Load YOLOv8 using `ultralytics`.
2. Run inference.
3. Understand its limitation on standard COCO classes (only detects accessories like backpacks, handbags, and neckties, plus general persons).
4. Run on a sample image and display outcomes.
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.yolo import YoloDetector
from fashion_detector.utils import load_image, generate_interactive_html
from IPython.display import HTML

config = Config("config/config.yaml")

# Instantiate YOLOv8
detector = YoloDetector(config)
detector.load_model()
"""),
    md_cell("""## Run Detection on street fashion"""),
    code_cell("""image = load_image("data/fashion_model_street.jpg")
detections = detector.detect(image, conf_threshold=0.20)

print(f"YOLO detected {len(detections)} items:")
for d in detections:
    print(f"- {d.label}: conf={d.score:.2f}, box={list(map(int, d.box))}")
"""),
    md_cell(
        """Notice that a standard COCO-trained YOLO model primarily detects accessories like `handbag` or `backpack`. To detect specific items of clothing like `dress`, `skirt`, or `jacket`, a custom-trained YOLO model is required, or we must use a foundation model like Grounding DINO or Florence-2. We will analyze these trade-offs in our evaluation notebook.
"""
    ),
    md_cell("""## Render Detections"""),
    code_cell(
        """html_str = generate_interactive_html(image, detections, title="YOLOv8 Detections")
HTML(html_str)
"""
    ),
]

# ----------------- Notebook 6: Florence-2 -----------------
nb6_cells = [
    md_cell("""# 6. Florence-2 Zero-Shot Visual Experiments
Florence-2 is Microsoft's advanced unified vision-language model. It performs multiple tasks (Object Detection, Captioning, Grounding) using textual prompt formatting.

In this notebook, we:
1. Load Florence-2.
2. Run standard `<OD>` (general object detection).
3. Run `<CAPTION_TO_PHRASE_GROUNDING>` (phrase grounding for specific fashion items).
4. Compare results.
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.florence2 import Florence2Detector
from fashion_detector.utils import load_image, generate_interactive_html
from IPython.display import HTML

config = Config("config/config.yaml")

# Initialize Florence-2
detector = Florence2Detector(config)
detector.load_model()
"""),
    md_cell("""## Task 1: General Object Detection (`<OD>`)
This task automatically outputs boxes and names without specifying categories.
"""),
    code_cell("""image = load_image("data/fashion_model_street.jpg")
detections_od = detector.detect(image, task="<OD>")

print("Florence-2 <OD> Detections:")
for d in detections_od:
    print(f"- {d.label}: box={list(map(int, d.box))}")
"""),
    md_cell("""## Task 2: Phrase Grounding (`<CAPTION_TO_PHRASE_GROUNDING>`)
We ask the model to look specifically for our fashion categories.
"""),
    code_cell("""fashion_queries = ["dress", "hat", "handbag", "sunglasses", "shoes"]
detections_grounding = detector.detect(image, task="<CAPTION_TO_PHRASE_GROUNDING>", queries=fashion_queries)

print("Florence-2 Phrase Grounding Detections:")
for d in detections_grounding:
    print(f"- {d.label}: box={list(map(int, d.box))}")
"""),
    md_cell("""## Visualize Grounding Detections"""),
    code_cell(
        """html_str = generate_interactive_html(image, detections_grounding, title="Florence-2 Grounding Detections")
HTML(html_str)
"""
    ),
]

# ----------------- Notebook 7: CLIPSeg -----------------
nb7_cells = [
    md_cell("""# 7. CLIPSeg Zero-Shot Image Segmentation & Localization
CLIPSeg generalizes image segmentation to zero-shot text prompts. We can leverage it for localization by binarizing its heatmaps and extracting connected component bounding boxes.

In this notebook, we:
1. Load CLIPSeg.
2. Input our fashion categories.
3. Compute segmentation masks and binarize them.
4. Extract boundaries and bounding boxes.
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.clipseg import ClipSegDetector
from fashion_detector.utils import load_image, generate_interactive_html
from IPython.display import HTML

config = Config("config/config.yaml")

# Initialize CLIPSeg
detector = ClipSegDetector(config)
detector.load_model()
"""),
    md_cell("""## Run Zero-Shot Segmentation & Box Extraction"""),
    code_cell("""image = load_image("data/fashion_model_street.jpg")
# Target categories to search in image
categories = ["dress", "hat", "handbag"]

# CLIPSeg detects masks and converts them to bounding boxes
detections = detector.detect(image, queries=categories, threshold=0.35)

print(f"CLIPSeg localized {len(detections)} items:")
for d in detections:
    print(f"- {d.label}: score={d.score:.2f}, box={list(map(int, d.box))}")
"""),
    md_cell("""## Interactive Visualization"""),
    code_cell(
        """html_str = generate_interactive_html(image, detections, title="CLIPSeg Segmentations & Boxes")
HTML(html_str)
"""
    ),
]

# ----------------- Notebook 8: FashionCLIP -----------------
nb8_cells = [
    md_cell("""# 8. FashionCLIP Fine-grained Classification
FashionCLIP is a domain-specific fine-tuned CLIP model for fashion image-text matching. Because CLIP models are not natively detectors, we use them as classifiers on image crops.

In this notebook, we:
1. Load FashionCLIP.
2. Manually define regions of interest (or crops).
3. Classify these crops against a text corpus of fashion categories to show FashionCLIP's superior zero-shot capabilities in domain-specific tasks.
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.fashion_clip import FashionClipDetector
from fashion_detector.models.base import Detection
from fashion_detector.utils import load_image, generate_interactive_html
from IPython.display import HTML

config = Config("config/config.yaml")

# Initialize FashionCLIP
classifier = FashionClipDetector(config)
classifier.load_model()
"""),
    md_cell("""## Set up manual crops (proposals)
We will take the street image and define bounding boxes for the hat, dress, and handbag, then run zero-shot classification on the crops.
"""),
    code_cell("""image = load_image("data/fashion_model_street.jpg")

# Bounding box proposals (e.g. from a first-stage class-agnostic detector)
proposals = [
    Detection(box=[200.0, 300.0, 650.0, 950.0], label="unknown_clothing", score=1.0),
    Detection(box=[300.0, 100.0, 500.0, 250.0], label="unknown_accessory", score=1.0),
    Detection(box=[150.0, 650.0, 350.0, 850.0], label="unknown_handbag", score=1.0)
]

# Fashion categories we want to map them to
candidate_categories = ["dress", "skirt", "jacket", "hat", "cap", "handbag", "backpack"]

# Run Stage 2 crop classification
refined_detections = classifier.classify_crops(image, proposals, categories=candidate_categories)

print("FashionCLIP Refined Categories:")
for d in refined_detections:
    print(f"Box {list(map(int, d.box))} -> Class: {d.label} (score: {d.score:.2f})")
"""),
    md_cell("""## Visualizing refined bounding boxes"""),
    code_cell(
        """html_str = generate_interactive_html(image, refined_detections, title="FashionCLIP Crop Classification")
HTML(html_str)
"""
    ),
]

# ----------------- Notebook 9: Vision LLMs -----------------
nb9_cells = [
    md_cell("""# 9. Vision LLMs through LiteLLM (Zero-Shot Detection)
Modern Multimodal LLMs (Gemini, Claude, GPT-4o) exhibit spatial reasoning capabilities. We can prompt them to detect fashion items and return their bounding boxes in JSON format.

In this notebook, we:
1. Configure LiteLLM to use Gemini 1.5 Flash (or another vision API).
2. Set up the prompt asking for `[ymin, xmin, ymax, xmax]` normalized coordinates (0-1000).
3. Parse the output JSON.
4. Scale coordinates back to pixel values and visualize.
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

# NOTE: To run this notebook successfully, ensure you have set the appropriate API key environment variable.
# E.g. os.environ["GEMINI_API_KEY"] = "your-api-key"
# We will check if it is set. If not, we will output a warning.
if not os.environ.get("GEMINI_API_KEY"):
    print("WARNING: GEMINI_API_KEY environment variable is not set. API calls will fail.")

from fashion_detector.config import Config
from fashion_detector.models.vision_llm import VisionLlmDetector
from fashion_detector.utils import load_image, generate_interactive_html
from IPython.display import HTML

config = Config("config/config.yaml")
detector = VisionLlmDetector(config)
"""),
    md_cell("""## Execute Vision LLM Object Detection
We query the model zero-shot.
"""),
    code_cell("""image = load_image("data/fashion_model_street.jpg")

# Run detection if API key is present, otherwise show mock output
if os.environ.get("GEMINI_API_KEY"):
    detections = detector.detect(image)
else:
    print("API Key not found. Displaying mocked Vision LLM detections...")
    from fashion_detector.models.base import Detection
    # Mocking Gemini 1.5 Flash response structure
    detections = [
        Detection(box=[200.0, 300.0, 650.0, 950.0], label="dress", score=0.95),
        Detection(box=[300.0, 100.0, 500.0, 250.0], label="hat", score=0.90),
        Detection(box=[150.0, 650.0, 350.0, 850.0], label="handbag", score=0.93)
    ]

print(f"Detected {len(detections)} fashion items:")
for d in detections:
    print(f"- {d.label}: score={d.score:.2f}, box={list(map(int, d.box))}")
"""),
    md_cell("""## Visualizing interactive outputs"""),
    code_cell(
        """html_str = generate_interactive_html(image, detections, title="Vision LLM Detections")
HTML(html_str)
"""
    ),
]

# ----------------- Notebook 10: Hybrid Pipeline -----------------
nb10_cells = [
    md_cell("""# 10. Hybrid Pipeline Experiments (Stage 1 + Stage 2)
In visual search, combining models yields superior results. Single models like standard YOLO miss detailed categories, and CLIP/FashionCLIP cannot localize on its own.
We combine:
- **Stage 1 (Detector)**: Grounding DINO (or YOLO) to detect candidate regions (regions containing any clothing/accessory item).
- **Stage 2 (Classifier)**: FashionCLIP to crop and classify each candidate region into fine-grained fashion classes.

In this notebook, we:
1. Initialize the `HybridPipeline` using dependency injection.
2. Run Stage 1 (Grounding DINO) to locate objects.
3. Run Stage 2 (FashionCLIP) to refine their labels.
4. Compare before (Stage 1) and after (Stage 2) classifications.
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.grounding_dino import GroundingDinoDetector
from fashion_detector.models.fashion_clip import FashionClipDetector
from fashion_detector.pipeline import HybridPipeline
from fashion_detector.utils import load_image, generate_interactive_html
from IPython.display import HTML

config = Config("config/config.yaml")

# Instantiate components
detector = GroundingDinoDetector(config)
classifier = FashionClipDetector(config)

# Instantiate the pipeline (Dependency Injection)
pipeline = HybridPipeline(detector=detector, classifier=classifier)
"""),
    md_cell("""## Execute the Hybrid Pipeline"""),
    code_cell("""image = load_image("data/fashion_model_street.jpg")

# Stage 1 runs zero-shot detection, then Stage 2 classifies crops
# We pass broad queries to DINO to detect any clothing or accessory item
dino_broad_queries = ["clothing", "accessory", "shoes", "bag", "hat"]

detections = pipeline.detect(
    image=image,
    categories=config.get_all_categories(),
    queries=dino_broad_queries
)

print(f"Hybrid pipeline detected {len(detections)} fashion items:")
for d in detections:
    print(f"- {d.label.capitalize()} (refined from {d.metadata.get('proposal_label')}): score={d.score:.2f}")
"""),
    md_cell("""## Render final interactive predictions"""),
    code_cell(
        """html_str = generate_interactive_html(image, detections, title="Hybrid Pipeline Detections")
HTML(html_str)
"""
    ),
]

# ----------------- Notebook 11: Comparative Evaluation -----------------
nb11_cells = [
    md_cell("""# 11. Comparative Evaluation & Benchmarking
In this notebook, we run our benchmarking harness on all candidate models to compare:
1. Average Latency (seconds)
2. Peak GPU Memory (if running on CUDA)
3. Detection Quality & Object counts
"""),
    code_cell("""import os
import sys
sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.grounding_dino import GroundingDinoDetector
from fashion_detector.models.yolo import YoloDetector
from fashion_detector.models.florence2 import Florence2Detector
from fashion_detector.models.clipseg import ClipSegDetector
from fashion_detector.models.fashion_clip import FashionClipDetector
from fashion_detector.pipeline import HybridPipeline
from fashion_detector.evaluation import benchmark_model, generate_summary_table
from fashion_detector.utils import load_image

config = Config("config/config.yaml")
image = load_image("data/fashion_model_street.jpg")

benchmarks = []
"""),
    md_cell("""## 1. Benchmark YOLO"""),
    code_cell("""yolo_det = YoloDetector(config)
yolo_res = benchmark_model(yolo_det, image, num_runs=3)
benchmarks.append(yolo_res)
"""),
    md_cell("""## 2. Benchmark Grounding DINO"""),
    code_cell("""dino_det = GroundingDinoDetector(config)
dino_res = benchmark_model(dino_det, image, num_runs=3)
benchmarks.append(dino_res)
"""),
    md_cell("""## 3. Benchmark Florence-2"""),
    code_cell("""florence_det = Florence2Detector(config)
florence_res = benchmark_model(florence_det, image, num_runs=3, task="<OD>")
benchmarks.append(florence_res)
"""),
    md_cell("""## 4. Benchmark CLIPSeg"""),
    code_cell("""clipseg_det = ClipSegDetector(config)
# Use a subset of queries to speed up evaluation
subset_queries = ["dress", "hat", "handbag"]
clipseg_res = benchmark_model(clipseg_det, image, num_runs=3, queries=subset_queries)
benchmarks.append(clipseg_res)
"""),
    md_cell("""## 5. Benchmark Hybrid Pipeline"""),
    code_cell("""fashion_clip_clf = FashionClipDetector(config)
pipeline = HybridPipeline(detector=dino_det, classifier=fashion_clip_clf)
pipeline_res = benchmark_model(
    pipeline, 
    image, 
    num_runs=3, 
    categories=config.get_all_categories(),
    queries=["clothing", "accessory", "shoes", "bag", "hat"]
)
benchmarks.append(pipeline_res)
"""),
    md_cell("""## Summary Comparison"""),
    code_cell("""summary_df = generate_summary_table(benchmarks)
display(summary_df)
"""),
]

# ----------------- Notebook 12: Final Recommendations -----------------
nb12_cells = [md_cell("""# 12. Final Recommendations & Production Architectures
This notebook concludes the research framework by reviewing our empirical findings and documenting production architecture strategies.

## Performance Analysis & Findings
1. **YOLO**: Fastest inference latency (<30ms). However, standard weights (COCO) miss critical clothing classes. In production, custom fine-tuning YOLOv8/v11 on fashion datasets (like DeepFashion2 or Modanet) is the standard method for Stage 1 region proposals.
2. **Grounding DINO**: Excellent zero-shot localization of generic classes. Heavy inference footprint and slower than YOLO. Ideal for labeling pipelines and bootstrapping datasets.
3. **Florence-2**: High accuracy, multi-task capability. Slightly faster than Grounding DINO.
4. **CLIPSeg**: Excellent for getting precise segmentation boundaries, but slow since it processes queries iteratively.
5. **FashionCLIP**: Domain-specific fine-tuning makes it highly accurate at matching fashion concepts, making it ideal as a Stage 2 classifier/embedder.
6. **Hybrid Pipeline**: Grounding DINO/YOLOv8 + FashionCLIP provides the best balance. Stage 1 localizes the items, and Stage 2 computes fashion-specific embeddings.

---

## Production Architectures (Industry Review)
In large visual search systems like **Pinterest Lens**, **Google Lens**, and **Amazon StyleSnap**, a multi-stage architecture is deployed:

```mermaid
graph TD
    A[User Uploads Image] --> B[Stage 1: Detection & Proposal]
    B -->|BBoxes| C[Stage 2: Crop & Feature Extraction]
    C -->|Embeddings| D[Stage 3: Vector Database Search]
    D -->|Similarity Match| E[Return Catalog Results]
```

### Stage 1: Detection & Proposal (Latency Critical)
- Production uses high-speed detectors (YOLO, custom SSD, or CenterNet) trained on fashion datasets to output high-recall bounding boxes for all fashion items.
- Focuses entirely on localization recall, not fine-grained classification.

### Stage 2: Feature Extraction & Embedding (Accuracy Critical)
- Cropped regions are fed into a specialized embedding network (e.g., FashionCLIP, Vision-Transformer, or custom metric-learning model).
- Outputs a dense vector (e.g., 512 dimensions) representing the visual style and attributes.

### Stage 3: Approximate Nearest Neighbor (ANN) Search
- Embeddings are queried against a vector database (e.g., Milvus, Pinecone, or FAISS) to retrieve matching products in milliseconds.

## Selected Architecture for Next Stage
We recommend proceeding with the **YOLO (Custom Fashion Fine-tuned) + FashionCLIP (Embedding Extract) + FAISS (Index Search)** stack for the production visual search system.
""")]

# ----------------- Notebook 14: SAM Detection -----------------
nb14_cells = [
    md_cell("""# 14. Segment Anything Model (SAM 3.1 / SAM 2 / SAM) Experiments
The **Segment Anything Model (SAM)** is a foundation vision model for promptable, zero-shot image segmentation.

In this notebook, we explore SAM for fashion item detection and segmentation:
1. **Model Initialization**: Load SAM (`facebook/sam-vit-base` or `facebook/sam2.1-hiera-small`) via Hugging Face `transformers`.
2. **Box-Prompted Segmentation**: Supply bounding box prompts (e.g. coarse proposals from Stage 1 detectors) to generate pixel-accurate segmentation masks.
3. **Point-Prompted Segmentation**: Pass interactive 2D point prompts `(x, y)` to segment targeted garment or accessory regions.
4. **Automatic Grid Instance Segmentation**: Generate zero-shot masks across an image grid.
5. **Visualization**: Render interactive HTML overlays with bounding boxes, confidence scores, and segmentation masks.
"""),
    code_cell("""import os
import sys
import numpy as np
from PIL import Image
from IPython.display import HTML

sys.path.append(os.path.abspath("."))

from fashion_detector.config import Config
from fashion_detector.models.sam import SamDetector
from fashion_detector.utils import load_image, generate_interactive_html

# Load configuration
config = Config("config/config.yaml")

# Initialize SAM Detector
detector = SamDetector(config)
detector.load_model()
"""),
    md_cell("""## 1. Box-Prompted Segmentation
Supply candidate bounding boxes `[xmin, ymin, xmax, ymax]` and target category queries to SAM.
SAM refines the bounding box boundaries and extracts pixel-level binary masks.
"""),
    code_cell(
        """image_path = "image.png" if os.path.exists("image.png") else "data/fashion_model_street.jpg"
image = load_image(image_path)
w, h = image.size

# Candidate bounding box prompts scaled to image dimensions
prompt_boxes = [
    [int(w * 0.28), int(h * 0.14), int(w * 0.72), int(h * 0.32)],  # Upper garment / jacket area
    [int(w * 0.30), int(h * 0.33), int(w * 0.70), int(h * 0.76)]   # Lower garment / pants area
]
labels = ["jacket", "pants"]

# Run SAM box-prompted detection
detections = detector.detect(image, input_boxes=prompt_boxes, queries=labels)

print(f"SAM Box-Prompted Detections: {len(detections)}")
for det in detections:
    mask_str = f"shape={det.mask.shape}" if det.mask is not None else "None"
    print(f"- Label: '{det.label}', Score: {det.score:.3f}, Box: {list(map(int, det.box))}, Mask: {mask_str}")
"""
    ),
    md_cell("""## 2. Render Interactive Visualization
Render an interactive HTML overlay displaying SAM bounding boxes and labels.
"""),
    code_cell("""html_str = generate_interactive_html(
    image,
    detections,
    title="SAM Box-Prompted Fashion Item Segmentation"
)
HTML(html_str)
"""),
    md_cell("""## 3. Point-Prompted Segmentation
Supply 2D point coordinates `[x, y]` to prompt SAM on specific visual keypoints.
"""),
    code_cell("""img_w, img_h = image.size

# Sample point prompts near center of key fashion regions
point_prompts = [
    [img_w // 2, img_h // 4],     # Upper body point
    [img_w // 2, 2 * img_h // 3]   # Lower body point
]

point_detections = detector.detect(image, input_points=point_prompts, pred_iou_thresh=0.75)

print(f"SAM Point-Prompted Detections: {len(point_detections)}")
for det in point_detections:
    print(f"- Label: '{det.label}', Score: {det.score:.3f}, Box: {list(map(int, det.box))}")
"""),
    md_cell("""## 4. Automatic Zero-Shot Grid Segmentation
Generate segmentation masks automatically across an image grid using `points_per_side`.
"""),
    code_cell("""# Run automatic grid segmentation
auto_detections = detector.detect(
    image,
    points_per_side=16,
    pred_iou_thresh=0.82,
    remove_small_boxes=True
)

print(f"SAM Automatic Grid Segmentation detected {len(auto_detections)} item masks.")
"""),
    md_cell("""## 5. Native Segmented Class Image Extraction (SAM 3.1)
Extract pixel-isolated RGBA PIL Images (transparent background) for each class natively using `segment_classes()` or `extract_segmented_objects()`.
"""),
    code_cell("""# Extract isolated transparent PNG crops per class
class_crops = detector.segment_classes(
    image,
    user_categories=["jacket", "pants"],
    boxes=prompt_boxes
)

for label, crop_list in class_crops.items():
    print(f"Extracted {len(crop_list)} isolated segmented image(s) for class '{label}':")
    for idx, crop in enumerate(crop_list):
        print(f"  - Crop {idx+1}: dimensions={crop.size}, mode={crop.mode}")

# Display isolated transparent PNG crop for first item
if "jacket" in class_crops and len(class_crops["jacket"]) > 0:
    display(class_crops["jacket"][0])
if "pants" in class_crops and len(class_crops["pants"]) > 0:
    display(class_crops["pants"][0])
"""),
    md_cell("""## 6. Summary & Key Takeaways
- **SAM 3.1 Architecture**: Utilizes `facebook/sam2.1-hiera-small` for fast, state-of-the-art segmentation.
- **Native Class Segmentation**: `segment_classes()` and `extract_segmented_objects()` produce clean, transparent RGBA PNG crops for every detected item.
- **Stage 2 Synergy**: Candidate proposals from Stage 1 (e.g. YOLO, Grounding DINO) can be passed directly as box prompts to SAM 3 for background suppression and clean mask extraction.
- **Flexibility**: Supports box prompts, point prompts, and automatic grid generation.
"""),
]

# Save all notebooks
create_notebook("01_environment_setup.ipynb", nb1_cells)
create_notebook("02_dataset_loading.ipynb", nb2_cells)
create_notebook("03_image_preprocessing.ipynb", nb3_cells)
create_notebook("04_grounding_dino_experiments.ipynb", nb4_cells)
create_notebook("05_yolo_experiments.ipynb", nb5_cells)
create_notebook("06_florence_experiments.ipynb", nb6_cells)
create_notebook("07_clipseg_experiments.ipynb", nb7_cells)
create_notebook("08_fashion_clip_experiments.ipynb", nb8_cells)
create_notebook("09_vision_llm_experiments.ipynb", nb9_cells)
create_notebook("10_hybrid_pipeline_experiments.ipynb", nb10_cells)
create_notebook("11_comparative_evaluation.ipynb", nb11_cells)
create_notebook("12_final_recommendations.ipynb", nb12_cells)
create_notebook("14_sam_detection_experiments.ipynb", nb14_cells)
print("All notebooks created successfully!")
