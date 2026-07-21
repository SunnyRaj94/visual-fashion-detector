# Fashion Item Detection & Localization Research Framework

A modular, configuration-driven research framework to detect and localize fashion items (clothing, accessories, and shoes) from unconstrained real-world images. The framework allows users to compare state-of-the-art vision models (Grounding DINO, YOLOv8, Florence-2, CLIPSeg, and Vision LLMs) and combine them into hybrid pipelines (e.g., Detector + FashionCLIP embedder/classifier).

---

## 🌟 Key Features

- **SOLID & Modular Architecture**: Easy to extend with new models by implementing the `BaseDetector` interface.
- **Dynamic Clickable Overlays**: Generates interactive HTML visualization widgets with clickable bounding boxes and tooltips directly inside Jupyter notebooks.
- **Domain-Specific Classification**: Uses FashionCLIP to perform zero-shot classification on cropped region proposals from Stage-1 detectors.
- **Comprehensive Benchmarking**: Automatic latency profiling, memory logging, and IoU metric calculation for model comparison.
- **Structured Logging & Error Handling**: Unified logging with timing decorators and context managers to inspect bottleneck steps.

---

## 📂 Project Structure

```text
├── config/
│   └── config.yaml           # Centralized configuration (model names, parameters, categories)
├── fashion_detector/
│   ├── __init__.py
│   ├── config.py             # YAML Config parser & environment setup
│   ├── logging.py            # Centralized structured logger & execution timer
│   ├── utils.py              # Visualizations, PIL operations, and Interactive HTML builder
│   ├── pipeline.py           # Hybrid two-stage orchestrator (Stage 1 Detection + Stage 2 Embedding)
│   ├── evaluation.py         # Benchmarking harness (latency, IoU mapping)
│   └── models/
│       ├── __init__.py
│       ├── base.py           # Base abstract class for all detectors
│       ├── grounding_dino.py # Grounding DINO Zero-shot Object Detector (HF transformers)
│       ├── yolo.py           # YOLO detector (Ultralytics API)
│       ├── florence2.py      # Florence-2 unified VL model (Microsoft)
│       ├── clipseg.py        # CLIPSeg Zero-shot segmenter & contour-box extractor
│       ├── fashion_clip.py   # FashionCLIP Stage-2 crop classifier
│       └── vision_llm.py     # Vision LLM zero-shot detector (via LiteLLM)
├── data/                     # Downloaded street fashion images for testing
├── annotations/              # Mock ground-truth annotations for metric evaluation
├── pyproject.toml            # Project packaging & dependency definitions
├── README.md                 # Documentation
└── 01_... to 12_...ipynb     # Step-by-step Jupyter notebooks
```

---

## 🚀 Installation & Setup

1. **Create Virtual Environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   Install the package in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

3. **API Keys (Optional for Vision LLMs)**:
   To run experiments with Google Gemini, set your API key:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

---

## 📓 Notebook Walkthroughs

The framework includes 12 incrementally built, executable Jupyter notebooks:

1. **`01_environment_setup.ipynb`**: Verifies directories, PyTorch hardware acceleration, and parses config.
2. **`02_dataset_loading.ipynb`**: Sets up test image suite and mock ground-truths.
3. **`03_image_preprocessing.ipynb`**: Tests image manipulations and the clickable HTML visualization widget.
4. **`04_grounding_dino_experiments.ipynb`**: Evaluates Grounding DINO zero-shot text-prompted localization.
5. **`05_yolo_experiments.ipynb`**: Explores YOLOv8 and analyzes COCO class limitations.
6. **`06_florence_experiments.ipynb`**: Investigates Microsoft Florence-2 `<OD>` and `<CAPTION_TO_PHRASE_GROUNDING>`.
7. **`07_clipseg_experiments.ipynb`**: Runs zero-shot segmentation and extracts bounding boxes from contours.
8. **`08_fashion_clip_experiments.ipynb`**: Runs zero-shot classification on custom region crops.
9. **`09_vision_llm_experiments.ipynb`**: Queries Gemini 1.5 Flash using structured 2D coordinate prompts.
10. **`10_hybrid_pipeline_experiments.ipynb`**: Integrates Detector (Stage 1) + FashionCLIP (Stage 2) in a hybrid pipeline.
11. **`11_comparative_evaluation.ipynb`**: Benchmarks all models for latency and memory usage.
12. **`12_final_recommendations.ipynb`**: Evaluates production trade-offs and reviews Google Lens / Pinterest style architectures.

---

## 🛠️ Configuration

All hyperparameters, thresholds, model paths, and category taxonomies are configured in `config/config.yaml`.
Example configuration snippet:
```yaml
device: "auto"
cache_dir: "./cache"

models:
  grounding_dino:
    name: "IDEA-Research/grounding-dino-tiny"
    box_threshold: 0.25
    text_threshold: 0.25
  yolo:
    name: "yolov8m.pt"
    conf_threshold: 0.25
    iou_threshold: 0.45
```
