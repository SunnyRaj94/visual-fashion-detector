import time
import gc
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fashion_detector.models.base import BaseDetector, Detection
from fashion_detector.logging import logger


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]

    Returns:
        IoU value between 0.0 and 1.0.
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Intersection coordinates
    ixmin = max(xmin1, xmin2)
    iymin = max(ymin1, ymin2)
    ixmax = min(xmax1, xmax2)
    iymax = min(ymax1, ymax2)

    iw = max(0.0, ixmax - ixmin)
    ih = max(0.0, iymax - iymin)

    intersection_area = iw * ih

    # Union Area
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - intersection_area

    if union_area == 0.0:
        return 0.0

    return intersection_area / union_area


def match_detections(
    predictions: List[Detection],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Tuple[
    List[Tuple[Detection, Dict[str, Any]]], List[Detection], List[Dict[str, Any]]
]:
    """Matches prediction bounding boxes to ground truths based on class and IoU.

    Args:
        predictions: List of Detection predictions.
        ground_truths: List of ground truth dicts with keys 'box' and 'label'.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        Tuple of (matched_pairs, unmatched_predictions, unmatched_ground_truths).
    """
    matched = []
    unmatched_preds = list(predictions)
    unmatched_gts = list(ground_truths)

    # Sort predictions by score descending to prioritize best matches first
    sorted_preds = sorted(predictions, key=lambda x: x.score, reverse=True)

    for pred in sorted_preds:
        best_gt_idx = -1
        best_iou = -1.0

        for idx, gt in enumerate(unmatched_gts):
            # Must match category
            if pred.label.lower() != gt["label"].lower():
                continue

            iou = calculate_iou(pred.box, gt["box"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx != -1:
            gt_match = unmatched_gts.pop(best_gt_idx)
            matched.append((pred, gt_match))
            unmatched_preds.remove(pred)

    return matched, unmatched_preds, unmatched_gts


def benchmark_model(
    detector: BaseDetector, image: Image.Image, num_runs: int = 5, **kwargs: Any
) -> Dict[str, Any]:
    """Benchmarks a model for latency and memory usage.

    Args:
        detector: The detector instance to benchmark.
        image: The PIL Image to run detection on.
        num_runs: Number of inference runs to average latency over.

    Returns:
        Dictionary containing benchmark metrics.
    """
    logger.info(
        f"Starting benchmark for {detector.__class__.__name__} over {num_runs} runs..."
    )

    # Warmup run (also triggers downloading/caching on first run)
    try:
        detector.detect(image, **kwargs)
    except Exception as e:
        logger.error(f"Warmup detection failed: {e}")
        return {"error": str(e)}

    # Latency tracking
    latencies = []

    # Clear cache before memory tracking if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    elif torch.backends.mps.is_available():
        gc.collect()
        initial_memory = 0.0
    else:
        gc.collect()
        initial_memory = 0.0

    # Run loop
    for run in range(num_runs):
        start_time = time.perf_counter()
        detections = detector.detect(image, **kwargs)
        duration = time.perf_counter() - start_time
        latencies.append(duration)

    # Peak memory tracking (CUDA specific, as MPS has no simple memory allocation function in torch)
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        memory_delta = peak_memory - initial_memory
    else:
        peak_memory = 0.0
        memory_delta = 0.0

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))

    result = {
        "model_class": detector.__class__.__name__,
        "avg_latency_sec": avg_latency,
        "std_latency_sec": std_latency,
        "peak_gpu_mem_mb": peak_memory,
        "gpu_mem_delta_mb": memory_delta,
        "num_detections": len(detections),
        "detections": [d.to_dict() for d in detections],
    }

    logger.info(
        f"Benchmark results for {detector.__class__.__name__}: "
        f"Latency={avg_latency:.4f}s ± {std_latency:.4f}s, "
        f"Detections={len(detections)}"
    )

    return result


def generate_summary_table(benchmarks: List[Dict[str, Any]]) -> pd.DataFrame:
    """Combines multiple benchmark results into a single pandas DataFrame."""
    rows = []
    for bench in benchmarks:
        if "error" in bench:
            continue
        rows.append(
            {
                "Model / Pipeline": bench["model_class"],
                "Avg Latency (s)": round(bench["avg_latency_sec"], 4),
                "Latency Std (s)": round(bench["std_latency_sec"], 4),
                "Avg Detections": bench["num_detections"],
                "Peak GPU Memory (MB)": (
                    round(bench["peak_gpu_mem_mb"], 2)
                    if bench["peak_gpu_mem_mb"] > 0
                    else "N/A"
                ),
            }
        )
    return pd.DataFrame(rows)
