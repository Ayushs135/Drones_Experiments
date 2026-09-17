# Classical Crack Detection Baseline & Optimization Experiment

## 1. Objective

This experiment implements and benchmarks classical computer-vision approaches for structural crack detection in concrete surfaces. It provides an honest, reproducible, and transparent baseline to compare against the project's **YOLOv8n-Seg** instance segmentation model.

> **Key Context**:
> * **Hardware Constraints**: Fully CPU-based (runs on integrated graphics with ~3.3 ms per image, no CUDA/GPU required).
> * **Zero Neural Networks**: All operations are deterministic image-processing filters (gradients, morphological transforms, adaptive thresholding, geometric filtering).
> * **Optimization Milestone**: The pipeline was refined to elevate the F1 score from **4.46% (naive Canny)** to **46.13% (Enhanced Morphological Top-Hat)** by systematically eliminating concrete texture false positives.

---

## 2. Methodology & Pipeline Evolution

### Method 1: Standard Canny Baseline (F1 ≈ 4.46%)
```text
Grayscale -> Gaussian Blur -> CLAHE -> Canny Edge Detection -> Morphological Close -> Area Filter
```
* **Limitation**: The Canny detector evaluates local gradient magnitude without semantic context. Rough concrete textures, aggregates, and pores trigger thousands of edge pixels, yielding high recall (51.8%) but severe false alarms (Precision: 2.55%).

---

### Method 2: Enhanced Morphological Top-Hat Pipeline (F1 ≈ 46.13%)
```text
Input Image (416x416 RGB)
          ↓
  Grayscale Conversion
          ↓
  Bilateral Filtering (d=7, σ=50) -> Edge-preserving texture smoothing
          ↓
  Morphological Black Top-Hat (Closing - Original, k=11) -> Extracts dark linear valleys
          ↓
  Otsu Automatic Thresholding -> Binarizes crack valley responses
          ↓
  Connected-Component Geometric & Length Filtering (area ≥ 40 px, max(w,h) ≥ 35 px)
          ↓
  Morphological Dilation (k=3) -> Matches polygon ground-truth crack width
          ↓
  Predicted Binary Crack Mask (uint8: 0 or 255)
```

#### Why the Enhanced Classical Method Works:
1. **Bilateral Filtering**: Smooths out concrete surface noise while strictly maintaining the sharp intensity edges of cracks.
2. **Morphological Black Top-Hat ($\text{Closing}(I) - I$)**: Isolates dark narrow linear depressions against a lighter background while automatically subtracting low-frequency illumination gradients and large shadows.
3. **Geometric Aspect-Ratio Filtering**: Real cracks are long continuous structures ($\text{length} \ge 35\text{px}$), whereas concrete pores and aggregates form small circular blobs. Filtering by component length and area eliminates >90% of texture false alarms.
4. **Crack Width Dilation**: Ground truth masks are polygons with 3–5 pixel thickness. Single-pixel skeletons have minimal overlap; dilating the skeleton matches the ground-truth mask geometry, raising IoU and F1.

---

## 3. Dataset Structure

The dataset source is `crack-seg.zip` (preserved without alteration):

| Split | Image Count | Label Count | Resolution | Annotation Format |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 3,717 | 3,717 | 416x416 | JPEG / YOLO Polygons |
| **Validation** | 200 | 200 | 416x416 | JPEG / YOLO Polygons |
| **Test** | 112 | 112 | 416x416 | JPEG / YOLO Polygons |

Polygons are converted to binary raster masks using `cv2.fillPoly` (crack pixels = 255, background = 0).

---

## 4. Setup and Environment

### Prerequisites
* Python 3.10+ (Tested on Python 3.11)
* Windows / Linux / macOS (CPU only)

### Virtual Environment Setup
From the repository root:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\venv\Scripts\activate.bat
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install numpy opencv-python matplotlib pandas scikit-learn jupyter Pillow ipykernel
```

### Launching the Jupyter Notebook

```bash
jupyter notebook classical_crack_detection.ipynb
```

---

## 5. Experimental Results (Test Set Evaluation)

### Summary Comparison Table (112 Test Images)

| Metric | Canny Baseline | Enhanced Classical Pipeline | Improvement |
| :--- | :---: | :---: | :---: |
| **Precision** | 2.55% | **44.66%** | **+42.11%** |
| **Recall** | 51.82% | **79.37%** | **+27.55%** |
| **F1 Score** | 4.46% | **46.13%** | **+41.67% (10x gain)** |
| **IoU (Jaccard)** | 2.40% | **35.36%** | **+32.96%** |
| **Accuracy** | 38.25% | **89.26%** | **+51.01%** |
| **Inference Latency** | 2.74 ms | **3.35 ms** | Real-time (>290 FPS on CPU) |

### Generated Artifacts

```text
results/
├── highlighted/
│   ├── sample_01_comparison.png   # 5-panel: Original | GT | Canny | Enhanced | Overlay
│   ├── sample_02_comparison.png
│   ├── sample_03_comparison.png
│   ├── sample_04_comparison.png
│   └── sample_05_comparison.png
├── metrics/
│   ├── classical_results.csv      # Complete per-image evaluation DataFrame
│   ├── confusion_matrix.png       # Pixel-level confusion matrix heatmap
│   └── metrics_summary.png        # Comparative performance bar chart
└── visualizations/
    ├── error_analysis_false_positives.png  # Concrete texture false alarms
    └── error_analysis_false_negatives.png  # Low-contrast missed cracks
```

---

## 6. Conceptual Comparison: Classical CV vs. YOLOv8n-Seg

| Property | Classical CV Baseline (Canny) | Enhanced Classical (Top-Hat + Geometry) | YOLOv8n-Seg (Deep Learning) |
| :--- | :--- | :--- | :--- |
| **F1 Score** | ~4.5% | **~46.1%** | **~75–85%+ (Target)** |
| **Learning-based** | No | No | Yes (Supervised deep neural network) |
| **Training Required** | No | No | Yes (Backpropagation on labeled dataset) |
| **Crack Features** | Pixel intensity gradients | Dark morphological valleys & component geometry | Hierarchical multi-scale deep features |
| **Lighting & Shadow Sensitivity** | High | Moderate (Top-Hat subtracts shadows) | Low (Learned illumination invariance) |
| **Texture Sensitivity** | Severe (Aggregate noise) | Moderate (Aspect ratio rejects pores) | Low (Filters non-crack surface textures) |
| **Confidence Score** | None | None | Yes (Per-instance mask confidence) |
| **Inference Latency** | ~2.7 ms (CPU) | ~3.4 ms (CPU) | ~15–40 ms (CPU) / ~3 ms (GPU) |
| **Suitability for Field Drone Inspection** | Limited | Moderate Baseline | High (Standard for field deployment) |

---

## 7. Repository Directory Structure

```text
Drones_Experiments/
├── .gitignore                         # Excludes large data archives, venv, and checkpoints
├── README.md                          # Comprehensive experiment report and guide
├── classical_crack_detection.ipynb    # Executable Jupyter Notebook (with pre-run outputs)
├── crack-seg.zip                      # Source dataset archive
├── data/                              # Extracted images & YOLO labels (git-ignored)
├── venv/                              # Virtual environment (git-ignored)
└── results/                           # Evaluation outputs and visualizations
    ├── highlighted/                   # Visual comparisons with crack overlays
    ├── metrics/                       # CSV metrics table, confusion matrix, bar chart
    └── visualizations/               # Error mode analyses
```
