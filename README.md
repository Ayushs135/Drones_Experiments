# Classical Crack Detection Experiment

## 1. Overview

This experiment implements a **CPU-based classical computer-vision pipeline for structural crack detection and segmentation**. It serves as an interpretable baseline alongside the project's YOLOv8n-Seg approach.

The method uses image-processing and morphological operations rather than neural-network training, making it lightweight and suitable for systems without dedicated GPU hardware.

---

## 2. Architecture

The complete pipeline is:

```text
Input Image (416 × 416 RGB)
            ↓
    Grayscale Conversion
            ↓
    Bilateral Filtering
       (d = 7, σ = 50)
            ↓
 Morphological Black Top-Hat
       (Kernel = 11 × 11)
            ↓
    Otsu Thresholding
            ↓
  Connected Components
            ↓
 Area & Length Filtering
 (Area ≥ 40 px, Length ≥ 35 px)
            ↓
   Morphological Dilation
       (Kernel = 3 × 3)
            ↓
   Binary Crack Mask
      0 = Background
      255 = Crack
```

### Processing Steps

**Bilateral Filtering**
Reduces fine concrete texture while preserving important intensity boundaries.

**Black Top-Hat Transformation**
Highlights dark, narrow structures against the surrounding concrete surface.

$$
Black\ TopHat = Closing(I) - I
$$

**Otsu Thresholding**
Automatically converts the morphological response into a binary candidate crack mask.

**Geometric Filtering**
Connected components are filtered using minimum area and length constraints to remove small surface artifacts, pores, and texture.

**Morphological Dilation**
Expands the detected crack structures so that the predicted regions better represent the annotated crack width.

---

## 3. Results

The enhanced pipeline was evaluated on the **112-image held-out test set**.

| Metric          |            Result |
| --------------- | ----------------: |
| Precision       |        **44.66%** |
| Recall          |        **79.37%** |
| F1 Score        |        **46.13%** |
| IoU (Jaccard)   |        **35.36%** |
| Pixel Accuracy  |        **89.26%** |
| Average Latency | **3.35 ms/image** |

The method achieves approximately **290+ images/second** under the measured CPU conditions.

The relatively high recall indicates that the pipeline detects a substantial portion of annotated crack pixels, while the precision and IoU values also show the remaining difficulty of separating cracks from concrete texture and other surface structures.

---

## 4. Limitations

The method is based on manually designed image-processing rules and therefore has several limitations:

* **Lighting sensitivity:** Changes in illumination can affect intensity-based crack detection.
* **Surface texture:** Aggregates, pores, stains, and rough concrete can produce crack-like responses.
* **Parameter dependence:** Kernel sizes and geometric thresholds may need adjustment for different imaging conditions.
* **Appearance assumptions:** Very faint, wide, or unusual cracks may not satisfy the selected morphological characteristics.
* **No learned representation:** The method does not learn crack features from training data.
* **No native confidence score:** Unlike a trained detection model, it does not provide a learned confidence value for each detection.

These limitations motivate the use of a learned segmentation approach such as YOLOv8n-Seg for more variable inspection environments.

---

## 5. Project Structure

```text
Drones_Experiments/
│
├── README.md
├── classical_crack_detection.ipynb
├── crack-seg.zip
│
├── data/
│   └── extracted dataset
│
├── venv/
│   └── Python virtual environment
│
└── results/
    │
    ├── highlighted/
    │   ├── sample_01_comparison.png
    │   ├── sample_02_comparison.png
    │   ├── sample_03_comparison.png
    │   ├── sample_04_comparison.png
    │   └── sample_05_comparison.png
    │
    ├── metrics/
    │   ├── classical_results.csv
    │   ├── confusion_matrix.png
    │   └── metrics_summary.png
    │
    └── visualizations/
        ├── error_analysis_false_positives.png
        └── error_analysis_false_negatives.png
```

---

## 6. Installation

### Requirements

* Python 3.10+
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Pillow

The experiment is **CPU-only** and does not require CUDA or a dedicated GPU.

### Create Virtual Environment

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install numpy opencv-python matplotlib pandas scikit-learn jupyter Pillow ipykernel
```

---

## 7. Running the Experiment

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
classical_crack_detection.ipynb
```

Run the notebook from top to bottom.

The notebook performs:

1. Dataset loading and inspection
2. Ground-truth mask generation
3. Classical crack detection
4. Crack-mask visualization
5. Pixel-level metric calculation
6. Confusion-matrix generation
7. Error analysis
8. Runtime measurement
9. Result export

Generated results are stored in the `results/` directory.
