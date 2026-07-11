# Week 4 — Computer Vision: ANN vs CNN

**Notebook:** `week4_saksham_saraogi_jecrc.ipynb`  
**Dataset:** CIFAR-10 (loaded via `tensorflow.keras.datasets`)  
**Intern:** Saksham Saraogi · JECRC University · Celebal Technologies CEI

---

## Overview

Builds and benchmarks multiple neural network architectures on the CIFAR-10 image classification dataset. Progresses from a flat dense ANN baseline to a scaled-up CNN with data augmentation and early stopping — systematically comparing how architectural choices affect validation accuracy.

---

## Dataset

| Property | Value |
|----------|-------|
| Name | CIFAR-10 |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32 × 32 × 3 (RGB) |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |

Pixel values normalized from [0, 255] to [0.0, 1.0] before training. Flattened vectors (3072-dim) used for ANN models; 3D tensors preserved for CNN models.

---

## Models Built

| # | Model | Architecture | Epochs |
|---|-------|-------------|--------|
| 1 | Baseline ANN | Dense + Dropout | 10 |
| 2 | Baseline CNN | Conv2D + BatchNorm + MaxPooling + Dense | 10 |
| 3 | Data-Augmented CNN | Same as CNN + RandomFlip + RandomRotation + RandomZoom | 10 |
| 4 | Improved ANN | Deeper Dense layout (Student Task 1) | 10 |
| 5 | Scaled-Up CNN | 32→64→128 filters + 20 epochs + EarlyStopping (Tasks 2–4) | 20 |

---

## Sections Covered

| Section | Topic |
|---------|-------|
| 1 | Imports — TensorFlow, Keras, NumPy, Matplotlib |
| 2 | Load CIFAR-10 — shape verification, class label definitions |
| 3 | Preprocessing — normalization, ANN flattening |
| 4 | Baseline ANN — Dense(512) + Dropout(0.3) + Dense(256) + output |
| 5 | Baseline CNN — Conv2D(32) + BatchNorm + Pool → Conv2D(64) + BatchNorm + Pool → Dense |
| 6 | Accuracy comparison: ANN vs CNN at 10 epochs |
| 7 | Data-Augmented CNN — RandomFlip, RandomRotation, RandomZoom in-pipeline |
| 8 | Student Tasks 1–5 implementation |
| 9 | Final comparison table — all 5 model variants |

---

## Student Tasks

| Task | Description | Implementation |
|------|-------------|----------------|
| 1 | Increase ANN layers | Added extra Dense layers; showed depth alone doesn't close the gap vs CNN |
| 2 | Change CNN filters 32→64→128 | Progressive filter scaling for richer feature maps |
| 3 | Increase epochs to 20 | Extended training on scaled-up CNN |
| 4 | Add EarlyStopping | `patience=3` on validation loss — prevents overfitting |
| 5 | Add data augmentation | RandomFlip + RandomRotation + RandomZoom (Section 7) |

---

## Key Findings

- CNNs significantly outperform ANNs on image data by preserving spatial structure through convolutional filters
- Batch Normalization stabilizes training and accelerates convergence
- Data augmentation improves generalization by exposing the model to transformed views during training
- EarlyStopping prevents wasted compute and overfitting — the model halts when validation loss plateaus
- Scaling filters (32→64→128) progressively captures low-level (edges), mid-level (textures), and high-level (objects) features

---

## Libraries Used

```python
tensorflow, keras
numpy, matplotlib
```

---

## How to Run

1. Open `week4_saksham_saraogi_jecrc.ipynb` in Google Colab
2. No dataset upload needed — CIFAR-10 downloads automatically via `keras.datasets.cifar10.load_data()`
3. GPU runtime recommended (`Runtime → Change runtime type → T4 GPU`)
4. Run all cells top to bottom

---

*Week 4 · Celebal Excellence Internship · Data Science · 2026*
