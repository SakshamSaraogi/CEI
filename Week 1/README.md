# Week 1 — ML Foundations

**Notebook:** `week1_saksham_saraogi_jecrc.ipynb`  
**Intern:** Saksham Saraogi · JECRC University · Celebal Technologies CEI

---

## Overview

This notebook covers the foundational Python and mathematical toolkit required for machine learning. The work is structured across six parts — from core Python syntax to probability distributions and visualization — building up the essential skillset used in all subsequent weeks.

---

## Topics Covered

| Part | Topic | Key Concepts |
|------|-------|-------------|
| 1 | Python Fundamentals | Data types, control flow, data structures, exceptions, lambdas |
| 2 | NumPy | Array creation, indexing, broadcasting, matrix ops, dot product |
| 3 | Pandas | DataFrames, Series, filtering, groupby, merging, pivot tables |
| 4 | Linear Algebra | Vectors, matrix multiplication, eigenvalues, determinants, SVD |
| 5 | Statistics | Central tendency, dispersion, IQR, Z-scores, outlier detection |
| 6 | Probability & Distributions | Normal, Binomial, Poisson, PSI, distribution shift |

---

## Sections Breakdown

### Part 1 — Python Fundamentals
- `classify_number(n)` — control flow with conditional returns
- Word frequency counter using loops and dicts (no `Counter`)
- `safe_divide(a, b)` — exception handling with `ZeroDivisionError` and `TypeError`
- Higher-order functions and lambda composition (`apply_twice`)

### Part 2 — NumPy
- 1D → 2D → 3D array reshaping
- Boolean indexing and advanced slicing
- Element-wise vs matrix multiplication (`*` vs `@`)
- Broadcasting, aggregation, and linear algebra ops

### Part 3 — Pandas
- Series vs DataFrame extraction
- `.loc` / `.iloc` selection
- GroupBy aggregation, merging DataFrames, pivot tables
- Handling missing values — detection, imputation, and dropping

### Part 4 — Linear Algebra
- Vector addition and scalar multiplication
- Matrix dot product and transpose
- Eigenvalue decomposition
- Determinant and matrix inverse
- Singular Value Decomposition (SVD) — visualization of principal directions

### Part 5 — Statistics
- Mean, median, mode, variance, standard deviation
- IQR-based outlier detection and box plots
- Z-score normalization
- Salary distribution analysis with histogram + KDE overlay (bimodal gap visualization)
- Population Stability Index (PSI) — measuring distribution drift

### Part 6 — Probability & Distributions
- Normal Distribution — PDF curve
- Binomial Distribution — bar chart
- Poisson Distribution — bar chart
- Area chart comparing training vs production distribution drift
- Central Limit Theorem — histogram + normal overlay demonstrating CLT convergence

---

## Key Visualizations

- **Section 4.1** — Eigenvector arrow: single diagonal upward-sloping vector from origin
- **Section 4.3** — Two eigenvector arrows from the same origin pointing in different directions
- **Section 5.1** — Salary histogram with KDE overlay showing a clear bimodal gap between low-salary and high-salary clusters
- **Section 5.5** — Area chart comparing training vs production feature distributions (PSI drift visualization)
- **Section 6** — Normal (line), Binomial (bar), Poisson (bar) distribution plots
- **Section 6.4** — Central Limit Theorem histogram with fitted normal curve overlay

---

## Libraries Used

```python
numpy, pandas, matplotlib, seaborn, scipy.stats, sklearn.preprocessing
```

---

## How to Run

1. Open `week1_saksham_saraogi_jecrc.ipynb` in Google Colab or Jupyter
2. Run all cells top to bottom (`Runtime → Run all`)
3. No external dataset required — all data is generated inline

---

*Week 1 · Celebal Excellence Internship · Data Science · 2025*
