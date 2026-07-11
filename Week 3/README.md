# Week 3 — Unsupervised Learning

**Notebook:** `week3_saksham_saraogi_jecrc.ipynb`  
**Dataset:** `Country-data.csv`  
**Intern:** Saksham Saraogi · JECRC University · Celebal Technologies CEI

---

## Overview

Applies unsupervised clustering techniques to group 167 countries by socio-economic indicators. Uses K-Means and DBSCAN to identify patterns across child mortality, income, GDP, life expectancy, and other development metrics — with PCA-based 2D visualization to interpret cluster structure.

---

## Dataset

| Property | Value |
|----------|-------|
| File | `Country-data.csv` |
| Rows | 167 countries |
| Features | 9 numeric + 1 identifier (`country`) |
| Source | HELP International — aid prioritization data |

**Features:** `child_mort`, `exports`, `health`, `imports`, `income`, `inflation`, `life_expec`, `total_fer`, `gdpp`

---

## Sections Covered

| Section | Topic |
|---------|-------|
| 1–2 | Install libraries and imports |
| 3 | Load dataset from `/content/Country-data.csv` |
| 4 | Data cleaning — strip column whitespace, drop duplicates, force numeric types, median imputation |
| 5 | Feature isolation (drop `country`) and StandardScaler normalization |
| 6 | Elbow Method — inertia loop over k ∈ [2, 10], line plot with best_k marker |
| 7 | K-Means training with `best_k = 3` |
| 8 | Silhouette Score evaluation and interpretation |
| 9 | DBSCAN comparative model (eps=1.5, min_samples=5) |
| 10 | PCA 2D projection — variance explained per component |
| 11 | Color-coded PCA scatterplot of K-Means clusters |
| 12 | Cluster mean profiles across all features |
| 13 | Sample country listings per cluster |
| 14 | Written observations — socio-economic analysis |

---

## Key Results

| Metric | Value |
|--------|-------|
| Optimal k (Elbow) | 3 |
| Silhouette Score | 0.2833 |
| PCA variance captured | ~62.7% (PC1: 46.0%, PC2: 16.7%) |
| DBSCAN clusters | 1 core cluster + noise points |

---

## Cluster Profiles

| Cluster | Label | Avg Child Mortality | Avg GDP per Capita | Avg Income | Avg Life Expectancy |
|---------|-------|--------------------|--------------------|------------|---------------------|
| 0 | High Income / Developed | ~5 | ~$42,000 | ~$45,672 | ~80 years |
| 1 | Low Income / High Mortality | ~93 | ~$1,922 | ~$2,800 | ~59 years |
| 2 | Middle Income / Developing | ~22 | ~$6,486 | ~$12,305 | ~73 years |

---

## Observations Summary (Section 14)

1. **Cluster 1** captures the most underdeveloped nations — child mortality ~18× higher than Cluster 0, predominantly Sub-Saharan Africa and South Asia
2. **Cluster 0** represents top-tier economic zones — Western Europe, North America, Australia, East Asia — with near-zero child mortality and life expectancy above 80
3. **Cluster 2** is a transitional middle-income band — Latin America, North Africa, Southeast Asia — showing improving but still constrained development
4. **DBSCAN noise points** identify economic outliers (Gulf oil states, small island economies) whose feature vectors don't fit any dense neighborhood
5. **PCA scatterplot** shows clear separation between Clusters 0 and 1 along PC1 (income/GDP axis); Cluster 2 sits centrally, confirming its intermediate status

---

## Libraries Used

```python
pandas, numpy, matplotlib, seaborn
sklearn (StandardScaler, KMeans, DBSCAN, PCA, silhouette_score)
```

---

## How to Run

1. Upload `Country-data.csv` to your Colab session (via the file sidebar)
2. Open `week3_saksham_saraogi_jecrc.ipynb` in Google Colab
3. Run all cells top to bottom (`Runtime → Run all`)

---

*Week 3 · Celebal Excellence Internship · Data Science · 2026*
