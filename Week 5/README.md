# Week 5 — Deep Learning: Text Generation

**Notebook:** `week5_saksham_saraogi_jecrc.ipynb`  
**Intern:** Saksham Saraogi · JECRC University · Celebal Technologies CEI

---

## Overview

Designs and trains three recurrent neural network architectures — Vanilla RNN, LSTM, and GRU — to learn the structure and contextual dependencies of a custom text corpus. The models are trained for next-word prediction and evaluated on their ability to generate coherent text sequences from a seed phrase.

---

## Problem Statement

> Design and implement a DL model capable of learning the underlying structure, grammar, and contextual dependencies of a given text corpus to generate coherent and meaningful text sequences using Vanilla RNN, LSTM, and GRU.

---

## Model Architectures

| Model | Key Characteristic | Handles Long-Range Dependencies |
|-------|--------------------|--------------------------------|
| Vanilla RNN | Basic sequential recurrence | ❌ Vanishing gradient problem |
| LSTM | Input, forget, output gates | ✅ Long-term memory cells |
| GRU | Update and reset gates (lighter than LSTM) | ✅ More efficient than LSTM |

All three models share identical configuration for fair comparison:
- **Embedding dim:** 64 (upscaled from 32 — Student Task 2)
- **Hidden units:** 128 (widened from 64 — Student Task 4)
- **Output layer:** `Dense(total_words, activation='softmax')`
- **Optimizer:** Adam with sparse categorical cross-entropy loss

---

## Sections Covered

| Step | Topic |
|------|-------|
| 1 | Imports and environment setup |
| 2 | Custom text corpus — AI/Data Science paragraph (Student Task 1) |
| 3 | Tokenization, n-gram sequence creation, `pad_sequences` for uniform length |
| 4 | Build three models — SimpleRNN, LSTM, GRU with upscaled dimensions |
| 5 | Train all three models for 100 epochs |
| 6 | Training loss comparison plot — cross-entropy loss across 100 epochs |
| 7 | Text generation function using `np.argmax` over softmax probability arrays |
| 8 | Generate 10-word samples from seed phrase `"deep learning"` (all three models) |
| 9 | Extended training to 200 epochs (Student Task 3) + regenerate text |
| Conclusion | Summary table of all student tasks |

---

## Student Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1 | Replace boilerplate corpus with custom paragraph | ✅ Custom AI/DL paragraph corpus |
| 2 | Upscale embedding dimensions | ✅ 32 → 64 |
| 3 | Expand training to 200 epochs | ✅ 100 + 100 continued training |
| 4 | Widen hidden layers | ✅ 64 → 128 units |
| 5 | Generate 10 words per prompt | ✅ Applied across all models |

---

## Text Generation

The generation function uses `np.argmax` to select the word with the highest predicted probability at each step from the model's softmax output. The selected index is mapped back to a word token and appended to the sequence, which rolls forward as the new input for the next prediction step.

**Seed phrase:** `"deep learning"`  
**Words generated per call:** 10  
**Models compared:** SimpleRNN · LSTM · GRU (at both 100 and 200 epochs)

---

## Key Observations

- LSTM and GRU produce more coherent sequences than Vanilla RNN due to gated memory mechanisms
- Extended training (200 epochs) improves word coherence and reduces repetition
- GRU achieves comparable output quality to LSTM with fewer parameters, making it more efficient for small corpora
- Vanilla RNN suffers from vanishing gradients and tends to repeat or lose context over longer generation sequences

---

## Libraries Used

```python
tensorflow, keras
numpy
```

---

## How to Run

1. Open `week5_saksham_saraogi_jecrc.ipynb` in Google Colab
2. No external dataset needed — corpus is defined inline
3. GPU runtime recommended for faster training (`Runtime → Change runtime type → T4 GPU`)
4. Run all cells top to bottom

---

*Week 5 · Celebal Excellence Internship · Data Science · 2026*
