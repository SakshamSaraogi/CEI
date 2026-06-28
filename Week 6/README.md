# MNIST Denoising Autoencoder

![](https://badgen.net/badge/Code/Python/blue?icon=https://simpleicons.org/icons/python.svg&labelColor=cyan&label) ![](https://badgen.net/badge/Library/PyTorch/blue?icon=https://simpleicons.org/icons/pytorch.svg&labelColor=cyan&label) ![](https://badgen.net/badge/Tools/NumPy/blue?icon=https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg&labelColor=cyan&label) ![](https://badgen.net/badge/Tools/Matplotlib/blue?icon=https://upload.wikimedia.org/wikipedia/en/5/56/Matplotlib_logo.svg&labelColor=cyan&label)

A Convolutional Denoising Autoencoder (DAE) built with PyTorch that learns to reconstruct clean MNIST digits from heavily corrupted (Gaussian noise, σ = 0.4) inputs — achieving a **+5.99 dB PSNR improvement** and **Test MSE of 0.020**.

---

## What is a Denoising Autoencoder?

A standard autoencoder compresses an input into a compact latent representation and then reconstructs it. A **Denoising Autoencoder** extends this by training the model to map a *noisy* version of the input back to the *clean* original. This forces the encoder to learn a robust, noise-invariant representation of the underlying structure — it can't just memorize pixel values, it has to understand what a digit actually looks like.

```
Noisy Image ──► Encoder ──► Bottleneck ──► Decoder ──► Clean Reconstruction
```

---

## Dataset — MNIST

The [MNIST database](http://yann.lecun.com/exdb/mnist/) contains 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels.

- **60,000** training samples
- **10,000** test samples
- Pixel values normalized to **[0, 1]** via `torchvision.transforms.ToTensor()`

**Noise Model:** Additive White Gaussian Noise (AWGN) with σ = 0.4, clamped to [0, 1]:

```python
noisy = torch.clamp(clean + 0.4 * torch.randn_like(clean), 0.0, 1.0)
```

---

## Model Architecture

A lightweight **3-block Convolutional Autoencoder** with ~9K parameters.

### Encoder
| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| enc1 | Conv2d(1→32, 3×3, pad=1) → ReLU → MaxPool(2×2) | 32 × 14 × 14 |
| enc2 | Conv2d(32→16, 3×3, pad=1) → ReLU → MaxPool(2×2) | 16 × 7 × 7 |
| enc3 | Conv2d(16→8, 3×3, pad=1) → ReLU | **8 × 7 × 7 (bottleneck)** |

### Decoder
| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| dec1 | ConvTranspose2d(8→16, 2×2, stride=2) → ReLU | 16 × 14 × 14 |
| dec2 | ConvTranspose2d(16→32, 2×2, stride=2) → ReLU | 32 × 28 × 28 |
| dec3 | Conv2d(32→1, 3×3, pad=1) → **Sigmoid** | **1 × 28 × 28** |

**Design choices:**
- MaxPool compression (28×28 → 7×7) forces the model to learn a compact, noise-invariant representation. Random noise can't be efficiently encoded into the bottleneck — it gets dropped.
- Sigmoid on the output layer keeps reconstructed pixel values in [0, 1], matching the normalized input range.
- ConvTranspose2d (not upsampling+conv) for symmetric, clean upscaling with learned weights.

---

## Training Strategy — Two-Phase Approach

Training directly on noisy inputs caused the model to plateau at MSE ≈ 0.111 for many epochs (outputting near-mean blurry images). A two-phase strategy solved this:

### Phase 1 — Clean Reconstruction Warmup
Train the autoencoder with **clean input → clean target**. This bootstraps the encoder/decoder weights to represent digit structure before any noise is introduced.

```python
output = model(clean_imgs)
loss   = criterion(output, clean_imgs)   # clean → clean
```

### Phase 2 — Denoising Fine-Tune
Switch to the DAE objective — **noisy input → clean target**:

```python
noisy  = add_gaussian_noise(clean_imgs, noise_factor=0.4)
output = model(noisy)
loss   = criterion(output, clean_imgs)   # noisy → clean
```

The model converged cleanly in just 6 denoising epochs after the warmup.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Noise factor (σ) | 0.4 |
| Batch size | 256 |
| Optimizer | Adam |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 5e-4 |
| Loss function | MSELoss |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |

### Why MSELoss?
We're comparing pixel values in input and output images — this is a **regression task**, not classification. MSE directly penalizes pixel-level reconstruction error and is differentiable everywhere, making it ideal for training autoencoders on normalized [0, 1] imagery.

```python
loss = criterion(output, clean_imgs)
```

---

## Results

| Metric | Value |
|--------|-------|
| Test MSE | **0.02003** |
| PSNR — noisy vs. clean | 10.99 dB |
| PSNR — denoised vs. clean | 16.98 dB |
| **PSNR Improvement** | **+5.99 dB** |

### Loss Curves
![Loss Curves](assets/loss_curves.png)

Train and validation loss converge together with no overfitting, confirming the model generalizes well.

### Clean | Noisy | Denoised
![Comparison Grid](assets/comparison_grid.png)

Digit structure that is visually unreadable in the noisy images (row 2) is clearly recovered in the denoised output (row 3) — strokes, loops, and endpoints are all preserved.

### Noise Robustness
![Noise Robustness](assets/noise_robustness.png)

The model was trained at σ = 0.4. Performance across different noise levels:
- **σ ≤ 0.3** — near-perfect reconstruction
- **σ = 0.4** — trained level, PSNR ~17 dB
- **σ ≥ 0.6** — model reverts to a blurry mean estimate (expected behaviour for OOD noise)

### Residual Error Maps
![Residual Map](assets/residual_map.png)

Residual heatmaps (|Clean − Denoised|) show the largest errors at stroke edges and in structurally complex digits (e.g., 8, 3). Background pixels are reconstructed almost perfectly, confirming the model learned to distinguish signal from noise.

---

## Repository Structure

```
├── denoising_autoencoder_MNIST.ipynb   # Full runnable notebook
├── best_dae.pth                        # Saved model weights (best checkpoint)
├── assets/
│   ├── comparison_grid.png
│   ├── loss_curves.png
│   ├── noise_robustness.png
│   └── residual_map.png
├── data/
│   └── MNIST/                          # Auto-downloaded by torchvision
└── README.md
```

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib
```

### Run the Notebook
```bash
jupyter notebook denoising_autoencoder_MNIST.ipynb
```

MNIST will be downloaded automatically to `data/` on first run via `torchvision.datasets.MNIST(download=True)`.

### Load the Pretrained Model
```python
import torch
from model import DenoisingAutoencoder   # or copy the class from the notebook

model = DenoisingAutoencoder()
model.load_state_dict(torch.load('best_dae.pth', map_location='cpu'))
model.eval()
```

---

## Key Observations

1. **Two-phase training was essential.** Without the warmup, the model output near-mean blurry images for many epochs. Warming up on clean reconstruction first solved convergence.

2. **The bottleneck is the denoising mechanism.** Compressing 28×28 → 7×7 into 8 channels (392 values) forces the model to discard information it can't compress efficiently — random noise has no structure, so it's naturally filtered out.

3. **PSNR is a better metric than MSE alone** for evaluating denoising. A +5.99 dB gain indicates the denoised output is ~4× closer to the clean image in terms of mean squared error.

4. **Graceful degradation at high noise.** The model handles moderate out-of-distribution noise well, but blurs at very high σ — a known characteristic of MSE-trained autoencoders.

---

## Potential Improvements

| Idea | Expected Benefit |
|------|-----------------|
| BatchNorm in all blocks | More stable training, faster convergence |
| Skip connections (U-Net style) | Better detail preservation, sharper strokes |
| SSIM or Perceptual loss | Visually sharper output |
| Noise level annealing during training | Better generalization across σ values |
| Train on full 60k dataset | Lower test MSE, better generalization |
| Deeper architecture (more channels) | Higher capacity for complex reconstructions |

---

## References

- [Denoising Autoencoders — Pascal Vincent et al. (2008)](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
- [MNIST Database — Yann LeCun](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Checkerboard Artifacts in ConvTranspose — Odena et al. (Distill)](https://distill.pub/2016/deconv-checkerboard/)
- Reference repo: [NvsYashwanth/MNIST-Autoencoder](https://github.com/NvsYashwanth/MNIST-Autoecncoder)
