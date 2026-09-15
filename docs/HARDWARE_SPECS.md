# Hardware Benchmarks & Test Environment

This document details the exact hardware test bench used to validate and benchmark **AI-Powered 3D Model Creator**, along with memory profiles and inference measurements.

---

##  Test Bench Specifications

| Component | Specification |
|---|---|
| **GPU** | **AMD Radeon RX 6600** (8GB GDDR6 VRAM) |
| **GPU Architecture** | AMD RDNA 2 (Navi 23 / gfx1032) |
| **Interface / Bus** | PCIe 4.0 x8 |
| **Backend Framework** | PyTorch 2.4.1 (CPU core) + `torch-directml 0.2.5.dev240914` |
| **Operating System** | Windows 11 / Windows 10 (64-bit) |
| **Host Python** | Python 3.11 (64-bit) |
| **Storage** | NVMe M.2 SSD |

---

##  VRAM Memory Footprint Breakdown

The AMD Radeon RX 6600 has an 8,192 MB (8 GB) physical VRAM limit. Standard pipelines attempt to load the entire graph onto GPU, causing immediate Out-of-Memory (OOM) errors. AI-Powered 3D Model Creator uses sequential lifecycle management:

```text
Phase 1: DINOv2 Vision Encoder
┌───────────────────────────────────────────────┐
│ GPU VRAM: ~2.4 GB (FP16 Weights + Chunked Act) │
└───────────────────────────────────────────────┘
    ↓ (VRAM fully freed to CPU host memory)
Phase 2: DiT Flow Matching Denoiser
┌───────────────────────────────────────────────┐
│ GPU VRAM: ~1.2 GB (FP16 Model) + ~0.8 GB Act  │
└───────────────────────────────────────────────┘
    ↓ (VRAM fully freed to CPU host memory)
Phase 3: VAE Volume Decoder (geo_decoder)
┌───────────────────────────────────────────────┐
│ GPU VRAM: ~0.4 GB (Weights) + ~1.6 GB KV-Cache │
└───────────────────────────────────────────────┘
    ↓ (Marching Cubes executed on CPU)
Phase 4: Post-Processing & Export
┌───────────────────────────────────────────────┐
│ GPU VRAM: 0 MB (idle)                         │
└───────────────────────────────────────────────┘
```

**Peak VRAM utilization:** Never exceeds **~3.2 GB**, comfortably leaving headroom on an 8 GB graphics card.

---

##  Benchmark Comparison Matrix

Tested using the standardized Roblox character avatar (512x512 PNG, centered):

| Pipeline Stage | Unoptimized CPU Baseline | First DirectML Port | AI-3D-Creator (MAX GPU) | Speedup Factor |
|---|---|---|---|---|
| **DINOv2 Encoder (1.1B)** | 150.3s | 150.3s (CPU Fallback) | **~30.2s** (GPU Chunked) | **5.0x faster** |
| **DiT Diffusion (5 steps)** | ~250.0s | 137.3s (PCIe sync) | **~112.5s** (Pure GPU Euler) | **2.2x faster** |
| **DiT Diffusion (4 steps, Ultra)** | — | — | **~89.8s** (Pure GPU Euler) | **2.8x faster** |
| **VAE Volume Decoding** | 280.3s | 44.3s (10k Chunks) | **~25.1s** (32k Chunks + KV) | **11.2x faster** |
| **Marching Cubes (Skimage)** | ~10.0s | Broken (Missing Args) | **~4.9s** (Fixed) | 2.0x faster |
| **TOTAL RUNTIME** | **~11.5 minutes** | **~5.5 minutes** | **~2.0 – 2.5 minutes** | **~5x faster overall** |

### TripoSR Model Benchmark:
- **Inference Time:** ~3.5 seconds
- **Marching Cubes (res=192):** ~7.8 seconds
- **Total Generation Time:** **~12 – 15 seconds**
