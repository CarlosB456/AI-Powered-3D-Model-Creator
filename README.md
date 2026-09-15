<div align="center">

# AI-powered 3D model creator compatible with (RX 6600, among others) – Open source

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DirectML](https://img.shields.io/badge/DirectML-AMD_RDNA_Accelerated-ED1C24.svg?logo=amd&logoColor=white)](https://github.com/microsoft/DirectML)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-RX_6600_%7C_RTX_%7C_Arc_%7C_CPU-8A2BE2.svg)](#verified-hardware-compatibility)
[![Gradio](https://img.shields.io/badge/UI-Bilingual_Gradio_Studio-FF7C00.svg?logo=gradio&logoColor=white)](https://gradio.app/)

**Generate production-ready 3D meshes (.glb & .obj) from a single 2D image locally on your computer.**  
Engineered specifically to solve the 8GB VRAM bottleneck, bringing state-of-the-art multi-billion parameter 3D models to consumer gaming GPUs like the **AMD Radeon RX 6600** as well as NVIDIA GeForce, Intel Arc, and CPU.

**Author & Lead Developer:** **Carlos B (eLdarqO)**

[Español (README_ES.md)](README_ES.md) | [Engineering Log: Challenges & Solutions](docs/CHALLENGES_AND_SOLUTIONS.md) | [How It Works](docs/HOW_IT_WORKS.md) | [Hardware Benchmarks](docs/HARDWARE_SPECS.md) | [Architecture](docs/ARCHITECTURE.md)

</div>

---

## Visual Interface Preview

Below is a live screenshot of the web studio running on an **AMD Radeon RX 6600 (8GB)**, showcasing automated background removal and 3D reconstruction into an interactive 3D viewport:

![AI-Powered 3D Model Creator Web Studio](docs/images/preview.png)

---

## Why This Project Exists

State-of-the-art foundation models for Image-to-3D generation (like Tencent's **Hunyuan3D-2 Turbo**) combine vision foundation models (DINOv2 with 1.14B parameters), Flow-Matching Diffusion Transformers (560M parameters), and spatial cross-attention decoders (328M parameters).

Running these models out-of-the-box requires over **14 GB of VRAM**. On consumer 8 GB GPUs (such as the AMD Radeon RX 6600, RX 7600, or RTX 3060/4060 8GB), standard pipelines crash immediately with out-of-memory errors, while CPU fallback takes an excruciating **11 to 15 minutes** per model.

This project introduces **Phased Memory Lifecycle Swapping**, **Chunked Self-Attention (DINOv2)**, **Zero-Copy PCIe Euler Integration**, and **Latent KV-Cached Volume Decoding**, slashing generation time on an 8GB AMD RX 6600 down to **~2.0 – 2.5 minutes** with a peak VRAM footprint of just **~3.2 GB**.

---

## Features

- **Multi-Model Engine (`app/models/`):**
  - **Hunyuan3D-2 Turbo**: High-fidelity geometric foundation model with intricate surface detail (~2.0 - 2.5 min on RX 6600).
  - **TripoSR**: Instant feed-forward transformer generation (~12 seconds) for rapid prototyping.
  - **Pluggable Architecture**: Modular `Base3DModel` contract for integrating future generative backends.
- **AMD RDNA & DirectML Native Acceleration:**
  - Full DirectML FP16 pipeline that runs natively on Windows without requiring Linux, WSL2, or experimental ROCm setups.
- **Automated Background Removal:**
  - Built-in `rembg` (U2-Net) foreground extraction with automatic bounding-box centering and canvas padding.
- **Polygon Decimation & Optimization (PyMeshLab):**
  - Quadric Edge Collapse Decimation allows setting exact target face budgets (e.g. 10k, 25k, 50k faces) for direct export to Roblox Studio, Blender, Unity, or Unreal Engine.
- **Modern Studio Web UI (`app/web_ui.py`):**
  - Unified dark studio aesthetic.
  - **Interactive 3D Viewport (`gr.Model3D`)**: Rotate in 360 degrees, zoom, and inspect wireframes directly in the browser.
  - **Dynamic Language Switcher**: Switch between **Español** and **English** in real-time.
  - One-click downloads for `.glb` and `.obj`.

---

## Understanding Quality Presets & Generation Times

Generation time is determined by the **Octree Resolution** used in spatial volume decoding:

| Preset | Octree Resolution | 3D Spatial Points Sampled | Time on RX 6600 | Recommended Use Case |
|---|---|---|---|---|
| **Ultra** | 128 (4 steps) | 2,146,689 (~2.1M) | **~2.0 minutes** | Recommended for Roblox, games, and quick prototypes. |
| **Rapida / Fast** | 128 (5 steps) | 2,146,689 (~2.1M) | **~2.5 minutes** | Optimal balance of fidelity and speed. |
| **Media / Medium** | 192 (5 steps) | 7,189,057 (~7.1M) | **~4.5 minutes** | High surface curvature definition. |
| **Alta / High** | 256 (5 steps) | 16,974,593 (~17.0M) | **~8.0 minutes** | Extreme geometric density (150k+ faces) for detailed 3D printing and offline renders. |

### The Cubic Scaling Law
Because 3D space scales cubically ($(N)^3$):
$$\frac{(256)^3}{(128)^3} = 2^3 = 8\times \text{ more 3D spatial points}$$
Selecting `alta` computes 8 times more spatial query points than `rapida`, which is why generation takes ~8 minutes and produces over 149,000 polygon faces.

---

## Verified Hardware Compatibility

| GPU / Platform | Architecture | Acceleration Backend | Typical Generation Time | Status |
|---|---|---|---|---|
| **AMD Radeon RX 6600 (8GB)** | RDNA 2 (Navi 23) | DirectML FP16 | **~2.0 – 2.5 min** | Primary Test Bench |
| **AMD Radeon RX 6700 / 6800 / 6900** | RDNA 2 | DirectML FP16 | **~1.5 – 2.0 min** | Verified |
| **AMD Radeon RX 7600 / 7700 / 7800 / 7900** | RDNA 3 | DirectML FP16 | **~1.0 – 1.8 min** | Verified |
| **NVIDIA GeForce RTX 3060 / 4060 (8GB)** | Ampere / Ada | CUDA / DirectML | **~1.2 – 1.8 min** | Compatible |
| **NVIDIA GeForce RTX 3080 / 4080 / 4090** | Ampere / Ada | CUDA FP16 | **~45s – 1.2 min** | Compatible |
| **Intel Arc A750 / A770 (8GB/16GB)** | Alchemist | DirectML FP16 | **~2.0 – 3.0 min** | Compatible |
| **CPU Only (x86_64 / Apple Silicon)** | CPU Core | PyTorch CPU FP32 | ~11 – 15 min | Fallback Mode |

---

## Benchmark Comparison (AMD Radeon RX 6600 8GB)

| Stage | Unoptimized CPU Baseline | First DirectML Port | This Project (MAX GPU) | Total Acceleration |
|---|---|---|---|---|
| **DINOv2 Encoder (1.1B)** | 150.3s (CPU) | 150.3s (CPU Fallback) | **~30.2s (GPU Chunked)** | **5.0x faster** |
| **DiT Diffusion (4-5 steps)** | ~250.0s (CPU) | 137.3s (PCIe Sync) | **~89.8s – 112.5s (Zero-Copy GPU)**| **2.5x faster** |
| **VAE Volume Decoding** | 280.3s (CPU) | 44.3s (10k Chunks) | **~25.1s (KV-Cache 8k Chunks)** | **11.2x faster** |
| **Marching Cubes** | ~10.0s (CPU) | Argument Error | **~4.9s (Fixed)** | **2.0x faster** |
| **TOTAL RUNTIME** | **~11.5 minutes** | **~5.5 minutes** | **~2.0 – 2.5 minutes** | **~5x faster overall** |

---

## Quickstart (Windows)

### Option 1: Automatic Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/CarlosB456/AI-Powered-3D-Model-Creator.git
   cd AI-Powered-3D-Model-Creator
   ```
2. Double-click **`INSTALL.bat`** to automatically set up dependencies.
3. Double-click **`START_WEB_UI.bat`** to launch the studio in your browser (`http://localhost:7860`).

### Option 2: Drag-and-Drop Generation
Drag any `.png` or `.jpg` image directly onto **`CLI_GENERATE.bat`**. The 3D model will be exported automatically to the `output/` folder.

### Option 3: Command Line Interface (CLI)
```bash
# High-quality generation with Hunyuan3D-2 Turbo (~2 min, 35k target faces, auto-rembg)
python app/cli.py my_image.png --model "Hunyuan3D-2 Turbo" --calidad ultra --rembg --decimate 35000

# Ultra-fast generation with TripoSR (~12 sec)
python app/cli.py my_image.png --model "TripoSR" --rembg
```

---

## Manual Installation

### 1. Requirements
* Windows 10 / 11 or Linux 64-bit
* Python 3.10 or 3.11 (Python 3.11 recommended)
* Git installed

### 2. Install PyTorch with DirectML (For AMD & Intel GPUs)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
```
*(For NVIDIA users: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`)*

### 3. Install Project Dependencies
```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
AI-Powered-3D-Model-Creator/
├── app/
│   ├── web_ui.py                 # Modern Studio Web Interface (Bilingual ES/EN, 3D Viewport)
│   ├── cli.py                    # Unified command-line interface
│   ├── models/
│   │   ├── base.py               # Abstract Base3DModel definition
│   │   ├── hunyuan3d_model.py    # Hunyuan3D-2 Turbo adapter (DirectML FP16, chunked attention, KV-cache)
│   │   └── triposr_model.py      # TripoSR fast adapter (~12s generation)
│   └── processors/
│       ├── background.py         # rembg automated background removal
│       └── mesh_optimizer.py     # PyMeshLab polygon decimation & normalization
├── docs/
│   ├── images/
│   │   └── preview.png           # User studio interface screenshot
│   ├── CHALLENGES_AND_SOLUTIONS.md # Deep dive into all 10 problems solved for RX 6600
│   ├── HOW_IT_WORKS.md           # Mathematical and architectural pipeline guide
│   ├── HARDWARE_SPECS.md         # Hardware benchmark specifications and VRAM profiles
│   └── ARCHITECTURE.md           # System diagrams & lifecycle flow
├── Hunyuan3D-2/                  # Core Hunyuan3D-2 engine with DirectML patches
├── TripoSR/                      # Core TripoSR engine
├── output/                       # Generated .glb and .obj models
├── START_WEB_UI.bat              # 1-click Web Studio launcher
├── CLI_GENERATE.bat              # Drag-and-drop CLI launcher
├── INSTALL.bat                   # Automated setup script
├── requirements.txt              # Pinned dependencies
├── LICENSE                       # Apache 2.0 Open-Source License
├── README.md                     # English documentation
└── README_ES.md                  # Spanish documentation
```

---

## Author & Credits

* **Creator & Lead Developer:** **Carlos B (eLdarqO)**

Special thanks to the researchers and teams behind:
- **Tencent Hunyuan Team** for [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)
- **Stability AI & Tripo AI** for [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- **Microsoft DirectML Team** for [torch-directml](https://github.com/microsoft/DirectML)
- **Daniel Gatis** for [rembg](https://github.com/danielgatis/rembg)
- **Visual Computing Lab** for [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab)

---

## License

Distributed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.
