# AI-Powered 3D Model Creator Architecture

AI-Powered 3D Model Creator is structured as a modular, hardware-agnostic 3D generation framework designed to accommodate multiple AI model engines while providing unified pre-processing, post-processing, and interactive presentation.

---

## System Architecture Diagram

```mermaid
flowchart TD
    subgraph Input["Input Pipeline"]
        IMG["Input Image(s): Single View or Multi-Angle (0°, 90°, 180°, 270°)"]
        REMBG["Automated Background Remover (rembg)"]
        PREP["Foreground Centering & Aspect Normalization"]
        IMG --> REMBG --> PREP
    end

    subgraph Core["Model Engine Dispatcher (Base3DModel)"]
        PREP --> ROUTER{"Model Selector"}
        ROUTER -->|"Hunyuan3D-2 Multi-View Turbo"| MV3D["Hunyuan3D Multi-View Adapter"]
        ROUTER -->|"Hunyuan3D-2 Turbo"| H3D["Hunyuan3D Adapter"]
        ROUTER -->|"TripoSR"| T3D["TripoSR Adapter"]
        ROUTER -.->|"Future Model"| F3D["Pluggable Engine"]
    end

    subgraph MV_Pipeline["Hunyuan3D Multi-View Pipeline"]
        MV3D --> MVDINO["DinoImageEncoderMV (1.1B)<br/>Sinusoidal Camera Angle Embeddings (GPU)"]
        MVDINO --> MVDIT["MV DiT Flow-Matching Denoiser (560M)<br/>DirectML Zero-Copy Integration (GPU)"]
        MVDIT --> MVVAE["VAE geo_decoder (328M)<br/>KV-Cached 8k Chunks (GPU)"]
        MVVAE --> MVMC["Marching Cubes Extraction (CPU)"]
    end

    subgraph H3D_Pipeline["Hunyuan3D-2 Turbo Execution Flow"]
        H3D --> DINO["DINOv2 (1.1B)<br/>Chunked Attention FP16 (GPU)"]
        DINO --> DIT["DiT Denoiser (560M)<br/>Zero-Copy Euler Flow Matching (GPU)"]
        DIT --> VAE["VAE geo_decoder (328M)<br/>KV-Cached 8k Chunks (GPU)"]
        VAE --> MC["Marching Cubes<br/>Dual / Skimage (CPU)"]
    end

    subgraph Post["Post-Processing & Optimization"]
        MC --> NORM[" Centering & Scale Normalization"]
        T3D --> NORM
        NORM --> DEC{"Decimate Faces?"}
        DEC -->|"Yes"| PYM["PyMeshLab Quadric Edge Collapse"]
        DEC -->|"No"| EXP["Export Pipeline"]
        PYM --> EXP
        EXP --> GLB[" Model.glb"]
        EXP --> OBJ[" Model.obj"]
    end

    subgraph UI["User Interface Layer"]
        EXP --> WEB[" Gradio Web UI (Bilingual)"]
        EXP --> CLI[" Unified CLI Interface"]
        EXP --> V3D[" Interactive 3D Viewport (gr.Model3D)"]
    end

    style DINO fill:#2e7d32,color:#fff
    style DIT fill:#2e7d32,color:#fff
    style VAE fill:#2e7d32,color:#fff
    style PYM fill:#1565c0,color:#fff
    style V3D fill:#6a1b9a,color:#fff
```

---

## Key Architecture Principles

### 1. Unified Model Abstraction (`app/models/base.py`)
All 3D generative backends implement the `Base3DModel` contract:
- `load(device)`: Initializes weight checkpoints and applies necessary kernel patches.
- `generate(image, quality, progress_callback, **kwargs) -> trimesh.Trimesh`: Converts 2D pixel matrices into a watertight 3D triangle mesh.
- `unload()`: Releases host and device buffers cleanly to allow sequential multi-model evaluation without memory accumulation.

### 2. Isolated Pre/Post Processors (`app/processors/`)
- **`background.py`**: Separates foreground silhouettes from cluttered imagery using deep boundary extraction (`u2net` via `rembg`), standardizing inputs across all models.
- **`mesh_optimizer.py`**: Interacts with `PyMeshLab` to execute topologically-aware quadric surface reduction, allowing artists to export game-ready low-poly assets or uncompressed high-poly source meshes.

### 3. Progressive Device Memory Swapping
To execute on budget gaming GPUs (8GB VRAM), memory is scheduled in discrete phases:
1. Load Encoder $\rightarrow$ Execute $\rightarrow$ Offload to Host RAM.
2. Load Denoiser $\rightarrow$ Execute $\rightarrow$ Offload to Host RAM.
3. Load Decoder $\rightarrow$ Execute $\rightarrow$ Offload to Host RAM.
This prevents concurrent residency of incompatible weight tensors while keeping per-generation latency around 2 minutes.
