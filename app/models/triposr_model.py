"""
AI-Powered 3D Model Creator - TripoSR Model Adapter
Ultra-fast feed-forward NeRF transformer for instant 3D generation (~10-15s).
"""
import os
import sys
import time
from typing import Optional, Callable, Any
from PIL import Image
import torch
import numpy as np
import trimesh

from .base import Base3DModel

# Add TripoSR to path
tsr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "TripoSR"))
if tsr_path not in sys.path:
    sys.path.insert(0, tsr_path)

class TripoSRModel(Base3DModel):
    QUALITIES = {
        'rapida': {'mc_res': 128, 'label': 'Rápida (MC 128, ~8s)'},
        'media':  {'mc_res': 192, 'label': 'Media (MC 192, ~12s)'},
        'alta':   {'mc_res': 256, 'label': 'Alta (MC 256, ~18s)'},
    }

    def __init__(self):
        super().__init__(
            name="TripoSR",
            description="Modelo feed-forward instantáneo (Stability AI / Tripo). Generación de malla 3D completa en 10-15 segundos."
        )
        self.model = None

    def load(self, device: Optional[str] = None) -> None:
        if self.is_loaded and self.model is not None:
            return

        from tsr.system import TSR
        print("[TripoSR] Cargando modelo...", flush=True)
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        self.model.renderer.set_chunk_size(8192)
        self.model.to("cpu")
        self.is_loaded = True
        print("[TripoSR] Modelo listo en CPU/DirectML!", flush=True)

    def get_supported_qualities(self) -> list[str]:
        return list(self.QUALITIES.keys())

    def generate(
        self,
        image: Image.Image,
        quality: str = "media",
        progress_callback: Optional[Callable] = None,
        **kwargs: Any
    ) -> trimesh.Trimesh:
        if not self.is_loaded:
            self.load()

        cfg = self.QUALITIES.get(quality, self.QUALITIES['media'])
        mc_res = cfg['mc_res']

        # Preprocess image
        if progress_callback:
            progress_callback(0.2, "Preprocesando imagen...")

        img = image.convert("RGBA")
        img_np = np.array(img).astype(np.float32) / 255.0
        if img.mode == "RGBA":
            img_np = img_np[:, :, :3] * img_np[:, :, 3:4] + (1 - img_np[:, :, 3:4]) * 0.5
        processed_img = Image.fromarray((img_np * 255.0).astype(np.uint8))

        if progress_callback:
            progress_callback(0.4, "Inferencia triplane (TripoSR)...")

        with torch.no_grad():
            scene_codes = self.model([processed_img], device="cpu")

        if progress_callback:
            progress_callback(0.7, f"Extrayendo malla (Marching Cubes res={mc_res})...")

        meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_res)
        mesh = meshes[0]

        if progress_callback:
            progress_callback(1.0, "Listo!")

        return mesh

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
