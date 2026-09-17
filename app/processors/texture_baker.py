"""
AI-Powered 3D Model Creator - Texture Baker Module
Generates and bakes high-resolution PBR UV textures from multi-angle or single-view images
using a high-performance Numba-accelerated CPU rasterizer.
Zero CUDA, nvcc, or MSVC requirements. 100% compatible with AMD Radeon RX 6600 and consumer hardware.
Created by Carlos B (eLdarqO)
"""
import os
import sys
import time
import torch
import trimesh
import numpy as np
from PIL import Image
from typing import Union, Dict, Optional, Callable

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
h3d_path = os.path.join(root_dir, "Hunyuan3D-2")
texgen_rasterizer_path = os.path.join(h3d_path, "hy3dgen", "texgen", "custom_rasterizer")

for p in [root_dir, h3d_path, texgen_rasterizer_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
    _TEXTURE_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Texture engine import issue: {e}")
    _TEXTURE_ENGINE_AVAILABLE = False


def bake_textures_onto_mesh(
    mesh: trimesh.Trimesh,
    views: Union[Image.Image, Dict[str, Optional[Image.Image]]],
    texture_resolution: int = 1024,
    progress_callback: Optional[Callable] = None
) -> trimesh.Trimesh:
    """
    Bakes multi-view or single-view images onto the 3D mesh via camera back-projection,
    generating a UV-unwrapped mesh with an embedded PBR diffuse texture.
    """
    if not _TEXTURE_ENGINE_AVAILABLE:
        print("[TEXTURE] Motor de textura no disponible, manteniendo malla sin textura.")
        return mesh

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh

    t_start = time.time()
    print(f"[TEXTURE] Iniciando baking de textura PBR ({texture_resolution}x{texture_resolution})...", flush=True)

    try:
        if progress_callback:
            progress_callback(0.85, "Desplegando coordenadas UV (xatlas)...")

        # 1. UV Unwrapping
        mesh = mesh_uv_wrap(mesh)

        if progress_callback:
            progress_callback(0.88, "Inicializando renderizador de proyeccion...")

        # 2. Setup CPU MeshRender
        render = MeshRender(
            default_resolution=texture_resolution,
            texture_size=texture_resolution,
            device='cpu'
        )
        render.load_mesh(mesh)

        # 3. Formulate projection views
        # Supported viewpoints with orthogonal orientations:
        # front (0 deg), left (90 deg), back (180 deg), right (270 deg)
        view_tuples = []
        if isinstance(views, dict):
            if views.get('front') is not None:
                view_tuples.append((views['front'], 0, 0, 1.0))
            if views.get('left') is not None:
                view_tuples.append((views['left'], 0, 90, 0.85))
            if views.get('back') is not None:
                view_tuples.append((views['back'], 0, 180, 1.0))
            if views.get('right') is not None:
                view_tuples.append((views['right'], 0, 270, 0.85))
        elif isinstance(views, Image.Image):
            view_tuples.append((views, 0, 0, 1.0))

        if not view_tuples:
            print("[TEXTURE] No se recibieron imagenes validas para proyectar textura.")
            return mesh

        if progress_callback:
            progress_callback(0.90, f"Proyectando {len(view_tuples)} angulos sobre la malla...")

        # 4. Multi-angle Back-projection
        project_textures = []
        project_weighted_cos_maps = []

        for img, elev, azim, weight in view_tuples:
            img_rgb = img.convert('RGB')
            img_resized = img_rgb.resize((texture_resolution, texture_resolution), Image.Resampling.LANCZOS)
            proj_tex, proj_cos, _ = render.back_project(img_resized, elev, azim)
            proj_cos = weight * (proj_cos ** 4)
            project_textures.append(proj_tex)
            project_weighted_cos_maps.append(proj_cos)

        if progress_callback:
            progress_callback(0.93, "Fusionando texturas y calculando oclusion...")

        # 5. Fast texture blending
        texture, trust_mask = render.fast_bake_texture(project_textures, project_weighted_cos_maps)

        if progress_callback:
            progress_callback(0.96, "Pintando oclusiones y bordes UV (inpaint)...")

        # 6. Inpainting of unprojected seams/crevices
        mask_np = (trust_mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_inp = render.uv_inpaint(texture, mask_np)
        texture_inp_t = torch.tensor(texture_inp / 255.0).float().to(texture.device)

        # 7. Apply texture to mesh
        render.set_texture(texture_inp_t)
        textured_mesh = render.save_mesh()

        t_elapsed = time.time() - t_start
        print(f"[TEXTURE] Textura PBR horneada exitosamente en {t_elapsed:.2f}s!", flush=True)
        return textured_mesh

    except Exception as e:
        print(f"[WARN] Error durante el proceso de texturizado: {e}. Manteniendo geometría base.", flush=True)
        return mesh
