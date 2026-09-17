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
import cv2
from PIL import Image
from typing import Union, Dict, Optional, Callable, Tuple

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
    output_dir: Optional[str] = None,
    base_name: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[trimesh.Trimesh, Optional[str]]:
    """
    Bakes multi-view or single-view images onto the 3D mesh via camera back-projection
    with automatic silhouette alignment and alpha-bleed masking.
    Returns (textured_mesh, texture_png_path).
    """
    if not _TEXTURE_ENGINE_AVAILABLE:
        print("[TEXTURE] Motor de textura no disponible, manteniendo malla sin textura.")
        return mesh, None

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh, None

    t_start = time.time()
    print(f"[TEXTURE] Iniciando baking de textura PBR ({texture_resolution}x{texture_resolution})...", flush=True)

    try:
        # 1. Geometry safety: pre-decimate if face count is extremely high to prevent slow UV unwrapping
        if len(mesh.faces) > 45000:
            from app.processors.mesh_optimizer import decimate_mesh
            mesh = decimate_mesh(mesh, 35000)

        if progress_callback:
            progress_callback(0.85, "Desplegando mapa UV (xatlas)...")

        # 2. UV Unwrapping
        mesh = mesh_uv_wrap(mesh)

        if progress_callback:
            progress_callback(0.88, "Inicializando renderizador de proyeccion...")

        # 3. Setup CPU MeshRender
        render = MeshRender(
            default_resolution=texture_resolution,
            texture_size=texture_resolution,
            device='cpu'
        )
        render.load_mesh(mesh)

        # 4. Formulate projection views
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
            return mesh, None

        if progress_callback:
            progress_callback(0.90, f"Alineando y proyectando {len(view_tuples)} angulos sobre la malla...")

        project_textures = []
        project_weighted_cos_maps = []

        # 5. Multi-angle Back-projection with Silhouette Alignment
        for img, elev, azim, weight in view_tuples:
            img_rgba = img.convert('RGBA')

            # Render depth from camera to find exact 3D mesh silhouette on screen
            depth = render.render_depth(elev, azim, return_type='pl')
            depth_arr = np.array(depth)
            mesh_mask = depth_arr > 0
            y_idx, x_idx = np.where(mesh_mask)
            if len(y_idx) == 0:
                continue

            mesh_ymin, mesh_ymax = y_idx.min(), y_idx.max()
            mesh_xmin, mesh_xmax = x_idx.min(), x_idx.max()
            mesh_w = max(1, mesh_xmax - mesh_xmin + 1)
            mesh_h = max(1, mesh_ymax - mesh_ymin + 1)

            # Crop subject from image based on alpha
            arr = np.array(img_rgba)
            alpha = arr[:, :, 3]
            sub_y, sub_x = np.where(alpha > 10)
            if len(sub_y) > 0:
                sub_ymin, sub_ymax = sub_y.min(), sub_y.max()
                sub_xmin, sub_xmax = sub_x.min(), sub_x.max()
                cropped = img_rgba.crop((sub_xmin, sub_ymin, sub_xmax + 1, sub_ymax + 1))
            else:
                cropped = img_rgba

            # Resize cropped subject to match the mesh silhouette exactly
            resized_sub = cropped.resize((mesh_w, mesh_h), Image.Resampling.LANCZOS)

            # Place on transparent canvas matching exact screen coordinates
            aligned_canvas = Image.new('RGBA', (texture_resolution, texture_resolution), (0, 0, 0, 0))
            aligned_canvas.paste(resized_sub, (mesh_xmin, mesh_ymin))

            # RGB base with neutral background
            aligned_rgb = Image.new('RGB', (texture_resolution, texture_resolution), (240, 240, 240))
            aligned_rgb.paste(aligned_canvas, mask=aligned_canvas.split()[3])

            # Back-project
            proj_tex, proj_cos, _ = render.back_project(aligned_rgb, elev, azim)

            # Mask out transparent background so background pixels never bleed onto the mesh
            alpha_arr = np.array(aligned_canvas)[:, :, 3]
            alpha_mask_t = torch.from_numpy(alpha_arr > 10).float().to(proj_cos.device).unsqueeze(-1)
            proj_cos = (weight * (proj_cos ** 4)) * alpha_mask_t

            project_textures.append(proj_tex)
            project_weighted_cos_maps.append(proj_cos)

        if progress_callback:
            progress_callback(0.93, "Fusionando texturas y calculando oclusion...")

        # 6. Fast texture blending
        texture, trust_mask = render.fast_bake_texture(project_textures, project_weighted_cos_maps)

        if progress_callback:
            progress_callback(0.96, "Pintando oclusiones y bordes UV (inpaint)...")

        # 7. Fast OpenCV Telea Inpainting of seams and occlusions
        texture_np = (texture.cpu().numpy() * 255).astype(np.uint8)
        trust_np = (trust_mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        inpaint_mask = 255 - trust_np

        inpainted_bgr = cv2.inpaint(texture_np[:, :, ::-1], inpaint_mask, 7, cv2.INPAINT_TELEA)
        inpainted_rgb = inpainted_bgr[:, :, ::-1]

        render.set_texture(torch.tensor(inpainted_rgb / 255.0).float().to(texture.device))
        textured_mesh = render.save_mesh()

        # 8. Save texture image separately if requested
        texture_path = None
        if output_dir and base_name:
            os.makedirs(output_dir, exist_ok=True)
            texture_path = os.path.join(output_dir, f"{base_name}_texture.png")
            Image.fromarray(inpainted_rgb).save(texture_path)

        t_elapsed = time.time() - t_start
        print(f"[TEXTURE] Textura PBR horneada exitosamente en {t_elapsed:.2f}s!", flush=True)
        return textured_mesh, texture_path

    except Exception as e:
        print(f"[WARN] Error durante el proceso de texturizado: {e}. Manteniendo geometría base.", flush=True)
        return mesh, None
