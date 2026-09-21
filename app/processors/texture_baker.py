"""
AI-Powered 3D Model Creator - Texture Baker Module
Generates and bakes high-resolution PBR UV textures from multi-angle or single-view images
using a high-performance Numba-accelerated CPU rasterizer with exact screen-space alpha masking
and 3D vertex-guided texture inpainting.
Zero CUDA, nvcc, or MSVC requirements. 100% compatible with AMD Radeon RX 6600 and consumer hardware.
Created by Carlos B (eLdarqO)
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
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
    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender, linear_grid_put_2d
    from hy3dgen.texgen.differentiable_renderer.camera_utils import get_mv_matrix, transform_pos
    _TEXTURE_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Texture engine import issue: {e}")
    _TEXTURE_ENGINE_AVAILABLE = False


def recenter_and_pad_view(
    image: Image.Image,
    target_size: int = 1024,
    border_ratio: float = 0.15
) -> Image.Image:
    """
    Centers the subject in a square canvas without stretching or aspect ratio distortion.
    Pads with a transparent background so only the genuine foreground subject is projected.
    Author: Carlos B (eLdarqO)
    """
    img_rgba = image.convert("RGBA")
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]

    non_zero = np.argwhere(alpha > 10)
    if non_zero.size == 0:
        return img_rgba.resize((target_size, target_size), Image.Resampling.LANCZOS)

    min_row, min_col = non_zero.min(axis=0)
    max_row, max_col = non_zero.max(axis=0)

    cropped = img_rgba.crop((min_col, min_row, max_col + 1, max_row + 1))
    w, h = cropped.size

    # Maintain exact aspect ratio; add uniform proportional padding
    pad = int(max(w, h) * border_ratio)
    square_dim = max(w, h) + 2 * pad

    canvas = Image.new("RGBA", (square_dim, square_dim), (0, 0, 0, 0))
    paste_x = (square_dim - w) // 2
    paste_y = (square_dim - h) // 2
    canvas.paste(cropped, (paste_x, paste_y))

    return canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)


def project_view_with_alpha(
    render: MeshRender,
    image_rgba: Image.Image,
    elev: float,
    azim: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects an RGBA view onto the mesh in UV space.
    Critically masks the cosine confidence map in SCREEN space with the image alpha channel
    before projecting to UV coordinates, ensuring transparent background pixels never bleed
    or contaminate the 3D surface.
    Author: Carlos B (eLdarqO)
    """
    img_arr = np.array(image_rgba) / 255.0
    image_t = torch.from_numpy(img_arr).float().to(render.device)
    rgb = image_t[..., :3]
    alpha = image_t[..., 3:4]

    resolution = rgb.shape[:2]
    proj = render.camera_proj_mat

    r_mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=render.camera_distance)
    pos_camera = transform_pos(r_mv, render.vtx_pos, keepdim=True)
    pos_clip = transform_pos(proj, pos_camera)
    pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]

    v0 = pos_camera[render.pos_idx[:, 0], :]
    v1 = pos_camera[render.pos_idx[:, 1], :]
    v2 = pos_camera[render.pos_idx[:, 2], :]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    vertex_normals = trimesh.geometry.mean_vertex_normals(
        vertex_count=render.vtx_pos.shape[0],
        faces=render.pos_idx.cpu(),
        face_normals=face_normals.cpu()
    )
    vertex_normals = torch.from_numpy(vertex_normals).float().to(render.device).contiguous()

    rast_out, _ = render.raster_rasterize(pos_clip, render.pos_idx, resolution=resolution)
    visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]

    normal, _ = render.raster_interpolate(vertex_normals[None, ...], rast_out, render.pos_idx)
    normal = normal[0, ...]
    uv, _ = render.raster_interpolate(render.vtx_uv[None, ...], rast_out, render.uv_idx)

    lookat = torch.tensor([[0, 0, -1]], device=render.device)
    cos_image = F.cosine_similarity(lookat, normal.view(-1, 3)).view(normal.shape[0], normal.shape[1], 1)
    cos_thres = np.cos(render.bake_angle_thres / 180.0 * np.pi)
    cos_image[cos_image < cos_thres] = 0

    # Mask in SCREEN space: only genuine foreground pixels receive projection weight
    cos_image = cos_image * (alpha > 0.08).float()
    cos_image[visible_mask == 0] = 0

    proj_mask = (visible_mask != 0).view(-1)
    uv_flat = uv.squeeze(0).contiguous().view(-1, 2)[proj_mask]
    rgb_flat = rgb.contiguous().view(-1, 3)[proj_mask]
    cos_flat = cos_image.contiguous().view(-1, 1)[proj_mask]

    proj_tex = linear_grid_put_2d(render.texture_size[1], render.texture_size[0], uv_flat[..., [1, 0]], rgb_flat)
    proj_cos = linear_grid_put_2d(render.texture_size[1], render.texture_size[0], uv_flat[..., [1, 0]], cos_flat)

    return proj_tex, proj_cos


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
    with automatic silhouette alignment, screen-space alpha masking, and 3D vertex-guided inpainting.
    Returns (textured_mesh, texture_png_path).
    Author: Carlos B (eLdarqO)
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

        # 5. Multi-angle Back-projection with Screen Alpha Masking
        for img, elev, azim, weight in view_tuples:
            centered_rgba = recenter_and_pad_view(img, target_size=texture_resolution, border_ratio=0.15)
            proj_tex, proj_cos = project_view_with_alpha(render, centered_rgba, elev=elev, azim=azim)
            proj_cos = weight * (proj_cos ** 4)
            project_textures.append(proj_tex)
            project_weighted_cos_maps.append(proj_cos)

        if progress_callback:
            progress_callback(0.93, "Fusionando texturas y calculando oclusion...")

        # 6. Fast texture blending
        texture, trust_mask = render.fast_bake_texture(project_textures, project_weighted_cos_maps)

        if progress_callback:
            progress_callback(0.96, "Pintando oclusiones y bordes UV (3D vertex inpaint)...")

        # 7. 3D Vertex-Guided Inpainting (preserves geometric color continuity across UV seams)
        trust_np = (trust_mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        inpainted_rgb = render.uv_inpaint(texture, trust_np)

        render.set_texture(torch.tensor(inpainted_rgb / 255.0).float().to(render.device))
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
        import traceback
        print(f"[WARN] Error durante el proceso de texturizado: {e}. Manteniendo geometria base.", flush=True)
        traceback.print_exc()
        return mesh, None
