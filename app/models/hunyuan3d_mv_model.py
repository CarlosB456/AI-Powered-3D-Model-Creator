"""
AI-Powered 3D Model Creator - Hunyuan3D-2 Multi-View Turbo Model Adapter
Multi-angle Image-to-3D generation (Front, Left, Back, Right) on DirectML (AMD Radeon RX 6600).
"""
import os
import sys
import time
import math
from typing import Optional, Callable, Any, Union
from PIL import Image
import torch
import torch_directml
import trimesh
from einops import repeat

# Ensure subfolder path is accessible
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
h3d_path = os.path.join(root_dir, "Hunyuan3D-2")
if h3d_path not in sys.path:
    sys.path.insert(0, h3d_path)

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.pipelines import export_to_trimesh, retrieve_timesteps
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
from app.models.base import Base3DModel

_orig_dino_forward = None

def chunked_dino_forward(self, hidden_states, head_mask=None, output_attentions=False):
    """
    Chunked Attention for DINOv2 to prevent DirectML allocation crashes on 8GB GPUs.
    Properly matches HuggingFace Dinov2SelfAttention signature and attributes.
    """
    mixed_query_layer = self.query(hidden_states)
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    B, H, L, D = query_layer.shape
    chunk_size = 512
    scale = 1.0 / math.sqrt(D)

    context_chunks = []
    for i in range(0, L, chunk_size):
        q_chunk = query_layer[:, :, i:i+chunk_size, :]
        attn_scores = torch.matmul(q_chunk, key_layer.transpose(-1, -2)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        ctx_chunk = torch.matmul(attn_probs, value_layer)
        context_chunks.append(ctx_chunk)

    context_layer = torch.cat(context_chunks, dim=2)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_shape)
    return (context_layer, None) if output_attentions else (context_layer,)


class Hunyuan3DMultiViewModel(Base3DModel):
    """
    Hunyuan3D-2 Multi-View Turbo adapter for multi-angle Image-to-3D generation.
    Accepts Front, Left, Back, and Right views to reconstruct complete 360-degree geometry.
    """
    QUALITIES = {
        'ultra':   {'octree': 128, 'steps': 4, 'chunks': 8192, 'label': 'Ultra (~2 min, octree 128 - 2.1M puntos)'},
        'rapida':  {'octree': 128, 'steps': 5, 'chunks': 8192, 'label': 'Rapida (~2.5 min, octree 128 - 2.1M puntos)'},
        'media':   {'octree': 192, 'steps': 5, 'chunks': 8192, 'label': 'Media (~4.5 min, octree 192 - 7.1M puntos)'},
        'alta':    {'octree': 256, 'steps': 5, 'chunks': 8192, 'label': 'Alta (~8 min, octree 256 - 17.0M puntos)'},
    }

    def __init__(self):
        super().__init__(
            name="Hunyuan3D-2 Multi-View Turbo",
            description="Modelo generativo 3D multi-angulo (Tencent). Soporta multiples vistas (Frontal, Izquierda, Trasera, Derecha)."
        )
        self.pipe = None
        self.device = torch_directml.device()
        self.device_name = torch_directml.device_name(0)

    def load(self, device: Optional[str] = None) -> None:
        if self.is_loaded and self.pipe is not None:
            return

        print(f"[Hunyuan3D Multi-View] Cargando pipeline en {self.device_name}...", flush=True)
        global _orig_dino_forward
        from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention
        if _orig_dino_forward is None:
            _orig_dino_forward = Dinov2SelfAttention.forward
        Dinov2SelfAttention.forward = chunked_dino_forward

        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mv',
            subfolder='hunyuan3d-dit-v2-mv-turbo',
            device='cpu',
            dtype=torch.float32
        )
        self.is_loaded = True
        print("[Hunyuan3D Multi-View] Pipeline listo!", flush=True)

    def get_supported_qualities(self) -> list[str]:
        return list(self.QUALITIES.keys())

    def _encode_multiview(self, images: Union[dict, Image.Image]) -> dict:
        d = self.device
        if isinstance(images, Image.Image):
            images = {"front": images}

        # Format input images to RGBA
        image_dict = {}
        for view_tag, img in images.items():
            if img is not None:
                image_dict[view_tag.lower()] = img.convert("RGBA")

        if not image_dict:
            raise ValueError("No valid input images provided for multi-view encoding.")

        cond_inputs = self.pipe.prepare_image(image_dict)
        img_tensor = cond_inputs.pop('image')

        try:
            self.pipe.conditioner.to(d, dtype=torch.float16)
            img_gpu = img_tensor.to(d, dtype=torch.float16)
            cond_gpu_inputs = {
                k: v.to(d, dtype=torch.float16 if v.is_floating_point() else v.dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_inputs.items()
            }
            with torch.no_grad():
                cond = self.pipe.encode_cond(
                    image=img_gpu,
                    additional_cond_inputs=cond_gpu_inputs,
                    do_classifier_free_guidance=True,
                    dual_guidance=False,
                )
            cond_cpu = {
                k: v.cpu().float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in cond.items()
            }
            self.pipe.conditioner.to('cpu')
            return cond_cpu
        except Exception as e:
            print(f"[Hunyuan3D Multi-View] Fallback DINOv2 CPU ({e})...", flush=True)
            self.pipe.conditioner.to('cpu', dtype=torch.float32)
            from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention
            orig_fwd = Dinov2SelfAttention.forward
            if _orig_dino_forward is not None:
                Dinov2SelfAttention.forward = _orig_dino_forward
            try:
                with torch.no_grad():
                    cond = self.pipe.encode_cond(
                        image=img_tensor,
                        additional_cond_inputs=cond_inputs,
                        do_classifier_free_guidance=True,
                        dual_guidance=False,
                    )
            finally:
                Dinov2SelfAttention.forward = orig_fwd
            return cond

    def _diffuse(self, cond: dict, num_steps: int, progress_callback: Optional[Callable] = None) -> torch.Tensor:
        d = self.device
        self.pipe.model.to(d, dtype=torch.float16)
        self.pipe.model.eval()

        def to_device_recursive(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(d, dtype=torch.float16 if obj.is_floating_point() else obj.dtype)
            elif isinstance(obj, dict):
                return {k: to_device_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_device_recursive(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(to_device_recursive(v) for v in obj)
            return obj

        cond_gpu = to_device_recursive(cond)

        batch_size = 1
        sigmas = [i / num_steps for i in range(num_steps + 1)]
        timesteps, _ = retrieve_timesteps(self.pipe.scheduler, num_steps, 'cpu', sigmas=sigmas)
        latents = self.pipe.prepare_latents(batch_size, torch.float16, "cpu", None).to(d)

        guidance = None
        if hasattr(self.pipe.model, 'guidance_embed') and self.pipe.model.guidance_embed is True:
            guidance = torch.tensor([5.0] * batch_size, device=d, dtype=torch.float16)

        sigmas_tensor = self.pipe.scheduler.sigmas_
        with torch.no_grad():
            for i, t in enumerate(timesteps[:num_steps]):
                if progress_callback:
                    progress_callback(0.25 + 0.35 * (i / num_steps), f"Difusion DiT Multi-View GPU: paso {i+1}/{num_steps}")
                t_step = time.time()
                latent_input = torch.cat([latents] * 2)
                t_val = t.expand(latent_input.shape[0]).to(d, dtype=torch.float16) / self.pipe.scheduler.config.num_train_timesteps
                noise_pred = self.pipe.model(latent_input, t_val, cond_gpu, guidance=guidance)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)

                dt = float(sigmas_tensor[i + 1] - sigmas_tensor[i])
                latents = latents + dt * noise_pred
                print(f"      [Multi-View GPU] Paso {i+1}/{num_steps}: {time.time() - t_step:.1f}s", flush=True)

        self.pipe.model.to('cpu')
        del cond_gpu
        import gc
        gc.collect()
        return latents.cpu().float()

    def _decode_volume_and_mesh(
        self,
        latents: torch.Tensor,
        octree_res: int,
        num_chunks: int,
        progress_callback: Optional[Callable] = None
    ) -> trimesh.Trimesh:
        import numpy as np
        import gc
        d = self.device

        self.pipe.vae.to('cpu', dtype=torch.float32)
        latents_scaled = 1.0 / self.pipe.vae.scale_factor * latents
        latents_dec = self.pipe.vae(latents_scaled)

        bounds = 1.01
        bounds_arr = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min, bbox_max = np.array(bounds_arr[0:3]), np.array(bounds_arr[3:6])

        from hy3dgen.shapegen.models.autoencoders.volume_decoders import generate_dense_grid_points
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min, bbox_max=bbox_max,
            octree_resolution=octree_res, indexing="ij"
        )
        xyz_samples = torch.from_numpy(xyz_samples).contiguous().reshape(-1, 3)
        total_points = xyz_samples.shape[0]

        batch_size = latents_dec.shape[0]
        grid_logits = None

        try:
            geo_decoder = self.pipe.vae.geo_decoder
            geo_decoder.to(d).half().eval()

            if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
                geo_decoder.cross_attn_decoder.attn.kv_cache = True
                geo_decoder.cross_attn_decoder.attn.data = None

            latents_gpu = latents_dec.to(d, dtype=torch.float16)

            batch_logits = []
            with torch.no_grad():
                for start in range(0, total_points, num_chunks):
                    chunk_queries = xyz_samples[start:start+num_chunks, :].to(d, dtype=torch.float16)
                    chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
                    logits = geo_decoder(queries=chunk_queries, latents=latents_gpu)
                    batch_logits.append(logits.cpu().float())

                    if progress_callback:
                        done = min(start + num_chunks, total_points)
                        progress_callback(0.65 + 0.25 * (done / total_points), f"Decodificacion Multi-View GPU ({done:,}/{total_points:,})")

            grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, *grid_size)).float()

            if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
                geo_decoder.cross_attn_decoder.attn.data = None
            geo_decoder.to('cpu').float()
            gc.collect()

        except Exception as e:
            print(f"[WARN] Multi-View volume decode GPU fallo ({e}), usando fallback CPU...", flush=True)
            self.pipe.vae.geo_decoder.to('cpu').float()
            gc.collect()
            from hy3dgen.shapegen.models.autoencoders.volume_decoders import VanillaVolumeDecoder
            vd = VanillaVolumeDecoder()
            grid_logits = vd(
                latents_dec, self.pipe.vae.geo_decoder,
                bounds=bounds, num_chunks=num_chunks,
                octree_resolution=octree_res, enable_pbar=False
            )

        # Marching Cubes surface extraction
        if progress_callback:
            progress_callback(0.92, "Extraccion de malla (Marching Cubes)...")
        surface_extractor = SurfaceExtractors['mc']()
        outputs = surface_extractor(grid_logits, mc_level=0.0, bounds=1.01, octree_resolution=octree_res)
        return export_to_trimesh(outputs)[0]

    def generate(
        self,
        image: Union[Image.Image, dict],
        quality: str = "rapida",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any
    ) -> trimesh.Trimesh:
        if not self.is_loaded:
            self.load()

        cfg = self.QUALITIES.get(quality, self.QUALITIES['rapida'])

        if progress_callback:
            progress_callback(0.05, "Codificando caracteristicas visuales multi-angulo (DINOv2 MV)...")
        t0 = time.time()
        cond = self._encode_multiview(image)
        print(f"[Hunyuan3D Multi-View] DINOv2 MV completado en {time.time() - t0:.1f}s", flush=True)

        if progress_callback:
            progress_callback(0.25, f"Difusion DiT Multi-View ({cfg['steps']} pasos)...")
        t1 = time.time()
        latents = self._diffuse(cond, cfg['steps'], progress_callback)
        print(f"[Hunyuan3D Multi-View] DiT completado en {time.time() - t1:.1f}s", flush=True)

        if progress_callback:
            progress_callback(0.65, f"Decodificando volumen 3D (octree {cfg['octree']})...")
        t2 = time.time()
        mesh = self._decode_volume_and_mesh(latents, cfg['octree'], cfg['chunks'], progress_callback)
        print(f"[Hunyuan3D Multi-View] Malla generada en {time.time() - t2:.1f}s", flush=True)

        return mesh

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.is_loaded = False
        import gc
        gc.collect()
