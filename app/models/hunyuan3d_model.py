"""
AI-Powered 3D Model Creator - Hunyuan3D-2 Turbo Model Adapter
Optimized for AMD Radeon RX 6600 (8GB VRAM) and NVIDIA via DirectML / PyTorch.
Features:
- DINOv2 chunked attention (GPU FP16)
- DiT Flow Matching with pure GPU Euler step
- VAE geo_decoder with KV-cache and 32k chunks
"""
import os
import sys
import math
import time
from typing import Optional, Callable, Any
from PIL import Image
import torch
import torch_directml
from einops import repeat
import trimesh

from .base import Base3DModel

# Import Hunyuan3D components
# Ensure Hunyuan3D-2 package root is in path
hy3d_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Hunyuan3D-2"))
if hy3d_path not in sys.path:
    sys.path.insert(0, hy3d_path)

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.pipelines import export_to_trimesh, retrieve_timesteps
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors

def chunked_dino_forward(self, hidden_states, head_mask=None, output_attentions=False):
    """
    Chunked query attention for Dinov2SelfAttention.
    Compatible with transformers >= 4.46 (no transpose_for_scores).
    Uses view+transpose to reshape QKV, then chunks Q to limit peak VRAM.
    Author: Carlos B (eLdarqO)
    """
    batch_size = hidden_states.shape[0]
    new_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)

    key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
    query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

    # query_layer shape: (B, H, L, D)
    B, H, L, D = query_layer.shape
    chunk_size = 512
    scale = 1.0 / math.sqrt(D)

    context_chunks = []
    for i in range(0, L, chunk_size):
        q_chunk = query_layer[:, :, i:i+chunk_size, :]
        attn_scores = torch.matmul(q_chunk, key_layer.transpose(-1, -2)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        ctx_chunk = torch.matmul(attn_probs, value_layer)
        context_chunks.append(ctx_chunk)

    context_layer = torch.cat(context_chunks, dim=2)
    # (B, H, L, D) -> (B, L, H, D) -> (B, L, H*D)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.reshape(batch_size, -1, self.all_head_size)
    return context_layer, None

class Hunyuan3DModel(Base3DModel):
    QUALITIES = {
        'ultra':   {'octree': 128, 'steps': 4, 'chunks': 8192, 'label': 'Ultra (~2 min, octree 128 - 2.1M puntos)'},
        'rapida':  {'octree': 128, 'steps': 5, 'chunks': 8192, 'label': 'Rapida (~2.5 min, octree 128 - 2.1M puntos)'},
        'media':   {'octree': 192, 'steps': 5, 'chunks': 8192, 'label': 'Media (~4.5 min, octree 192 - 7.1M puntos)'},
        'alta':    {'octree': 256, 'steps': 5, 'chunks': 8192, 'label': 'Alta (~8 min, octree 256 - 17.0M puntos)'},
    }

    def __init__(self):
        super().__init__(
            name="Hunyuan3D-2 Turbo",
            description="Modelo insignia de alta fidelidad geométrica (Tencent). Flujo DiT con aceleración MAX GPU en RX 6600."
        )
        self.pipe = None
        self.device = torch_directml.device()
        self.device_name = torch_directml.device_name(0)

    def load(self, device: Optional[str] = None) -> None:
        if self.is_loaded and self.pipe is not None:
            return
        
        print(f"[Hunyuan3D] Cargando pipeline en {self.device_name}...", flush=True)
        # Apply chunked attention patch to DINOv2
        from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention
        Dinov2SelfAttention.forward = chunked_dino_forward

        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mini',
            subfolder='hunyuan3d-dit-v2-mini-turbo',
            device='cpu',
            dtype=torch.float32
        )
        self.is_loaded = True
        print("[Hunyuan3D] Pipeline listo!", flush=True)

    def get_supported_qualities(self) -> list[str]:
        return list(self.QUALITIES.keys())

    def _encode_image(self, image: Image.Image) -> dict:
        d = self.device
        cond_inputs = self.pipe.prepare_image(image.convert("RGBA"))
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
            print(f"[Hunyuan3D] Fallback DINOv2 CPU ({e})...", flush=True)
            self.pipe.conditioner.to('cpu', dtype=torch.float32)
            with torch.no_grad():
                cond = self.pipe.encode_cond(
                    image=img_tensor,
                    additional_cond_inputs=cond_inputs,
                    do_classifier_free_guidance=True,
                    dual_guidance=False,
                )
            return cond

    def _diffuse(
        self,
        cond: dict,
        num_steps: int,
        progress_callback: Optional[Callable] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 5.0
    ) -> torch.Tensor:
        d = self.device
        self.pipe.model.to(d, dtype=torch.float16)
        self.pipe.model.eval()

        cond_gpu = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor):
                cond_gpu[k] = v.to(d, dtype=torch.float16 if v.is_floating_point() else v.dtype)
            elif isinstance(v, dict):
                cond_gpu[k] = {
                    subk: subv.to(d, dtype=torch.float16 if subv.is_floating_point() else subv.dtype)
                    if isinstance(subv, torch.Tensor) else subv
                    for subk, subv in v.items()
                }
            else:
                cond_gpu[k] = v

        batch_size = 1
        sigmas = [i / num_steps for i in range(num_steps + 1)]
        timesteps, _ = retrieve_timesteps(self.pipe.scheduler, num_steps, 'cpu', sigmas=sigmas)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = self.pipe.prepare_latents(batch_size, torch.float16, torch.device("cpu"), generator).to(d)

        guidance = None
        if hasattr(self.pipe.model, 'guidance_embed') and self.pipe.model.guidance_embed is True:
            guidance = torch.tensor([float(guidance_scale)] * batch_size, device=d, dtype=torch.float16)

        sigmas_tensor = self.pipe.scheduler.sigmas_
        with torch.no_grad():
            for i, t in enumerate(timesteps[:num_steps]):
                if progress_callback:
                    progress_callback(0.25 + 0.35 * (i / num_steps), f"Difusión DiT GPU: paso {i+1}/{num_steps}")
                t_step = time.time()
                latent_input = torch.cat([latents] * 2)
                t_val = t.expand(latent_input.shape[0]).to(d, dtype=torch.float16) / self.pipe.scheduler.config.num_train_timesteps
                noise_pred = self.pipe.model(latent_input, t_val, cond_gpu, guidance=guidance)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_cond - noise_pred_uncond)

                dt = float(sigmas_tensor[i + 1] - sigmas_tensor[i])
                latents = latents + dt * noise_pred
                print(f"      [GPU] Paso {i+1}/{num_steps}: {time.time() - t_step:.1f}s", flush=True)

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

        # Intentar en GPU primero, con fallback automático a CPU
        grid_logits = None
        try:
            geo_decoder = self.pipe.vae.geo_decoder
            geo_decoder.to(d).half().eval()

            if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
                geo_decoder.cross_attn_decoder.attn.kv_cache = True
                geo_decoder.cross_attn_decoder.attn.data = None

            latents_gpu = latents_dec.to(d, dtype=torch.float16)

            batch_logits = []
            t0 = time.time()
            with torch.no_grad():
                for start in range(0, total_points, num_chunks):
                    chunk_queries = xyz_samples[start:start+num_chunks, :].to(d, dtype=torch.float16)
                    chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
                    logits = geo_decoder(queries=chunk_queries, latents=latents_gpu)
                    batch_logits.append(logits.cpu().float())

                    if progress_callback:
                        done = min(start + num_chunks, total_points)
                        progress_callback(0.65 + 0.25 * (done / total_points), f"Volume Decode GPU ({done:,}/{total_points:,})")

            grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, *grid_size)).float()

            if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
                geo_decoder.cross_attn_decoder.attn.data = None
            geo_decoder.to('cpu').float()
            gc.collect()

        except Exception as e:
            print(f"[WARN] Volume decode GPU falló ({e}), usando fallback CPU...", flush=True)
            self.pipe.vae.geo_decoder.to('cpu').float()
            gc.collect()
            from hy3dgen.shapegen.models.autoencoders.volume_decoders import VanillaVolumeDecoder
            vd = VanillaVolumeDecoder()
            grid_logits = vd(
                latents_dec, self.pipe.vae.geo_decoder,
                bounds=bounds, num_chunks=num_chunks,
                octree_resolution=octree_res, enable_pbar=False
            )

        # Marching Cubes
        if progress_callback:
            progress_callback(0.92, "Extracción de malla (Marching Cubes)...")
        surface_extractor = SurfaceExtractors['mc']()
        outputs = surface_extractor(grid_logits, mc_level=0.0, bounds=1.01, octree_resolution=octree_res)
        return export_to_trimesh(outputs)[0]

    def generate(
        self,
        image: Image.Image,
        quality: str = "rapida",
        progress_callback: Optional[Callable] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 5.0,
        steps: Optional[int] = None,
        **kwargs: Any
    ) -> trimesh.Trimesh:
        if not self.is_loaded:
            self.load()

        cfg = self.QUALITIES.get(quality, self.QUALITIES['rapida'])
        num_steps = int(steps) if steps is not None and int(steps) > 0 else cfg['steps']
        
        if progress_callback:
            progress_callback(0.05, "Codificando imagen con DINOv2 (GPU FP16)...")
        cond = self._encode_image(image)

        if progress_callback:
            progress_callback(0.25, f"Difusión DiT ({num_steps} pasos)...")
        latents = self._diffuse(cond, num_steps, progress_callback, seed=seed, guidance_scale=guidance_scale)

        mesh = self._decode_volume_and_mesh(latents, cfg['octree'], cfg['chunks'], progress_callback)
        return mesh

    def unload(self) -> None:
        if self.pipe is not None:
            self.pipe.model.to('cpu')
            self.pipe.conditioner.to('cpu')
            self.pipe.vae.to('cpu')
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
