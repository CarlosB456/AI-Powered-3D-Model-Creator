"""
HUNYUAN3D-2 TURBO - Generador 3D desde Imágenes (MAX GPU)
GPU: AMD Radeon RX 6600 (DirectML)
Uso: python generar_3d.py <imagen.png> [--calidad ultra|rapida|media|alta]
"""
import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn
import torch_directml
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.pipelines import export_to_trimesh, retrieve_timesteps
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
from einops import repeat

CALIDAD = {
    'ultra':   {'octree': 128, 'steps': 4, 'chunks': 32768, 'desc': 'Ultra Rápida (~2 min)'},
    'rapida':  {'octree': 128, 'steps': 5, 'chunks': 32768, 'desc': 'Rápida (~2.5 min)'},
    'media':   {'octree': 192, 'steps': 5, 'chunks': 32768, 'desc': 'Media (~4 min)'},
    'alta':    {'octree': 256, 'steps': 5, 'chunks': 32768, 'desc': 'Alta (~8 min)'},
}

# ══════════════════════════════════════════════════════════
#  DINOv2 Atención Chunked para GPU con 8GB VRAM
# ══════════════════════════════════════════════════════════
def chunked_dino_forward(self, hidden_states, head_mask=None, output_attentions=False):
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
    return (context_layer,)

def patch_dinov2():
    from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention
    Dinov2SelfAttention.forward = chunked_dino_forward

def cargar_pipeline():
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini-turbo',
        device='cpu',
        dtype=torch.float32
    )
    return pipe

def codificar_imagen_gpu(pipe, imagen_path, d):
    cache_dir = os.path.join("output", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    nombre_base = os.path.splitext(os.path.basename(imagen_path))[0]
    cache_path = os.path.join(cache_dir, f"{nombre_base}_cond.pt")

    if os.path.exists(cache_path):
        print(f"      Cache cargado: {cache_path}", flush=True)
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    patch_dinov2()
    img = Image.open(imagen_path).convert("RGBA")
    cond_inputs = pipe.prepare_image(img)
    img_tensor = cond_inputs.pop('image')

    try:
        pipe.conditioner.to(d, dtype=torch.float16)
        img_gpu = img_tensor.to(d, dtype=torch.float16)
        cond_gpu_inputs = {
            k: v.to(d, dtype=torch.float16 if v.is_floating_point() else v.dtype) if isinstance(v, torch.Tensor) else v
            for k, v in cond_inputs.items()
        }
        with torch.no_grad():
            cond = pipe.encode_cond(
                image=img_gpu,
                additional_cond_inputs=cond_gpu_inputs,
                do_classifier_free_guidance=True,
                dual_guidance=False,
            )
        cond_cpu = {
            k: v.cpu().float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in cond.items()
        }
        pipe.conditioner.to('cpu')
        torch.save(cond_cpu, cache_path)
        return cond_cpu
    except Exception as e:
        print(f"      Fallback DINOv2 CPU ({e})...", flush=True)
        pipe.conditioner.to('cpu', dtype=torch.float32)
        with torch.no_grad():
            cond = pipe.encode_cond(
                image=img_tensor,
                additional_cond_inputs=cond_inputs,
                do_classifier_free_guidance=True,
                dual_guidance=False,
            )
        torch.save(cond, cache_path)
        return cond

def difusion_gpu(pipe, cond, d, num_steps=5):
    pipe.model.to(d, dtype=torch.float16)
    pipe.model.eval()

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
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, 'cpu', sigmas=sigmas)
    latents = pipe.prepare_latents(batch_size, torch.float16, "cpu", None).to(d)

    guidance = None
    if hasattr(pipe.model, 'guidance_embed') and pipe.model.guidance_embed is True:
        guidance = torch.tensor([5.0] * batch_size, device=d, dtype=torch.float16)

    sigmas_tensor = pipe.scheduler.sigmas_
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_step = time.time()
            latent_model_input = torch.cat([latents] * 2)
            timestep_val = t.expand(latent_model_input.shape[0]).to(d, dtype=torch.float16) / pipe.scheduler.config.num_train_timesteps
            noise_pred = pipe.model(latent_model_input, timestep_val, cond_gpu, guidance=guidance)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)
            dt = float(sigmas_tensor[i + 1] - sigmas_tensor[i])
            latents = latents + dt * noise_pred
            print(f"      [GPU] Paso {i+1}/{num_steps}: {time.time() - t_step:.1f}s", flush=True)

    pipe.model.to('cpu')
    del cond_gpu
    return latents.cpu().float()

def decodificar_malla_gpu(pipe, latents, d, octree_res=128, num_chunks=32768):
    import numpy as np
    pipe.vae.to('cpu', dtype=torch.float32)
    latents_scaled = 1.0 / pipe.vae.scale_factor * latents
    latents_dec = pipe.vae(latents_scaled)

    geo_decoder = pipe.vae.geo_decoder
    geo_decoder.to(d).half().eval()

    if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
        geo_decoder.cross_attn_decoder.attn.kv_cache = True
        geo_decoder.cross_attn_decoder.attn.data = None

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
    print(f"      Puntos 3D: {total_points:,} (chunks de {num_chunks:,})", flush=True)

    batch_size = latents_dec.shape[0]
    latents_gpu = latents_dec.to(d, dtype=torch.float16)

    batch_logits = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, total_points, num_chunks):
            chunk_queries = xyz_samples[start:start+num_chunks, :].to(d, dtype=torch.float16)
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents_gpu)
            batch_logits.append(logits.cpu().float())

            done = min(start + num_chunks, total_points)
            pct = done / total_points * 100
            if done < total_points:
                elapsed = time.time() - t0
                eta = elapsed / done * (total_points - done)
                print(f"\r      [GPU] Volume: {pct:.0f}% ETA: {eta:.0f}s   ", end="", flush=True)

    print(f"\r      [GPU] Volume: 100% en {time.time()-t0:.1f}s          ", flush=True)
    grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, *grid_size)).float()

    if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
        geo_decoder.cross_attn_decoder.attn.data = None
    geo_decoder.to('cpu').float()

    # Marching Cubes on CPU
    surface_extractor = SurfaceExtractors['mc']()
    outputs = surface_extractor(grid_logits, mc_level=0.0, bounds=1.01, octree_resolution=octree_res)
    return export_to_trimesh(outputs)[0]

def main():
    parser = argparse.ArgumentParser(description='Generar modelo 3D desde una imagen (MAX GPU)')
    parser.add_argument('imagen', help='Ruta a la imagen PNG/JPG')
    parser.add_argument('--calidad', choices=['ultra', 'rapida', 'media', 'alta'], default='rapida',
                        help='Calidad de generacion (default: rapida)')
    parser.add_argument('--nombre', default=None, help='Nombre del archivo de salida (sin extension)')
    args = parser.parse_args()

    if not os.path.exists(args.imagen):
        print(f"ERROR: No se encontro la imagen: {args.imagen}")
        sys.exit(1)

    cfg = CALIDAD[args.calidad]
    nombre = args.nombre or os.path.splitext(os.path.basename(args.imagen))[0]

    print("=" * 60, flush=True)
    print("  HUNYUAN3D-2 TURBO - GPU AMD RADEON RX 6600 (MAX GPU)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Imagen:  {args.imagen}", flush=True)
    print(f"  Calidad: {args.calidad} - {cfg['desc']}", flush=True)
    print(f"  Salida:  output/{nombre}.glb", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()
    d = torch_directml.device()
    print(f"\n[1/5] GPU: {torch_directml.device_name(0)} (DirectML)", flush=True)

    print("[2/5] Cargando modelo...", flush=True)
    t0 = time.time()
    pipe = cargar_pipeline()
    print(f"      Listo en {time.time() - t0:.1f}s", flush=True)

    print("[3/5] DINOv2 codificación (GPU/Cache)...", flush=True)
    t0 = time.time()
    cond = codificar_imagen_gpu(pipe, args.imagen, d)
    print(f"      Listo en {time.time() - t0:.1f}s", flush=True)

    print(f"[4/5] Difusion en GPU AMD ({cfg['steps']} pasos)...", flush=True)
    t0 = time.time()
    latents = difusion_gpu(pipe, cond, d, cfg['steps'])
    print(f"      Difusion total: {time.time() - t0:.1f}s", flush=True)

    print(f"[5/5] Generando malla 3D (res={cfg['octree']}, chunks={cfg['chunks']:,})...", flush=True)
    t0 = time.time()
    mesh = decodificar_malla_gpu(pipe, latents, d, cfg['octree'], cfg['chunks'])
    print(f"      Malla generada en {time.time() - t0:.1f}s", flush=True)

    os.makedirs("output", exist_ok=True)
    out_obj = f"output/{nombre}.obj"
    out_glb = f"output/{nombre}.glb"
    mesh.export(out_obj)
    mesh.export(out_glb)

    print("\n" + "=" * 60, flush=True)
    print("  MODELO 3D GENERADO EXITOSAMENTE!", flush=True)
    print("=" * 60, flush=True)
    print(f"  Vertices:     {len(mesh.vertices):,}", flush=True)
    print(f"  Caras:        {len(mesh.faces):,}", flush=True)
    print(f"  GLB:          {out_glb} ({os.path.getsize(out_glb)/1024:.1f} KB)", flush=True)
    print(f"  OBJ:          {out_obj} ({os.path.getsize(out_obj)/1024:.1f} KB)", flush=True)
    print(f"  Tiempo total: {time.time() - t_total:.1f}s", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
