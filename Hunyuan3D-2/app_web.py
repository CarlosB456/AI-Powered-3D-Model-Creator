"""
HUNYUAN3D-2 TURBO — Interfaz Web Local (MAX GPU)
GPU: AMD Radeon RX 6600 (DirectML)
Optimizaciones:
  - DINOv2: GPU FP16 con atención chunked (parcheo dinámico)
  - DiT: GPU FP16 (difusión)
  - VAE geo_decoder: GPU FP16 (volume decoding)
  - Marching Cubes: CPU (operación simple)
"""
import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch_directml
import gradio as gr
from PIL import Image
from typing import Optional, Tuple, Union
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.pipelines import export_to_trimesh, retrieve_timesteps
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
from einops import repeat

# ── Config ──
DEVICE = torch_directml.device()
GPU_NAME = torch_directml.device_name(0)
PIPE = None
DINO_PATCHED = False

CALIDAD_CONFIG = {
    'Ultra Rápida (~2 min)': {'octree': 128, 'steps': 4, 'chunks': 32768},
    'Rápida (~2.5 min)':     {'octree': 128, 'steps': 5, 'chunks': 32768},
    'Media (~4 min)':        {'octree': 192, 'steps': 5, 'chunks': 32768},
    'Alta (~8 min)':         {'octree': 256, 'steps': 5, 'chunks': 32768},
}

# ══════════════════════════════════════════════════════════
#  PARCHE 1: DINOv2 Atención Chunked para GPU con 8GB VRAM
# ══════════════════════════════════════════════════════════
def chunked_dino_forward(self, hidden_states, head_mask=None, output_attentions=False):
    """
    Reemplaza Dinov2SelfAttention.forward con atención chunked.
    En vez de crear la matriz completa QK^T (5330x5330x24 = 1.3GB),
    procesamos la query en chunks para mantener el uso de VRAM bajo.
    """
    mixed_query_layer = self.query(hidden_states)
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    # B, H, L, D
    B, H, L, D = query_layer.shape
    chunk_size = 512  # Procesar 512 tokens a la vez
    scale = 1.0 / math.sqrt(D)

    context_chunks = []
    for i in range(0, L, chunk_size):
        q_chunk = query_layer[:, :, i:i+chunk_size, :]  # B, H, chunk, D
        attn_scores = torch.matmul(q_chunk, key_layer.transpose(-1, -2)) * scale  # B, H, chunk, L
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        ctx_chunk = torch.matmul(attn_probs, value_layer)  # B, H, chunk, D
        context_chunks.append(ctx_chunk)

    context_layer = torch.cat(context_chunks, dim=2)  # B, H, L, D
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_shape)

    return (context_layer,)


def patch_dinov2_attention():
    """Monkey-patch DINOv2 para usar atención chunked en GPU"""
    global DINO_PATCHED
    if DINO_PATCHED:
        return
    from transformers.models.dinov2.modeling_dinov2 import Dinov2SelfAttention
    Dinov2SelfAttention.forward = chunked_dino_forward
    DINO_PATCHED = True
    print("      [PATCH] DINOv2 atención chunked activada", flush=True)


# ══════════════════════════════════════════════════════════
#  Pipeline Functions
# ══════════════════════════════════════════════════════════

def cargar_modelo():
    global PIPE
    if PIPE is not None:
        return
    print("[INIT] Cargando pipeline...", flush=True)
    t0 = time.time()
    PIPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini-turbo',
        device='cpu',
        dtype=torch.float32
    )
    print(f"[INIT] Listo en {time.time()-t0:.1f}s", flush=True)


def codificar_imagen_gpu(pipe, imagen_path):
    """DINOv2 en GPU con atención chunked"""
    d = DEVICE
    patch_dinov2_attention()

    if isinstance(imagen_path, str):
        img = Image.open(imagen_path).convert("RGBA")
    else:
        img = Image.fromarray(imagen_path).convert("RGBA")

    cond_inputs = pipe.prepare_image(img)
    img_tensor = cond_inputs.pop('image')

    # Move conditioner to GPU FP16
    pipe.conditioner.to(d, dtype=torch.float16)

    img_gpu = img_tensor.to(d, dtype=torch.float16)
    cond_gpu_inputs = {}
    for k, v in cond_inputs.items():
        if isinstance(v, torch.Tensor):
            cond_gpu_inputs[k] = v.to(d, dtype=torch.float16 if v.is_floating_point() else v.dtype)
        else:
            cond_gpu_inputs[k] = v

    with torch.no_grad():
        cond = pipe.encode_cond(
            image=img_gpu,
            additional_cond_inputs=cond_gpu_inputs,
            do_classifier_free_guidance=True,
            dual_guidance=False,
        )

    # Move results to CPU, free VRAM
    cond_cpu = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor):
            cond_cpu[k] = v.cpu().float()
        elif isinstance(v, dict):
            cond_cpu[k] = {
                subk: subv.cpu().float() if isinstance(subv, torch.Tensor) and subv.is_floating_point()
                else subv.cpu() if isinstance(subv, torch.Tensor) else subv
                for subk, subv in v.items()
            }
        else:
            cond_cpu[k] = v

    pipe.conditioner.to('cpu')
    return cond_cpu


def codificar_imagen_cpu(pipe, imagen_path):
    """Fallback: DINOv2 en CPU"""
    if isinstance(imagen_path, str):
        img = Image.open(imagen_path).convert("RGBA")
    else:
        img = Image.fromarray(imagen_path).convert("RGBA")

    cond_inputs = pipe.prepare_image(img)
    img_tensor = cond_inputs.pop('image')

    pipe.conditioner.to('cpu', dtype=torch.float32)
    with torch.no_grad():
        cond = pipe.encode_cond(
            image=img_tensor,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=True,
            dual_guidance=False,
        )
    return cond


def difusion_gpu(pipe, cond, num_steps=5):
    """DiT en GPU AMD FP16"""
    d = DEVICE
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
        guidance = torch.tensor([5.0], device=d, dtype=torch.float16)

    sigmas_tensor = pipe.scheduler.sigmas_
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_step = time.time()
            latent_model_input = torch.cat([latents] * 2)
            timestep_val = t.expand(latent_model_input.shape[0]).to(d, dtype=torch.float16) / pipe.scheduler.config.num_train_timesteps
            noise_pred = pipe.model(latent_model_input, timestep_val, cond_gpu, guidance=guidance)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)
            # Paso Euler 100% en GPU (sin transferencias PCIe intermedias)
            dt = float(sigmas_tensor[i + 1] - sigmas_tensor[i])
            latents = latents + dt * noise_pred
            print(f"      [GPU] Paso {i+1}/{num_steps}: {time.time()-t_step:.1f}s", flush=True)

    pipe.model.to('cpu')
    del cond_gpu
    return latents.cpu().float()


def volume_decode_gpu(latents_dec, geo_decoder, bounds, num_chunks, octree_res):
    """geo_decoder en GPU FP16 con KV-cache y chunks masivos de 32k"""
    import numpy as np
    d = DEVICE

    geo_decoder.to(d)
    geo_decoder.half()
    geo_decoder.eval()

    # Activar KV cache para no recalcular claves/valores 65+ veces
    if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
        geo_decoder.cross_attn_decoder.attn.kv_cache = True
        geo_decoder.cross_attn_decoder.attn.data = None

    batch_size = latents_dec.shape[0]
    latents_gpu = latents_dec.to(d, dtype=torch.float16)

    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])

    from hy3dgen.shapegen.models.autoencoders.volume_decoders import generate_dense_grid_points
    xyz_samples, grid_size, length = generate_dense_grid_points(
        bbox_min=bbox_min, bbox_max=bbox_max,
        octree_resolution=octree_res, indexing="ij"
    )
    xyz_samples = torch.from_numpy(xyz_samples).contiguous().reshape(-1, 3)
    total_points = xyz_samples.shape[0]
    print(f"      Puntos 3D: {total_points:,} (chunks: {num_chunks:,})", flush=True)

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

    grid_logits = torch.cat(batch_logits, dim=1)
    grid_logits = grid_logits.view((batch_size, *grid_size)).float()

    # Liberar memoria de cache
    if hasattr(geo_decoder.cross_attn_decoder, 'attn'):
        geo_decoder.cross_attn_decoder.attn.data = None
    geo_decoder.to('cpu').float()
    return grid_logits


def generar_3d(imagen, calidad):
    global PIPE
    if PIPE is None:
        cargar_modelo()

    if imagen is None:
        return None, None, "❌ Sube una imagen primero"

    t_total = time.time()
    os.makedirs("output", exist_ok=True)
    cfg = CALIDAD_CONFIG[calidad]

    # ── PASO 1: DINOv2 ──
    print(f"\n{'='*60}", flush=True)
    print(f"[1/3] DINOv2 (codificación de imagen)...", flush=True)
    t0 = time.time()
    dino_device = "?"
    try:
        cond = codificar_imagen_gpu(PIPE, imagen)
        dino_device = "GPU "
        print(f"      *** DINOv2 en GPU! *** {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"      GPU fallo: {e}", flush=True)
        print(f"      Usando CPU...", flush=True)
        PIPE.conditioner.to('cpu', dtype=torch.float32)
        cond = codificar_imagen_cpu(PIPE, imagen)
        dino_device = "CPU"
        print(f"      DINOv2 en CPU: {time.time()-t0:.1f}s", flush=True)
    t_enc = time.time() - t0

    # ── PASO 2: Difusión DiT ──
    print(f"[2/3] Difusión DiT ({cfg['steps']} pasos)...", flush=True)
    t0 = time.time()
    latents = difusion_gpu(PIPE, cond, cfg['steps'])
    t_diff = time.time() - t0
    print(f"      Difusión GPU total: {t_diff:.1f}s", flush=True)

    # ── PASO 3: VAE + Volume Decode + MC ──
    print(f"[3/3] VAE + Volume Decode + Marching Cubes...", flush=True)
    t0 = time.time()

    PIPE.vae.to('cpu', dtype=torch.float32)
    latents_scaled = 1.0 / PIPE.vae.scale_factor * latents

    # VAE forward (transformer) - CPU
    print("      VAE transformer (CPU)...", flush=True)
    latents_dec = PIPE.vae(latents_scaled)

    # Volume decode - GPU!
    vae_device = "?"
    print("      Volume decode (GPU)...", flush=True)
    try:
        grid_logits = volume_decode_gpu(
            latents_dec, PIPE.vae.geo_decoder,
            bounds=1.01, num_chunks=cfg['chunks'],
            octree_res=cfg['octree']
        )
        vae_device = "GPU "
    except Exception as e:
        print(f"      GPU fallo: {e}", flush=True)
        print(f"      Fallback CPU...", flush=True)
        PIPE.vae.geo_decoder.to('cpu').float()
        from hy3dgen.shapegen.models.autoencoders.volume_decoders import VanillaVolumeDecoder
        vd = VanillaVolumeDecoder()
        grid_logits = vd(latents_dec, PIPE.vae.geo_decoder,
                         bounds=1.01, num_chunks=cfg['chunks'],
                         octree_resolution=cfg['octree'], enable_pbar=True)
        vae_device = "CPU"

    # Marching Cubes - CPU
    print("      Marching Cubes (CPU)...", flush=True)
    surface_extractor = SurfaceExtractors['mc']()
    outputs = surface_extractor(grid_logits, mc_level=0.0, bounds=1.01, octree_resolution=cfg['octree'])
    mesh = export_to_trimesh(outputs)[0]
    t_vae = time.time() - t0
    print(f"      VAE+MC total: {t_vae:.1f}s", flush=True)

    # ── GUARDAR ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_glb = f"output/modelo_{timestamp}.glb"
    out_obj = f"output/modelo_{timestamp}.obj"
    mesh.export(out_glb)
    mesh.export(out_obj)

    t_final = time.time() - t_total

    info = f""" **Modelo 3D Generado!**

| Etapa | Dispositivo | Tiempo |
|-------|------------|--------|
| DINOv2 | {dino_device} | {t_enc:.1f}s |
| Difusión DiT | GPU  | {t_diff:.1f}s |
| VAE + MC | {vae_device} | {t_vae:.1f}s |
| **TOTAL** | | **{t_final:.1f}s** |

| Modelo | Valor |
|--------|-------|
| Vértices | {len(mesh.vertices):,} |
| Caras | {len(mesh.faces):,} |
| GLB | {os.path.getsize(out_glb)/1024:.1f} KB |
"""
    print(f"\n{'='*60}", flush=True)
    print(f"  TOTAL: {t_final:.1f}s | V:{len(mesh.vertices):,} F:{len(mesh.faces):,}", flush=True)
    print(f"  DINOv2:{dino_device} DiT:GPU VAE:{vae_device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return out_glb, out_obj, info


# ── Interfaz Gradio ──
with gr.Blocks(
    title="Hunyuan3D-2 Turbo — GPU AMD",
    theme=gr.themes.Soft(),
) as app:
    gr.Markdown(f"""
    #  Hunyuan3D-2 Turbo — Imagen a 3D
    ### GPU: {GPU_NAME} · DirectML · Aceleración Máxima
    """)

    with gr.Row():
        with gr.Column(scale=1):
            imagen_input = gr.Image(label=" Imagen", type="filepath", height=400)
            calidad_input = gr.Radio(
                choices=list(CALIDAD_CONFIG.keys()),
                value='Rápida (~3 min)',
                label=" Calidad",
            )
            generar_btn = gr.Button(" Generar Modelo 3D", variant="primary", size="lg")

        with gr.Column(scale=1):
            modelo_glb = gr.File(label=" Modelo GLB")
            modelo_obj = gr.File(label=" Modelo OBJ")
            info_output = gr.Markdown(label=" Resultado")

    gr.Markdown("""
    ---
    **GPU Acelerada:** DINOv2  | DiT  | Volume Decoder  | Marching Cubes (CPU)

    **Tips:** Usa fondo blanco/transparente · Abre `.glb` en [gltf-viewer](https://gltf-viewer.donmccurdy.com/)
    """)

    generar_btn.click(
        fn=generar_3d,
        inputs=[imagen_input, calidad_input],
        outputs=[modelo_glb, modelo_obj, info_output],
    )

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Hunyuan3D-2 Turbo — MAX GPU")
    print(f"  GPU: {GPU_NAME}")
    print(f"{'='*60}")
    cargar_modelo()
    print(f"  Modelo listo! → http://localhost:7860")
    print(f"{'='*60}\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
