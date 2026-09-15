import os
import time
import torch
import torch_directml
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.pipelines import export_to_trimesh

def main():
    print("=" * 60, flush=True)
    print("  HUNYUAN3D-2 TURBO - GPU AMD RX 6600 (OPTIMIZADO)", flush=True)
    print("=" * 60, flush=True)

    d = torch_directml.device()
    print(f"[1/6] GPU: {torch_directml.device_name(0)} (DirectML)", flush=True)

    print("[2/6] Cargando Pipeline...", flush=True)
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini-turbo',
        device='cpu',
        dtype=torch.float32
    )
    print(f"      Cargado en {time.time() - t0:.1f}s", flush=True)

    os.makedirs("output", exist_ok=True)

    # Step 3: Load cached conditioning (skips 150s DINOv2 pass)
    cache_path = "output/cond_cache.pt"
    print(f"[3/6] Cargando embeddings cacheados...", flush=True)
    cond = torch.load(cache_path, map_location='cpu', weights_only=False)
    print("      OK!", flush=True)

    # Step 4: Move DiT to GPU
    print("[4/6] Moviendo DiT a GPU AMD (FP16)...", flush=True)
    pipe.model.to(d, dtype=torch.float16)
    pipe.model.eval()
    print("      DiT en VRAM!", flush=True)

    # Move cond to GPU
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

    # Step 5: Diffusion on GPU
    num_steps = 5
    print(f"[5/6] Difusion Flow Matching en GPU AMD ({num_steps} pasos)...", flush=True)
    t_diff = time.time()

    batch_size = 1
    sigmas = [i / num_steps for i in range(num_steps + 1)]
    from hy3dgen.shapegen.pipelines import retrieve_timesteps
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, 'cpu', sigmas=sigmas)

    latents = pipe.prepare_latents(batch_size, torch.float16, "cpu", None).to(d)

    guidance = None
    if hasattr(pipe.model, 'guidance_embed') and pipe.model.guidance_embed is True:
        guidance = torch.tensor([5.0] * batch_size, device=d, dtype=torch.float16)

    do_cfg = True
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_step = time.time()
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            timestep_val = t.expand(latent_model_input.shape[0]).to(d, dtype=torch.float16) / pipe.scheduler.config.num_train_timesteps
            noise_pred = pipe.model(latent_model_input, timestep_val, cond_gpu, guidance=guidance)

            if do_cfg:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)

            outputs = pipe.scheduler.step(noise_pred.cpu().float(), t, latents.cpu().float())
            latents = outputs.prev_sample.to(d, dtype=torch.float16)
            print(f"      Paso {i+1}/{num_steps}: {time.time() - t_step:.1f}s", flush=True)

    print(f"      DIFUSION TOTAL: {time.time() - t_diff:.1f}s en GPU AMD!", flush=True)

    # Step 6: VAE decode + Marching Cubes (CPU, lower resolution for speed)
    print("[6/6] Decodificando VAE + Marching Cubes (CPU)...", flush=True)
    t_vae = time.time()

    # Free GPU memory
    pipe.model.to('cpu')
    del cond_gpu
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    latents_cpu = latents.cpu().float()
    pipe.vae.to('cpu', dtype=torch.float32)

    # Use regular Marching Cubes (fast, no diso dependency)
    from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
    pipe.vae.surface_extractor = SurfaceExtractors['mc']()

    latents_scaled = 1.0 / pipe.vae.scale_factor * latents_cpu
    latents_dec = pipe.vae(latents_scaled)

    # Use resolution 128 for fast decode (~2min vs ~20min at 256)
    octree_res = 128
    print(f"      Octree resolution: {octree_res}", flush=True)
    mesh_outputs = pipe.vae.latents2mesh(
        latents_dec,
        bounds=1.01,
        mc_level=0.0,
        num_chunks=10000,
        octree_resolution=octree_res,
        mc_algo=None,
        enable_pbar=True,
    )
    mesh = export_to_trimesh(mesh_outputs)[0]
    print(f"      VAE + MC completado en {time.time() - t_vae:.1f}s!", flush=True)

    out_obj = "output/cell_hunyuan_gpu.obj"
    out_glb = "output/cell_hunyuan_gpu.glb"
    mesh.export(out_obj)
    mesh.export(out_glb)

    print("\n" + "=" * 60, flush=True)
    print("  RESULTADO - HUNYUAN3D-2 EN GPU AMD RADEON RX 6600", flush=True)
    print("=" * 60, flush=True)
    print(f"  Vertices: {len(mesh.vertices):,}", flush=True)
    print(f"  Caras:    {len(mesh.faces):,}", flush=True)
    print(f"  GLB: {out_glb} ({os.path.getsize(out_glb)/1024:.1f} KB)", flush=True)
    print(f"  OBJ: {out_obj} ({os.path.getsize(out_obj)/1024:.1f} KB)", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
