import time
import torch
import torch_directml
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

print("[1] Verificando DirectML GPU AMD...", flush=True)
d = torch_directml.device()
print(f"    GPU: {torch_directml.device_name(0)}", flush=True)

print("[2] Cargando Hunyuan3D-2mini en CPU...", flush=True)
t0 = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini-turbo',
    device='cpu',
    dtype=torch.float32  # CPU uses FP32 for fast AVX2 execution
)
print(f"    Pipeline cargado en {time.time() - t0:.2f}s", flush=True)

# 1. Encode image on CPU with FP32
img_path = "c:/Users/Benja/Desktop/IA/TripoSR/output/test/input_processed.png"
image = Image.open(img_path).convert("RGBA")

print("[3] Codificando imagen con DINOv2 en CPU (FP32)...", flush=True)
t_enc = time.time()
cond_inputs = pipe.prepare_image(image)
img_tensor = cond_inputs.pop('image')
cond = pipe.encode_cond(
    image=img_tensor,
    additional_cond_inputs=cond_inputs,
    do_classifier_free_guidance=True,
    dual_guidance=False,
)
print(f"    Codificacion completada en {time.time() - t_enc:.2f}s!", flush=True)

# 2. Move ONLY DiT model to DirectML GPU in FP16
print("[4] Moviendo DiT (560M params / 1.1GB) a la GPU AMD RX 6600 (FP16)...", flush=True)
t_move = time.time()
pipe.model.to(d, dtype=torch.float16)
print(f"    DiT en GPU listo en {time.time() - t_move:.2f}s!", flush=True)

# 3. Move cond dictionary to DirectML GPU
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

# 4. Test 1 forward step of DiT on GPU
batch_size = 1
latents = pipe.prepare_latents(batch_size, torch.float16, "cpu", None)
latent_model_input = torch.cat([latents] * 2).to(d)
timestep = torch.tensor([0.5, 0.5], device=d, dtype=torch.float16)

guidance = None
if hasattr(pipe.model, 'guidance_embed') and pipe.model.guidance_embed is True:
    guidance = torch.tensor([5.0] * batch_size, device=d, dtype=torch.float16)

print("[5] Ejecutando 1 paso de DiT en la GPU AMD Radeon RX 6600...", flush=True)
t_step = time.time()
with torch.no_grad():
    noise_pred = pipe.model(latent_model_input, timestep, cond_gpu, guidance=guidance)

print(f"    >>> EXITO! 1 paso de DiT en GPU AMD ejecutado en {time.time() - t_step:.3f}s! <<<", flush=True)
print(f"    Shape: {noise_pred.shape}, device: {noise_pred.device}, dtype: {noise_pred.dtype}", flush=True)
