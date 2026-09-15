import time
import torch
import torch_directml
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

print("[1] Inicializando DirectML...")
d = torch_directml.device()
print(f"    Dispositivo: {torch_directml.device_name(0)}")

print("[2] Cargando pipeline en CPU (FP16)...")
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini-turbo',
    device='cpu',
    dtype=torch.float16
)

print("[3] Codificando imagen en CPU...")
img_path = "c:/Users/Benja/Desktop/IA/TripoSR/output/test/input_processed.png"
image = Image.open(img_path).convert("RGBA")

t0 = time.time()
cond_inputs = pipe.prepare_image(image)
img_tensor = cond_inputs.pop('image')
cond = pipe.encode_cond(
    image=img_tensor,
    additional_cond_inputs=cond_inputs,
    do_classifier_free_guidance=True,
    dual_guidance=False,
)
print(f"    Condicionamiento listo en {time.time() - t0:.2f}s")

print("[4] Moviendo DiT model y tensores a DirectML (RX 6600)...")
pipe.model.to(d, dtype=torch.float16)

# Move cond dict to GPU
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
latents = pipe.prepare_latents(batch_size, torch.float16, "cpu", None)
latent_model_input = torch.cat([latents] * 2).to(d)
timestep = torch.tensor([0.5, 0.5], device=d, dtype=torch.float16)

guidance = None
if hasattr(pipe.model, 'guidance_embed') and pipe.model.guidance_embed is True:
    guidance = torch.tensor([5.0] * batch_size, device=d, dtype=torch.float16)

print("[5] Probando 1 paso de inferencia de DiT en la GPU AMD...")
t1 = time.time()
with torch.no_grad():
    noise_pred = pipe.model(latent_model_input, timestep, cond_gpu, guidance=guidance)

print(f"    Paso DiT completado exitosamente en {time.time() - t1:.3f}s!")
print(f"    Forma del tensor predicho: {noise_pred.shape} en dispositivo {noise_pred.device}")
