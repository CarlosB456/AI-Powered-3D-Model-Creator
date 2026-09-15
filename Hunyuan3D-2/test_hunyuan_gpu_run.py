import time
from PIL import Image
import torch
import torch_directml
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

d = torch_directml.device()
print(f"[1/4] Dispositivo DirectML: {torch_directml.device_name(0)}")

print("[2/4] Cargando Hunyuan3D-2mini en memoria...")
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini-turbo',
    device='cpu',
    dtype=torch.float32
)

print("[3/4] Moviendo modelo a la GPU AMD Radeon RX 6600...")
pipe.to(d)
pipe.device = d

image_path = "c:/Users/Benja/Desktop/IA/TripoSR/output/test/input_processed.png"
image = Image.open(image_path).convert("RGBA")

print("[4/4] Ejecutando Difusión en la GPU AMD (5 pasos)...")
t0 = time.time()
try:
    outputs = pipe(
        image=image,
        num_inference_steps=5,
        octree_resolution=256,
        mc_algo="dmc"
    )
    mesh = outputs[0]
    dt = time.time() - t0
    print(f"[EXITO TOTAL EN GPU] Generacion 3D completada en {dt:.2f} segundos!")
    mesh.export("output/cell_hunyuan_gpu.obj")
    mesh.export("output/cell_hunyuan_gpu.glb")
    print(f"Modelo guardado: output/cell_hunyuan_gpu.glb ({len(mesh.vertices):,} vertices, {len(mesh.faces):,} caras)")
except Exception as e:
    print(f"[ERROR EN GPU]: {e}")
    import traceback
    traceback.print_exc()
