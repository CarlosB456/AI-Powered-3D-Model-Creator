import os
import sys
import time
from PIL import Image
import torch

def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml
        d = torch_directml.device()
        print(f"[INFO] AMD GPU detectada via DirectML: {torch_directml.device_name(0)}")
        return d
    except Exception as e:
        print(f"[INFO] DirectML no disponible ({e}), usando CPU.")
        return "cpu"

def main():
    image_path = "c:/Users/Benja/Desktop/IA/TripoSR/output/test/input_processed.png"
    if not os.path.exists(image_path):
        image_path = "assets/demo.png"

    print(f"[1/4] Cargando imagen: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    print("[2/4] Cargando Hunyuan3D-2mini-Turbo...")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    t0 = time.time()
    device = "cpu"  # Empezar con CPU para maxima estabilidad multihilo
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        device=device,
        dtype=torch.float32
    )
    print(f"      Pipeline cargado en {time.time() - t0:.2f}s")

    print("[3/4] Generando forma 3D con Flow Matching DiT (Turbo: 5 pasos)...")
    t1 = time.time()
    # Turbo model uses 5-8 steps
    outputs = pipe(
        image=image,
        num_inference_steps=5,
        octree_resolution=256,
        mc_algo="dmc"  # Dual Marching Cubes para aristas mas definidas
    )
    mesh = outputs[0]
    print(f"      Generacion completada en {time.time() - t1:.2f}s!")

    print("[4/4] Guardando modelo 3D...")
    os.makedirs("output", exist_ok=True)
    out_obj = "output/cell_hunyuan.obj"
    out_glb = "output/cell_hunyuan.glb"
    mesh.export(out_obj)
    mesh.export(out_glb)

    print("\n" + "="*60)
    print("      RESULTADO HUNYUAN3D-2 (TENCENT)")
    print("="*60)
    print(f"Vértices generados: {len(mesh.vertices):,}")
    print(f"Caras poligonales: {len(mesh.faces):,}")
    print(f"Archivo OBJ: {out_obj} ({os.path.getsize(out_obj)/1024:.1f} KB)")
    print(f"Archivo GLB: {out_glb} ({os.path.getsize(out_glb)/1024:.1f} KB)")
    print("="*60)

if __name__ == "__main__":
    main()
