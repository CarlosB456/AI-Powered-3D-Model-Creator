import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import torch

def get_best_device(force_device=None):
    if force_device:
        return force_device
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        import torch_directml
        d = torch_directml.device()
        print(f"[INFO] AMD GPU detectada via DirectML: {torch_directml.device_name(0)}")
        return d
    except Exception as e:
        print(f"[INFO] DirectML no disponible ({e}), utilizando CPU.")
        return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Prueba de generación Image-to-3D")
    parser.add_argument("--image", default="examples/chair.png", help="Ruta de la imagen de entrada")
    parser.add_argument("--device", default=None, help="Dispositivo (cpu, dml, cuda:0)")
    parser.add_argument("--mc-res", type=int, default=256, help="Resolución de Marching Cubes")
    parser.add_argument("--output-dir", default="output/test", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_best_device(args.device)
    print(f"[INFO] Dispositivo seleccionado: {device}")

    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    t0 = time.time()
    print("[1/5] Cargando modelo TripoSR...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(8192)
    print(f"      Modelo cargado en {time.time() - t0:.2f}s")

    test_device = device
    try:
        model.to(device)
    except Exception as err:
        print(f"[WARN] Error al mover modelo a {device}: {err}. Usando CPU...")
        model.to("cpu")
        test_device = "cpu"

    print(f"[2/5] Procesando imagen de entrada: {args.image}")
    input_img = Image.open(args.image)
    if input_img.mode == "RGBA":
        img_np = np.array(input_img).astype(np.float32) / 255.0
        img_np = img_np[:, :, :3] * img_np[:, :, 3:4] + (1 - img_np[:, :, 3:4]) * 0.5
        processed_img = Image.fromarray((img_np * 255.0).astype(np.uint8))
    else:
        processed_img = remove_background(input_img.convert("RGB"))
        processed_img = resize_foreground(processed_img, 0.85)
        img_np = np.array(processed_img).astype(np.float32) / 255.0
        img_np = img_np[:, :, :3] * img_np[:, :, 3:4] + (1 - img_np[:, :, 3:4]) * 0.5
        processed_img = Image.fromarray((img_np * 255.0).astype(np.uint8))

    processed_path = os.path.join(args.output_dir, "input_processed.png")
    processed_img.save(processed_path)
    print(f"      Imagen preparada guardada en: {processed_path}")

    print("[3/5] Ejecutando inferencia (Image-to-3D Triplane)...")
    t_inf = time.time()
    with torch.no_grad():
        try:
            scene_codes = model([processed_img], device=test_device)
        except Exception as e:
            print(f"[WARN] Inferencia en {test_device} no soportó ciertos operadores ({e}). Reintentando en CPU...")
            model.to("cpu")
            test_device = "cpu"
            scene_codes = model([processed_img], device="cpu")
    inf_time = time.time() - t_inf
    print(f"      Inferencia completada en {inf_time:.2f}s!")

    print(f"[4/5] Extrayendo malla 3D (Marching Cubes res={args.mc_res})...")
    t_mc = time.time()
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=args.mc_res)
    mesh = meshes[0]
    mc_time = time.time() - t_mc
    print(f"      Malla extraída en {mc_time:.2f}s: {len(mesh.vertices)} vértices, {len(mesh.faces)} caras")

    print("[5/5] Exportando archivos 3D...")
    obj_path = os.path.join(args.output_dir, "model.obj")
    glb_path = os.path.join(args.output_dir, "model.glb")
    
    mesh.export(obj_path)
    mesh.export(glb_path)

    obj_size_kb = os.path.getsize(obj_path) / 1024
    glb_size_kb = os.path.getsize(glb_path) / 1024

    print("\n" + "="*60)
    print("      RESULTADOS DE LA PRUEBA IMAGE-TO-3D")
    print("="*60)
    print(f"Dispositivo utilizado: {test_device}")
    print(f"Tiempo de inferencia:  {inf_time:.2f} s")
    print(f"Tiempo de extracción:  {mc_time:.2f} s")
    print(f"Tiempo total:          {time.time() - t0:.2f} s")
    print(f"Vértices 3D:           {len(mesh.vertices):,}")
    print(f"Caras poligonales:     {len(mesh.faces):,}")
    print(f"Archivo OBJ generado:  {obj_path} ({obj_size_kb:.1f} KB)")
    print(f"Archivo GLB generado:  {glb_path} ({glb_size_kb:.1f} KB)")
    print("="*60)
    print("[EXITO] ¡Modelo 3D generado correctamente!")

if __name__ == "__main__":
    main()
