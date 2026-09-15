"""
AI-Powered 3D Model Creator - Unified Command Line Interface (CLI)
Usage:
    python app/cli.py image.png [--model "Hunyuan3D-2 Turbo"|"TripoSR"] [--calidad ultra] [--rembg] [--decimate 35000]
"""
import os
import sys
import argparse
import time
from PIL import Image
import torch_directml

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.models import get_model, list_models
from app.processors import remove_background, prepare_foreground, decimate_mesh, normalize_mesh, export_mesh_files

def main():
    parser = argparse.ArgumentParser(description="AI-Powered 3D Model Creator CLI")
    parser.add_argument("image", help="Ruta de la imagen de entrada (PNG o JPG)")
    parser.add_argument("--model", choices=list_models(), default="Hunyuan3D-2 Turbo",
                        help="Modelo a utilizar (default: Hunyuan3D-2 Turbo)")
    parser.add_argument("--calidad", default="ultra",
                        help="Preset de calidad (ultra, rapida, media, alta)")
    parser.add_argument("--rembg", action="store_true", default=True,
                        help="Remover fondo automáticamente con rembg (activado por defecto)")
    parser.add_argument("--no-rembg", dest="rembg", action="store_false",
                        help="Desactivar remoción de fondo")
    parser.add_argument("--decimate", type=int, default=0,
                        help="Número objetivo de caras para simplificar la malla (0 = mantener original)")
    parser.add_argument("--output-dir", default="output",
                        help="Directorio de guardado (default: output/)")
    parser.add_argument("--name", default=None,
                        help="Nombre base del archivo de salida (sin extensión)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ ERROR: No se encontró la imagen: {args.image}")
        sys.exit(1)

    gpu_name = torch_directml.device_name(0)
    print("=" * 65)
    print("  AI-Powered 3D Model Creator CLI")
    print(f"  GPU:     {gpu_name}")
    print(f"  Modelo:  {args.model}")
    print(f"  Calidad: {args.calidad}")
    print(f"  Imagen:  {args.image}")
    print("=" * 65)

    t0 = time.time()
    img_pil = Image.open(args.image).convert("RGBA")

    # Preprocessing
    if args.rembg:
        print("[1/4] Removiendo fondo de la imagen...", flush=True)
        img_pil = remove_background(img_pil)
        img_prepared = prepare_foreground(img_pil, target_size=512)
    else:
        print("[1/4] Preparando imagen...", flush=True)
        img_prepared = prepare_foreground(img_pil, target_size=512)

    # Inference
    print(f"[2/4] Generando geometría 3D con {args.model}...", flush=True)
    model = get_model(args.model)
    mesh = model.generate(img_prepared, quality=args.calidad)

    # Postprocessing
    print(f"[3/4] Postprocesando malla ({len(mesh.vertices):,} vértices, {len(mesh.faces):,} caras)...", flush=True)
    mesh = normalize_mesh(mesh)
    if args.decimate > 0 and args.decimate < len(mesh.faces):
        mesh = decimate_mesh(mesh, target_faces=args.decimate)

    # Export
    print("[4/4] Exportando archivos 3D...", flush=True)
    base_name = args.name or f"{os.path.splitext(os.path.basename(args.image))[0]}_{args.model.lower().replace(' ', '_')}"
    glb_path, obj_path = export_mesh_files(mesh, args.output_dir, base_name)

    t_total = time.time() - t0
    glb_size = os.path.getsize(glb_path) / 1024

    print("\n" + "=" * 65)
    print("   ¡GENERACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 65)
    print(f"  Vértices:      {len(mesh.vertices):,}")
    print(f"  Caras:         {len(mesh.faces):,}")
    print(f"  Archivo GLB:   {glb_path} ({glb_size:.1f} KB)")
    print(f"  Archivo OBJ:   {obj_path}")
    print(f"  Tiempo Total:  {t_total:.1f} segundos")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
