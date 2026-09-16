"""
AI-Powered 3D Model Creator - Unified Command Line Interface (CLI)
Usage:
    # Single image generation:
    python app/cli.py image.png [--model "Hunyuan3D-2 Turbo"|"TripoSR"] [--calidad ultra] [--rembg] [--decimate 35000]

    # Multi-angle generation:
    python app/cli.py --front front.png --left left.png --back back.png [--calidad ultra]
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
    parser.add_argument("image", nargs="?", default=None, help="Ruta de la imagen de entrada (PNG o JPG)")
    parser.add_argument("--front", default=None, help="Ruta de la imagen frontal (0°)")
    parser.add_argument("--left", default=None, help="Ruta de la imagen lateral izquierda (90°)")
    parser.add_argument("--back", default=None, help="Ruta de la imagen posterior / trasera (180°)")
    parser.add_argument("--right", default=None, help="Ruta de la imagen lateral derecha (270°)")
    parser.add_argument("--model", choices=list_models(), default="Hunyuan3D-2 Turbo",
                        help="Modelo a utilizar (default: Hunyuan3D-2 Turbo)")
    parser.add_argument("--calidad", default="ultra",
                        help="Preset de calidad (ultra, rapida, media, alta)")
    parser.add_argument("--rembg", action="store_true", default=True,
                        help="Remover fondo automaticamente con rembg (activado por defecto)")
    parser.add_argument("--no-rembg", dest="rembg", action="store_false",
                        help="Desactivar remocion de fondo")
    parser.add_argument("--decimate", type=int, default=0,
                        help="Numero objetivo de caras para simplificar la malla (0 = mantener original)")
    parser.add_argument("--output-dir", default="output",
                        help="Directorio de guardado (default: output/)")
    parser.add_argument("--name", default=None,
                        help="Nombre base del archivo de salida (sin extension)")
    args = parser.parse_args()

    # Determine input mode
    raw_images = {}
    if args.front: raw_images['front'] = args.front
    elif args.image: raw_images['front'] = args.image
    if args.left: raw_images['left'] = args.left
    if args.back: raw_images['back'] = args.back
    if args.right: raw_images['right'] = args.right

    if not raw_images:
        print("[Error] Debes especificar al menos una imagen (ej: python app/cli.py imagen.png o --front imagen.png)")
        sys.exit(1)

    for tag, p in raw_images.items():
        if not os.path.exists(p):
            print(f"[Error] No se encontro la imagen para vista '{tag}': {p}")
            sys.exit(1)

    model_to_use = args.model
    is_multi = len(raw_images) > 1 or args.front is not None and len(raw_images) > 1
    if is_multi and "Multi-View" not in model_to_use:
        model_to_use = "Hunyuan3D-2 Multi-View Turbo"

    gpu_name = torch_directml.device_name(0)
    print("=" * 65)
    print("  AI-Powered 3D Model Creator CLI")
    print(f"  GPU:     {gpu_name}")
    print(f"  Modelo:  {model_to_use}")
    print(f"  Calidad: {args.calidad}")
    print(f"  Vistas:  {list(raw_images.keys())}")
    print("=" * 65)

    t0 = time.time()

    # Preprocessing
    processed_images = {}
    print("[1/4] Procesando imagenes (rembg + centrado)...", flush=True)
    for tag, p in raw_images.items():
        pil = Image.open(p).convert("RGBA")
        if args.rembg:
            pil = remove_background(pil)
        pil = prepare_foreground(pil, target_size=512)
        processed_images[tag] = pil

    # 3D Generation
    print(f"[2/4] Generando malla 3D con {model_to_use} ({args.calidad})...", flush=True)
    model = get_model(model_to_use)

    if "Multi-View" in model_to_use:
        mesh = model.generate(processed_images, quality=args.calidad)
    else:
        single_pil = processed_images.get('front', next(iter(processed_images.values())))
        mesh = model.generate(single_pil, quality=args.calidad)

    # Postprocessing
    print("[3/4] Optimizando y normalizando geometria...", flush=True)
    caras_ini = len(mesh.faces)
    mesh = normalize_mesh(mesh)

    if args.decimate > 0 and args.decimate < caras_ini:
        print(f"      Reduciendo poligonos: {caras_ini:,} -> {args.decimate:,} caras...", flush=True)
        mesh = decimate_mesh(mesh, target_faces=args.decimate)

    # Export
    print("[4/4] Exportando archivos 3D...", flush=True)
    base_name = args.name or f"mesh_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    glb_path, obj_path = export_mesh_files(mesh, args.output_dir, base_name)

    t_total = time.time() - t0
    glb_kb = os.path.getsize(glb_path) / 1024

    print("=" * 65)
    print("  Generacion completada exitosamente!")
    print(f"  Vertices:     {len(mesh.vertices):,}")
    print(f"  Caras:        {len(mesh.faces):,}")
    print(f"  Archivo GLB:  {glb_path} ({glb_kb:.1f} KB)")
    print(f"  Archivo OBJ:  {obj_path}")
    print(f"  Tiempo total: {t_total:.1f} segundos")
    print("=" * 65)

if __name__ == "__main__":
    main()
