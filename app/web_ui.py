"""
AI-Powered 3D Model Creator (RX 6600, among others)
Universal AI Image-to-3D Studio - Open Source
Created by Carlos B (eLdarqO)
"""
import os
import sys
import time
import random
from PIL import Image
import gradio as gr
import torch_directml

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

h3d_path = os.path.join(root_dir, "Hunyuan3D-2")

from app.models import get_model, list_models
from app.processors import remove_background, prepare_foreground, decimate_mesh, normalize_mesh, export_mesh_files, remove_floaters

GPU_NAME = torch_directml.device_name(0)

# Diccionario i18n
I18N = {
    "es": {
        "title": "AI-Powered 3D Model Creator",
        "subtitle": "Estudio Local de Generación Image-to-3D",
        "desc": f"Aceleración por GPU: {GPU_NAME} · DirectML FP16",
        "model_label": "Modelo de IA",
        "tab_single": "Vista Unica",
        "tab_multi": "Multi-Angulos (360°)",
        "image_label": "Imagen Frontal / Principal",
        "view_front": "Vista Frontal (0°) - Principal",
        "view_left": "Vista Izquierda (90°)",
        "view_back": "Vista Trasera (180°)",
        "view_right": "Vista Derecha (270°)",
        "multi_hint": "Sube al menos la vista frontal y trasera (o lateral) para capturar el modelo en 360° con máxima precisión.",
        "rembg_label": "Eliminar fondo automáticamente",
        "mesh_header": "Configuración de Geometría",
        "quality_label": "Calidad (Resolución de Octree y Pasos DiT)",
        "quality_info": "Ultra (~2 min, octree 128) | Rápida (~2.5 min, octree 128) | Media (~4.5 min, octree 192) | Alta (~8 min, octree 256)",
        "decimate_label": "Reducir polígonos (Decimation)",
        "target_faces_label": "Polígonos objetivo (Caras)",
        "normalize_label": "Centrar y normalizar escala",
        "adv_header": "Opciones Avanzadas (Generación y Limpieza)",
        "seed_label": "Semilla (Seed)",
        "random_seed_label": "Semilla aleatoria",
        "guidance_label": "Escala de Guía (Guidance Scale / CFG)",
        "guidance_info": "Controla fidelidad a la imagen (5.0 recomendado para Turbo)",
        "steps_label": "Sobreescribir pasos DiT (0 = usar preset)",
        "clean_floaters_label": "Eliminar fragmentos flotantes (Floater Removal)",
        "examples_single_label": "Ejemplos de Prueba Rápida (Vista Única)",
        "examples_multi_label": "Ejemplos de Prueba Rápida (Multi-Ángulo)",
        "generate_btn": "Generar Modelo 3D",
        "viewport_label": "Visor 3D Interactivo (Rotar 360° y Zoom)",
        "cutout_label": "Recorte Procesado",
        "download_glb": "Descargar GLB",
        "download_obj": "Descargar OBJ",
        "no_image_err": "Por favor sube al menos una imagen.",
        "status_generating": "Generando modelo 3D con",
        "r_title": "Modelo 3D Generado",
        "r_model": "Modelo",
        "r_views": "Modo de Vista",
        "r_gpu": "GPU",
        "r_verts": "Vértices",
        "r_faces": "Caras",
        "r_seed": "Semilla",
        "r_size": "Tamaño GLB",
        "r_time": "Tiempo Total",
        "footer": f"Creado por Carlos B (eLdarqO) · Hardware: {GPU_NAME} (8GB VRAM) · Compatible con Roblox Studio, Blender, Unity y Unreal Engine",
    },
    "en": {
        "title": "AI-Powered 3D Model Creator",
        "subtitle": "Local AI Image-to-3D Studio",
        "desc": f"GPU Acceleration: {GPU_NAME} · DirectML FP16",
        "model_label": "AI Model",
        "tab_single": "Single View",
        "tab_multi": "Multi-Angle (360°)",
        "image_label": "Front / Main Image",
        "view_front": "Front View (0°) - Primary",
        "view_left": "Left View (90°)",
        "view_back": "Back View (180°)",
        "view_right": "Right View (270°)",
        "multi_hint": "Upload at least Front and Back (or Left) views to capture 360° geometry without hallucination.",
        "rembg_label": "Auto-remove background",
        "mesh_header": "Geometry Settings",
        "quality_label": "Quality (Octree Resolution & DiT Steps)",
        "quality_info": "Ultra (~2 min, octree 128) | Fast (~2.5 min, octree 128) | Medium (~4.5 min, octree 192) | High (~8 min, octree 256)",
        "decimate_label": "Reduce polygon count (Decimation)",
        "target_faces_label": "Target polygon count (Faces)",
        "normalize_label": "Center and normalize scale",
        "adv_header": "Advanced Options (Generation & Mesh Cleanup)",
        "seed_label": "Seed",
        "random_seed_label": "Randomize Seed",
        "guidance_label": "Guidance Scale (CFG)",
        "guidance_info": "Controls fidelity to reference image (5.0 recommended for Turbo)",
        "steps_label": "Override DiT Steps (0 = use preset default)",
        "clean_floaters_label": "Remove floating mesh debris (Floater Removal)",
        "examples_single_label": "Quick Test Examples (Single View)",
        "examples_multi_label": "Quick Test Examples (Multi-Angle)",
        "generate_btn": "Generate 3D Model",
        "viewport_label": "Interactive 3D Viewport (360° Rotate & Zoom)",
        "cutout_label": "Processed Cutout",
        "download_glb": "Download GLB",
        "download_obj": "Download OBJ",
        "no_image_err": "Please upload at least one image.",
        "status_generating": "Generating 3D model with",
        "r_title": "3D Model Ready",
        "r_model": "Model",
        "r_views": "View Mode",
        "r_gpu": "GPU",
        "r_verts": "Vertices",
        "r_faces": "Faces",
        "r_seed": "Seed",
        "r_size": "GLB Size",
        "r_time": "Total Time",
        "footer": f"Created by Carlos B (eLdarqO) · Hardware: {GPU_NAME} (8GB VRAM) · Compatible with Roblox Studio, Blender, Unity, and Unreal Engine",
    }
}

# Estilos CSS
STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Reset & Global Colors ── */
html, body, .gradio-container {
    background-color: #0b0d14 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e2e8f0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
    padding: 24px 32px !important;
}

/* ── Universal Container Overrides (No white boxes) ── */
.gradio-container .bg-white,
.gradio-container [class*="bg-white"],
.gradio-container [class*="bg-gray-"],
.gradio-container [class*="bg-slate-"],
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container fieldset,
.gradio-container .block,
.gradio-container div[data-testid="box"],
.gradio-container div[data-testid="panel"] {
    background-color: #121520 !important;
    border: 1px solid #1e2336 !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
}

/* ── Typography & Headers ── */
.studio-hdr {
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #1e2336;
    margin-bottom: 18px;
}
.studio-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f8fafc;
    margin: 0 0 4px 0;
}
.studio-sub {
    font-size: 1.05rem;
    color: #a5b4fc;
    font-weight: 600;
    margin: 0 0 6px 0;
}
.studio-desc {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
}

.section-label {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Tabs Styling ── */
.tab-nav {
    border-bottom: 1px solid #1e2336 !important;
    margin-bottom: 12px !important;
}
.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 8px 16px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    color: #f8fafc !important;
    border-bottom: 2px solid #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.08) !important;
}

/* ── Labels & Text ── */
label, .gr-form label, .block label span {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}

/* ── Inputs & Dropdowns ── */
input, select, textarea, .gr-input, .gr-dropdown {
    background-color: #181c2b !important;
    border: 1px solid #282e44 !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
    font-family: inherit !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
    outline: none !important;
}

/* ── Checkboxes & Sliders ── */
input[type="checkbox"] {
    width: 18px !important;
    height: 18px !important;
    accent-color: #8b5cf6 !important;
    cursor: pointer !important;
}
.gr-check-radio {
    background-color: transparent !important;
    border: none !important;
}

input[type="range"] {
    accent-color: #8b5cf6 !important;
}

/* ── Primary Action Button ── */
#btn-generate-main {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    border: 1px solid #8b5cf6 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    font-family: inherit !important;
    letter-spacing: 0.02em !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    margin-top: 14px !important;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.35) !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
}
#btn-generate-main:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.55) !important;
    filter: brightness(1.1) !important;
}

/* ── Image & 3D Viewport Cards ── */
div[data-testid="image"],
div[data-testid="model3d"],
.model3d {
    background-color: #0e1018 !important;
    border: 1px solid #1e2336 !important;
    border-radius: 14px !important;
}

/* ── Language Switcher Box ── */
.lang-switcher-row {
    margin-bottom: -10px !important;
}
.lang-switcher-row fieldset {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Footer ── */
.studio-footer {
    padding-top: 18px;
    margin-top: 20px;
    border-top: 1px solid #1e2336;
    color: #64748b;
    font-size: 0.82rem;
    line-height: 1.5;
}
"""

def render_hdr(lang: str) -> str:
    t = I18N[lang]
    return f"""
    <div class="studio-hdr">
        <div class="studio-title">{t["title"]}</div>
        <div class="studio-sub">{t["subtitle"]}</div>
        <div class="studio-desc">{t["desc"]}</div>
    </div>
    """

def render_ftr(lang: str) -> str:
    t = I18N[lang]
    return f'<div class="studio-footer">{t["footer"]}</div>'

def ejecutar_generacion(
    img_single, img_front, img_left, img_back, img_right,
    modelo, rembg_on, calidad, dec_on, dec_target, norm_on,
    semilla_val, semilla_rand, guidance_val, custom_steps_val, limpiar_flotantes_on,
    lang
):
    t = I18N.get(lang, I18N["es"])
    t_start = time.time()
    out_dir = os.path.join(root_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Determine input mode: Multi-view or Single Image
    raw_images = {}
    if img_front is not None: raw_images['front'] = img_front
    if img_left is not None: raw_images['left'] = img_left
    if img_back is not None: raw_images['back'] = img_back
    if img_right is not None: raw_images['right'] = img_right

    is_multiview = len(raw_images) > 0
    if not is_multiview and img_single is None:
        return None, None, None, None, f"[Error] {t['no_image_err']}"

    # Auto-switch to Multi-View model if multiple views provided and default single-view model was selected
    actual_model_name = modelo
    if is_multiview and len(raw_images) > 1 and "Multi-View" not in actual_model_name:
        actual_model_name = "Hunyuan3D-2 Multi-View Turbo"
        print(f"[AI-Powered 3D Model Creator] Multiples angulos detectados ({list(raw_images.keys())}). Usando {actual_model_name}...", flush=True)

    # 2. Process images (rembg + centering)
    preview_cutout = None
    input_payload = None
    views_info = ""

    if is_multiview:
        processed_dict = {}
        for view_tag, raw_img in raw_images.items():
            pil = raw_img.convert("RGBA")
            if rembg_on:
                pil = remove_background(pil)
            pil = prepare_foreground(pil, target_size=512)
            processed_dict[view_tag] = pil
            if view_tag == 'front' or preview_cutout is None:
                preview_cutout = pil

        # If model expects single image but we are in multi-view, pass front image
        if "Multi-View" in actual_model_name:
            input_payload = processed_dict
            views_info = f"Multi-Angulo ({', '.join(processed_dict.keys())})"
        else:
            input_payload = processed_dict.get('front', next(iter(processed_dict.values())))
            views_info = f"Vista Unica (Frontal)"
    else:
        pil = img_single.convert("RGBA")
        if rembg_on:
            pil = remove_background(pil)
        pil = prepare_foreground(pil, target_size=512)
        preview_cutout = pil

        if "Multi-View" in actual_model_name:
            input_payload = {"front": pil}
            views_info = "Multi-View (Frontal)"
        else:
            input_payload = pil
            views_info = "Vista Unica"

    # Seed calculation
    if semilla_rand or semilla_val is None:
        actual_seed = random.randint(1, 10000000)
    else:
        actual_seed = int(semilla_val)

    steps_override = int(custom_steps_val) if custom_steps_val and int(custom_steps_val) > 0 else None

    # 3. Model generation
    print(f"[AI-Powered 3D Model Creator] {t['status_generating']} {actual_model_name} ({calidad}) [Seed: {actual_seed}, CFG: {guidance_val}, Pasos: {steps_override or 'preset'}]...", flush=True)
    model = get_model(actual_model_name)
    mesh = model.generate(
        input_payload,
        quality=calidad,
        seed=actual_seed,
        guidance_scale=float(guidance_val) if guidance_val is not None else 5.0,
        steps=steps_override
    )

    vi = len(mesh.vertices)
    fi = len(mesh.faces)

    if limpiar_flotantes_on:
        mesh = remove_floaters(mesh)

    if norm_on:
        mesh = normalize_mesh(mesh)
    if dec_on and dec_target > 0 and dec_target < len(mesh.faces):
        mesh = decimate_mesh(mesh, int(dec_target))

    vf = len(mesh.vertices)
    ff = len(mesh.faces)

    ts = time.strftime("%Y%m%d_%H%M%S")
    glb_path, obj_path = export_mesh_files(mesh, out_dir, f"mesh_{ts}")
    t_total = time.time() - t_start
    glb_kb = os.path.getsize(glb_path) / 1024

    info_md = f"""### {t['r_title']}

| Metrica / Metric | Detalle / Detail |
|---|---|
| **{t['r_model']}** | `{actual_model_name}` |
| **{t['r_views']}** | `{views_info}` |
| **{t['r_gpu']}** | `{GPU_NAME}` |
| **{t['r_seed']}** | `{actual_seed}` |
| **{t['r_verts']}** | **{vf:,}** (original: {vi:,}) |
| **{t['r_faces']}** | **{ff:,}** (original: {fi:,}) |
| **{t['r_size']}** | **{glb_kb:.1f} KB** |
| **{t['r_time']}** | **{t_total:.1f} segundos** |
"""
    return preview_cutout, glb_path, glb_path, obj_path, info_md

def cambiar_idioma(sel):
    lang = "es" if "Español" in sel else "en"
    t = I18N[lang]
    return (
        lang,
        render_hdr(lang),
        gr.Dropdown.update(label=t["model_label"]),
        gr.Image.update(label=t["image_label"]),
        gr.Image.update(label=t["view_front"]),
        gr.Image.update(label=t["view_left"]),
        gr.Image.update(label=t["view_back"]),
        gr.Image.update(label=t["view_right"]),
        gr.Checkbox.update(label=t["rembg_label"]),
        gr.Dropdown.update(label=t["quality_label"], info=t["quality_info"]),
        gr.Checkbox.update(label=t["decimate_label"]),
        gr.Slider.update(label=t["target_faces_label"]),
        gr.Checkbox.update(label=t["normalize_label"]),
        gr.Accordion.update(label=t["adv_header"]),
        gr.Number.update(label=t["seed_label"]),
        gr.Checkbox.update(label=t["random_seed_label"]),
        gr.Slider.update(label=t["guidance_label"], info=t["guidance_info"]),
        gr.Slider.update(label=t["steps_label"]),
        gr.Checkbox.update(label=t["clean_floaters_label"]),
        gr.Button.update(value=t["generate_btn"]),
        gr.Model3D.update(label=t["viewport_label"]),
        gr.Image.update(label=t["cutout_label"]),
        gr.File.update(label=t["download_glb"]),
        gr.File.update(label=t["download_obj"]),
        render_ftr(lang)
    )

def actualizar_calidad(modelo, lang):
    t = I18N.get(lang, I18N["es"])
    if "Hunyuan" in modelo:
        return gr.Dropdown.update(
            choices=['ultra', 'rapida', 'media', 'alta'],
            value='ultra',
            label=t["quality_label"],
            info=t["quality_info"]
        )
    return gr.Dropdown.update(
        choices=['rapida', 'media', 'alta'],
        value='media',
        label=t["quality_label"],
        info="TripoSR: Rapida (~10s) | Media (~15s) | Alta (~20s)"
    )

# Interfaz Gradio
with gr.Blocks(
    title="AI-Powered 3D Model Creator",
    css=STUDIO_CSS
) as app:

    lang_state = gr.State(value="es")

    # Language Switcher
    with gr.Row(elem_classes=["lang-switcher-row"]):
        with gr.Column(scale=4):
            pass
        with gr.Column(scale=1, min_width=180):
            lang_radio = gr.Radio(
                choices=["Español", "English"],
                value="Español",
                label="",
                interactive=True
            )

    # Hero Header
    header_html = gr.HTML(value=render_hdr("es"))

    with gr.Row():
        # Left Column: Input and Controls
        with gr.Column(scale=1):
            with gr.Box():
                modelo_selector = gr.Dropdown(
                    choices=list_models(),
                    value=list_models()[0],
                    label=I18N["es"]["model_label"],
                    interactive=True
                )

                # Input Mode Tabs: Single View vs Multi-Angle
                with gr.Tabs() as input_tabs:
                    with gr.Tab(I18N["es"]["tab_single"]):
                        imagen_input_single = gr.Image(
                            label=I18N["es"]["image_label"],
                            type="pil",
                            height=280
                        )
                        ex_single_paths = [
                            os.path.join(h3d_path, "assets", "example_images", "052.png"),
                            os.path.join(h3d_path, "assets", "example_images", "101.png"),
                            os.path.join(h3d_path, "assets", "example_images", "1123.png"),
                            os.path.join(h3d_path, "assets", "example_images", "1493.png"),
                        ]
                        ex_single_valid = [[p] for p in ex_single_paths if os.path.exists(p)]
                        if ex_single_valid:
                            gr.Examples(
                                examples=ex_single_valid,
                                inputs=[imagen_input_single],
                                label=I18N["es"]["examples_single_label"],
                                examples_per_page=4
                            )

                    with gr.Tab(I18N["es"]["tab_multi"]):
                        gr.HTML(f'<div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px;">{I18N["es"]["multi_hint"]}</div>')
                        with gr.Row():
                            imagen_front = gr.Image(label=I18N["es"]["view_front"], type="pil", height=150)
                            imagen_left = gr.Image(label=I18N["es"]["view_left"], type="pil", height=150)
                        with gr.Row():
                            imagen_back = gr.Image(label=I18N["es"]["view_back"], type="pil", height=150)
                            imagen_right = gr.Image(label=I18N["es"]["view_right"], type="pil", height=150)
                        ex_multi_paths = [
                            [
                                os.path.join(h3d_path, "assets", "example_mv_images", "1", "front.png"),
                                os.path.join(h3d_path, "assets", "example_mv_images", "1", "left.png"),
                                os.path.join(h3d_path, "assets", "example_mv_images", "1", "back.png"),
                                None
                            ],
                            [
                                os.path.join(h3d_path, "assets", "example_mv_images", "2", "front.png"),
                                os.path.join(h3d_path, "assets", "example_mv_images", "2", "left.png"),
                                os.path.join(h3d_path, "assets", "example_mv_images", "2", "back.png"),
                                None
                            ],
                        ]
                        ex_multi_valid = [p for p in ex_multi_paths if os.path.exists(p[0])]
                        if ex_multi_valid:
                            gr.Examples(
                                examples=ex_multi_valid,
                                inputs=[imagen_front, imagen_left, imagen_back, imagen_right],
                                label=I18N["es"]["examples_multi_label"],
                                examples_per_page=2
                            )

                quitar_fondo_cb = gr.Checkbox(
                    value=True,
                    label=I18N["es"]["rembg_label"]
                )

            with gr.Box():
                gr.HTML(f'<div class="section-label">{I18N["es"]["mesh_header"]}</div>')
                calidad_dropdown = gr.Dropdown(
                    choices=['ultra', 'rapida', 'media', 'alta'],
                    value='ultra',
                    label=I18N["es"]["quality_label"],
                    info=I18N["es"]["quality_info"]
                )
                reducir_poligonos_cb = gr.Checkbox(
                    value=False,
                    label=I18N["es"]["decimate_label"]
                )
                poligonos_slider = gr.Slider(
                    minimum=2000,
                    maximum=100000,
                    step=2000,
                    value=35000,
                    label=I18N["es"]["target_faces_label"]
                )
                normalizar_cb = gr.Checkbox(
                    value=True,
                    label=I18N["es"]["normalize_label"]
                )

            with gr.Accordion(I18N["es"]["adv_header"], open=False) as adv_accordion:
                with gr.Row():
                    semilla_input = gr.Number(value=1234, label=I18N["es"]["seed_label"], precision=0)
                    semilla_rand_cb = gr.Checkbox(value=True, label=I18N["es"]["random_seed_label"])
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=15.0, step=0.5, value=5.0,
                    label=I18N["es"]["guidance_label"],
                    info=I18N["es"]["guidance_info"]
                )
                pasos_slider = gr.Slider(
                    minimum=0, maximum=30, step=1, value=0,
                    label=I18N["es"]["steps_label"]
                )
                limpiar_flotantes_cb = gr.Checkbox(
                    value=True,
                    label=I18N["es"]["clean_floaters_label"]
                )

            generar_btn = gr.Button(
                I18N["es"]["generate_btn"],
                variant="primary",
                size="lg",
                elem_id="btn-generate-main"
            )

        # Right Column: Viewport & Results
        with gr.Column(scale=1):
            with gr.Box():
                visor_3d = gr.Model3D(
                    label=I18N["es"]["viewport_label"],
                    height=440
                )

            with gr.Row():
                preview_imagen = gr.Image(
                    label=I18N["es"]["cutout_label"],
                    type="pil",
                    height=160
                )
                with gr.Column():
                    descarga_glb = gr.File(label=I18N["es"]["download_glb"])
                    descarga_obj = gr.File(label=I18N["es"]["download_obj"])

            with gr.Box():
                info_resultado = gr.Markdown(label="Resumen")

    footer_html = gr.HTML(value=render_ftr("es"))

    # Force dark mode via JS execution on page load
    app.load(_js="""() => {
        document.documentElement.classList.add('dark');
        document.body.classList.add('dark');
        try { localStorage.setItem('theme', 'dark'); } catch(e) {}
    }""")

    # Events
    lang_radio.change(
        fn=cambiar_idioma,
        inputs=[lang_radio],
        outputs=[
            lang_state,
            header_html,
            modelo_selector,
            imagen_input_single,
            imagen_front,
            imagen_left,
            imagen_back,
            imagen_right,
            quitar_fondo_cb,
            calidad_dropdown,
            reducir_poligonos_cb,
            poligonos_slider,
            normalizar_cb,
            adv_accordion,
            semilla_input,
            semilla_rand_cb,
            guidance_slider,
            pasos_slider,
            limpiar_flotantes_cb,
            generar_btn,
            visor_3d,
            preview_imagen,
            descarga_glb,
            descarga_obj,
            footer_html
        ]
    )

    modelo_selector.change(
        fn=actualizar_calidad,
        inputs=[modelo_selector, lang_state],
        outputs=[calidad_dropdown]
    )

    generar_btn.click(
        fn=ejecutar_generacion,
        inputs=[
            imagen_input_single,
            imagen_front,
            imagen_left,
            imagen_back,
            imagen_right,
            modelo_selector,
            quitar_fondo_cb,
            calidad_dropdown,
            reducir_poligonos_cb,
            poligonos_slider,
            normalizar_cb,
            semilla_input,
            semilla_rand_cb,
            guidance_slider,
            pasos_slider,
            limpiar_flotantes_cb,
            lang_state
        ],
        outputs=[
            preview_imagen,
            visor_3d,
            descarga_glb,
            descarga_obj,
            info_resultado
        ]
    )

def main():
    print("=" * 65)
    print("  AI-Powered 3D Model Creator (RX 6600, among others)")
    print(f"  Creator: Carlos B (eLdarqO)")
    print(f"  GPU: {GPU_NAME}")
    print("=" * 65)
    print("  Iniciando estudio web en http://localhost:7860/?__theme=dark ...\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
