"""
AI-Powered 3D Model Creator - Background Processor Module
Automated background removal and foreground centering/padding using rembg.
"""
import numpy as np
from PIL import Image

def remove_background(image: Image.Image, threshold: int = 10) -> Image.Image:
    """
    Removes background from an image.
    Uses rembg if available, with robust alpha-based fallback.
    """
    try:
        import rembg
        # If image does not have alpha, convert to RGBA
        img_rgba = image.convert("RGBA")
        result = rembg.remove(img_rgba)
        return result
    except Exception as e:
        print(f"[WARN] rembg failed ({e}), using luminance-threshold fallback...")
        img_rgba = image.convert("RGBA")
        arr = np.array(img_rgba)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        # Treat near-white or near-black backgrounds if mostly uniform
        is_near_white = (r > 240) & (g > 240) & (b > 240)
        arr[is_near_white, 3] = 0
        return Image.fromarray(arr)

def prepare_foreground(image: Image.Image, target_size: int = 512, pad_ratio: float = 0.1) -> Image.Image:
    """
    Centers the subject, pads borders, and resizes to target_size.
    Ensures optimal input for 3D reconstruction models.
    """
    image = image.convert("RGBA")
    arr = np.array(image)
    alpha = arr[:, :, 3]
    
    # Find bounding box of non-transparent pixels
    bbox = np.where(alpha > 10)
    if len(bbox[0]) == 0 or len(bbox[1]) == 0:
        return image.resize((target_size, target_size), Image.LANCZOS)
    
    y_min, y_max = np.min(bbox[0]), np.max(bbox[0])
    x_min, x_max = np.min(bbox[1]), np.max(bbox[1])
    
    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    # Compute new square canvas with padding
    w, h = cropped.size
    max_dim = max(w, h)
    padding = int(max_dim * pad_ratio)
    canvas_size = max_dim + 2 * padding
    
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2
    canvas.paste(cropped, (paste_x, paste_y))
    
    return canvas.resize((target_size, target_size), Image.LANCZOS)
