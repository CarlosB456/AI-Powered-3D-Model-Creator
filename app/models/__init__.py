"""
AI-Powered 3D Model Creator Model Registry
"""
from typing import Dict, Type
from .base import Base3DModel
from .hunyuan3d_model import Hunyuan3DModel
from .hunyuan3d_mv_model import Hunyuan3DMultiViewModel
from .triposr_model import TripoSRModel

MODEL_REGISTRY: Dict[str, Type[Base3DModel]] = {
    "Hunyuan3D-2 Multi-View Turbo": Hunyuan3DMultiViewModel,
    "Hunyuan3D-2 Turbo": Hunyuan3DModel,
    "TripoSR": TripoSRModel,
}

_LOADED_INSTANCES: Dict[str, Base3DModel] = {}

def list_models() -> list[str]:
    """Returns list of registered model names."""
    return list(MODEL_REGISTRY.keys())

def get_model(name: str) -> Base3DModel:
    """Gets or instantiates a registered model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconocido: '{name}'. Disponibles: {list_models()}")
    
    if name not in _LOADED_INSTANCES:
        _LOADED_INSTANCES[name] = MODEL_REGISTRY[name]()
    
    return _LOADED_INSTANCES[name]

def unload_all_models() -> None:
    """Unloads all cached models to free VRAM."""
    for model in _LOADED_INSTANCES.values():
        model.unload()
    _LOADED_INSTANCES.clear()

get_available_models = list_models

__all__ = [
    "Base3DModel",
    "Hunyuan3DModel",
    "Hunyuan3DMultiViewModel",
    "TripoSRModel",
    "list_models",
    "get_available_models",
    "get_model",
    "unload_all_models",
]
