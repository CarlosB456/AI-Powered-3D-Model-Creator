"""
AI-Powered 3D Model Creator - Base Model Interface
Standardized interface for all 3D reconstruction models.
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from PIL import Image
import trimesh

class Base3DModel(ABC):
    """
    Abstract interface for all Image-to-3D generation models.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_loaded = False

    @abstractmethod
    def load(self, device: Optional[str] = None) -> None:
        """Loads weights and prepares models."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        quality: str = "rapida",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any
    ) -> trimesh.Trimesh:
        """
        Generates a 3D mesh from an input PIL Image.
        Returns a trimesh.Trimesh object.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Frees VRAM and clears cached tensors."""
        pass

    @abstractmethod
    def get_supported_qualities(self) -> list[str]:
        """Returns list of quality preset names."""
        pass
