"""
models/__init__.py
Clean imports for all SGLDv2 model components.
Raises ImportError with actionable messages if a component fails to load.
"""

import sys


def _import_structure_extractor():
    try:
        from .structure_gan import StructureExtractor
        return StructureExtractor
    except ImportError as e:
        raise ImportError(
            f"Failed to import StructureExtractor: {e}\n"
            "  Make sure opencv-python is installed: pip install opencv-python"
        ) from e
    except Exception as e:
        raise ImportError(f"Unexpected error importing structure_gan: {e}") from e


def _import_sketch_adapter():
    try:
        from .sketch_adapter import SketchAdapter, SketchTokenEncoder
        return SketchAdapter, SketchTokenEncoder
    except ImportError as e:
        raise ImportError(
            f"Failed to import SketchAdapter: {e}\n"
            "  Make sure torch is installed: pip install torch"
        ) from e
    except Exception as e:
        raise ImportError(f"Unexpected error importing sketch_adapter: {e}") from e


def _import_diffusion_models():
    try:
        from .diffusion_model import SGLDv2Model, SGLDv2InferencePipeline
        return SGLDv2Model, SGLDv2InferencePipeline
    except ImportError as e:
        raise ImportError(
            f"Failed to import diffusion model: {e}\n"
            "  Make sure diffusers and transformers are installed:\n"
            "  pip install diffusers transformers accelerate"
        ) from e
    except Exception as e:
        raise ImportError(
            f"Unexpected error importing diffusion_model: {e}"
        ) from e


# Run all imports at module load time with clear error messages
try:
    StructureExtractor                    = _import_structure_extractor()
    SketchAdapter, SketchTokenEncoder     = _import_sketch_adapter()
    SGLDv2Model, SGLDv2InferencePipeline  = _import_diffusion_models()
except ImportError as e:
    print(f"\n[Import Error] {e}")
    print("  Install missing packages:  pip install -r requirements.txt\n")
    raise

__all__ = [
    "StructureExtractor",
    "SketchAdapter",
    "SketchTokenEncoder",
    "SGLDv2Model",
    "SGLDv2InferencePipeline",
]