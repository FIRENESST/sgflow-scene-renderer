"""SGFlow：结构-几何解耦的场景图流生成网络（一句话 -> 三维场景）"""
from .config import SGFlowConfig
from .checkpoint import CheckpointError, load_checkpoint, read_checkpoint, save_checkpoint
from .pipeline import ScenePipeline
from .scene_graph import SceneGraph
from .texhead import TexHead

__all__ = [
    "SGFlowConfig",
    "CheckpointError",
    "load_checkpoint",
    "read_checkpoint",
    "save_checkpoint",
    "SceneGraph",
    "ScenePipeline",
    "TexAssetExporter",
    "TexHead",
]


def __getattr__(name):
    # Keep ``python -m sgflow.tex_assets`` free of runpy's already-imported
    # warning while preserving ``from sgflow import TexAssetExporter``.
    if name == "TexAssetExporter":
        from .tex_assets import TexAssetExporter
        return TexAssetExporter
    raise AttributeError(name)
