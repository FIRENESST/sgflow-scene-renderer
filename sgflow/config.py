"""SGFlow 全局配置"""
import math
from dataclasses import dataclass, field

PAD_ID = 0

DEFAULT_CATEGORIES = [
    "pad", "floor", "wall", "table", "chair", "sofa", "bed", "lamp",
    "shelf", "plant", "window", "door", "rug", "tv", "desk", "cabinet",
]


@dataclass
class SGFlowConfig:
    # ---- 文本条件 ----
    text_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text_dim: int = 384
    # ---- 网络 ----
    d_model: int = 256
    n_layers: int = 4
    n_latents: int = 32          # 潜瓶颈大小 K：文本注入与物体读取都经过它
    d_state: int = 16            # SSM 状态维度
    expand: int = 2              # SSM 通道扩展因子
    # ---- 场景 ----
    max_objects: int = 128
    max_generated_objects: int | None = None  # 默认 min(32, max_objects)，可显式放宽
    d_appearance: int = 16
    categories: list = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    # ---- 流匹配 ----
    flow_steps: int = 16         # 采样步数（DDPM 通常需 1000）
    # ---- 约束 ----
    room_size: tuple = (8.0, 8.0, 4.0)
    w_collision: float = 1.0
    w_boundary: float = 1.0
    w_support: float = 0.5
    # ---- 纹理策略（二选一）----
    texture_mode: str = "generated"        # "generated"=模型生成 | "library"=接纹理库
    texture_lib: str = "textures_lib"      # 库目录：<lib>/<category>/albedo.png[+rough.png+normal.png]
    texture_batch_size: int = 16
    texture_size_limit: int = 4096
    texture_train_size: int = 16            # 训练正则使用的小尺寸可微预览
    # ---- GPU 优化开关 ----
    use_amp: bool = True                   # 训练自动混合精度（仅 CUDA 生效）
    use_compile: bool = True               # torch.compile 图融合（仅 CUDA 生效）
    ssm_chunk: int = 64                    # PyTorch SSM 参考扫描的循环分组大小

    def __post_init__(self):
        positive_ints = (
            "text_dim", "d_model", "n_layers", "n_latents", "d_state", "expand",
            "max_objects", "d_appearance", "flow_steps", "ssm_chunk",
            "texture_batch_size", "texture_size_limit", "texture_train_size",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % 8:
            raise ValueError(f"d_model must be divisible by 8, got {self.d_model}")
        if self.max_generated_objects is None:
            self.max_generated_objects = min(32, self.max_objects)
        if (
            isinstance(self.max_generated_objects, bool)
            or not isinstance(self.max_generated_objects, int)
            or not 1 <= self.max_generated_objects <= self.max_objects
        ):
            raise ValueError("max_generated_objects must be in [1, max_objects]")
        if self.texture_size_limit < 2:
            raise ValueError("texture_size_limit must be at least 2")
        if self.texture_train_size < 2 or self.texture_train_size > self.texture_size_limit:
            raise ValueError("texture_train_size must be in [2, texture_size_limit]")
        if len(self.room_size) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v <= 0
            for v in self.room_size
        ):
            raise ValueError("room_size must contain three positive dimensions")
        if not self.categories or self.categories[PAD_ID] != "pad":
            raise ValueError("categories[PAD_ID] must be 'pad'")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must be unique")
        if len(self.categories) < 2 or any(not isinstance(name, str) or not name for name in self.categories):
            raise ValueError("categories must contain PAD and at least one non-empty object category")
        if self.texture_mode not in {"generated", "library"}:
            raise ValueError("texture_mode must be 'generated' or 'library'")
        if not isinstance(self.text_model, str) or not self.text_model:
            raise ValueError("text_model must be a non-empty string")
        if not isinstance(self.texture_lib, str) or not self.texture_lib:
            raise ValueError("texture_lib must be a non-empty path")
        for name in ("w_collision", "w_boundary", "w_support"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        for name in ("use_amp", "use_compile"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    @property
    def latent_dim(self) -> int:
        # 平移(3) + 旋转6D(6) + log尺寸(3) + 外观(A)
        return 3 + 6 + 3 + self.d_appearance
