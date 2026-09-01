"""参数化物体建模：按类别模板把物体拆成 primitive 部件，按全局精细等级重建。

设计约定：
- 每个部件是一个 axis-aligned 的盒子 / UV 球 / 圆柱，用局部坐标（物体中心为原点）
  表示，再按物体的 OBB 位姿放置。
- ``DETAIL_LEVELS`` 把全局 1-5 等级映射为 (parts_limit, subdiv, smooth) 三元组；
  等级越高保留的部件越多、球/柱的细分越高、越倾向平滑着色。
- 细节描述 ``detail`` 是一个可选 dict：``{"parts": [...], "smooth": bool}``，
  由 LLM 或本地检查点给出；缺省时按类别模板生成。
"""
from __future__ import annotations

import math
from typing import Any

# 全局精细等级 1-5 -> (部件数上限, 球/柱细分级数, 是否平滑着色)
DETAIL_LEVELS: dict[int, dict[str, Any]] = {
    1: {"parts_limit": 1, "subdiv": 0, "smooth": False},
    2: {"parts_limit": 3, "subdiv": 1, "smooth": False},
    3: {"parts_limit": 6, "subdiv": 2, "smooth": True},
    4: {"parts_limit": 12, "subdiv": 2, "smooth": True},
    5: {"parts_limit": None, "subdiv": 3, "smooth": True},
}

DEFAULT_LEVEL = 3

# primitive 类型 -> 默认细分基数（越高等级在其上乘细分增益）
_PRIMITIVES = {"box", "sphere", "cylinder", "cone"}


def clamp_level(level: Any) -> int:
    """把外部输入收敛到合法等级 1-5。"""
    try:
        value = int(level)
    except (TypeError, ValueError):
        return DEFAULT_LEVEL
    return max(1, min(5, value))


def _part(kind: str, offset, size, **extra) -> dict[str, Any]:
    if kind not in _PRIMITIVES:
        raise ValueError(f"unknown part kind {kind!r}")
    return {"kind": kind, "offset": tuple(offset), "size": tuple(size), **extra}


# ---- 类别模板：返回局部空间（半尺寸=0.5 归一化到物体 OBB）的部件列表 ----
# 坐标约定：局部 XY 平面是物体占地，+Z 向上；size 是占整体 OBB 的比例 (0-1)。

def _template_box(_size) -> list[dict[str, Any]]:
    return [_part("box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]


def _template_table(_size) -> list[dict[str, Any]]:
    # 归一化：offset/size 除以 OBB 半尺寸 (size/2)，因此 0.5 = 半尺寸，1.0 = 全尺寸
    parts = [
        _part("box", (0.0, 0.0, 0.49), (1.0, 1.0, 0.08)),            # 桌面
    ]
    for sxn in (-1, 1):
        for syn in (-1, 1):
            parts.append(_part("cylinder",
                               (sxn * 0.40, syn * 0.40, 0.225),
                               (0.08, 0.08, 0.45)))
    return parts


def _template_chair(_size) -> list[dict[str, Any]]:
    parts = [
        _part("box", (0.0, 0.0, 0.45), (1.0, 1.0, 0.08)),            # 座面
        _part("box", (0.0, 0.45, 0.70), (1.0, 0.07, 0.60)),          # 靠背
    ]
    for sxn in (-1, 1):
        for syn in (-1, 1):
            parts.append(_part("cylinder",
                               (sxn * 0.42, syn * 0.42, 0.225),
                               (0.07, 0.07, 0.45)))
    return parts


def _template_bed(_size) -> list[dict[str, Any]]:
    parts = [
        _part("box", (0.0, 0.0, 0.25), (1.0, 1.0, 0.50)),            # 床体
        _part("box", (0.0, 0.48, 0.60), (1.0, 0.06, 0.70)),          # 床头
        _part("box", (0.0, -0.10, 0.55), (0.90, 0.80, 0.15)),        # 床垫
    ]
    return parts


def _template_lamp(_size) -> list[dict[str, Any]]:
    return [
        _part("cylinder", (0.0, 0.0, 0.05), (0.50, 0.50, 0.10)),     # 底座
        _part("cylinder", (0.0, 0.0, 0.50), (0.03, 0.03, 0.90)),     # 杆
        _part("cone", (0.0, 0.0, 0.90), (0.30, 0.30, 0.25)),         # 灯罩
    ]


def _template_sofa(_size) -> list[dict[str, Any]]:
    parts = [
        _part("box", (0.0, 0.0, 0.35), (1.0, 1.0, 0.40)),            # 座体
        _part("box", (0.0, 0.45, 0.60), (1.0, 0.12, 0.60)),          # 靠背
    ]
    for sxn in (-1, 1):
        parts.append(_part("box", (sxn * 0.47, 0.0, 0.50),
                           (0.08, 1.0, 0.50)))                        # 扶手
    return parts


def _template_quadruped(_size) -> list[dict[str, Any]]:
    """四足动物（monkey/dog/cat/rabbit/horse 等）的通用体块。"""
    parts = [
        _part("sphere", (0.0, 0.0, 0.55), (0.50, 0.35, 0.30)),       # 躯干
        _part("sphere", (0.0, 0.42, 0.75), (0.22, 0.22, 0.22)),      # 头
    ]
    for sxn in (-1, 1):
        for syn in (-1, 1):
            parts.append(_part("cylinder",
                               (sxn * 0.30, syn * 0.30, 0.30),
                               (0.08, 0.08, 0.60)))                   # 腿
    parts.append(_part("cylinder", (0.0, -0.48, 0.60),
                       (0.05, 0.30, 0.05)))                          # 尾
    return parts


def _template_person(_size) -> list[dict[str, Any]]:
    return [
        _part("sphere", (0.0, 0.0, 0.92), (0.16, 0.16, 0.14)),       # 头
        _part("box", (0.0, 0.0, 0.55), (0.40, 0.24, 0.50)),          # 躯干
        _part("cylinder", (-0.12, 0.0, 0.25), (0.10, 0.10, 0.50)),   # 左腿
        _part("cylinder", (0.12, 0.0, 0.25), (0.10, 0.10, 0.50)),    # 右腿
    ]


_TEMPLATES = {
    "table": _template_table,
    "dining_table": _template_table,
    "coffee_table": _template_table,
    "desk": _template_table,
    "chair": _template_chair,
    "armchair": _template_chair,
    "stool": _template_chair,
    "bench": _template_chair,
    "bed": _template_bed,
    "lamp": _template_lamp,
    "floor_lamp": _template_lamp,
    "ceiling_lamp": _template_lamp,
    "sofa": _template_sofa,
    "monkey": _template_quadruped,
    "dog": _template_quadruped,
    "cat": _template_quadruped,
    "rabbit": _template_quadruped,
    "horse": _template_quadruped,
    "person": _template_person,
}


def build_detail(category: str, size, detail: dict[str, Any] | None) -> dict[str, Any]:
    """归一化一个物体的细节描述，返回 {"parts": [...], "smooth": bool}。

    ``detail`` 为 None 或缺字段时回落到类别模板；模板也没有则给单个盒子。
    """
    if isinstance(detail, dict) and isinstance(detail.get("parts"), list) and detail["parts"]:
        parts = []
        for index, raw in enumerate(detail["parts"]):
            if not isinstance(raw, dict):
                raise ValueError(f"detail.parts[{index}] must be an object")
            kind = raw.get("kind")
            if kind not in _PRIMITIVES:
                raise ValueError(f"detail.parts[{index}].kind must be one of {sorted(_PRIMITIVES)}")
            parts.append(_part(
                kind,
                _vec(raw.get("offset"), 3, f"detail.parts[{index}].offset"),
                _vec(raw.get("size"), 3, f"detail.parts[{index}].size", positive=True),
            ))
        return {"parts": parts, "smooth": bool(detail.get("smooth", True))}
    template = _TEMPLATES.get(category, _template_box)
    return {"parts": template(size), "smooth": True}


def apply_level(detail: dict[str, Any], level: int) -> dict[str, Any]:
    """按全局等级裁剪部件数量、细分和平滑标记。"""
    spec = DETAIL_LEVELS[clamp_level(level)]
    parts = detail["parts"]
    limit = spec["parts_limit"]
    if limit is not None:
        parts = parts[:limit]
    return {
        "parts": parts,
        "smooth": bool(detail.get("smooth", True)) and spec["smooth"],
        "subdiv": spec["subdiv"],
    }


def _vec(value, width, label, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} numbers")
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{label} contains non-finite values")
    if positive and any(v <= 0 for v in out):
        raise ValueError(f"{label} must contain only positive values")
    return out
