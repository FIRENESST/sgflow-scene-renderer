"""SGFlow Studio — Blender 侧边栏面板插件。

在 3D 视口侧边栏（N 面板）的 "SGFlow" 标签页中提供：
提示词生成场景（调用项目 venv 的 OpenAI 兼容后端）、导入场景 JSON、
导出纹理和一键渲染。重度计算通过子进程调用项目虚拟环境完成，
Blender 自带 Python 无需安装 torch/openai。
"""
from __future__ import annotations

bl_info = {
    "name": "SGFlow Studio",
    "author": "FIRENESST",
    "version": (0, 1, 1),
    "blender": (4, 2, 0),
    "location": "3D 视口 > 侧边栏 (N) > SGFlow",
    "description": "一句话生成结构化三维场景并桥接 SGFlow 管线",
    "category": "3D View",
}

import bpy

from . import operators, panel, prefs, props

_MODULES = (prefs, props, operators, panel)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    for mod in reversed(_MODULES):
        mod.unregister()


if __name__ == "__main__":
    register()
