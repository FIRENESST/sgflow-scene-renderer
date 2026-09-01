"""场景级属性组：面板上的所有输入状态保存在 Scene 上。"""
from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    StringProperty,
)


class SGFlowSceneSettings(bpy.types.PropertyGroup):
    prompt: StringProperty(
        name="提示词",
        description="自然语言场景描述，例如：一间温馨的卧室，有双人床和两个床头灯",
        default="一间温馨的卧室，有双人床、两个床头灯和一张书桌",
    )
    seed: IntProperty(name="随机种子", default=7)
    refine_steps: IntProperty(
        name="精修步数",
        description="空间求解器迭代步数",
        default=96,
        min=8,
        max=512,
    )
    device: EnumProperty(
        name="设备",
        items=(
            ("auto", "自动", "按 SGFlow 设备配置自动选择"),
            ("cpu", "CPU", "强制 CPU"),
            ("cuda", "CUDA", "强制 CUDA（校验失败会报错而不是回落）"),
        ),
        default="auto",
    )
    scene_json: StringProperty(
        name="场景 JSON",
        description="SGFlow 场景图 JSON 文件",
        default="",
        subtype="FILE_PATH",
    )
    materials_json: StringProperty(
        name="材质清单",
        description="可选：tex_assets 导出的 materials.json",
        default="",
        subtype="FILE_PATH",
    )
    texture_mode: EnumProperty(
        name="纹理模式",
        items=(
            ("library", "纹理库", "只查纹理库，缺失类别使用白模"),
            ("generated", "模型生成", "从检查点恢复 TexHead 生成程序纹理"),
        ),
        default="library",
    )
    texture_size: IntProperty(name="纹理尺寸", default=256, min=16, max=4096)
    checkpoint: StringProperty(
        name="检查点",
        description="generated 纹理模式所需的 sgflow_ckpt.pt",
        default="",
        subtype="FILE_PATH",
    )
    render_path: StringProperty(
        name="渲染输出",
        default="//sgflow_render.png",
        subtype="FILE_PATH",
    )
    resolution: IntProperty(name="分辨率", default=512, min=64, max=4096)
    replace_scene: BoolProperty(
        name="清空当前场景",
        description="导入前删除现有对象（后台模式总是清空）",
        default=True,
    )
    auto_render: BoolProperty(
        name="生成后直接渲染",
        default=False,
    )
    detail_level: IntProperty(
        name="精细等级",
        description="全局几何精细程度：1=单立方体占位，5=最多部件+最高细分+平滑着色",
        default=3,
        min=1,
        max=5,
    )
    status: StringProperty(
        name="状态",
        description="最近一次操作的结果",
        default="就绪",
    )
    busy: BoolProperty(name="任务进行中", default=False)


def register():
    bpy.utils.register_class(SGFlowSceneSettings)
    bpy.types.Scene.sgflow = bpy.props.PointerProperty(type=SGFlowSceneSettings)


def unregister():
    del bpy.types.Scene.sgflow
    bpy.utils.unregister_class(SGFlowSceneSettings)
