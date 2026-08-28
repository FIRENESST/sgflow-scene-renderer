"""插件偏好设置：项目路径、venv Python、OpenAI 兼容服务与纹理库。"""
from __future__ import annotations

import os

import bpy
from bpy.props import StringProperty

_DEFAULT_PROJECT = os.environ.get(
    "SGFLOW_PROJECT_DIR",
    r"C:\Users\43828\Desktop\MyProject\AIRenderPipeline",
)


def _default_python() -> str:
    return os.path.join(_DEFAULT_PROJECT, ".venv", "Scripts", "python.exe")


class SGFlowAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    project_dir: StringProperty(
        name="SGFlow 项目目录",
        description="包含 sgflow/ 包的项目根目录",
        default=_DEFAULT_PROJECT,
        subtype="DIR_PATH",
    )
    python_exe: StringProperty(
        name="项目 Python",
        description="项目虚拟环境中的 python.exe（需要已安装 torch/openai）",
        default=_default_python(),
        subtype="FILE_PATH",
    )
    openai_base_url: StringProperty(
        name="API Base URL",
        description="OpenAI 兼容服务地址；本地服务通常以 /v1 结尾",
        default=os.environ.get("OPENAI_BASE_URL", ""),
    )
    openai_model: StringProperty(
        name="模型名",
        description="OpenAI 兼容模型名（对应 OPENAI_MODEL）",
        default=os.environ.get("OPENAI_MODEL", ""),
    )
    openai_api_key: StringProperty(
        name="API Key",
        description="OpenAI 兼容服务密钥；本地无鉴权服务可填 not-needed。仅保存在本机 Blender 偏好中",
        default="",
        subtype="PASSWORD",
    )
    texture_lib: StringProperty(
        name="纹理库目录",
        description="library 模式使用的纹理库（<lib>/<category>/albedo.png）",
        default=os.path.join(_DEFAULT_PROJECT, "textures_lib"),
        subtype="DIR_PATH",
    )
    output_dir: StringProperty(
        name="输出目录",
        description="生成的场景 JSON、纹理和渲染图输出位置",
        default=os.path.join(_DEFAULT_PROJECT, "blender_out"),
        subtype="DIR_PATH",
    )

    def env_overrides(self) -> dict[str, str]:
        """转成传给子进程的环境变量增量。"""
        env: dict[str, str] = {"PYTHONDONTWRITEBYTECODE": "1"}
        if self.openai_base_url:
            env["OPENAI_BASE_URL"] = self.openai_base_url
        if self.openai_model:
            env["OPENAI_MODEL"] = self.openai_model
        if self.openai_api_key:
            env["OPENAI_API_KEY"] = self.openai_api_key
        return env

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "project_dir")
        layout.prop(self, "python_exe")
        layout.separator()
        layout.label(text="OpenAI 兼容服务（生成场景用）")
        layout.prop(self, "openai_model")
        layout.prop(self, "openai_base_url")
        layout.prop(self, "openai_api_key")
        layout.separator()
        layout.prop(self, "texture_lib")
        layout.prop(self, "output_dir")


def get_prefs(context) -> SGFlowAddonPreferences:
    entry = context.preferences.addons.get(__package__)
    if entry is None:
        raise KeyError(
            f"SGFlow 插件偏好未就绪（addons 中无 {__package__!r}）；"
            "请在 偏好设置 > 插件 中确认 SGFlow Studio 已勾选启用。"
        )
    return entry.preferences


def register():
    bpy.utils.register_class(SGFlowAddonPreferences)


def unregister():
    bpy.utils.unregister_class(SGFlowAddonPreferences)
