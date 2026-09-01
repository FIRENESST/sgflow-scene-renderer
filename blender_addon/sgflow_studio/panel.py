"""3D 视口侧边栏（N 面板）"SGFlow" 标签页。"""
from __future__ import annotations

import bpy


class SGFLOW_PT_main(bpy.types.Panel):
    bl_label = "SGFlow 场景生成"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SGFlow"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.sgflow

        box = layout.box()
        box.label(text="① 提示词生成", icon="TEXT")
        box.prop(settings, "prompt")
        box.prop(settings, "backend")
        if settings.backend == "checkpoint":
            box.prop(settings, "checkpoint")
        row = box.row(align=True)
        row.prop(settings, "seed")
        row.prop(settings, "refine_steps")
        box.prop(settings, "device")
        detail_row = box.row()
        if settings.backend == "checkpoint":
            detail_row.prop(settings, "detail_level", text="精细等级 (1-5)")
            detail_row.enabled = False
            box.label(text="本地检查点暂不支持 L6", icon="INFO")
        else:
            box.prop(settings, "detail_level")
        row = box.row(align=True)
        row.prop(settings, "auto_render")
        gen = box.row()
        gen.enabled = not settings.busy
        gen.operator("sgflow.generate_scene", icon="PLAY")

        box = layout.box()
        box.label(text="② 场景导入", icon="IMPORT")
        box.prop(settings, "scene_json")
        box.prop(settings, "materials_json")
        box.prop(settings, "replace_scene")
        box.operator("sgflow.import_scene")

        box = layout.box()
        box.label(text="③ 纹理导出", icon="TEXTURE")
        box.prop(settings, "texture_mode")
        row = box.row(align=True)
        row.prop(settings, "texture_size")
        if settings.texture_mode == "generated":
            box.prop(settings, "checkpoint")
        tex = box.row()
        tex.enabled = not settings.busy
        tex.operator("sgflow.export_textures")

        box = layout.box()
        box.label(text="④ 渲染", icon="RENDER_STILL")
        box.prop(settings, "render_path")
        box.prop(settings, "resolution")
        box.operator("sgflow.render_scene")

        row = layout.row(align=True)
        row.label(text=settings.status, icon="INFO")
        layout.operator("sgflow.open_output", icon="FILE_FOLDER")


_CLASSES = (SGFLOW_PT_main,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
