"""无界面验证：安装并启用 sgflow_studio，注册检查 + 导入 smoke_scene.json + 渲染。

用法：blender-launcher --background --factory-startup --python blender_addon_test.py
结果写入 blender_addon_test_result.json。
"""
import json
import os
import shutil
import sys

import bpy
import addon_utils

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(ROOT, "blender_addon_test_result.json")
report = {"steps": [], "ok": False}


def step(name, **kw):
    report["steps"].append({"name": name, **kw})


try:
    # 1. 安装到用户 addons 目录
    src = os.path.join(ROOT, "blender_addon", "sgflow_studio")
    addons_dir = os.path.join(bpy.utils.user_resource("SCRIPTS"), "addons")
    os.makedirs(addons_dir, exist_ok=True)
    dst = os.path.join(addons_dir, "sgflow_studio")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    step("install", dst=dst)

    # 2. 启用插件（新装插件必须先刷新模块缓存；default_set=True 才会创建偏好条目）
    addon_utils.modules_refresh()
    addon_utils.enable("sgflow_studio", default_set=True)
    mod = sys.modules.get("sgflow_studio")
    step("enable", ok=mod is not None)
    if mod is None:
        raise RuntimeError("插件模块未加载")

    # 3. 注册检查（bpy.ops 的 dir() 不可靠，用属性访问探测）
    ops = ["generate_scene", "import_scene",
           "export_textures", "render_scene", "open_output"]
    missing = []
    for name in ops:
        try:
            getattr(bpy.ops.sgflow, name)
        except AttributeError:
            missing.append(name)
    step("operators", missing=missing)
    if missing:
        raise RuntimeError(f"缺少操作符：{missing}")
    step("panel", registered=hasattr(bpy.types, "SGFLOW_PT_main"))

    # 4. 导入 smoke_scene.json
    scene_path = os.path.join(ROOT, "smoke_scene.json")
    with open(scene_path, encoding="utf-8") as f:
        expected = len(json.load(f)["objects"])
    bpy.context.scene.sgflow.scene_json = scene_path
    bpy.context.scene.sgflow.replace_scene = True
    rc = bpy.ops.sgflow.import_scene()
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    step("import", rc=list(rc), meshes=len(meshes), expected=expected)
    if rc != {"FINISHED"} or len(meshes) < expected:
        raise RuntimeError(f"导入失败 rc={rc} meshes={len(meshes)} expected>={expected}")

    # 5. 渲染 128px 验证引擎
    render_path = os.path.join(ROOT, "blender_out", "addon_test_render.png")
    bpy.context.scene.sgflow.render_path = render_path
    bpy.context.scene.sgflow.resolution = 128
    rc = bpy.ops.sgflow.render_scene()
    exists = os.path.isfile(render_path) and os.path.getsize(render_path) > 0
    step("render", rc=list(rc), exists=exists, engine=bpy.context.scene.render.engine)
    if rc != {"FINISHED"} or not exists:
        raise RuntimeError("渲染失败")

    report["ok"] = True
except Exception as exc:
    report["error"] = str(exc)
finally:
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
