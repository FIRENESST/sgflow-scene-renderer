"""操作符：生成 / 导入 / 纹理导出 / 渲染。

子进程任务使用模态定时器避免冻结 UI。场景重建直接复用项目里的
``sgflow/blender_importer.py``（它只依赖 bpy/mathutils/标准库，
可以在 Blender 自带 Python 中原样加载，无需 torch）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import bpy

from .prefs import get_prefs

_IMPORTER_CACHE: dict[str, object] = {}


def _load_importer(project_dir: str):
    """按路径加载项目中的 blender_importer.py，避免把 sgflow 包装进 sys.path。"""
    path = os.path.join(project_dir, "sgflow", "blender_importer.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到桥接模块：{path}")
    cached = _IMPORTER_CACHE.get(path)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("sgflow_blender_importer", path)
    module = importlib.util.module_from_spec(spec)
    # Python 3.13 的 @dataclass 处理 KW_ONLY 等注解时会查 sys.modules[cls.__module__]，
    # 因此 exec_module 前必须先注册，否则报 'NoneType' object has no attribute '__dict__'。
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    _IMPORTER_CACHE[path] = module
    return module


def build_scene_in_blender(context, project_dir: str, scene_path: str,
                           materials_path: str | None, replace: bool):
    """在进程内重建场景（复用桥接校验与材质逻辑）。

    importer 全部走 data API，不依赖 UI context，modal 回调里可直接调用。
    """
    importer = _load_importer(project_dir)
    with open(scene_path, encoding="utf-8") as f:
        scene = json.load(f)
    manifest = None
    if materials_path:
        with open(materials_path, encoding="utf-8") as f:
            manifest = json.load(f)
    return importer.build(
        scene, manifest,
        manifest_path=materials_path,
        replace_scene=replace,
    )


class _SubprocessMixin:
    """模态子进程基类：execute 启动进程，modal 轮询结束。"""

    _timer = None
    _proc: subprocess.Popen | None = None
    _stdout_path: str | None = None

    def _start(self, context, argv: list[str], env: dict[str, str], cwd: str):
        out = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", prefix="sgflow_", delete=False, encoding="utf-8"
        )
        self._stdout_path = out.name
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            argv, stdout=out, stderr=subprocess.STDOUT,
            env=env, cwd=cwd, creationflags=creation,
        )
        out.close()
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.3, window=context.window)
        self._set_status(context, "任务进行中……")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        rc = self._proc.poll()
        if rc is None:
            return {"RUNNING_MODAL"}
        self._finish_timer(context)
        log = self._read_log()
        if rc == 0:
            try:
                self._on_success(context, log)
            except Exception as exc:  # noqa: BLE001 - 必须反馈到面板而不是崩 UI
                self._set_status(context, f"导入失败：{exc}")
                self.report({"ERROR"}, f"SGFlow: {exc}")
                return {"CANCELLED"}
            return {"FINISHED"}
        tail = log.strip().splitlines()[-1] if log.strip() else f"exit={rc}"
        self._set_status(context, f"失败：{tail[:200]}")
        self.report({"ERROR"}, f"SGFlow 子进程失败（{tail[:200]}），日志：{self._stdout_path}")
        return {"CANCELLED"}

    def _finish_timer(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def _read_log(self) -> str:
        try:
            with open(self._stdout_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def _set_status(self, context, text: str):
        context.scene.sgflow.status = text
        context.scene.sgflow.busy = text == "任务进行中……"

    def _on_success(self, context, log: str):
        raise NotImplementedError

    def cancel(self, context):
        self._finish_timer(context)
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._set_status(context, "已取消")


class SGFLOW_OT_generate_scene(_SubprocessMixin, bpy.types.Operator):
    """调用项目 venv 的 sgflow.openai_compat 生成场景 JSON"""

    bl_idname = "sgflow.generate_scene"
    bl_label = "生成场景"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = get_prefs(context)
        settings = context.scene.sgflow
        if not os.path.isfile(prefs.python_exe):
            self.report({"ERROR"}, f"项目 Python 不存在：{prefs.python_exe}（请在插件偏好中修正）")
            return {"CANCELLED"}
        if not prefs.openai_model:
            self.report({"ERROR"}, "请先在插件偏好中填写模型名（OPENAI_MODEL）")
            return {"CANCELLED"}
        out_dir = bpy.path.abspath(prefs.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        scene_path = os.path.join(out_dir, "scene.json")
        argv = [
            prefs.python_exe, "-m", "sgflow.openai_compat",
            settings.prompt,
            "--output", scene_path,
            "--refine-steps", str(settings.refine_steps),
            "--seed", str(settings.seed),
        ]
        if settings.device != "auto":
            argv += ["--device", settings.device]
        env = dict(os.environ)
        env.update(prefs.env_overrides())
        self._scene_path = scene_path
        return self._start(context, argv, env, cwd=prefs.project_dir)

    def _on_success(self, context, log: str):
        settings = context.scene.sgflow
        settings.scene_json = self._scene_path
        prefs = get_prefs(context)
        build_scene_in_blender(
            context, prefs.project_dir, self._scene_path, None,
            replace=settings.replace_scene,
        )
        n = len([o for o in bpy.context.scene.objects if o.type == "MESH"])
        self._set_status(context, f"已生成并导入：{n} 个网格对象")
        self.report({"INFO"}, f"SGFlow 场景已导入（{n} 个对象）")
        if settings.auto_render:
            bpy.ops.sgflow.render_scene()


class SGFLOW_OT_import_scene(bpy.types.Operator):
    """导入已有场景 JSON（可选材质清单）"""

    bl_idname = "sgflow.import_scene"
    bl_label = "导入场景 JSON"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        prefs = get_prefs(context)
        settings = context.scene.sgflow
        scene_path = bpy.path.abspath(settings.scene_json)
        if not scene_path or not os.path.isfile(scene_path):
            self.report({"ERROR"}, "请先选择有效的场景 JSON 文件")
            return {"CANCELLED"}
        materials = bpy.path.abspath(settings.materials_json) if settings.materials_json else None
        if materials and not os.path.isfile(materials):
            self.report({"ERROR"}, f"材质清单不存在：{materials}")
            return {"CANCELLED"}
        try:
            build_scene_in_blender(
                context, prefs.project_dir, scene_path, materials,
                replace=settings.replace_scene,
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"导入失败：{exc}")
            settings.status = f"导入失败：{exc}"
            return {"CANCELLED"}
        n = len([o for o in context.scene.objects if o.type == "MESH"])
        settings.status = f"已导入 {n} 个网格对象"
        self.report({"INFO"}, f"SGFlow 场景已导入（{n} 个对象）")
        return {"FINISHED"}


class SGFLOW_OT_export_textures(_SubprocessMixin, bpy.types.Operator):
    """调用 sgflow.tex_assets 导出纹理并生成 materials.json"""

    bl_idname = "sgflow.export_textures"
    bl_label = "导出纹理"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = get_prefs(context)
        settings = context.scene.sgflow
        scene_path = bpy.path.abspath(settings.scene_json)
        if not scene_path or not os.path.isfile(scene_path):
            self.report({"ERROR"}, "请先选择有效的场景 JSON 文件")
            return {"CANCELLED"}
        if not os.path.isfile(prefs.python_exe):
            self.report({"ERROR"}, f"项目 Python 不存在：{prefs.python_exe}")
            return {"CANCELLED"}
        out_dir = bpy.path.abspath(prefs.output_dir)
        tex_dir = os.path.join(out_dir, "textures_out")
        argv = [
            prefs.python_exe, "-m", "sgflow.tex_assets",
            scene_path, tex_dir,
            "--mode", settings.texture_mode,
            "--size", str(settings.texture_size),
        ]
        if settings.texture_mode == "library":
            argv += ["--lib", bpy.path.abspath(prefs.texture_lib)]
        else:
            if not settings.checkpoint or not os.path.isfile(bpy.path.abspath(settings.checkpoint)):
                self.report({"ERROR"}, "generated 模式需要有效的检查点文件")
                return {"CANCELLED"}
            argv += ["--checkpoint", bpy.path.abspath(settings.checkpoint)]
            if settings.prompt:
                argv += ["--prompt", settings.prompt]
        env = dict(os.environ)
        env.update(prefs.env_overrides())
        self._materials_path = os.path.join(tex_dir, "materials.json")
        return self._start(context, argv, env, cwd=prefs.project_dir)

    def _on_success(self, context, log: str):
        settings = context.scene.sgflow
        if os.path.isfile(self._materials_path):
            settings.materials_json = self._materials_path
            self._set_status(context, f"纹理已导出：{self._materials_path}")
        else:
            self._set_status(context, "纹理导出完成（未找到 materials.json，请检查日志）")
        self.report({"INFO"}, "SGFlow 纹理导出完成")


class SGFLOW_OT_render_scene(bpy.types.Operator):
    """配置 Eevee 并渲染当前场景"""

    bl_idname = "sgflow.render_scene"
    bl_label = "渲染场景"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = get_prefs(context)
        settings = context.scene.sgflow
        scn = context.scene
        try:
            importer = _load_importer(prefs.project_dir)
            engine = importer.configure_render_engine(scn)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"渲染引擎配置失败：{exc}")
            return {"CANCELLED"}
        scn.render.resolution_x = scn.render.resolution_y = settings.resolution
        scn.render.resolution_percentage = 100
        render_path = bpy.path.abspath(settings.render_path)
        if settings.render_path.startswith("//") and not bpy.data.filepath:
            # 未保存的 .blend：// 会解析到 Blender 启动目录（如 C:\），通常不可写。
            # 兜底到偏好设置的输出目录。
            render_path = os.path.join(
                bpy.path.abspath(prefs.output_dir),
                os.path.basename(render_path) or "sgflow_render.png",
            )
        try:
            os.makedirs(os.path.dirname(render_path) or ".", exist_ok=True)
        except OSError as exc:
            self.report({"ERROR"}, f"无法创建渲染输出目录：{exc}")
            return {"CANCELLED"}
        scn.render.filepath = render_path
        try:
            bpy.ops.render.render(write_still=True)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"渲染失败：{exc}")
            return {"CANCELLED"}
        settings.status = f"渲染完成（{engine}）：{render_path}"
        self.report({"INFO"}, f"SGFlow 渲染完成：{render_path}")
        return {"FINISHED"}


class SGFLOW_OT_open_output(bpy.types.Operator):
    """在资源管理器中打开输出目录"""

    bl_idname = "sgflow.open_output"
    bl_label = "打开输出目录"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = get_prefs(context)
        out_dir = bpy.path.abspath(prefs.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(out_dir)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", out_dir])
        return {"FINISHED"}


_CLASSES = (
    SGFLOW_OT_generate_scene,
    SGFLOW_OT_import_scene,
    SGFLOW_OT_export_textures,
    SGFLOW_OT_render_scene,
    SGFLOW_OT_open_output,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
