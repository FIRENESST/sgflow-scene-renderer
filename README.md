# SGFlow — 一句话生成结构化三维场景

SGFlow（**S**cene **G**raph **Flow**）是一个面向 Blender 渲染流程的 Python 研究原型：输入自然语言，生成带类别、位置、旋转、尺寸和外观潜变量的场景图，再导出程序纹理或查找纹理库，最后由 Blender 子进程重建和渲染。

> 重要：仓库不包含预训练权重。真正的文本到场景推理必须使用训练后的检查点；`allow_untrained=True` 只用于代码烟测，不代表生成质量。

## 管线

```text
自然语言 -> 冻结文本后端 + 可训练投影
          -> 结构头（PAD/对象存在性 + 类别）
          -> 整流流匹配（位置 / 6D 旋转 / log 尺寸 / 外观）
          -> 旋转感知的 AABB 约束精修（碰撞 / 边界 / 支撑）
          -> 版本化 SceneGraph JSON
          -> TexHead 程序纹理或纹理库
          -> Blender 立方体代理 + Principled BSDF + 渲染
```

核心设计：

- 整流流匹配用直线插值路径训练，推理通常使用 8–16 个 ODE 步。
- 潜瓶颈注意力将文本注入 K 个潜向量，复杂度为 `O((N + L) * K)`。
- 选择性 SSM 以 `O(N * d_state)` 顺序扫描混合对象；当前使用 CPU/CUDA 通用、可求导的 PyTorch 参考实现。
- 6D 旋转表示用 Gram–Schmidt 转为 `SO(3)`，并对零向量/共线输入提供稳定正交补全。
- 程序纹理每个对象使用 17 个 `O(1)` 控制量（4 个类型混合、6 个颜色、7 个图案/材质标量），光栅化成 Albedo、单通道 Roughness 和 Normal。
- 约束层使用旋转后世界 AABB，并且屏蔽的 PAD 槽不会被当成隐形支撑物。

## 安装

Python 3.10+；Blender 桥接需要独立安装 Blender。

```powershell
python -m venv .venv

# CPU 版 Torch；GPU 环境请按 PyTorch 官方命令安装对应 CUDA 轮子
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

`sentence-transformers` 是可选文本后端：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[text]"
```

未安装时使用确定性 8,192 桶字符 3-gram 编码；检查点会记录文本后端类型，加载时若后端不匹配会明确报错，不会静默改变语义空间。

## 快速功能烟测

仓库附带一个极小样本，下面的 1 epoch 只验证训练—保存—加载闭环，不用于评估生成质量。

```powershell
# 1. 训练并保存自描述检查点
.\.venv\Scripts\python.exe -m sgflow.train `
  --data examples/scenes --epochs 1 --batch-size 1 --seed 7 `
  --output sgflow_ckpt.pt

# 2. 从同一检查点生成版本化场景 JSON
.\.venv\Scripts\python.exe -c "from sgflow import ScenePipeline; sg = ScenePipeline(checkpoint='sgflow_ckpt.pt').generate('a cozy bedroom with a bed and two lamps', seed=7); sg.to_json('scene.json')"
```

`ScenePipeline` 默认要求检查点。如果只想验证维度和代码路径，可显式使用：

```python
pipeline = ScenePipeline(SGFlowConfig(), allow_untrained=True)
```

## 纹理导出

### 模型/程序生成模式

Generated 模式会从同一检查点恢复文本投影和 TexHead，并分批光栅化以限制峰值内存。

```powershell
.\.venv\Scripts\python.exe -m sgflow.tex_assets `
  scene.json textures_out --mode generated --checkpoint sgflow_ckpt.pt `
  --prompt "a cozy bedroom" --size 256 --batch-size 16 --seed 7
```

无检查点时只能用 `--allow-untrained` 显式进入确定性烟测模式。

### 纹理库模式

Library 模式不构建 TexHead；缺少 `albedo.png` 的类别使用白模，并写入 `texture_report.json`。

```powershell
.\.venv\Scripts\python.exe -m sgflow.tex_assets `
  scene.json textures_out --mode library --lib textures_lib
```

```text
textures_lib/
└── bed/
    ├── albedo.png   # 必需
    ├── rough.png    # 可选
    └── normal.png   # 可选
```

`materials.json` 的纹理路径相对于清单文件，因此整个输出目录可移动。

## Blender 重建与渲染

```bash
# 无材质清单
blender --background --python sgflow/blender_importer.py -- scene.json render.png

# 使用材质清单
blender --background --python sgflow/blender_importer.py -- scene.json textures_out/materials.json render.png
```

桥接会在创建 Blender 对象之前校验材质索引、类别、重复项和贴图文件，并复用已加载图像/相同材质。后台 CLI 默认清空场景；交互模式不会默认删除原对象，需要时显式加 `--full-replace`。

当前重建器使用立方体代理，尚未实现按类别链接 Blender Collection 资产库。

## 训练数据

`--data` 目录下每个 JSON 是一个样本：

```json
{
  "prompt": "一间温馨的卧室，有双人床、床头柜和落地灯",
  "objects": [
    {
      "category": "bed",
      "position": [0.0, 0.5, 0.3],
      "rotation6d": [1, 0, 0, 0, 1, 0],
      "scale": [2.0, 1.6, 0.5],
      "appearance": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
  ]
}
```

数据载入器会拒绝未知/PAD 类别、非有限数值、非正尺寸、错误向量长度和超过 `max_objects` 的场景，不再静默截断。

检查点包含：

- 完整 `SGFlowConfig` 和类别词表
- SceneDenoiser 权重
- 文本后端类型与可训练投影（不重复保存冻结 MiniLM/哈希表）
- TexHead 权重
- optimizer、AMP scaler、epoch 和训练摘要

恢复训练：

```powershell
.\.venv\Scripts\python.exe -m sgflow.train `
  --data data/scenes --epochs 100 --resume sgflow_ckpt.pt `
  --output sgflow_ckpt.pt --seed 7
```

## 场景与材质协议

- `SceneGraph.to_json()` 写出 `schema_version: 1`，使用原子替换；读取器仍接受没有版本字段的旧 JSON。
- 空场景以形状正确的 `(0, *)` 张量表示，可排序、序列化和进入纹理导出。
- `materials.json` 写出 `manifest_version: 1`；Blender 侧仍兼容无版本的旧清单。
- 所有生成纹理随机性来自场景指纹 + 显式 `seed`，不依赖 Python 进程级盐化 `hash()`。

## 核心配置

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `max_objects` | `128` | 模型槽位上限 |
| `max_generated_objects` | `min(32, max_objects)` | 默认推理输出上限，防止低质量检查点填满所有槽位 |
| `flow_steps` | `16` | 整流 ODE 步数 |
| `d_model` / `n_layers` | `256` / `4` | 模型宽度 / 深度 |
| `n_latents` | `32` | 文本潜瓶颈 K |
| `room_size` | `(8, 8, 4)` | 约束层房间尺寸 |
| `texture_mode` | `generated` | `generated` / `library` |
| `texture_batch_size` | `16` | 生成纹理的分批对象数 |
| `texture_size_limit` | `4096` | 单边像素硬上限 |
| `texture_train_size` | `16` | 训练正则使用的可微纹理预览边长 |
| `use_amp` / `use_compile` | `True` / `True` | 仅 CUDA 训练/推理时生效 |

## 测试

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖 SSM 前向/反向传播、PAD 平衡结构损失、退化 6D 旋转、旋转边界约束、屏蔽支撑物、场景协议、检查点恢复、局部 RNG、分批纹理和 Blender 参数/材质校验。Blender 实机无界面渲染需在已安装 Blender 的环境另行验证。

## 项目结构

```text
sgflow/
├── checkpoint.py       # 版本化、原子保存的自描述检查点
├── config.py           # 全局配置与跨模块不变量校验
├── math3d.py           # 稳定 6D 旋转、Morton Z 序
├── text_encoder.py     # 冻结 MiniLM / 紧凑确定性哈希后端
├── scene_graph.py      # 严格校验的版本化 JSON 协议
├── models.py           # 潜瓶颈 + StructureHead + SceneDenoiser
├── cuda_ops.py         # CPU/CUDA 通用的可求导 SSM 参考扫描
├── flow_matching.py    # PAD 平衡结构目标 + 整流损失/ODE
├── constraints.py      # 旋转 AABB 碰撞/边界/支撑约束
├── pipeline.py         # 检查点驱动的端到端生成
├── train.py            # 可恢复、可复现的训练循环
├── texhead.py          # 程序纹理参数头
├── tex_raster.py       # 向量化纹理光栅器
├── tex_assets.py       # 分批 generated / library 导出
└── blender_importer.py # 材质校验、路径解析和 Blender 重建
```

## 已知边界

- 项目没有随附预训练模型、大规模数据集或质量基准；生成质量取决于你的数据和训练。
- TexHead 的 17 个控制量均进入参数先验，并通过小尺寸可微光栅预览接受图像空间正则；仍没有受监督的材质真值，写实材质需增加纹理标注/感知损失。
- PyTorch SSM 参考扫描优先保证数学与梯度正确；生产级长序列 GPU 性能可在完成前向/反向数值对齐后接入 `mamba-ssm` 等成熟内核。
- 支撑候选选择包含离散判定，因此约束是分段可微而非处处光滑。
- Blender 资产 Collection 实例化仍是待接入的适配器；当前使用代理几何体。
