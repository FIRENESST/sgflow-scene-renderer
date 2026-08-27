特别感谢：model Kimi-K3/ChatGPT-5.6Sol

# SGFlow — 一句话生成结构化三维场景

SGFlow（**S**cene **G**raph **Flow**）是一个面向 Blender 渲染流程的 Python 研究原型：输入自然语言，生成带类别、位置、旋转、尺寸和外观潜变量的场景图，再导出程序纹理或查找纹理库，最后由 Blender 子进程重建和渲染。它现在有两种可互换的场景后端：本地检查点驱动的整流流模型，以及无需 SGFlow 预训练权重的 OpenAI 兼容大模型规划器。

> 重要：仓库不包含 SGFlow 预训练权重。本地流模型需要训练后的检查点；`allow_untrained=True` 只用于代码烟测。OpenAI 兼容后端可以直接生成场景，但质量、费用、隐私和可用性取决于你配置的模型服务。

## 管线

```text
自然语言 ─┬─ [本地] 冻结文本后端 + 结构头 + 整流流匹配 ─┐
          └─ [API]  OpenAI Chat Completions + 严格计划校验 ─┤
                                                            ↓
                 稀疏空间关系图 + 15 轴 OBB-SAT 可微精修
                                                            ↓
                 SceneGraph JSON -> 纹理 -> Blender 重建/渲染
```

核心设计：

- 整流流匹配用直线插值路径训练，推理通常使用 8–16 个 ODE 步。
- 潜瓶颈注意力将文本注入 K 个潜向量，复杂度为 `O((N + L) * K)`。
- 选择性 SSM 以 `O(N * d_state)` 顺序扫描混合对象；当前使用 CPU/CUDA 通用、可求导的 PyTorch 参考实现。
- 6D 旋转表示用 Gram–Schmidt 转为 `SO(3)`，并对零向量/共线输入提供稳定正交补全。
- 程序纹理每个对象使用 17 个 `O(1)` 控制量（4 个类型混合、6 个颜色、7 个图案/材质标量），光栅化成 Albedo、单通道 Roughness 和 Normal。
- 默认碰撞层使用完整 15 分离轴 OBB-SAT；可切回旧版世界 AABB 快速近似。屏蔽的 PAD 槽不会被当成隐形支撑物。
- API 后端采用“语义规划与几何求解分离”：大模型给出物体、米制 OBB 初值和稀疏关系，本地优化器负责物理边界与关系落地。

## 安装

Python 3.10+；Blender 桥接需要独立安装 Blender。

```powershell
python -m venv .venv

# 二选一：CUDA 版同样保留 CPU 执行能力，RTX 30/40/50 建议 GPU 版
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
# .\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

`requirements.txt` 包含通用依赖和 OpenAI Python SDK；Torch 使用独立的 CPU/GPU requirements，以免 pip 从默认源误装 CPU-only 构建。当前 GPU 文件使用 PyTorch 2.13 + CUDA 13.0，适配 RTX 30/40/50 系常用显卡。NVIDIA 驱动必须满足该轮子的运行要求，但无需另外安装完整 CUDA Toolkit。

## CPU/GPU 选择与 RTX 优化

CUDA 版 PyTorch 同时包含 CPU 路径，因此只需一个 `.venv`。设备优先级为 `SGFLOW_DEVICE` 环境变量、项目根目录 `sgflow_device.json`、自动检测：

```powershell
# 每次 set 都会原子更新 sgflow_device.json；设置 CUDA 时默认先跑真实前向/反向测试
.\.venv\Scripts\sgflow-device.exe set cuda
.\.venv\Scripts\sgflow-device.exe set cpu
.\.venv\Scripts\sgflow-device.exe set auto

# 查看驱动、Torch CUDA 构建、SM 架构、显存、BF16/TF32 和当前解析结果
.\.venv\Scripts\sgflow-device.exe status --test
```

也可临时覆盖：

```powershell
$env:SGFLOW_DEVICE = "cuda:0"  # 多卡时选择序号
$env:SGFLOW_DEVICE = "cpu"
```

手动维护配置时，可参考仓库根目录的 `sgflow_device.example.json`，复制为 `sgflow_device.json` 并修改 `"device"` 字段（`auto` / `cpu` / `cuda` / `cuda:0`）。

显式配置 CUDA 时若轮子、驱动、设备序号或 SM 架构不兼容，程序会立即报错，避免训练任务无声回落到 CPU；只有 `auto` 模式允许安全回退。

RTX 30（Ampere）、RTX 40（Ada）和 RTX 50（Blackwell）共享以下自动优化：

- `amp_dtype="auto"`：运行时查询 BF16 能力，支持时优先 BF16，否则使用 FP16；FP16 才启用 GradScaler。
- RTX 30 及以上默认开启 TF32，加速仍需 FP32 的矩阵乘。
- 固定形状卷积启用 cuDNN benchmark；注意力继续由 PyTorch SDPA 自动选择 Flash/高效后端。
- 推理模型段使用 autocast，关系约束与 OBB 几何精修强制回到 FP32。
- `torch.compile` 若在 Windows、Triton或具体算子上首次执行失败，会警告并恢复 eager，不会让长任务直接中断。

需要完全保守的数值路径时可设置 `use_amp=False`、`cuda_allow_tf32=False` 和 `use_compile=False`。

对于 6–8GB 显存型号，建议训练从 `--batch-size 1` 或 `2` 起步，并保持较小的 `texture_train_size`；纹理导出可单独降低 `--batch-size`。CLI 捕获显存不足时会给出这些调整方向并清理缓存。

`sentence-transformers` 是可选文本后端：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[text]"
```

未安装时使用确定性 8,192 桶字符 3-gram 编码；检查点会记录文本后端类型，加载时若后端不匹配会明确报错，不会静默改变语义空间。

## OpenAI 兼容 API 生成

API 后端使用常见的 `POST /v1/chat/completions` 消息格式。官方 OpenAI、vLLM、LM Studio、Ollama 网关或其他兼容服务均可通过相同的环境变量接入；具体服务是否支持 JSON Schema 由服务端决定。

```powershell
# 官方服务：通常不需要设置 OPENAI_BASE_URL
$env:OPENAI_API_KEY = "你的密钥"
$env:OPENAI_MODEL = "你的模型名"

# 本地/第三方服务时再设置；通常需要保留末尾 /v1
$env:OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
# 无鉴权的本地服务可使用占位值
$env:OPENAI_API_KEY = "not-needed"

.\.venv\Scripts\sgflow-openai.exe `
  "一间温馨的卧室，有双人床、两个床头灯和一张书桌" `
  --output scene.json --seed 7
```

也可以直接使用 Python API：

```python
from sgflow import ScenePipeline

pipeline = ScenePipeline.from_openai()  # 默认读取 OPENAI_* 环境变量
scene = pipeline.generate(
    "a compact studio with a bed, a desk and a reading lamp",
    refine_steps=96,
    seed=7,
)
scene.to_json("scene.json")
```

可显式传入 `model`、`base_url`、`api_key`、`timeout`、`max_retries` 和 `structured_output`。`structured_output="auto"` 会先请求严格 JSON Schema；仅当服务端以 400/404/415/422 明确拒绝该格式时，依次退回 JSON Object 和纯文本 JSON。无论服务端采用哪种格式，本地仍会拒绝未知类别、重复 ID、非有限值、非法尺寸和悬空关系引用。密钥不会写入检查点或场景 JSON。

API 场景 JSON 的 `metadata` 会记录模型名、对象 ID、稀疏关系和空间求解器版本，方便复现与排错，但不会记录 API 密钥。

`seed` 会作为“variation key”写入规划提示，但 OpenAI 兼容服务未必提供确定性采样保证；需要严格复现时请保存生成出的场景 JSON。

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

桥接器会先完整校验场景几何，再修改 `.blend` 数据；渲染输出目录会自动创建。渲染引擎会在 Blender 4.x/5.x 的 `BLENDER_EEVEE_NEXT` 和 Blender 3.x 的 `BLENDER_EEVEE` 之间自动选择，摄像机使用目标跟踪姿态，避免后台模式下依赖 UI 上下文或易错的固定欧拉角。

运行时诊断：

```powershell
# 检查 Python、Torch/CUDA、OpenAI SDK 和 Blender 是否可见
.\.venv\Scripts\sgflow-doctor.exe

# 加做 CUDA 分配、autocast、矩阵乘和反向传播测试
.\.venv\Scripts\sgflow-doctor.exe --gpu-smoke-test

# 若 Blender 已在 PATH，实际启动一次 factory-startup 后台探针
.\.venv\Scripts\sgflow-doctor.exe --probe-blender

# Blender 不在 PATH 时可显式指定
.\.venv\Scripts\sgflow-doctor.exe --probe-blender --blender "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
```

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
- 可选的顶层 `metadata` 必须是有限、可 JSON 序列化的数据；旧版读取路径保持兼容。
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
| `collision_mode` | `obb` | `obb`=15 轴有向盒 SAT；`aabb`=旧快速近似 |
| `texture_mode` | `generated` | `generated` / `library` |
| `texture_batch_size` | `16` | 生成纹理的分批对象数 |
| `texture_size_limit` | `4096` | 单边像素硬上限 |
| `texture_train_size` | `16` | 训练正则使用的可微纹理预览边长 |
| `use_amp` / `use_compile` | `True` / `True` | 仅 CUDA 训练/推理时生效 |
| `amp_dtype` | `auto` | 自动 BF16/FP16，或固定 `float16` / `bfloat16` |
| `cuda_allow_tf32` | `True` | RTX 30+ FP32 Tensor Core 加速 |
| `cuda_cudnn_benchmark` | `True` | 固定形状卷积内核自动择优 |

## 测试

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖 SSM 前向/反向传播、CPU/GPU 显式切换、CUDA autocast 与模型反向传播、编译失败回退、PAD 平衡结构损失、退化 6D 旋转、OBB/AABB 碰撞差异、稀疏关系求解、OpenAI 请求与响应校验、场景协议、检查点恢复、局部 RNG、分批纹理，以及 Blender 参数/场景/材质/引擎兼容校验。真实 API 调用和 Blender 实机无界面渲染需要在具有对应服务与 Blender 的环境中另行验证。

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
├── device.py           # CPU/CUDA 配置、架构校验、AMP/TF32 调优与 CLI
├── flow_matching.py    # PAD 平衡结构目标 + 整流损失/ODE
├── constraints.py      # OBB/AABB 碰撞、边界与支撑约束
├── spatial.py          # 稀疏空间关系图 + 15 轴 OBB-SAT 布局求解
├── openai_compat.py    # OpenAI Chat Completions 兼容规划器与 CLI
├── runtime.py          # Python/Torch/OpenAI/Blender 运行时诊断
├── pipeline.py         # 本地检查点后端与 OpenAI 兼容工厂入口
├── train.py            # 可恢复、可复现的训练循环
├── texhead.py          # 程序纹理参数头
├── tex_raster.py       # 向量化纹理光栅器
├── tex_assets.py       # 分批 generated / library 导出
└── blender_importer.py # 材质校验、路径解析和 Blender 重建
```

## 已知边界

- 项目没有随附预训练模型、大规模数据集或质量基准；生成质量取决于你的数据和训练。
- API 后端把大模型用作语义/关系规划器，不生成网格；输出仍由代理几何体或后续资产检索阶段落地。不同兼容服务对 Structured Outputs、采样和模型名的支持并不一致。
- TexHead 的 17 个控制量均进入参数先验，并通过小尺寸可微光栅预览接受图像空间正则；仍没有受监督的材质真值，写实材质需增加纹理标注/感知损失。
- PyTorch SSM 参考扫描优先保证数学与梯度正确；生产级长序列 GPU 性能可在完成前向/反向数值对齐后接入 `mamba-ssm` 等成熟内核。
- 支撑候选选择和 SAT 近平行轴屏蔽包含离散判定，因此约束是分段可微而非处处光滑。
- Blender 资产 Collection 实例化仍是待接入的适配器；当前使用代理几何体。

## 空间建模路线说明

这次没有把“更大的生成网络”硬塞进当前无数据/无权重的仓库，而是采用更适合现阶段的分层方案：

1. 类别与数量规划；
2. 尺寸和初始 OBB；
3. 稀疏对象关系图；
4. OBB 位姿与物理约束求解。

这条路线吸收了 [LayoutGPT](https://arxiv.org/abs/2305.15393) 用 LLM 做布局规划、[Holodeck](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Holodeck_Language_Guided_Generation_of_3D_Embodied_AI_Environments_CVPR_2024_paper.html) 用 LLM 生成空间关系并做约束优化，以及 [CommonScenes](https://arxiv.org/abs/2305.16283) 显式建模场景图关系的共同思路。对于后续有规模化 3D-FRONT/SG-FRONT 数据的训练阶段，可以再把当前独立结构头升级为稀疏关系图条件的级联 OBB 扩散/流模型；现有 `SpatialPlan` 和 `SceneGraph` 可作为中间协议，不需要推翻 API 或 Blender 侧。
