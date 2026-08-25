"""设备选择：配置文件 / 环境变量 / 自动检测 三级控制

优先级：SGFLOW_DEVICE 环境变量 > sgflow_device.json > 自动检测（有 CUDA 用 GPU，否则 CPU）

配置文件 sgflow_device.json（可选）：
    {"device": "gpu"}   或   {"device": "cpu"}
"""
import json
import os
import tempfile

import torch

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sgflow_device.json")


def resolve_device(cfg=None) -> str:
    """返回 'cuda' 或 'cpu'"""
    # 1) 环境变量优先
    env = os.environ.get("SGFLOW_DEVICE", "").lower()
    if env in ("gpu", "cuda"):
        return "cuda" if torch.cuda.is_available() else _warn_cpu("环境变量指定 GPU 但 CUDA 不可用")
    if env == "cpu":
        return "cpu"

    # 2) 配置文件
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                dev = json.load(f).get("device", "").lower()
            if dev in ("gpu", "cuda"):
                return "cuda" if torch.cuda.is_available() else _warn_cpu("配置文件指定 GPU 但 CUDA 不可用")
            if dev == "cpu":
                return "cpu"
        except Exception:
            pass

    # 3) 自动检测
    return "cuda" if torch.cuda.is_available() else "cpu"


def _warn_cpu(msg: str) -> str:
    import warnings
    warnings.warn(f"{msg}，回退到 CPU")
    return "cpu"


def set_device(device: str):
    """写入配置文件，持久化选择"""
    if not isinstance(device, str) or device.lower() not in {"cpu", "gpu", "cuda"}:
        raise ValueError("device must be one of: cpu, gpu, cuda")
    normalized = device.lower()
    directory = os.path.dirname(_CONFIG_PATH)
    fd, tmp = tempfile.mkstemp(prefix=".sgflow_device.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"device": normalized}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _CONFIG_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
