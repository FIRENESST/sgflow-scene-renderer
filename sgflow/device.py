"""Device selection, CUDA capability validation, and runtime tuning.

Selection priority is ``SGFLOW_DEVICE`` > ``sgflow_device.json`` > auto.
An explicit CUDA request fails fast when CUDA cannot be used; automatic mode
falls back to CPU. A CUDA-enabled PyTorch build can execute both branches, so
one environment is sufficient for CPU/GPU switching.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any

import torch


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sgflow_device.json",
)


class DeviceUnavailableError(RuntimeError):
    """The requested accelerator is not usable by the current PyTorch runtime."""


@dataclass(frozen=True)
class DevicePreference:
    request: str
    source: str


def _normalize_device(value: str, *, allow_auto: bool = True) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device must be a non-empty string")
    normalized = value.strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    allowed = {"cpu", "cuda"}
    if allow_auto:
        allowed.add("auto")
    if normalized in allowed:
        return normalized
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", 1)[1]
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"
    options = "auto, cpu, cuda, cuda:<index>" if allow_auto else "cpu, cuda, cuda:<index>"
    raise ValueError(f"device must be one of: {options}")


def read_device_preference() -> DevicePreference:
    """Read the requested device and where the request came from."""
    env = os.environ.get("SGFLOW_DEVICE", "").strip()
    if env:
        return DevicePreference(_normalize_device(env), "environment")
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid device config {_CONFIG_PATH}: {exc}") from exc
        if not isinstance(payload, dict) or "device" not in payload:
            raise ValueError(f"device config {_CONFIG_PATH} must contain a 'device' field")
        return DevicePreference(_normalize_device(payload["device"]), "config")
    return DevicePreference("auto", "auto")


def _cuda_index(device: str | torch.device) -> int:
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {device!r}")
    return torch.cuda.current_device() if parsed.index is None else parsed.index


def _architecture_name(capability: tuple[int, int]) -> str:
    major, minor = capability
    if major >= 10:
        return "Blackwell-or-newer"
    if (major, minor) == (8, 9):
        return "Ada Lovelace"
    if major == 8:
        return "Ampere"
    if major == 7:
        return "Turing/Volta"
    return f"SM {major}.{minor}"


def validate_cuda_device(
    device: str | torch.device = "cuda",
    *,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Validate driver, wheel, device ordinal, architecture, and allocation."""
    if torch.version.cuda is None:
        raise DeviceUnavailableError(
            "the installed PyTorch build is CPU-only; install a CUDA wheel first"
        )
    if not torch.cuda.is_available():
        raise DeviceUnavailableError(
            "PyTorch has CUDA support but no CUDA device is available; check the NVIDIA driver"
        )
    index = _cuda_index(device)
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise DeviceUnavailableError(f"CUDA device index {index} is outside 0..{count - 1}")
    resolved = torch.device("cuda", index)
    properties = torch.cuda.get_device_properties(resolved)
    capability = torch.cuda.get_device_capability(resolved)
    arch_tag = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = list(torch.cuda.get_arch_list())
    native_arch = any(tag == arch_tag or tag.startswith(f"{arch_tag}a") for tag in compiled_arches)
    if compiled_arches and not native_arch:
        # Wheels may retain forward-compatible PTX instead of a native cubin.
        ptx_tag = f"compute_{capability[0]}{capability[1]}"
        ptx_arch = any(tag == ptx_tag or tag.startswith(f"{ptx_tag}a") for tag in compiled_arches)
        if not ptx_arch:
            raise DeviceUnavailableError(
                f"PyTorch CUDA wheel does not include {arch_tag}; compiled arches: {compiled_arches}"
            )

    with torch.cuda.device(resolved):
        bf16 = bool(torch.cuda.is_bf16_supported())
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if smoke_test:
            left = right = loss = None
            try:
                left = torch.randn(256, 256, device=resolved, requires_grad=True)
                right = torch.randn(256, 256, device=resolved)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16 if bf16 else torch.float16,
                ):
                    loss = (left @ right).square().mean()
                loss.backward()
                torch.cuda.synchronize(resolved)
                if left.grad is None or not torch.isfinite(left.grad).all():
                    raise RuntimeError("CUDA backward pass produced invalid gradients")
            except Exception as exc:
                raise DeviceUnavailableError(f"CUDA allocation/compute smoke test failed: {exc}") from exc
            finally:
                del left, right, loss
                torch.cuda.empty_cache()

    return {
        "device": str(resolved),
        "name": properties.name,
        "architecture": _architecture_name(capability),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "compiled_arches": compiled_arches,
        "bf16_supported": bf16,
        "tf32_supported": capability[0] >= 8,
        "total_memory_gib": round(total_bytes / 2**30, 2),
        "free_memory_gib": round(free_bytes / 2**30, 2),
        "smoke_test": smoke_test,
    }


def resolve_device(cfg=None, *, strict: bool | None = None) -> str:
    """Resolve ``cpu`` or a concrete ``cuda:<index>`` device.

    Explicit environment/config CUDA requests are strict by default so a long
    GPU job cannot silently run on CPU. Auto detection remains safely
    fallback-capable. Pass ``strict=False`` for warning+fallback behavior.
    """
    del cfg  # retained for API compatibility
    preference = read_device_preference()
    request = preference.request
    if request == "cpu":
        return "cpu"
    if request == "auto":
        if torch.version.cuda is None or not torch.cuda.is_available():
            return "cpu"
        try:
            return validate_cuda_device("cuda")["device"]
        except DeviceUnavailableError as exc:
            warnings.warn(f"CUDA auto-detection failed ({exc}); falling back to CPU", RuntimeWarning)
            return "cpu"

    strict = True if strict is None else strict
    try:
        return validate_cuda_device(request)["device"]
    except DeviceUnavailableError:
        if strict:
            raise
        warnings.warn("CUDA was requested but is unavailable; falling back to CPU", RuntimeWarning)
        return "cpu"


def validate_device(device: str | torch.device, *, smoke_test: bool = False) -> str:
    """Validate an explicit API/CLI device argument and return a canonical name."""
    request = _normalize_device(str(device))
    if request == "auto":
        return resolve_device()
    if request == "cpu":
        return "cpu"
    return validate_cuda_device(request, smoke_test=smoke_test)["device"]


def resolve_amp_dtype(cfg, device: str | torch.device) -> torch.dtype:
    """Select a safe autocast dtype for RTX 30/40/50-series GPUs."""
    if torch.device(device).type != "cuda":
        return torch.float32
    requested = getattr(cfg, "amp_dtype", "auto")
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise DeviceUnavailableError("amp_dtype='bfloat16' is unsupported by this GPU/runtime")
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def configure_cuda_runtime(cfg, device: str | torch.device) -> dict[str, Any] | None:
    """Apply architecture-safe CUDA performance settings and return the profile."""
    if torch.device(device).type != "cuda":
        return None
    profile = validate_cuda_device(device)
    index = _cuda_index(device)
    torch.cuda.set_device(index)
    allow_tf32 = bool(getattr(cfg, "cuda_allow_tf32", True)) and profile["tf32_supported"]
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(getattr(cfg, "cuda_cudnn_benchmark", True))
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    profile["amp_dtype"] = str(resolve_amp_dtype(cfg, device)).removeprefix("torch.")
    profile["tf32_enabled"] = allow_tf32
    profile["cudnn_benchmark"] = bool(getattr(cfg, "cuda_cudnn_benchmark", True))
    return profile


def set_device(device: str) -> None:
    """Atomically persist ``auto``, ``cpu``, ``cuda`` or ``cuda:<index>``."""
    normalized = _normalize_device(device)
    directory = os.path.dirname(_CONFIG_PATH)
    fd, temporary = tempfile.mkstemp(prefix=".sgflow_device.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"device": normalized}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, _CONFIG_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def device_report(*, smoke_test: bool = False) -> dict[str, Any]:
    preference = read_device_preference()
    report: dict[str, Any] = {
        "request": preference.request,
        "source": preference.source,
        "config_path": _CONFIG_PATH,
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        resolved = resolve_device()
        report["resolved"] = resolved
        if torch.device(resolved).type == "cuda":
            report["cuda"] = validate_cuda_device(resolved, smoke_test=smoke_test)
    except (DeviceUnavailableError, ValueError) as exc:
        report["error"] = str(exc)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Configure and validate the SGFlow compute device")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--test", action="store_true", help="run a CUDA forward/backward smoke test")
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("device", help="auto, cpu, cuda, gpu, or cuda:<index>")
    set_parser.add_argument("--no-test", action="store_true", help="persist CUDA without testing it")
    args = parser.parse_args(argv)

    performed_test = False
    if args.command == "set":
        normalized = _normalize_device(args.device)
        if normalized.startswith("cuda") and not args.no_test:
            validate_cuda_device(normalized, smoke_test=True)
            performed_test = True
        set_device(normalized)
    requested_test = performed_test or getattr(args, "test", False)
    report = device_report(smoke_test=requested_test)
    if requested_test and not str(report.get("resolved", "")).startswith("cuda"):
        report["error"] = "CUDA smoke test requested but the configured device is not CUDA"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if "error" in report:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
