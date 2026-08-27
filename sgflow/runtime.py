"""Portable runtime diagnostics for Python, Torch, OpenAI and Blender."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_report(
    *, probe_blender: bool = False, blender: str | None = None,
    gpu_smoke_test: bool = False,
) -> dict[str, Any]:
    py_ok = sys.version_info >= (3, 10)
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": py_ok,
        },
        "packages": {
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
            "Pillow": _package_version("Pillow"),
            "openai": _package_version("openai"),
            "sentence-transformers": _package_version("sentence-transformers"),
        },
    }
    try:
        import torch

        report["torch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        report["torch_runtime"] = {"error": str(exc)}
    try:
        from .device import device_report

        sgflow_device = device_report(smoke_test=gpu_smoke_test)
        if gpu_smoke_test and not str(sgflow_device.get("resolved", "")).startswith("cuda"):
            sgflow_device["error"] = "GPU smoke test requested but SGFlow did not resolve a CUDA device"
        report["sgflow_device"] = sgflow_device
    except Exception as exc:
        report["sgflow_device"] = {"error": str(exc)}

    blender_path = blender or shutil.which("blender")
    blender_report: dict[str, Any] = {"path": blender_path, "available": bool(blender_path)}
    if probe_blender and blender_path:
        command = [
            blender_path,
            "--background",
            "--factory-startup",
            "--python-expr",
            (
                "import bpy; "
                "print('SGFLOW_PROBE', bpy.app.version_string, bpy.app.background, "
                "bpy.context.scene.render.engine)"
            ),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                creationflags=creationflags,
            )
            combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            marker = next((line.strip() for line in combined.splitlines() if "SGFLOW_PROBE" in line), None)
            blender_report.update({
                "background_probe": completed.returncode == 0 and marker is not None,
                "returncode": completed.returncode,
                "details": marker or combined[-1000:],
            })
        except (OSError, subprocess.TimeoutExpired) as exc:
            blender_report.update({"background_probe": False, "error": str(exc)})
    report["blender"] = blender_report
    return report


def _print_human(report: dict[str, Any]) -> None:
    python = report["python"]
    print(f"Python: {python['version']} ({'OK' if python['supported'] else 'requires 3.10+'})")
    for name, version in report["packages"].items():
        status = version or "not installed"
        optional = " (optional)" if name == "sentence-transformers" else ""
        print(f"{name}: {status}{optional}")
    torch_runtime = report["torch_runtime"]
    if "error" in torch_runtime:
        print(f"Torch runtime: ERROR - {torch_runtime['error']}")
    else:
        print(
            "Torch runtime: "
            f"CUDA={torch_runtime['cuda_available']} "
            f"CUDA version={torch_runtime['cuda_version']} "
            f"devices={torch_runtime['device_count']}"
        )
    blender = report["blender"]
    print(f"Blender: {blender['path'] or 'not found on PATH'}")
    if "background_probe" in blender:
        print(f"Blender background probe: {'OK' if blender['background_probe'] else 'FAILED'}")
        if blender.get("details"):
            print(blender["details"])
    device = report.get("sgflow_device", {})
    if "error" in device:
        print(f"SGFlow device: ERROR - {device['error']}")
    else:
        print(
            f"SGFlow device: request={device.get('request')} "
            f"source={device.get('source')} resolved={device.get('resolved')}"
        )
        cuda = device.get("cuda")
        if cuda:
            print(
                f"GPU: {cuda['name']}  {cuda['architecture']}  "
                f"SM={cuda['compute_capability']}  BF16={cuda['bf16_supported']}"
            )
            if cuda.get("smoke_test"):
                print("GPU forward/backward smoke test: OK")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect SGFlow runtime compatibility")
    parser.add_argument("--probe-blender", action="store_true")
    parser.add_argument("--blender", default=None, help="explicit Blender executable path")
    parser.add_argument("--gpu-smoke-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = runtime_report(
        probe_blender=args.probe_blender,
        blender=args.blender,
        gpu_smoke_test=args.gpu_smoke_test,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    if args.gpu_smoke_test and "error" in report.get("sgflow_device", {}):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
