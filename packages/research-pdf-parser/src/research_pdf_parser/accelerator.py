"""Hardware capability detection without assuming a particular machine class."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUCapability:
    """One local GPU capability verdict suitable for routing and diagnostics."""

    available: bool
    backend: str
    devices: tuple[str, ...]
    detail: str


def _nvidia_devices() -> tuple[str, ...]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ()
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _paddle_cuda_count() -> int:
    if importlib.util.find_spec("paddle") is None:
        return 0
    try:
        import paddle

        if not paddle.device.is_compiled_with_cuda():
            return 0
        return int(paddle.device.cuda.device_count())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return 0


def detect_local_gpu() -> GPUCapability:
    """Detect locally visible CUDA hardware; no hostname implies a GPU."""
    devices = _nvidia_devices()
    paddle_count = _paddle_cuda_count()
    if devices or paddle_count:
        if not devices:
            devices = tuple(f"CUDA GPU {index}" for index in range(paddle_count))
        detail = f"CUDA devices={len(devices)}; "
        detail += "Paddle CUDA ready" if paddle_count else "Paddle CUDA unavailable"
        return GPUCapability(True, "cuda", devices, detail)
    return GPUCapability(False, "none", (), "no usable local CUDA GPU detected")


def resolve_formula_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/gpu to a concrete Paddle device string."""
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("formula device must be auto, cpu, or gpu")
    if requested == "cpu":
        return "cpu"
    cuda_count = _paddle_cuda_count()
    if cuda_count:
        return "gpu:0"
    if requested == "gpu":
        capability = detect_local_gpu()
        if capability.available:
            raise RuntimeError(
                "A CUDA GPU is visible, but the installed Paddle runtime has no CUDA support; "
                "install a matching paddlepaddle-gpu build or use --formula-device cpu."
            )
        raise RuntimeError("--formula-device gpu was requested, but no usable local CUDA GPU was detected.")
    return "cpu"
