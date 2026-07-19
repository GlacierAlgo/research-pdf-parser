"""Runtime capability checks for local and optional GPU profiles."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import asdict, dataclass

import pymupdf
from liteparse import FormulaAtom, LiteParse
from liteparse import __version__ as liteparse_version

from .accelerator import detect_local_gpu, resolve_formula_device
from .formula_service import formula_service_health
from .mineru_remote import RemoteMineruConfig, check_remote_gpu_connection


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    detail: str
    required: bool = False


def inspect_capabilities(
    *,
    gpu_host: str | None = None,
    remote_uvx: str = "~/.local/bin/uvx",
    formula_server_url: str | None = None,
) -> list[Capability]:
    paddle_available = importlib.util.find_spec("paddleocr") is not None
    local_gpu = detect_local_gpu()
    capabilities = [
        Capability(
            "liteparse-formula-atom",
            hasattr(LiteParse, "parse_with_formula_atoms") and FormulaAtom is not None,
            f"liteparse {liteparse_version}",
            required=True,
        ),
        Capability("pymupdf", True, f"PyMuPDF {pymupdf.VersionBind}", required=True),
        Capability(
            "native-vector-policy",
            True,
            "OCR disabled; scanned pages are logged and deferred",
            required=True,
        ),
        Capability(
            "formula-cpu",
            paddle_available,
            "PaddleOCR installed" if paddle_available else "install the formula-cpu extra",
        ),
        Capability(
            "local-gpu",
            local_gpu.available,
            local_gpu.detail,
        ),
    ]
    if paddle_available:
        capabilities.append(
            Capability(
                "formula-device",
                True,
                f"auto resolves to {resolve_formula_device('auto')}",
            )
        )
    if formula_server_url:
        try:
            health = formula_service_health(formula_server_url)
            detail = f"{health.get('model', 'unknown')} on {health.get('device', 'unknown')}"
            capabilities.append(Capability("formula-service", True, detail))
        except RuntimeError as exc:
            capabilities.append(Capability("formula-service", False, str(exc)))
    if gpu_host:
        ssh_available = shutil.which("ssh") is not None and shutil.which("scp") is not None
        capabilities.append(
            Capability(
                "ssh",
                ssh_available,
                "ssh and scp found" if ssh_available else "ssh/scp missing",
                required=True,
            )
        )
        try:
            detail = check_remote_gpu_connection(
                RemoteMineruConfig(host=gpu_host, uvx_path=remote_uvx)
            )
            available = "uvx=ok" in detail and "gpu=missing" not in detail
        except RuntimeError as exc:
            available = False
            detail = str(exc)
        capabilities.append(Capability("remote-gpu-worker", available, detail, required=True))
    return capabilities


def capability_dicts(capabilities: list[Capability]) -> list[dict[str, object]]:
    return [asdict(capability) for capability in capabilities]
