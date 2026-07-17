"""Runtime capability checks for local and optional DGX profiles."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import asdict, dataclass

import pymupdf
from liteparse import FormulaAtom, LiteParse
from liteparse import __version__ as liteparse_version

from .formula_service import formula_service_health
from .mineru_remote import RemoteMineruConfig, check_dgx_connection


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    detail: str
    required: bool = False


def inspect_capabilities(
    *,
    check_dgx: bool = False,
    dgx_host: str = "dgx-aliyun",
    remote_uvx: str = "~/.local/bin/uvx",
    formula_server_url: str | None = None,
) -> list[Capability]:
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
            importlib.util.find_spec("paddleocr") is not None,
            "PaddleOCR installed" if importlib.util.find_spec("paddleocr") else "install the formula-cpu extra",
        ),
        Capability(
            "ssh",
            shutil.which("ssh") is not None and shutil.which("scp") is not None,
            "ssh and scp found" if shutil.which("ssh") and shutil.which("scp") else "ssh/scp missing",
            required=check_dgx,
        ),
    ]
    if formula_server_url:
        try:
            health = formula_service_health(formula_server_url)
            detail = f"{health.get('model', 'unknown')} on {health.get('device', 'unknown')}"
            capabilities.append(Capability("formula-service", True, detail))
        except RuntimeError as exc:
            capabilities.append(Capability("formula-service", False, str(exc)))
    if check_dgx:
        try:
            detail = check_dgx_connection(RemoteMineruConfig(host=dgx_host, uvx_path=remote_uvx))
            available = "uvx=ok" in detail
        except RuntimeError as exc:
            available = False
            detail = str(exc)
        capabilities.append(Capability("dgx", available, detail, required=True))
    return capabilities


def capability_dicts(capabilities: list[Capability]) -> list[dict[str, object]]:
    return [asdict(capability) for capability in capabilities]
