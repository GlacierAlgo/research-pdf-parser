"""Runtime capability checks for local and optional DGX profiles."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import asdict, dataclass

import pymupdf
from liteparse import FormulaAtom, LiteParse
from liteparse import __version__ as liteparse_version


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    detail: str
    required: bool = False


def inspect_capabilities(*, check_dgx: bool = False, dgx_host: str = "dgx-aliyun") -> list[Capability]:
    capabilities = [
        Capability(
            "liteparse-formula-atom",
            hasattr(LiteParse, "parse_with_formula_atoms") and FormulaAtom is not None,
            f"liteparse {liteparse_version}",
            required=True,
        ),
        Capability("pymupdf", True, f"PyMuPDF {pymupdf.VersionBind}", required=True),
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
    if check_dgx:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", dgx_host, "true"],
            capture_output=True,
            text=True,
            check=False,
        )
        detail = "reachable" if result.returncode == 0 else (result.stderr.strip() or "unreachable")
        capabilities.append(Capability("dgx", result.returncode == 0, detail, required=True))
    return capabilities


def capability_dicts(capabilities: list[Capability]) -> list[dict[str, object]]:
    return [asdict(capability) for capability in capabilities]
