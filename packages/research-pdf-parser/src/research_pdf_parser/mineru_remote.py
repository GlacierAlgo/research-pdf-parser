from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


class RemoteMineruError(RuntimeError):
    """Raised when a remote MinerU conversion cannot be completed."""


@dataclass(frozen=True)
class RemoteMineruConfig:
    host: str = "dgx-aliyun"
    uvx_path: str = "~/.local/bin/uvx"
    backend: str = "hybrid-engine"
    effort: str = "high"
    language: str = "ch"
    model_source: str = "modelscope"
    keep_remote: bool = False


def _validate_host(host: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host):
        raise RemoteMineruError(f"Unsafe SSH host value: {host!r}")


def _run(
    args: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError as exc:
        raise RemoteMineruError(f"Required command is unavailable: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RemoteMineruError(f"Command failed ({args[0]}){suffix}") from exc


def build_mineru_command(
    remote_dir: str,
    uvx_path: str,
    config: RemoteMineruConfig,
) -> str:
    input_pdf = f"{remote_dir}/input.pdf"
    output_dir = f"{remote_dir}/output"
    mineru_args = [
        uvx_path,
        "--managed-python",
        "--python",
        "3.12",
        "--from",
        "mineru[all]",
        "mineru",
        "-p",
        input_pdf,
        "-o",
        output_dir,
        "-b",
        config.backend,
    ]
    if config.backend != "pipeline":
        mineru_args.extend(["--effort", config.effort])
    mineru_args.extend(
        [
            "-m",
            "txt",
            "-l",
            config.language,
            "-f",
            "true",
            "-t",
            "true",
            "--image-analysis",
            "false",
        ]
    )
    command = " ".join(shlex.quote(part) for part in mineru_args)
    return " && ".join(
        [
            "set -e",
            f"mkdir -p {shlex.quote(output_dir)}",
            f"MINERU_MODEL_SOURCE={shlex.quote(config.model_source)} /usr/bin/time -p {command}",
        ]
    )


def find_mineru_markdown(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("**/hybrid_txt/*.md"))
    if not candidates:
        candidates = sorted(output_dir.glob("**/pipeline/*.md"))
    if not candidates:
        candidates = sorted(output_dir.glob("**/*.md"))
    if len(candidates) != 1:
        raise RemoteMineruError(f"Expected one MinerU Markdown output, found {len(candidates)} in {output_dir}")
    return candidates[0]


def convert_pdf_on_dgx(
    pdf_path: Path,
    destination_dir: Path,
    config: RemoteMineruConfig,
) -> Path:
    """Run MinerU through SSH and return the downloaded raw Markdown path."""
    _validate_host(config.host)
    destination_dir.mkdir(parents=True, exist_ok=True)

    home_result = _run(
        ["ssh", config.host, 'printf "%s" "$HOME"'],
        capture_output=True,
    )
    remote_home = home_result.stdout.strip()
    if not remote_home.startswith("/"):
        raise RemoteMineruError("Could not resolve the remote home directory")
    uvx_path = config.uvx_path
    if uvx_path.startswith("~/"):
        uvx_path = f"{remote_home}/{uvx_path[2:]}"

    run_result = _run(
        [
            "ssh",
            config.host,
            "mkdir -p ~/.cache/research-pdf-parser/mineru-runs && "
            "mktemp -d ~/.cache/research-pdf-parser/mineru-runs/run.XXXXXX",
        ],
        capture_output=True,
    )
    remote_dir = run_result.stdout.strip()
    if not remote_dir.startswith(f"{remote_home}/.cache/research-pdf-parser/mineru-runs/run."):
        raise RemoteMineruError(f"Unexpected remote working directory: {remote_dir!r}")

    completed = False
    try:
        _run(["scp", str(pdf_path), f"{config.host}:{remote_dir}/input.pdf"])
        command = build_mineru_command(remote_dir, uvx_path, config)
        _run(["ssh", config.host, command])
        _run(["scp", "-r", f"{config.host}:{remote_dir}/output", str(destination_dir)])
        markdown = find_mineru_markdown(destination_dir / "output")
        completed = True
        return markdown
    finally:
        if completed and not config.keep_remote:
            _run(["ssh", config.host, f"rm -rf -- {shlex.quote(remote_dir)}"])
