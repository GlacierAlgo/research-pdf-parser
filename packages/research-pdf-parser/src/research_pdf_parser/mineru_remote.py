from __future__ import annotations

import re
import shlex
import subprocess
import time
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
    connect_timeout: int = 10
    transport_attempts: int = 3
    retry_delay: float = 1.0
    server_alive_interval: int = 15


def _validate_host(host: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host):
        raise RemoteMineruError(f"Unsafe SSH host value: {host!r}")


def _run(
    args: list[str],
    *,
    capture_output: bool = False,
    attempts: int = 1,
    retry_delay: float = 1.0,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return subprocess.run(
                args,
                check=True,
                text=True,
                capture_output=capture_output,
                input=input,
            )
        except FileNotFoundError as exc:
            raise RemoteMineruError(f"Required command is unavailable: {args[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            transient = any(
                marker in detail.lower()
                for marker in (
                    "banner exchange",
                    "connection reset",
                    "connection closed",
                    "connection timed out",
                    "operation timed out",
                    "broken pipe",
                )
            )
            if transient and attempt < attempts:
                time.sleep(retry_delay * attempt)
                continue
            suffix = f": {detail}" if detail else ""
            raise RemoteMineruError(f"Command failed ({args[0]}){suffix}") from exc
    raise AssertionError("unreachable")


def ssh_options(config: RemoteMineruConfig) -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.connect_timeout}",
        "-o",
        f"ServerAliveInterval={config.server_alive_interval}",
        "-o",
        "ServerAliveCountMax=3",
    ]


def ssh_command(config: RemoteMineruConfig, remote_command: str) -> list[str]:
    return ["ssh", *ssh_options(config), config.host, remote_command]


def scp_command(config: RemoteMineruConfig, *paths: str) -> list[str]:
    return ["scp", *ssh_options(config), *paths]


def check_dgx_connection(config: RemoteMineruConfig) -> str:
    """Verify SSH plus the configured uvx path without starting a model."""
    _validate_host(config.host)
    uvx_check = (
        f'"$HOME"/{shlex.quote(config.uvx_path[2:])}'
        if config.uvx_path.startswith("~/")
        else shlex.quote(config.uvx_path)
    )
    result = _run(
        ssh_command(
            config,
            'printf "home=%s\\n" "$HOME"; '
            f"test -x {uvx_check} && printf 'uvx=ok\\n' || printf 'uvx=missing\\n'",
        ),
        capture_output=True,
        attempts=config.transport_attempts,
        retry_delay=config.retry_delay,
    )
    return result.stdout.strip()


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
        ssh_command(config, 'printf "%s" "$HOME"'),
        capture_output=True,
        attempts=config.transport_attempts,
        retry_delay=config.retry_delay,
    )
    remote_home = home_result.stdout.strip()
    if not remote_home.startswith("/"):
        raise RemoteMineruError("Could not resolve the remote home directory")
    uvx_path = config.uvx_path
    if uvx_path.startswith("~/"):
        uvx_path = f"{remote_home}/{uvx_path[2:]}"

    run_result = _run(
        ssh_command(
            config,
            "mkdir -p ~/.cache/research-pdf-parser/mineru-runs && "
            "mktemp -d ~/.cache/research-pdf-parser/mineru-runs/run.XXXXXX",
        ),
        capture_output=True,
        attempts=config.transport_attempts,
        retry_delay=config.retry_delay,
    )
    remote_dir = run_result.stdout.strip()
    if not remote_dir.startswith(f"{remote_home}/.cache/research-pdf-parser/mineru-runs/run."):
        raise RemoteMineruError(f"Unexpected remote working directory: {remote_dir!r}")

    completed = False
    try:
        _run(
            scp_command(config, str(pdf_path), f"{config.host}:{remote_dir}/input.pdf"),
            attempts=config.transport_attempts,
            retry_delay=config.retry_delay,
        )
        command = build_mineru_command(remote_dir, uvx_path, config)
        _run(ssh_command(config, command))
        _run(
            scp_command(config, "-r", f"{config.host}:{remote_dir}/output", str(destination_dir)),
            attempts=config.transport_attempts,
            retry_delay=config.retry_delay,
        )
        markdown = find_mineru_markdown(destination_dir / "output")
        completed = True
        return markdown
    finally:
        if completed and not config.keep_remote:
            _run(
                ssh_command(config, f"rm -rf -- {shlex.quote(remote_dir)}"),
                attempts=config.transport_attempts,
                retry_delay=config.retry_delay,
            )
