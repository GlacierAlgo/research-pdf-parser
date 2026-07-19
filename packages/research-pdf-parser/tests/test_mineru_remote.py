import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research_pdf_parser.mineru_remote import (
    RemoteMineruConfig,
    _run,
    build_mineru_command,
    check_remote_gpu_connection,
    find_mineru_markdown,
    ssh_command,
)


class RemoteMineruTests(unittest.TestCase):
    def test_builds_high_accuracy_command_with_quoted_paths(self) -> None:
        config = RemoteMineruConfig(host="gpu-worker", backend="hybrid-engine", effort="high")

        command = build_mineru_command(
            "/home/user/run with space",
            "/home/user/bin/uvx",
            config,
        )

        self.assertIn("MINERU_MODEL_SOURCE=modelscope", command)
        self.assertIn("'mineru[all]'", command)
        self.assertIn("--managed-python", command)
        self.assertIn("--effort high", command)
        self.assertIn("'/home/user/run with space/input.pdf'", command)

    def test_finds_hybrid_markdown_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            markdown = output / "document" / "hybrid_txt" / "document.md"
            markdown.parent.mkdir(parents=True)
            markdown.write_text("result", encoding="utf-8")

            self.assertEqual(find_mineru_markdown(output), markdown)

    def test_ssh_command_has_batch_timeout_and_keepalive(self) -> None:
        command = ssh_command(
            RemoteMineruConfig(host="gpu-worker", connect_timeout=7, server_alive_interval=11),
            "true",
        )

        self.assertIn("BatchMode=yes", command)
        self.assertIn("ConnectTimeout=7", command)
        self.assertIn("ServerAliveInterval=11", command)
        self.assertEqual(command[-2:], ["gpu-worker", "true"])

    @mock.patch("research_pdf_parser.mineru_remote._run")
    def test_remote_capability_check_queries_gpu_and_uvx(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout="home=/home/user\nuvx=ok\ngpu=NVIDIA RTX 4090\n",
            stderr="",
        )

        detail = check_remote_gpu_connection(RemoteMineruConfig(host="gpu-worker"))

        self.assertIn("gpu=NVIDIA RTX 4090", detail)
        remote_command = run.call_args.args[0][-1]
        self.assertIn("nvidia-smi", remote_command)

    @mock.patch("research_pdf_parser.mineru_remote.time.sleep")
    @mock.patch("research_pdf_parser.mineru_remote.subprocess.run")
    def test_retries_transient_ssh_banner_failures(self, run: mock.Mock, sleep: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CalledProcessError(255, ["ssh"], stderr="banner exchange: invalid format"),
            subprocess.CompletedProcess(["ssh"], 0, stdout="ok", stderr=""),
        ]

        result = _run(["ssh"], capture_output=True, attempts=2, retry_delay=0.01)

        self.assertEqual(result.stdout, "ok")
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
