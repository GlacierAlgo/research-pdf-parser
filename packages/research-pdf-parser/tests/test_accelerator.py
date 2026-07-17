from __future__ import annotations

import unittest
from unittest import mock

from research_pdf_parser.accelerator import detect_local_gpu, resolve_formula_device


class AcceleratorTests(unittest.TestCase):
    @mock.patch("research_pdf_parser.accelerator._paddle_cuda_count", return_value=0)
    @mock.patch("research_pdf_parser.accelerator._nvidia_devices", return_value=())
    def test_auto_falls_back_to_cpu(self, _nvidia: mock.Mock, _paddle: mock.Mock) -> None:
        self.assertEqual(resolve_formula_device("auto"), "cpu")
        self.assertFalse(detect_local_gpu().available)

    @mock.patch("research_pdf_parser.accelerator._paddle_cuda_count", return_value=2)
    def test_auto_selects_first_usable_paddle_gpu(self, _paddle: mock.Mock) -> None:
        self.assertEqual(resolve_formula_device("auto"), "gpu:0")

    @mock.patch("research_pdf_parser.accelerator.detect_local_gpu")
    @mock.patch("research_pdf_parser.accelerator._paddle_cuda_count", return_value=0)
    def test_explicit_gpu_fails_when_unavailable(
        self,
        _paddle: mock.Mock,
        capability: mock.Mock,
    ) -> None:
        capability.return_value.available = False
        with self.assertRaisesRegex(RuntimeError, "no usable local CUDA GPU"):
            resolve_formula_device("gpu")


if __name__ == "__main__":
    unittest.main()
