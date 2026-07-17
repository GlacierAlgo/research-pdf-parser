"""Reusable PP-FormulaNet runtime used in-process or by the local service."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .accelerator import resolve_formula_device


@dataclass(frozen=True)
class FormulaPrediction:
    latex: str
    score: float | None


def paddle_result(result: Any) -> dict[str, Any]:
    payload: Any = result
    if not isinstance(payload, dict):
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            try:
                payload = dict(result)
            except (TypeError, ValueError):
                return {}
    payload = payload.get("res", payload)
    return payload if isinstance(payload, dict) else {}


class PaddleFormulaRuntime:
    """Load one PP-FormulaNet model once and reuse it across batches."""

    def __init__(self, model_name: str = "PP-FormulaNet_plus-S", device: str = "auto") -> None:
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import FormulaRecognition
        except ImportError as exc:
            raise RuntimeError(
                "Local formula runtime is unavailable; install with `uv sync --extra formula-cpu` "
                "or provide --formula-server-url http://HOST/formula_ocr."
            ) from exc

        started = time.perf_counter()
        resolved_device = resolve_formula_device(device)
        self._model = FormulaRecognition(
            model_name=model_name,
            device=resolved_device,
            engine="paddle_static",
        )
        self.init_seconds = time.perf_counter() - started
        self.model_name = model_name
        self.device = resolved_device

    def predict(self, paths: list[Path], batch_size: int = 4) -> tuple[list[FormulaPrediction], float]:
        started = time.perf_counter()
        raw_results = list(self._model.predict(input=[str(path) for path in paths], batch_size=batch_size))
        elapsed = time.perf_counter() - started
        predictions: list[FormulaPrediction] = []
        for raw in raw_results:
            payload = paddle_result(raw)
            latex = payload.get("rec_formula")
            score = payload.get("rec_score")
            parsed_score: float | None = None
            if score is not None:
                try:
                    parsed_score = float(score.item() if hasattr(score, "item") else score)
                except (TypeError, ValueError):
                    parsed_score = None
            predictions.append(
                FormulaPrediction(latex=latex if isinstance(latex, str) else "", score=parsed_score)
            )
        return predictions, elapsed
