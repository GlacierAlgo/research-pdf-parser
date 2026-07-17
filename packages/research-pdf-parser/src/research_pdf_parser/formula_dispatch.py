from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

FormulaRecord = dict[str, Any]
FormulaProcessor = Callable[[list[FormulaRecord], Path, bool], int]


class FormulaDispatchError(ValueError):
    pass


class FormulaProcessorRegistry:
    """Small explicit registry for formula OCR processor modules."""

    def __init__(self) -> None:
        self._processors: dict[str, FormulaProcessor] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._processors)

    def register(self, name: str, processor: FormulaProcessor) -> None:
        normalized = name.strip()
        if not normalized:
            raise FormulaDispatchError("Formula processor name must not be empty.")
        if normalized in self._processors:
            raise FormulaDispatchError(f"Formula processor is already registered: {normalized}")
        self._processors[normalized] = processor

    def dispatch(
        self,
        records: list[FormulaRecord],
        manifest_path: Path,
        *,
        processor_override: str | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        grouped: dict[str, list[FormulaRecord]] = {}
        for record in records:
            processor_name = processor_override or record.get("processor")
            if not isinstance(processor_name, str) or not processor_name.strip():
                formula_id = record.get("id", "<unknown>")
                raise FormulaDispatchError(f"{formula_id} has no formula processor route.")
            grouped.setdefault(processor_name, []).append(record)

        counts: dict[str, int] = {}
        for processor_name, processor_records in grouped.items():
            processor = self._processors.get(processor_name)
            if processor is None:
                available = ", ".join(self.names) or "<none>"
                raise FormulaDispatchError(
                    f"Unknown formula processor {processor_name!r}; registered processors: {available}."
                )
            counts[processor_name] = processor(processor_records, manifest_path, force)
        return counts
