"""CPU-first PDF-to-Markdown parsing for native-vector research documents."""

from .contracts import DocumentBlock, DocumentProbe, ParseResult
from .facade import parse_pdf
from .probe import probe_pdf
from .version import __version__

__all__ = [
    "__version__",
    "DocumentBlock",
    "DocumentProbe",
    "ParseResult",
    "parse_pdf",
    "probe_pdf",
]
