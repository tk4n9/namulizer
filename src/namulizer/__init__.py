"""namulizer: convert academic PDFs/Markdown into NamuWiki-style HTML."""

from .document import Document, Footnote, Reference, Section
from .md_parser import parse_markdown
from .renderer import render_html

__version__ = "0.3.0"

__all__ = [
    "Document",
    "Footnote",
    "Reference",
    "Section",
    "parse_markdown",
    "render_html",
    "__version__",
]
