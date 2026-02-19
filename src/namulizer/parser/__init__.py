"""Parser package."""

from .base import Paper, Section, Footnote, Reference, Paragraph, FigureBlock, TableBlock, EquationBlock
from .pdf_parser import PDFParser
from .tex_parser import TeXParser

__all__ = [
    "Paper",
    "Section",
    "Footnote",
    "Reference",
    "Paragraph",
    "FigureBlock",
    "TableBlock",
    "EquationBlock",
    "PDFParser",
    "TeXParser",
]
