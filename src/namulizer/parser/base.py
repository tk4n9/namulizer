"""Core intermediate representation (IR) for parsed papers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class Paragraph:
    text: str


@dataclass(slots=True)
class FigureBlock:
    caption: str = ""
    source_path: str | None = None
    data_uri: str | None = None


@dataclass(slots=True)
class TableBlock:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""


@dataclass(slots=True)
class EquationBlock:
    latex: str
    display: bool = True


ContentBlock = Paragraph | FigureBlock | TableBlock | EquationBlock


@dataclass(slots=True)
class Section:
    level: int
    title: str
    content: list[ContentBlock] = field(default_factory=list)


@dataclass(slots=True)
class Footnote:
    id: int
    content: str


@dataclass(slots=True)
class Reference:
    key: str
    text: str
    url: str | None = None


@dataclass(slots=True)
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    date: str | None = None
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    footnotes: list[Footnote] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)


class Parser(Protocol):
    def parse(self, input_path: Path) -> Paper:  # pragma: no cover - structural protocol
        """Parse an input document into Paper IR."""
