"""Core document model shared by the parser and the renderer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Section:
    """A rendered document section (flattened tree node)."""

    title: str
    level: int  # 1 = top-level section
    number: str = ""  # "1", "1.2", ...
    anchor: str = ""  # "s-1", "s-1.2", ...
    html: str = ""  # rendered body HTML, heading excluded


@dataclass
class Footnote:
    """A footnote / citation entry shown in the bottom block and hover popup."""

    id: str
    html: str


@dataclass
class Reference:
    """A bibliography entry rendered in the 참고문헌 block (no popup)."""

    id: str
    html: str


@dataclass
class Document:
    title: str
    authors: list[str] = field(default_factory=list)
    abstract_html: str | None = None
    sections: list[Section] = field(default_factory=list)
    footnotes: list[Footnote] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
