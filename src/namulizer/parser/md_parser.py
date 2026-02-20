"""Markdown parser with heuristic extraction into Paper IR."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

from .base import EquationBlock, FigureBlock, Footnote, Paper, Paragraph, Reference, Section, TableBlock


class MarkdownParser:
    """Parse a Markdown file into the unified Paper IR."""

    def __init__(self, embed_images: bool = True) -> None:
        self.embed_images = embed_images

    def parse(self, input_path: Path) -> Paper:
        input_path = Path(input_path)
        raw = input_path.read_text(encoding="utf-8", errors="ignore")

        frontmatter, body = _split_frontmatter(raw)
        meta = _parse_frontmatter(frontmatter)

        title = meta.get("title", "")
        authors = meta.get("authors", [])
        date = meta.get("date")
        abstract = meta.get("abstract", "")

        # If no title from frontmatter, use the first # heading.
        if not title:
            title, body = _extract_title_heading(body)

        if not title:
            title = input_path.stem

        footnote_defs = _extract_footnote_definitions(body)
        body = _remove_footnote_definitions(body)

        sections = _parse_sections(body, input_path.parent, self.embed_images)

        if not sections:
            blocks = _parse_blocks(body, input_path.parent, self.embed_images)
            sections = [Section(level=1, title="본문", content=blocks)]

        footnotes = [
            Footnote(id=i + 1, content=content)
            for i, (_, content) in enumerate(footnote_defs)
        ]

        # Rewrite footnote references in section content to use numeric ids.
        label_to_id = {label: i + 1 for i, (label, _) in enumerate(footnote_defs)}
        for section in sections:
            for block in section.content:
                if isinstance(block, Paragraph):
                    block.text = _rewrite_footnote_refs(block.text, label_to_id)

        return Paper(
            title=title,
            authors=authors,
            date=date,
            abstract=abstract,
            sections=sections,
            footnotes=footnotes,
            references=[],
        )


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------

def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split leading YAML frontmatter from body text."""
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    # Find the actual end of the closing --- line.
    closing_end = text.index("\n", end + 1) + 1 if end + 4 < len(text) and text[end + 4:end + 5] == "\n" else end + 4
    frontmatter = text[3:end].strip()
    body = text[closing_end:] if closing_end <= len(text) else text[end + 4:]
    return frontmatter, body


def _parse_frontmatter(raw: str) -> dict:
    """Minimal YAML-like frontmatter parser (no pyyaml dependency)."""
    if not raw:
        return {}

    result: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Handle "key: value" pairs.
        m = re.match(r"^([a-zA-Z_]\w*)\s*:\s*(.*)", line)
        if not m:
            continue

        key = m.group(1).lower()
        value = m.group(2).strip()

        # Strip surrounding quotes.
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]

        if key == "title":
            result["title"] = value
        elif key == "date":
            result["date"] = value if value else None
        elif key == "abstract":
            result["abstract"] = value
        elif key in ("author", "authors"):
            result["authors"] = _parse_author_value(value)

    return result


def _parse_author_value(value: str) -> list[str]:
    """Parse an author value, which may be a comma/and-separated string or YAML list."""
    # Inline YAML list: [Alice, Bob]
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    parts = re.split(r",\s*|\s+and\s+", value)
    return [p.strip().strip("\"'") for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Title extraction from body
# ---------------------------------------------------------------------------

def _extract_title_heading(body: str) -> tuple[str, str]:
    """Extract a top-level ``# Title`` heading and return (title, remaining_body)."""
    m = re.match(r"^\s*#\s+(.+?)(?:\s*\{[^}]*\})?\s*$", body, re.MULTILINE)
    if m:
        title = m.group(1).strip()
        remaining = body[:m.start()] + body[m.end():]
        return title, remaining
    return "", body


# ---------------------------------------------------------------------------
# Footnote definitions
# ---------------------------------------------------------------------------

_FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]]+)\]:\s*(.+)$", re.MULTILINE)


def _extract_footnote_definitions(text: str) -> list[tuple[str, str]]:
    """Extract ``[^label]: content`` definitions."""
    return [(m.group(1), m.group(2).strip()) for m in _FOOTNOTE_DEF_RE.finditer(text)]


def _remove_footnote_definitions(text: str) -> str:
    return _FOOTNOTE_DEF_RE.sub("", text)


def _rewrite_footnote_refs(text: str, label_to_id: dict[str, int]) -> str:
    """Replace ``[^label]`` references with ``[fn:N]`` markers."""
    def _replace(m: re.Match[str]) -> str:
        label = m.group(1)
        fn_id = label_to_id.get(label)
        if fn_id is not None:
            return f"[fn:{fn_id}]"
        return m.group(0)

    return re.sub(r"\[\^([^\]]+)\]", _replace, text)


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)(?:\s*\{[^}]*\})?\s*$", re.MULTILINE)


def _parse_sections(body: str, asset_root: Path, embed_images: bool) -> list[Section]:
    """Split body into sections based on ``##``–``######`` headings."""
    spans: list[tuple[int, int, int, str]] = []
    for m in _HEADING_RE.finditer(body):
        level = len(m.group(1)) - 1  # ## → 1, ### → 2, #### → 3
        title = m.group(2).strip()
        spans.append((m.start(), m.end(), level, title))

    if not spans:
        return []

    sections: list[Section] = []

    # Prelude before first heading.
    prelude = body[:spans[0][0]].strip()
    if prelude:
        blocks = _parse_blocks(prelude, asset_root, embed_images)
        if blocks:
            sections.append(Section(level=1, title="개요", content=blocks))

    for idx, (start, content_start, level, title) in enumerate(spans):
        content_end = spans[idx + 1][0] if idx + 1 < len(spans) else len(body)
        content = body[content_start:content_end]
        blocks = _parse_blocks(content, asset_root, embed_images)
        sections.append(Section(level=level, title=title or "(제목 없음)", content=blocks))

    return sections


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------

def _parse_blocks(
    text: str, asset_root: Path, embed_images: bool
) -> list[Paragraph | FigureBlock | TableBlock | EquationBlock]:
    """Parse a chunk of Markdown into content blocks."""
    blocks: list[Paragraph | FigureBlock | TableBlock | EquationBlock] = []
    i = 0

    while i < len(text):
        hits: list[tuple[int, str]] = []

        # Display math: $$ ... $$
        pos = text.find("$$", i)
        if pos != -1:
            hits.append((pos, "math_dollar"))

        # Fenced code block (``` or ~~~).
        fence_m = re.search(r"^(`{3,}|~{3,})", text[i:], re.MULTILINE)
        if fence_m:
            hits.append((i + fence_m.start(), "fenced"))

        # Image: ![alt](src)
        img_m = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", text[i:])
        if img_m:
            hits.append((i + img_m.start(), "image"))

        # Table: line of | cells |
        tbl_m = re.search(r"^\|(.+\|)\s*\n\|[\s:|-]+\|\s*\n", text[i:], re.MULTILINE)
        if tbl_m:
            hits.append((i + tbl_m.start(), "table"))

        if not hits:
            blocks.extend(_text_to_paragraphs(text[i:]))
            break

        next_pos, kind = min(hits, key=lambda h: h[0])

        if next_pos > i:
            blocks.extend(_text_to_paragraphs(text[i:next_pos]))

        if kind == "math_dollar":
            close = text.find("$$", next_pos + 2)
            if close == -1:
                blocks.extend(_text_to_paragraphs(text[next_pos:]))
                break
            latex = text[next_pos + 2:close].strip()
            blocks.append(EquationBlock(latex=latex, display=True))
            i = close + 2
            continue

        if kind == "fenced":
            fence_match = re.match(r"^(`{3,}|~{3,})(.*)", text[next_pos:])
            if not fence_match:
                blocks.extend(_text_to_paragraphs(text[next_pos:next_pos + 3]))
                i = next_pos + 3
                continue
            marker = fence_match.group(1)
            info = fence_match.group(2).strip().lower()
            after_open = next_pos + len(marker) + len(fence_match.group(2))
            # Skip past the newline after the opening fence.
            if after_open < len(text) and text[after_open] == "\n":
                after_open += 1
            close = text.find(f"\n{marker[0] * len(marker)}", after_open)
            if close == -1:
                close_end = len(text)
                content = text[after_open:]
            else:
                content = text[after_open:close]
                # Skip past closing fence line.
                line_end = text.find("\n", close + 1)
                close_end = line_end + 1 if line_end != -1 else len(text)

            if info in ("math", "latex", "tex"):
                blocks.append(EquationBlock(latex=content.strip(), display=True))
            else:
                # Treat as a code paragraph.
                blocks.append(Paragraph(text=content.rstrip()))
            i = close_end
            continue

        if kind == "image":
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", text[next_pos:])
            if m:
                caption = m.group(1)
                src = m.group(2).strip()
                fig = _make_figure(src, caption, asset_root, embed_images)
                blocks.append(fig)
                i = next_pos + m.end()
                continue

        if kind == "table":
            tbl, end = _parse_table(text, next_pos)
            blocks.append(tbl)
            i = end
            continue

    return [b for b in blocks if not isinstance(b, Paragraph) or b.text.strip()]


def _text_to_paragraphs(text: str) -> list[Paragraph]:
    """Split plain text on blank lines into paragraph blocks."""
    parts = re.split(r"\n\s*\n", text)
    result: list[Paragraph] = []
    for part in parts:
        cleaned = _clean_inline_md(part.strip())
        if cleaned:
            result.append(Paragraph(text=cleaned))
    return result


def _clean_inline_md(text: str) -> str:
    """Strip common inline Markdown formatting to plain text."""
    # Bold / italic.
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"___(.+?)___", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Inline code.
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strikethrough.
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Normalise whitespace within each line, preserve paragraph structure.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _parse_table(text: str, start: int) -> tuple[TableBlock, int]:
    """Parse a Markdown pipe-table starting at *start*."""
    lines: list[str] = []
    pos = start
    for line in text[start:].splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("|"):
            lines.append(stripped)
            pos += len(line)
        elif not stripped and not lines:
            pos += len(line)
        else:
            if lines:
                break
            pos += len(line)

    if len(lines) < 2:
        return TableBlock(), pos

    def split_row(row: str) -> list[str]:
        row = row.strip().strip("|")
        return [cell.strip() for cell in row.split("|")]

    header_cells = split_row(lines[0])
    # lines[1] is the separator row.
    body_rows = [split_row(line) for line in lines[2:] if line.strip()]

    return TableBlock(headers=header_cells, rows=body_rows), pos


# ---------------------------------------------------------------------------
# Figure / image helpers
# ---------------------------------------------------------------------------

def _make_figure(
    src: str, caption: str, asset_root: Path, embed_images: bool
) -> FigureBlock:
    source_path = (asset_root / src).resolve() if not Path(src).is_absolute() else Path(src)
    data_uri = None
    if embed_images and source_path.exists():
        mime, _ = mimetypes.guess_type(source_path.name)
        mime = mime or "application/octet-stream"
        data = base64.b64encode(source_path.read_bytes()).decode("ascii")
        data_uri = f"data:{mime};base64,{data}"

    return FigureBlock(caption=caption, source_path=str(source_path), data_uri=data_uri)
