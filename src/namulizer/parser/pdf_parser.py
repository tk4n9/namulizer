"""PDF parser using PyMuPDF with heuristic section reconstruction."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

from .base import FigureBlock, Paper, Paragraph, Section

try:  # pragma: no cover - optional import guard for environments without pymupdf
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None


_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")


class PDFParser:
    """Parse PDFs into the unified Paper IR."""

    def __init__(self, embed_images: bool = True) -> None:
        self.embed_images = embed_images

    def parse(self, input_path: Path) -> Paper:
        if fitz is None:
            raise RuntimeError("pymupdf is required for PDF parsing")

        doc = fitz.open(str(input_path))
        metadata = doc.metadata or {}

        authors = _split_authors((metadata.get("author") or "").strip())
        date = _normalize_date(metadata.get("creationDate") or metadata.get("modDate"))

        lines: list[dict[str, Any]] = []
        font_sizes: list[float] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_lines = _extract_page_lines(page)
            lines.extend(page_lines)
            font_sizes.extend([float(line["font_size"]) for line in page_lines if float(line["font_size"]) > 0])

        body_font = _median(font_sizes) if font_sizes else 10.0
        inferred_title = _infer_title(lines, body_font)
        title = (metadata.get("title") or "").strip() or inferred_title or input_path.stem

        abstract = _extract_abstract(lines, body_font)
        sections = _build_sections_from_lines(lines, body_font)

        if self.embed_images:
            figures = self._extract_document_images(doc)
            if figures:
                if not sections:
                    sections = [Section(level=1, title="본문", content=[])]
                sections[-1].content.extend(figures)

        if not sections:
            sections = [Section(level=1, title="본문", content=[Paragraph(text="(내용 추출 실패)")])]

        return Paper(
            title=title,
            authors=authors,
            date=date,
            abstract=abstract,
            sections=sections,
            footnotes=[],
            references=[],
        )

    def _extract_document_images(self, doc: "fitz.Document") -> list[FigureBlock]:
        figures: list[FigureBlock] = []
        seen_xrefs: set[int] = set()

        for page_index in range(len(doc)):
            page = doc[page_index]
            for img in page.get_images(full=True):
                xref = int(img[0])
                width = int(img[2] or 0)
                height = int(img[3] or 0)

                # Skip tiny decorative assets.
                if width < 120 or height < 120 or width * height < 20_000:
                    continue

                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                image_data = doc.extract_image(xref)
                ext = image_data.get("ext") or "png"
                data = image_data.get("image")
                if not data:
                    continue

                mime, _ = mimetypes.guess_type(f"x.{ext}")
                mime = mime or "application/octet-stream"
                data_uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
                figures.append(
                    FigureBlock(
                        caption=f"Figure (page {page_index + 1})",
                        source_path=f"pdf:xref:{xref}",
                        data_uri=data_uri,
                    )
                )

        return figures


def _extract_page_lines(page: "fitz.Page") -> list[dict[str, Any]]:
    d = page.get_text("dict")
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    raw_lines: list[dict[str, Any]] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = sorted(line.get("spans", []), key=lambda s: float((s.get("bbox") or [0])[0]))
            if not spans:
                continue

            pieces = []
            max_size = 0.0
            for span in spans:
                txt = (span.get("text") or "").strip()
                if txt:
                    pieces.append(txt)
                max_size = max(max_size, float(span.get("size") or 0.0))

            text = _normalize_space(" ".join(pieces))
            if not text:
                continue

            bbox = line.get("bbox") or block.get("bbox") or [0, 0, 0, 0]
            x0, y0, x1, y1 = map(float, bbox)
            column = 0 if x0 < page_width * 0.53 else 1

            raw_lines.append(
                {
                    "text": text,
                    "font_size": max_size,
                    "x": x0,
                    "x_end": x1,
                    "y": y0,
                    "y_end": y1,
                    "line_height": max(1.0, y1 - y0),
                    "column": column,
                    "page": int(page.number),
                    "page_height": page_height,
                }
            )

    merged = _merge_line_fragments(raw_lines)
    filtered = [line for line in merged if not _should_drop_line(line)]
    filtered.sort(key=lambda it: (int(it["page"]), int(it["column"]), float(it["y"]), float(it["x"])))
    return filtered


def _merge_line_fragments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lines:
        return []

    lines = sorted(lines, key=lambda it: (int(it["page"]), int(it["column"]), float(it["y"]), float(it["x"])))
    out: list[dict[str, Any]] = []

    current: dict[str, Any] | None = None
    for line in lines:
        if current is None:
            current = dict(line)
            continue

        same_page = int(line["page"]) == int(current["page"])
        same_column = int(line["column"]) == int(current["column"])
        close_y = abs(float(line["y"]) - float(current["y"])) <= 1.2
        close_font = abs(float(line["font_size"]) - float(current["font_size"])) <= 2.5
        x_gap = float(line["x"]) - float(current["x_end"])

        if same_page and same_column and close_y and close_font and x_gap <= 260:
            current["text"] = _normalize_space(f"{current['text']} {line['text']}")
            current["x_end"] = max(float(current["x_end"]), float(line["x_end"]))
            current["y_end"] = max(float(current["y_end"]), float(line["y_end"]))
            current["font_size"] = max(float(current["font_size"]), float(line["font_size"]))
            current["line_height"] = max(float(current["line_height"]), float(line["line_height"]))
        else:
            out.append(current)
            current = dict(line)

    if current is not None:
        out.append(current)

    return out


def _should_drop_line(line: dict[str, Any]) -> bool:
    text = _normalize_space(str(line["text"]))
    lower = text.lower()

    if not text:
        return True

    # Drop page number-only footer.
    if re.fullmatch(r"\d{1,3}", text) and float(line["y"]) > float(line["page_height"]) * 0.9:
        return True

    # Common first-page and footer boilerplate for scholarly PDFs.
    noise_fragments = (
        "provided proper attribution is provided",
        "solely for use in journalistic or scholarly works",
        "conference on neural information processing systems",
        "long beach, ca, usa",
    )
    if any(fragment in lower for fragment in noise_fragments):
        return True

    return False


def _infer_title(lines: list[dict[str, Any]], body_font: float) -> str | None:
    if not lines:
        return None

    candidates: list[tuple[float, int, str]] = []
    threshold = body_font * 1.55

    for line in lines:
        text = _normalize_space(str(line["text"]))
        fs = float(line["font_size"])

        if int(line["page"]) > 0:
            continue
        if fs < threshold:
            continue
        if len(text) < 6 or len(text) > 150:
            continue
        if "@" in text:
            continue
        if text.lower().startswith("arxiv:"):
            continue

        alpha_ratio = sum(1 for ch in text if ch.isalpha()) / max(1, len(text))
        if alpha_ratio < 0.45:
            continue

        candidates.append((fs, int(line["page"]), text))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    return candidates[0][2]


def _extract_abstract(lines: list[dict[str, Any]], body_font: float) -> str:
    if not lines:
        return ""

    start_idx = -1
    for i, line in enumerate(lines):
        text = _normalize_space(str(line["text"]))
        if text.lower() == "abstract":
            start_idx = i + 1
            break

    if start_idx == -1:
        return ""

    abstract_lines: list[dict[str, Any]] = []
    for line in lines[start_idx:]:
        text = _normalize_space(str(line["text"]))
        if _is_numbered_heading(text):
            break
        # Skip low-value contribution footnote details.
        if float(line["font_size"]) < body_font * 0.9:
            continue
        abstract_lines.append(line)

    paragraphs = _lines_to_paragraphs(abstract_lines, body_font)
    return paragraphs[0] if paragraphs else ""


def _build_sections_from_lines(lines: list[dict[str, Any]], body_font: float) -> list[Section]:
    if not lines:
        return []

    numbered_headings: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        text = _normalize_space(str(line["text"]))
        parsed = _parse_numbered_heading(text)
        if not parsed:
            continue

        number, title = parsed
        if not _is_valid_heading_candidate(number, title, float(line["font_size"]), body_font):
            continue

        level = min(3, number.count(".") + 1)
        numbered_headings.append({"idx": i, "level": level, "title": title})

    sections: list[Section] = []

    if numbered_headings:
        for pos, heading in enumerate(numbered_headings):
            start = int(heading["idx"]) + 1
            end = int(numbered_headings[pos + 1]["idx"]) if pos + 1 < len(numbered_headings) else len(lines)

            content_lines = lines[start:end]
            paragraphs = _lines_to_paragraphs(content_lines, body_font)
            blocks = [Paragraph(text=p) for p in paragraphs]

            sections.append(
                Section(
                    level=int(heading["level"]),
                    title=str(heading["title"]),
                    content=blocks,
                )
            )
    else:
        # Fallback: create one section with all textual content.
        paragraphs = _lines_to_paragraphs(lines, body_font)
        blocks = [Paragraph(text=p) for p in paragraphs]
        sections.append(Section(level=1, title="본문", content=blocks))

    return [section for section in sections if section.content or section.title]


def _parse_numbered_heading(text: str) -> tuple[str, str] | None:
    m = _NUMBERED_HEADING_RE.match(text)
    if not m:
        return None
    number = m.group(1)
    title = _normalize_space(m.group(2))
    return number, title


def _is_numbered_heading(text: str) -> bool:
    return bool(_NUMBERED_HEADING_RE.match(text))


def _is_valid_heading_candidate(number: str, title: str, font_size: float, body_font: float) -> bool:
    if not title or len(title) > 120:
        return False
    if title.endswith("."):
        return False
    if "@" in title:
        return False
    if title.lower().startswith("arxiv"):
        return False

    first_alpha = next((ch for ch in title if ch.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return False

    # Reject table-like numeric strings that accidentally match heading patterns.
    alpha_count = sum(1 for ch in title if ch.isalpha())
    digit_count = sum(1 for ch in title if ch.isdigit())
    alpha_ratio = alpha_count / max(1, len(title))
    if alpha_count < 3 or alpha_ratio < 0.22:
        return False
    if digit_count > max(3, int(alpha_count * 1.2)):
        return False

    words = [word for word in re.split(r"[\s/:;,\-()]+", title) if word]
    if not any(sum(1 for ch in word if ch.isalpha()) >= 2 for word in words):
        return False

    parts = number.split(".")
    if len(parts) > 4:
        return False

    try:
        int_parts = [int(part) for part in parts]
    except ValueError:
        return False

    if any(part <= 0 for part in int_parts):
        return False

    top_num = int_parts[0]
    if top_num > 20:
        return False

    if len(int_parts) == 1:
        if font_size < body_font * 1.08:
            return False
    else:
        if any(part > 20 for part in int_parts[1:]):
            return False
        if font_size < body_font * 0.95:
            return False

    return True


def _lines_to_paragraphs(lines: list[dict[str, Any]], body_font: float) -> list[str]:
    if not lines:
        return []

    lines = sorted(lines, key=lambda it: (int(it["page"]), int(it["column"]), float(it["y"]), float(it["x"])))

    paragraphs: list[str] = []
    buffer = ""
    prev_line: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal buffer
        text = _normalize_space(buffer)
        if text:
            paragraphs.append(text)
        buffer = ""

    for line in lines:
        text = _normalize_space(str(line["text"]))
        if not text:
            flush()
            prev_line = None
            continue

        if prev_line is None:
            buffer = text
            prev_line = line
            continue

        page_changed = int(line["page"]) != int(prev_line["page"])
        column_changed = int(line["column"]) != int(prev_line["column"])
        y_gap = float(line["y"]) - float(prev_line["y_end"])
        para_break = page_changed or column_changed or y_gap > max(body_font * 0.95, 11.5)

        if para_break:
            flush()
            buffer = text
            prev_line = line
            continue

        if buffer.endswith("-") and text and text[0].islower():
            buffer = buffer[:-1] + text
        else:
            buffer = f"{buffer} {text}"

        prev_line = line

    flush()
    return paragraphs


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _split_authors(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r";|,| and ", raw)
    return [p.strip() for p in parts if p.strip()]


def _normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("D:"):
        # PDF metadata format e.g. D:20240204121000
        year = clean[2:6]
        month = clean[6:8] if len(clean) >= 8 else "01"
        day = clean[8:10] if len(clean) >= 10 else "01"
        if year.isdigit() and month.isdigit() and day.isdigit():
            return f"{year}-{month}-{day}"
    return clean
