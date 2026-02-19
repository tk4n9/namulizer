"""Render Paper IR into a self-contained NamuWiki-style HTML file."""

from __future__ import annotations

import base64
import html
import mimetypes
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from namulizer.parser.base import EquationBlock, FigureBlock, Footnote, Paper, Paragraph, Section, TableBlock


@dataclass(slots=True)
class RenderedSection:
    level: int
    heading_tag: int
    title: str
    number: str
    anchor: str
    html: str


class HTMLRenderer:
    """Render parsed IR into the NamuWiki-style template."""

    def __init__(self, template_path: Path | None = None) -> None:
        if template_path is None:
            template_path = Path(__file__).resolve().parent.parent / "template" / "namuwiki.html"

        loader = FileSystemLoader(str(template_path.parent))
        self._env = Environment(loader=loader, autoescape=True, trim_blocks=True, lstrip_blocks=True)
        self._template_name = template_path.name

    def render(
        self,
        paper: Paper,
        *,
        title_override: str | None = None,
        dark_mode: bool = False,
        enable_fold: bool = True,
        embed_images: bool = True,
        math_engine: str = "none",
    ) -> str:
        page_title = title_override or paper.title or "Untitled"
        footnote_map = {fn.id: fn for fn in paper.footnotes}

        rendered_sections = self._render_sections(
            paper.sections,
            footnote_map=footnote_map,
            embed_images=embed_images,
            math_engine=math_engine,
        )

        toc_items = [
            {
                "level": section.level,
                "title": section.title,
                "number": section.number,
                "anchor": section.anchor,
            }
            for section in rendered_sections
        ]

        footnotes = [
            {
                "id": fn.id,
                "html": self._render_inline_text(fn.content),
            }
            for fn in paper.footnotes
        ]

        modified_text = paper.date or date.today().isoformat()

        template = self._env.get_template(self._template_name)
        return template.render(
            page_title=page_title,
            authors=paper.authors,
            modified_text=modified_text,
            modified_datetime=modified_text,
            abstract=paper.abstract,
            toc_items=toc_items,
            sections=[asdict(s) for s in rendered_sections],
            footnotes=footnotes,
            references=paper.references,
            dark_mode=dark_mode,
            enable_fold=enable_fold,
        )

    def _render_sections(
        self,
        sections: list[Section],
        *,
        footnote_map: dict[int, Footnote],
        embed_images: bool,
        math_engine: str,
    ) -> list[RenderedSection]:
        counters = [0, 0, 0]
        used_anchors: set[str] = set()
        rendered: list[RenderedSection] = []

        for section in sections:
            level = max(1, min(3, section.level))
            counters[level - 1] += 1
            for idx in range(level, len(counters)):
                counters[idx] = 0

            number = ".".join(str(n) for n in counters[:level] if n > 0)
            base_anchor = f"s-{number}" if number else "s"
            anchor = _dedupe_anchor(base_anchor, used_anchors)
            heading_tag = min(level + 1, 6)

            body_parts: list[str] = []
            for block in section.content:
                body_parts.append(
                    self._render_block(
                        block,
                        footnote_map=footnote_map,
                        embed_images=embed_images,
                        math_engine=math_engine,
                    )
                )

            rendered.append(
                RenderedSection(
                    level=level,
                    heading_tag=heading_tag,
                    title=section.title,
                    number=number,
                    anchor=anchor,
                    html="\n".join(part for part in body_parts if part),
                )
            )

        return rendered

    def _render_block(
        self,
        block: Paragraph | FigureBlock | TableBlock | EquationBlock,
        *,
        footnote_map: dict[int, Footnote],
        embed_images: bool,
        math_engine: str,
    ) -> str:
        if isinstance(block, Paragraph):
            return f'<div class="wiki-paragraph">{self._render_paragraph_text(block.text, footnote_map)}</div>'

        if isinstance(block, FigureBlock):
            return self._render_figure(block, embed_images=embed_images)

        if isinstance(block, TableBlock):
            return self._render_table(block)

        if isinstance(block, EquationBlock):
            content = html.escape(block.latex)
            return (
                f'<pre class="wiki-equation" data-engine="{html.escape(math_engine)}" '
                f'data-display="{"true" if block.display else "false"}"><code>{content}</code></pre>'
            )

        return ""

    def _render_paragraph_text(self, text: str, footnote_map: dict[int, Footnote]) -> str:
        math_chunks: list[str] = []
        explicit_footnotes: list[int] = []

        def stash_math(match: re.Match[str]) -> str:
            math_chunks.append(match.group(1).strip())
            return f"@@MATH_{len(math_chunks)-1}@@"

        def stash_explicit_footnote(match: re.Match[str]) -> str:
            fn_id = int(match.group(1))
            if fn_id not in footnote_map:
                return match.group(0)
            explicit_footnotes.append(fn_id)
            return f"@@FN_{len(explicit_footnotes)-1}@@"

        text_with_footnote_tokens = re.sub(r"\[fn:(\d+)\]", stash_explicit_footnote, text)
        text_with_math_tokens = re.sub(r"\$([^$]+)\$", stash_math, text_with_footnote_tokens)
        escaped = html.escape(text_with_math_tokens)

        for idx, latex in enumerate(math_chunks):
            token = f"@@MATH_{idx}@@"
            fragment = '<code class="wiki-math-inline">' + html.escape(latex) + "</code>"
            escaped = escaped.replace(token, fragment)

        if footnote_map:
            escaped = re.sub(
                r"\[(\d+)\]",
                lambda match: self._render_footnote_ref(int(match.group(1)))
                if int(match.group(1)) in footnote_map
                else match.group(0),
                escaped,
            )

        for idx, fn_id in enumerate(explicit_footnotes):
            token = f"@@FN_{idx}@@"
            escaped = escaped.replace(token, self._render_footnote_ref(fn_id))

        return escaped

    def _render_footnote_ref(self, fn_id: int) -> str:
        return (
            f'<sup class="wiki-fn-content" id="fnref-{fn_id}"><a class="wiki-footnote-ref" '
            f'href="#fn-{fn_id}" data-footnote-id="{fn_id}" aria-describedby="wiki-footnote-popper">[{fn_id}]</a></sup>'
        )

    def _render_figure(self, block: FigureBlock, *, embed_images: bool) -> str:
        src = block.data_uri
        if not src and embed_images and block.source_path:
            src = _maybe_embed_image(Path(block.source_path))
        if not src:
            src = html.escape(block.source_path or "")

        if src:
            image_html = (
                f'<img src="{src}" alt="{html.escape(block.caption or "Figure")}" '
                'loading="lazy" class="wiki-image" />'
            )
            image_html = f'<span class="wiki-image-wrapper">{image_html}</span>'
        else:
            image_html = '<span class="wiki-image-wrapper"><span class="wiki-caption">(이미지를 찾을 수 없습니다)</span></span>'

        caption = html.escape(block.caption) if block.caption else ""
        caption_html = f'<div class="wiki-caption">{caption}</div>' if caption else ""
        return (
            '<div class="wiki-paragraph">'
            f'<span class="wiki-image-align-center">{image_html}</span>{caption_html}'
            '</div>'
        )

    def _render_table(self, block: TableBlock) -> str:
        head_html = ""
        if block.headers:
            head_cells = "".join(f"<th>{html.escape(cell)}</th>" for cell in block.headers)
            head_html = f"<thead><tr>{head_cells}</tr></thead>"

        row_html = ""
        if block.rows:
            rows = []
            for row in block.rows:
                cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
                rows.append(f"<tr>{cells}</tr>")
            row_html = "<tbody>" + "".join(rows) + "</tbody>"

        caption_html = f'<div class="wiki-caption">{html.escape(block.caption)}</div>' if block.caption else ""
        return f'<div class="wiki-table-wrap"><table class="wiki-table">{head_html}{row_html}</table>{caption_html}</div>'

    def _render_inline_text(self, text: str) -> str:
        return html.escape(text)


def _dedupe_anchor(anchor: str, used: set[str]) -> str:
    if anchor not in used:
        used.add(anchor)
        return anchor

    idx = 2
    while True:
        candidate = f"{anchor}-{idx}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def _maybe_embed_image(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"
