"""Document -> standalone NamuWiki-style HTML rendering."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
_TOC_MAX_LEVEL = 3


def _template():
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template("namuwiki.html")


def render_html(
    document,
    *,
    dark_mode: bool = False,
    enable_fold: bool = True,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now()
    toc_items = [
        {
            "level": min(s.level, _TOC_MAX_LEVEL),
            "anchor": s.anchor,
            "number": s.number,
            "title": s.title,
        }
        for s in document.sections
        if s.level <= _TOC_MAX_LEVEL
    ]
    sections = [
        {
            "anchor": s.anchor,
            "heading_tag": min(s.level + 1, 6),
            "level": min(s.level, 3),
            "number": s.number,
            "title": s.title,
            "html": s.html,
        }
        for s in document.sections
    ]
    return _template().render(
        page_title=document.title or "문서",
        modified_datetime=now.isoformat(timespec="seconds"),
        modified_text=now.strftime("%Y-%m-%d %H:%M:%S"),
        authors=document.authors,
        abstract_html=document.abstract_html,
        toc_items=toc_items,
        sections=sections,
        footnotes=[{"id": f.id, "html": f.html} for f in document.footnotes],
        references=[{"id": r.id, "html": r.html} for r in document.references],
        dark_mode=dark_mode,
        enable_fold=enable_fold,
    )
