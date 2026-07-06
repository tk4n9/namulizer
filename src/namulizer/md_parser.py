"""Markdown -> Document parsing built on markdown-it-py tokens.

The parser understands generic Markdown as well as the dialect emitted by
pymupdf4llm for PDFs (page-break rules, bold-only heading lines). Sections
are numbered NamuWiki-style ("1", "1.1", ...), an Abstract section is lifted
into the page intro, and a References/Bibliography section is converted into
NamuWiki 각주 footnotes so inline citations like ``[3]`` get hover popups.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.token import Token
from mdit_py_plugins.footnote import footnote_plugin

from .document import Document, Footnote, Reference, Section

# ---------------------------------------------------------------------------
# Preprocessing of pymupdf4llm-flavoured Markdown

_PAGE_RULE = re.compile(r"^-{3,}\s*$")
_BOLD_LINE = re.compile(r"^\*\*(?!\s)(.+?)\*\*\s*$")
_NUMBERED_TITLE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
_KNOWN_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "background",
    "bibliography",
    "conclusion",
    "conclusions",
    "discussion",
    "evaluation",
    "introduction",
    "methods",
    "references",
    "related work",
    "results",
    "초록",
    "서론",
    "결론",
    "참고문헌",
}


def _is_blank(lines: list[str], index: int) -> bool:
    if index < 0 or index >= len(lines):
        return True
    return not lines[index].strip()


def _preprocess(text: str, from_pdf: bool) -> str:
    if not from_pdf:
        return text
    # pymupdf4llm emits a handful of raw HTML tags; html is disabled in the
    # parser (they would show up escaped), so reduce them to plain Markdown.
    text = re.sub(r"</?su[bp]>", "", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # pymupdf4llm separates PDF pages with dash rules; a rule directly
        # after a text line would otherwise parse as a setext heading.
        if _PAGE_RULE.match(stripped):
            out.append("")
            continue
        # Some papers style section titles as bold text instead of headings.
        match = _BOLD_LINE.match(stripped)
        if match and _is_blank(lines, i - 1) and _is_blank(lines, i + 1):
            inner = match.group(1).strip()
            numbered = _NUMBERED_TITLE.match(inner)
            if numbered:
                depth = numbered.group(1).count(".") + 1
                out.append("#" * min(depth + 1, 6) + " " + inner)
                continue
            if inner.rstrip(".:").lower() in _KNOWN_HEADINGS:
                out.append("## " + inner.rstrip(".:"))
                continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# markdown-it renderer rules producing NamuWiki markup

def _render_paragraph_open(self, tokens, idx, options, env):
    if tokens[idx].hidden:
        return ""
    return '<p class="wiki-paragraph">'


def _render_paragraph_close(self, tokens, idx, options, env):
    if tokens[idx].hidden:
        return ""
    return "</p>\n"


def _render_table_open(self, tokens, idx, options, env):
    return '<div class="wiki-table-wrap"><table class="wiki-table">\n'


def _render_table_close(self, tokens, idx, options, env):
    return "</table></div>\n"


def _embed_local_image(src: str, env: dict) -> str:
    if not env.get("_nmz_embed_images", True):
        return src
    if src.startswith(("data:", "http://", "https://", "//")):
        return src
    path = Path(src)
    if not path.is_absolute():
        base = env.get("_nmz_base_path")
        if base is None:
            return src
        path = Path(base) / src
    if not path.is_file():
        return src
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _render_image(self, tokens, idx, options, env):
    token = tokens[idx]
    src = _embed_local_image(token.attrGet("src") or "", env)
    alt = self.renderInlineAsText(token.children or [], options, env)
    img = f'<img src="{escapeHtml(src)}" alt="{escapeHtml(alt)}" loading="lazy" />'
    return f'<span class="wiki-image-align-center">{img}</span>'


def _footnote_anchor_html(fid: str, env: dict) -> str:
    seen = env.setdefault("_nmz_fnref_seen", set())
    id_attr = ""
    if fid not in seen:
        seen.add(fid)
        id_attr = f' id="fnref-{fid}"'
    return (
        f'<a{id_attr} class="wiki-fn-content wiki-footnote-ref" '
        f'data-footnote-id="{fid}" href="#fn-{fid}">[{fid}]</a>'
    )


def _render_footnote_ref(self, tokens, idx, options, env):
    fid = str(tokens[idx].meta["id"] + 1)
    return _footnote_anchor_html(fid, env)


def _render_footnote_anchor(self, tokens, idx, options, env):
    # The template renders its own [n] backlink; drop the default ↩ anchor.
    return ""


def _build_md() -> MarkdownIt:
    md = MarkdownIt("gfm-like", {"html": False}).use(footnote_plugin)
    md.add_render_rule("paragraph_open", _render_paragraph_open)
    md.add_render_rule("paragraph_close", _render_paragraph_close)
    md.add_render_rule("table_open", _render_table_open)
    md.add_render_rule("table_close", _render_table_close)
    md.add_render_rule("image", _render_image)
    md.add_render_rule("footnote_ref", _render_footnote_ref)
    md.add_render_rule("footnote_anchor", _render_footnote_anchor)
    return md


# ---------------------------------------------------------------------------
# Token stream analysis

@dataclass
class _Chunk:
    title: str
    level: int
    tokens: list[Token] = field(default_factory=list)


def _split_footnote_block(tokens: list[Token]) -> tuple[list[Token], list[Token]]:
    start = next(
        (i for i, t in enumerate(tokens) if t.type == "footnote_block_open"), None
    )
    if start is None:
        return tokens, []
    end = next(i for i, t in enumerate(tokens) if t.type == "footnote_block_close")
    return tokens[:start] + tokens[end + 1 :], tokens[start + 1 : end]


def _inline_text(md: MarkdownIt, token: Token | None, env: dict) -> str:
    if token is None:
        return ""
    return md.renderer.renderInlineAsText(token.children or [], md.options, env).strip()


def _extract_title(
    tokens: list[Token], md: MarkdownIt, env: dict
) -> tuple[str | None, list[Token]]:
    """Treat a unique h1 as the page title, not a section.

    PDFs often put boilerplate (copyright lines) before the actual title,
    so the h1 does not have to be the very first block.
    """
    h1_starts = [
        i for i, t in enumerate(tokens) if t.type == "heading_open" and t.tag == "h1"
    ]
    if len(h1_starts) == 1:
        i = h1_starts[0]
        title = _inline_text(md, tokens[i + 1] if i + 1 < len(tokens) else None, env)
        if title:
            return title, tokens[:i] + tokens[i + 3 :]
    return None, tokens


def _demoted_heading_tokens(inline: Token | None) -> list[Token]:
    """Turn a bogus heading back into a plain paragraph."""
    p_open = Token("paragraph_open", "p", 1)
    p_open.block = True
    p_close = Token("paragraph_close", "p", -1)
    p_close.block = True
    return [p_open, inline, p_close] if inline is not None else []


def _chunk_sections(
    tokens: list[Token], md: MarkdownIt, env: dict, *, from_pdf: bool = False
) -> tuple[list[Token], list[_Chunk]]:
    preamble: list[Token] = []
    chunks: list[_Chunk] = []
    current: _Chunk | None = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            title = _inline_text(md, inline, env)
            # Display equations sometimes get misdetected as headings by
            # font-size heuristics; no real section title contains "=".
            if from_pdf and "=" in title:
                container = current.tokens if current is not None else preamble
                container.extend(_demoted_heading_tokens(inline))
                i += 3
                continue
            current = _Chunk(title=title, level=int(token.tag[1]))
            chunks.append(current)
            i += 3  # heading_open, inline, heading_close
            continue
        (current.tokens if current is not None else preamble).append(token)
        i += 1
    return preamble, chunks


# ---------------------------------------------------------------------------
# Abstract / references handling

_LEAD_SECTION_NUM = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
_EMBEDDED_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_ABSTRACT_TITLES = {"abstract", "초록", "요약"}
_REFERENCE_TITLES = {
    "references",
    "reference",
    "bibliography",
    "works cited",
    "참고문헌",
    "참고 문헌",
}


def _norm_title(title: str) -> str:
    return _LEAD_SECTION_NUM.sub("", title).strip().rstrip(".:").lower()


@dataclass
class _RefEntry:
    id: str
    html: str


_REF_LEAD = re.compile(r"^\[?(\d{1,3})[\].)]{0,2}\s+")
_REF_MARKER = re.compile(r"\[(\d{1,3})\]")


def _reference_texts(tokens: list[Token]) -> list[str]:
    """Collect one raw-markdown string per bibliography entry candidate."""
    texts: list[str] = []
    item_depth = 0
    item_buf: list[str] | None = None
    for token in tokens:
        if token.type == "list_item_open":
            item_depth += 1
            if item_depth == 1:
                item_buf = []
        elif token.type == "list_item_close":
            item_depth -= 1
            if item_depth == 0 and item_buf is not None:
                texts.append(" ".join(item_buf))
                item_buf = None
        elif token.type == "inline":
            if item_depth > 0 and item_buf is not None:
                item_buf.append(token.content)
            else:
                texts.append(token.content)
    return texts


def _split_entry_text(text: str) -> list[str]:
    """Split a paragraph that packs several [n]-prefixed entries together."""
    text = text.strip()
    if not text:
        return []
    starts = [
        m
        for m in _REF_MARKER.finditer(text)
        if m.start() == 0 or text[m.start() - 1] == "\n"
    ]
    if len(starts) >= 2:
        indices = [m.start() for m in starts] + [len(text)]
        return [text[a:b].strip() for a, b in zip(indices, indices[1:]) if text[a:b].strip()]
    return [text]


def _parse_reference_entries(
    chunk: _Chunk, md: MarkdownIt, env: dict
) -> list[_RefEntry]:
    entries: list[_RefEntry] = []
    for raw in _reference_texts(chunk.tokens):
        for part in _split_entry_text(raw):
            part = " ".join(part.split("\n"))
            # stray page numbers land between entries in PDF extractions
            if re.fullmatch(r"[\d\s.\-–]*", part):
                continue
            lead = _REF_LEAD.match(part)
            if lead:
                ref_id = lead.group(1)
                part = part[lead.end() :].strip()
            else:
                ref_id = str(len(entries) + 1)
            if not part:
                continue
            entries.append(_RefEntry(id=ref_id, html=md.renderInline(part, env).strip()))
    return entries


def _collect_footnotes(
    fn_tokens: list[Token], md: MarkdownIt, env: dict
) -> list[Footnote]:
    notes: list[Footnote] = []
    i = 0
    while i < len(fn_tokens):
        token = fn_tokens[i]
        if token.type != "footnote_open":
            i += 1
            continue
        depth = 1
        j = i + 1
        while j < len(fn_tokens) and depth:
            if fn_tokens[j].type == "footnote_open":
                depth += 1
            elif fn_tokens[j].type == "footnote_close":
                depth -= 1
            j += 1
        body = fn_tokens[i + 1 : j - 1]
        html = md.renderer.render(body, md.options, env).strip()
        fid = str(token.meta.get("id", len(notes)) + 1)
        notes.append(Footnote(id=fid, html=_strip_outer_paragraph(html)))
        i = j
    return notes


def _strip_outer_paragraph(html: str) -> str:
    open_tag = '<p class="wiki-paragraph">'
    if html.startswith(open_tag) and html.endswith("</p>") and html.count("<p") == 1:
        return html[len(open_tag) : -len("</p>")].strip()
    return html


# ---------------------------------------------------------------------------
# Inline citation linking ([3] -> footnote anchor with hover popup)

_TAG_SPLIT = re.compile(r"(<a\b[^>]*>.*?</a>|<[^>]+>)", re.S)


def _link_citations(html: str, valid_ids: set[str], env: dict) -> str:
    parts = _TAG_SPLIT.split(html)

    def repl(match: re.Match) -> str:
        fid = match.group(1)
        if fid not in valid_ids:
            return match.group(0)
        return _footnote_anchor_html(fid, env)

    for i, part in enumerate(parts):
        if part and not part.startswith("<"):
            parts[i] = _REF_MARKER.sub(repl, part)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public API

def parse_markdown(
    text: str,
    *,
    from_pdf: bool = False,
    base_path: str | Path | None = None,
    embed_images: bool = True,
) -> Document:
    md = _build_md()
    env: dict = {
        "_nmz_base_path": Path(base_path) if base_path else None,
        "_nmz_embed_images": embed_images,
        "_nmz_fnref_seen": set(),
    }
    tokens = md.parse(_preprocess(text, from_pdf), env)
    tokens, fn_tokens = _split_footnote_block(tokens)
    title, tokens = _extract_title(tokens, md, env)
    preamble, chunks = _chunk_sections(tokens, md, env, from_pdf=from_pdf)

    abstract_chunk = next(
        (c for c in chunks if _norm_title(c.title) in _ABSTRACT_TITLES), None
    )
    ref_chunk = next(
        (c for c in chunks if _norm_title(c.title) in _REFERENCE_TITLES), None
    )
    body_chunks = [c for c in chunks if c is not abstract_chunk and c is not ref_chunk]

    md_footnotes = _collect_footnotes(fn_tokens, md, env)
    ref_entries = _parse_reference_entries(ref_chunk, md, env) if ref_chunk else []

    # Abstract = anything before the first heading + an explicit Abstract section.
    abstract_parts = []
    if preamble:
        abstract_parts.append(md.renderer.render(preamble, md.options, env).strip())
    if abstract_chunk and abstract_chunk.tokens:
        abstract_parts.append(
            md.renderer.render(abstract_chunk.tokens, md.options, env).strip()
        )
    abstract_html = "\n".join(p for p in abstract_parts if p) or None

    sections: list[Section] = []
    if body_chunks:
        base_level = min(c.level for c in body_chunks)
        counters: list[int] = []
        for chunk in body_chunks:
            display_title = chunk.title
            level = chunk.level - base_level + 1
            numbered = _EMBEDDED_NUMBER.match(chunk.title)
            if numbered:
                # NamuWiki renumbers sections itself; drop the original
                # number and, for PDFs, trust it for the nesting depth
                # (font-size heading detection flattens 3.2/3.2.1 levels).
                display_title = numbered.group(2).strip()
                if from_pdf:
                    level = numbered.group(1).count(".") + 1
            level = min(level, len(counters) + 1)
            if level > len(counters):
                counters.append(0)
            counters[level - 1] += 1
            del counters[level:]
            number = ".".join(str(n) for n in counters)
            sections.append(
                Section(
                    title=display_title or "(제목 없음)",
                    level=level,
                    number=number,
                    anchor=f"s-{number}",
                    html=md.renderer.render(chunk.tokens, md.options, env).strip(),
                )
            )

    footnotes: list[Footnote]
    references: list[Reference]
    if ref_entries and not md_footnotes:
        # No native footnotes: promote bibliography entries to 각주 so that
        # inline [n] citations get NamuWiki hover popups.
        footnotes = [Footnote(id=e.id, html=e.html) for e in ref_entries]
        valid_ids = {f.id for f in footnotes}
        # Page order: abstract first, so its citation gets the fnref anchor.
        if abstract_html:
            abstract_html = _link_citations(abstract_html, valid_ids, env)
        for section in sections:
            section.html = _link_citations(section.html, valid_ids, env)
        references = []
    else:
        footnotes = md_footnotes
        references = [Reference(id=e.id, html=e.html) for e in ref_entries]

    return Document(
        title=title or "",
        abstract_html=abstract_html,
        sections=sections,
        footnotes=footnotes,
        references=references,
    )
