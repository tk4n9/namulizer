"""TeX source parser with heuristic extraction into Paper IR."""

from __future__ import annotations

import base64
import mimetypes
import re
import tempfile
from pathlib import Path
from zipfile import ZipFile

from .base import EquationBlock, FigureBlock, Footnote, Paper, Paragraph, Reference, Section, TableBlock

_SECTION_LEVELS = {
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
}

_MATH_ENVS = ("equation", "align", "gather", "multline")


class TeXParser:
    """Parse TeX input (single .tex or zip of TeX source tree)."""

    def __init__(self, embed_images: bool = True) -> None:
        self.embed_images = embed_images
        self._footnote_id = 1
        self._footnotes: list[Footnote] = []

    def parse(self, input_path: Path) -> Paper:
        input_path = Path(input_path)
        self._footnote_id = 1
        self._footnotes = []

        if input_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory(prefix="namulizer_tex_") as tmp:
                tmp_path = Path(tmp)
                with ZipFile(input_path) as zf:
                    zf.extractall(tmp_path)
                return self._parse_tex_tree(tmp_path)

        if input_path.suffix.lower() == ".tex":
            return self._parse_tex_file(input_path)

        raise ValueError(f"Unsupported TeX input: {input_path}")

    def _parse_tex_tree(self, root: Path) -> Paper:
        tex_files = sorted(root.rglob("*.tex"))
        if not tex_files:
            raise ValueError("No .tex file found in archive")

        main_tex = self._detect_main_tex(tex_files)
        paper = self._parse_tex_file(main_tex)
        paper.references.extend(self._parse_bib_files(root))
        return paper

    def _detect_main_tex(self, tex_files: list[Path]) -> Path:
        named_main = [p for p in tex_files if p.name.lower() in {"main.tex", "paper.tex"}]
        if named_main:
            return named_main[0]

        best_score = -1
        best_path = tex_files[0]
        for path in tex_files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            score = text.count("\\begin{document}") * 10 + text.count("\\section")
            if score > best_score:
                best_score = score
                best_path = path
        return best_path

    def _parse_tex_file(self, tex_path: Path) -> Paper:
        raw = tex_path.read_text(encoding="utf-8", errors="ignore")
        text = _strip_comments(raw)

        title = _clean_inline_tex(_extract_command_value(text, "title") or tex_path.stem)
        authors_raw = _extract_command_value(text, "author") or ""
        date = _clean_inline_tex(_extract_command_value(text, "date") or "").strip() or None
        authors = self._split_authors(authors_raw)
        abstract = _clean_inline_tex(_extract_environment_body(text, "abstract") or "").strip()

        body = _extract_document_body(text)
        body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.DOTALL)
        sections = self._parse_sections(body, tex_path.parent)

        if not sections:
            sections = [Section(level=1, title="본문", content=self._parse_content_blocks(body, tex_path.parent))]

        return Paper(
            title=title,
            authors=authors,
            date=date,
            abstract=abstract,
            sections=sections,
            footnotes=self._footnotes.copy(),
            references=[],
        )

    def _split_authors(self, raw: str) -> list[str]:
        parts = re.split(r"\\and|\\\\|,", raw)
        return [_clean_inline_tex(p).strip() for p in parts if _clean_inline_tex(p).strip()]

    def _parse_sections(self, body: str, asset_root: Path) -> list[Section]:
        spans = []
        pattern = re.compile(r"\\(section|subsection|subsubsection)\*?\{")
        for match in pattern.finditer(body):
            kind = match.group(1)
            title, title_end = _read_balanced_braces(body, match.end() - 1)
            spans.append((match.start(), title_end, _SECTION_LEVELS[kind], _clean_inline_tex(title).strip()))

        if not spans:
            return []

        sections: list[Section] = []

        prelude = body[: spans[0][0]].strip()
        if prelude:
            prelude_blocks = self._parse_content_blocks(prelude, asset_root)
            if prelude_blocks:
                sections.append(Section(level=1, title="개요", content=prelude_blocks))

        for idx, (start, content_start, level, title) in enumerate(spans):
            content_end = spans[idx + 1][0] if idx + 1 < len(spans) else len(body)
            content = body[content_start:content_end]
            blocks = self._parse_content_blocks(content, asset_root)
            sections.append(Section(level=level, title=title or "(제목 없음)", content=blocks))

        return sections

    def _parse_content_blocks(self, text: str, asset_root: Path) -> list[Paragraph | FigureBlock | TableBlock | EquationBlock]:
        blocks: list[Paragraph | FigureBlock | TableBlock | EquationBlock] = []

        i = 0
        while i < len(text):
            hits: list[tuple[int, str]] = []

            for env in ("figure", "table", *_MATH_ENVS):
                token = f"\\begin{{{env}}}"
                pos = text.find(token, i)
                if pos != -1:
                    hits.append((pos, env))

            for token, label in (("\\[", "math_bracket"), ("$$", "math_dollar")):
                pos = text.find(token, i)
                if pos != -1:
                    hits.append((pos, label))

            if not hits:
                blocks.extend(self._parse_paragraphs(text[i:]))
                break

            next_pos, kind = min(hits, key=lambda item: item[0])

            if next_pos > i:
                blocks.extend(self._parse_paragraphs(text[i:next_pos]))

            if kind == "figure":
                env_text, end = _extract_environment(text, "figure", next_pos)
                blocks.append(self._parse_figure(env_text, asset_root))
                i = end
                continue

            if kind == "table":
                env_text, end = _extract_environment(text, "table", next_pos)
                blocks.append(self._parse_table(env_text))
                i = end
                continue

            if kind in _MATH_ENVS:
                env_text, end = _extract_environment(text, kind, next_pos)
                blocks.append(EquationBlock(latex=_clean_math_block(env_text), display=True))
                i = end
                continue

            if kind == "math_bracket":
                end = text.find("\\]", next_pos + 2)
                if end == -1:
                    blocks.extend(self._parse_paragraphs(text[next_pos:]))
                    break
                latex = text[next_pos + 2 : end]
                blocks.append(EquationBlock(latex=_clean_math_block(latex), display=True))
                i = end + 2
                continue

            if kind == "math_dollar":
                end = text.find("$$", next_pos + 2)
                if end == -1:
                    blocks.extend(self._parse_paragraphs(text[next_pos:]))
                    break
                latex = text[next_pos + 2 : end]
                blocks.append(EquationBlock(latex=_clean_math_block(latex), display=True))
                i = end + 2
                continue

        return [b for b in blocks if not isinstance(b, Paragraph) or b.text.strip()]

    def _parse_paragraphs(self, text: str) -> list[Paragraph]:
        cleaned = text.replace("\r", "")
        parts = re.split(r"\n\s*\n", cleaned)
        paragraphs: list[Paragraph] = []
        for part in parts:
            sentence = _normalize_whitespace(part)
            if not sentence:
                continue
            sentence = _remove_low_value_commands(sentence)
            sentence = self._extract_footnotes_from_text(sentence)
            sentence = _clean_inline_tex(sentence)
            if sentence:
                paragraphs.append(Paragraph(text=sentence))
        return paragraphs

    def _extract_footnotes_from_text(self, text: str) -> str:
        token = "\\footnote{"
        out = []
        i = 0
        while i < len(text):
            start = text.find(token, i)
            if start == -1:
                out.append(text[i:])
                break
            out.append(text[i:start])
            content, end = _read_balanced_braces(text, start + len(token) - 1)
            fn_content = _clean_inline_tex(content).strip()
            fn_id = self._footnote_id
            self._footnote_id += 1
            self._footnotes.append(Footnote(id=fn_id, content=fn_content))
            out.append(f" [fn:{fn_id}] ")
            i = end
        return "".join(out)

    def _parse_figure(self, env_text: str, asset_root: Path) -> FigureBlock:
        caption = _clean_inline_tex(_extract_command_value(env_text, "caption") or "")
        source = _extract_includegraphics_path(env_text)

        if not source:
            return FigureBlock(caption=caption)

        source_path = (asset_root / source).resolve() if not Path(source).is_absolute() else Path(source)
        data_uri = None
        if self.embed_images and source_path.exists():
            data_uri = _file_to_data_uri(source_path)

        return FigureBlock(caption=caption, source_path=str(source_path), data_uri=data_uri)

    def _parse_table(self, env_text: str) -> TableBlock:
        caption = _clean_inline_tex(_extract_command_value(env_text, "caption") or "")
        tabular = _extract_environment_body(env_text, "tabular") or ""
        tabular = re.sub(r"^\s*\{[^{}]*\}\s*", "", tabular, count=1)
        rows_raw = [r.strip() for r in re.split(r"\\\\", tabular) if r.strip()]

        rows: list[list[str]] = []
        for row in rows_raw:
            row = re.sub(r"\\hline", "", row)
            cells = [_clean_inline_tex(c).strip() for c in row.split("&")]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)

        headers: list[str] = []
        body_rows: list[list[str]] = []
        if rows:
            headers = rows[0]
            body_rows = rows[1:]

        return TableBlock(headers=headers, rows=body_rows, caption=caption)

    def _parse_bib_files(self, root: Path) -> list[Reference]:
        refs: list[Reference] = []
        for bib in sorted(root.rglob("*.bib")):
            text = bib.read_text(encoding="utf-8", errors="ignore")
            for entry_match in re.finditer(r"@\w+\{\s*([^,]+),(.+?)\n\}\s*", text, flags=re.DOTALL):
                key = entry_match.group(1).strip()
                body = entry_match.group(2)
                title = _extract_bib_field(body, "title") or key
                refs.append(Reference(key=key, text=_clean_inline_tex(title)))
        return refs


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(stripped)
    return "\n".join(lines)


def _extract_document_body(text: str) -> str:
    doc_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, flags=re.DOTALL)
    return doc_match.group(1) if doc_match else text


def _extract_command_value(text: str, command: str) -> str | None:
    match = re.search(rf"\\{command}\*?\{{", text)
    if not match:
        return None
    value, _end = _read_balanced_braces(text, match.end() - 1)
    return value


def _extract_environment(text: str, env: str, start: int) -> tuple[str, int]:
    begin = f"\\begin{{{env}}}"
    end_token = f"\\end{{{env}}}"
    begin_idx = text.find(begin, start)
    if begin_idx == -1:
        return "", start
    content_start = begin_idx + len(begin)
    end_idx = text.find(end_token, content_start)
    if end_idx == -1:
        return text[content_start:], len(text)
    return text[content_start:end_idx], end_idx + len(end_token)


def _extract_environment_body(text: str, env: str) -> str | None:
    match = re.search(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", text, flags=re.DOTALL)
    return match.group(1) if match else None


def _read_balanced_braces(text: str, brace_start: int) -> tuple[str, int]:
    if brace_start >= len(text) or text[brace_start] != "{":
        return "", brace_start
    depth = 0
    chars = []
    i = brace_start
    while i < len(text):
        ch = text[i]
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return "".join(chars), i + 1
            chars.append(ch)
        else:
            if depth >= 1:
                chars.append(ch)
        i += 1
    return "".join(chars), i


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _remove_low_value_commands(text: str) -> str:
    text = re.sub(r"\\(label|ref|cite|citep|citet)\{[^{}]*\}", "", text)
    text = re.sub(r"\\(maketitle|tableofcontents|newpage|clearpage)\b", "", text)
    text = re.sub(r"\\(emph|textit|textbf)\{([^{}]*)\}", r"\2", text)
    text = text.replace("~", " ")
    text = text.replace("\\\n", " ")
    return text


def _clean_inline_tex(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    return _normalize_whitespace(text)


def _clean_math_block(text: str) -> str:
    text = re.sub(r"\\label\{[^{}]*\}", "", text)
    return _normalize_whitespace(text)


def _extract_includegraphics_path(text: str) -> str | None:
    match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", text)
    if not match:
        return None
    return match.group(1).strip()


def _file_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _extract_bib_field(entry_body: str, field: str) -> str | None:
    match = re.search(rf"{field}\s*=\s*[\"\{{]([^\"\}}]+)[\"\}}]", entry_body, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None
