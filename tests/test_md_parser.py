"""Tests for the Markdown parser.

Covers:
- YAML frontmatter (title, authors, date, abstract)
- Title extraction from # heading
- Section splitting (##, ###, ####)
- Paragraph parsing and inline cleanup
- Display math ($$...$$) and fenced math blocks
- Images (![alt](src))
- Markdown tables
- Footnotes ([^label] / [^label]: definition)
- CLI parser selection for .md/.markdown extensions
- End-to-end rendering
"""

from __future__ import annotations

from pathlib import Path

from namulizer.parser.base import EquationBlock, FigureBlock, Paragraph, TableBlock
from namulizer.parser.md_parser import MarkdownParser


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

def test_frontmatter_metadata(tmp_path: Path) -> None:
    md = """\
---
title: "My Paper Title"
author: Alice, Bob
date: 2025-06-01
abstract: A short abstract.
---

## Introduction
Hello world.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert paper.title == "My Paper Title"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.date == "2025-06-01"
    assert paper.abstract == "A short abstract."


def test_frontmatter_authors_yaml_list(tmp_path: Path) -> None:
    md = """\
---
title: Test
authors: [Alice, Bob, Carol]
---

## Body
Text.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert paper.authors == ["Alice", "Bob", "Carol"]


def test_frontmatter_authors_with_and(tmp_path: Path) -> None:
    md = """\
---
title: Test
author: Alice and Bob
---

## Body
Text.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert paper.authors == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# Title from # heading
# ---------------------------------------------------------------------------

def test_title_from_h1(tmp_path: Path) -> None:
    md = """\
# Paper Title From Heading

## Introduction
Body text.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert paper.title == "Paper Title From Heading"


def test_title_fallback_to_filename(tmp_path: Path) -> None:
    md = """\
## Section Only
Some text.
"""
    p = tmp_path / "my_paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert paper.title == "my_paper"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def test_section_levels(tmp_path: Path) -> None:
    md = """\
# Title

## Introduction
Intro text.

### Motivation
Why we did this.

## Method
Method text.

### Details
Detail text.

#### Sub-details
Sub-detail text.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    titles = [(s.level, s.title) for s in paper.sections]

    assert (1, "Introduction") in titles
    assert (2, "Motivation") in titles
    assert (1, "Method") in titles
    assert (2, "Details") in titles
    assert (3, "Sub-details") in titles


def test_prelude_before_first_heading(tmp_path: Path) -> None:
    md = """\
# Title

Some preamble text before any section.

## Section One
Body.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert paper.sections[0].title == "개요"
    assert any(
        isinstance(b, Paragraph) and "preamble" in b.text
        for b in paper.sections[0].content
    )


def test_no_sections_fallback(tmp_path: Path) -> None:
    md = "Just some plain text without any headings."
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert len(paper.sections) == 1
    assert paper.sections[0].title == "본문"


# ---------------------------------------------------------------------------
# Paragraphs & inline formatting
# ---------------------------------------------------------------------------

def test_paragraphs_split_on_blank_lines(tmp_path: Path) -> None:
    md = """\
## Section

First paragraph.

Second paragraph.

Third paragraph.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    paragraphs = [b for b in paper.sections[0].content if isinstance(b, Paragraph)]
    assert len(paragraphs) == 3


def test_inline_formatting_stripped(tmp_path: Path) -> None:
    md = """\
## Section

This is **bold** and *italic* and `code` and [a link](http://x.com).
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    para = [b for b in paper.sections[0].content if isinstance(b, Paragraph)][0]
    assert "**" not in para.text
    assert "*italic*" not in para.text
    assert "`code`" not in para.text
    assert "bold" in para.text
    assert "italic" in para.text
    assert "code" in para.text
    assert "a link" in para.text
    assert "http://x.com" not in para.text


# ---------------------------------------------------------------------------
# Display math
# ---------------------------------------------------------------------------

def test_display_math_dollar(tmp_path: Path) -> None:
    md = """\
## Theory

Some text.

$$
E = mc^2
$$

More text.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    blocks = paper.sections[0].content
    equations = [b for b in blocks if isinstance(b, EquationBlock)]
    assert len(equations) == 1
    assert "E = mc^2" in equations[0].latex
    assert equations[0].display is True


def test_fenced_math_block(tmp_path: Path) -> None:
    md = """\
## Theory

```math
\\nabla \\cdot E = \\frac{\\rho}{\\epsilon_0}
```
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    blocks = paper.sections[0].content
    equations = [b for b in blocks if isinstance(b, EquationBlock)]
    assert len(equations) == 1
    assert "nabla" in equations[0].latex


def test_fenced_latex_block(tmp_path: Path) -> None:
    md = """\
## Theory

```latex
x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}
```
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    equations = [b for s in paper.sections for b in s.content if isinstance(b, EquationBlock)]
    assert len(equations) == 1


def test_regular_code_block_is_paragraph(tmp_path: Path) -> None:
    md = """\
## Code

```python
print("hello")
```
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    blocks = paper.sections[0].content
    # Regular code block should be a paragraph, not equation.
    assert not any(isinstance(b, EquationBlock) for b in blocks)
    paragraphs = [b for b in blocks if isinstance(b, Paragraph)]
    assert any('print("hello")' in b.text for b in paragraphs)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def test_image_extraction(tmp_path: Path) -> None:
    md = """\
## Results

![Architecture diagram](figures/arch.png)
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    figures = [b for s in paper.sections for b in s.content if isinstance(b, FigureBlock)]
    assert len(figures) == 1
    assert figures[0].caption == "Architecture diagram"


def test_image_embed(tmp_path: Path) -> None:
    # Create a tiny 1x1 PNG.
    import struct, zlib
    def make_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        raw = b"\x00\x00\x00\x00"
        idat_data = zlib.compress(raw)
        idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
        idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return sig + ihdr + idat + iend

    img_path = tmp_path / "test.png"
    img_path.write_bytes(make_png())

    md = f"## Sec\n\n![cap]({img_path.name})\n"
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=True).parse(p)
    figures = [b for s in paper.sections for b in s.content if isinstance(b, FigureBlock)]
    assert len(figures) == 1
    assert figures[0].data_uri is not None
    assert figures[0].data_uri.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_table_parsing(tmp_path: Path) -> None:
    md = """\
## Results

| Model | BLEU | Params |
|-------|------|--------|
| Base  | 27.3 | 65M    |
| Big   | 28.4 | 213M   |
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    tables = [b for s in paper.sections for b in s.content if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].headers == ["Model", "BLEU", "Params"]
    assert len(tables[0].rows) == 2
    assert tables[0].rows[0] == ["Base", "27.3", "65M"]


# ---------------------------------------------------------------------------
# Footnotes
# ---------------------------------------------------------------------------

def test_footnotes(tmp_path: Path) -> None:
    md = """\
## Introduction

This has a footnote[^1] and another[^note].

[^1]: First footnote content.
[^note]: Second footnote content.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    paper = MarkdownParser(embed_images=False).parse(p)
    assert len(paper.footnotes) == 2
    assert paper.footnotes[0].id == 1
    assert paper.footnotes[0].content == "First footnote content."
    assert paper.footnotes[1].id == 2
    assert paper.footnotes[1].content == "Second footnote content."

    # References in text should be rewritten to [fn:N].
    para = [b for s in paper.sections for b in s.content if isinstance(b, Paragraph)][0]
    assert "[fn:1]" in para.text
    assert "[fn:2]" in para.text
    assert "[^1]" not in para.text


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_selects_md_parser() -> None:
    from namulizer.cli import _select_parser
    from namulizer.parser.md_parser import MarkdownParser as MP

    for ext in (".md", ".markdown"):
        parser = _select_parser(Path(f"/tmp/fake{ext}"), embed_images=False)
        assert isinstance(parser, MP), f"Failed for extension {ext}"


# ---------------------------------------------------------------------------
# End-to-end rendering
# ---------------------------------------------------------------------------

def test_end_to_end_render(tmp_path: Path) -> None:
    md = """\
---
title: "Self-Attention for NLP"
author: Alice Kim, Bob Lee
date: 2025-03-15
abstract: We present a novel attention mechanism.
---

## Introduction

Attention mechanisms[^attn] have revolutionized NLP.

## Method

Our method uses relative position encodings.

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
$$

### Architecture

| Layer | Dim | Heads |
|-------|-----|-------|
| 1     | 512 | 8     |
| 6     | 512 | 8     |

## Results

Our model achieves state-of-the-art results.

[^attn]: Bahdanau et al., 2014.
"""
    p = tmp_path / "paper.md"
    p.write_text(md)

    from namulizer.renderer.html_renderer import HTMLRenderer

    parser = MarkdownParser(embed_images=False)
    paper = parser.parse(p)
    renderer = HTMLRenderer()
    html = renderer.render(paper, math_engine="katex")

    # Metadata.
    assert "Self-Attention for NLP" in html
    assert "Alice Kim" in html
    assert "2025-03-15" in html

    # Structure.
    assert "wiki-macro-toc" in html
    assert "Introduction" in html
    assert "Method" in html
    assert "Architecture" in html
    assert "Results" in html

    # Content.
    assert "wiki-table" in html
    assert "wiki-equation" in html
    assert "wiki-footnote-ref" in html
