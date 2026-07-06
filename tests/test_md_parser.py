import base64

from namulizer.md_parser import parse_markdown

PAPER_MD = """# Attention Is Not All You Need

## Abstract

We revisit the transformer architecture [1].

## 1 Introduction

Transformers changed everything [1]. Some doubt remains [2].

### 1.1 Motivation

Why another paper?

## 2 Method

We stack more layers.

## References

[1] A. Vaswani et al. Attention is all you need. NeurIPS 2017.
[2] J. Doe. Skeptical takes on attention. https://example.org/skeptic
"""


def test_title_from_single_h1():
    doc = parse_markdown(PAPER_MD)
    assert doc.title == "Attention Is Not All You Need"


def test_abstract_lifted_out_of_sections():
    doc = parse_markdown(PAPER_MD)
    assert doc.abstract_html is not None
    assert "revisit the transformer" in doc.abstract_html
    assert all("abstract" != s.title.lower() for s in doc.sections)


def test_section_numbering_and_anchors():
    doc = parse_markdown(PAPER_MD)
    numbers = [(s.number, s.level, s.title) for s in doc.sections]
    # original heading numbers are stripped; NamuWiki renumbers sections
    assert numbers == [
        ("1", 1, "Introduction"),
        ("1.1", 2, "Motivation"),
        ("2", 1, "Method"),
    ]
    assert [s.anchor for s in doc.sections] == ["s-1", "s-1.1", "s-2"]


def test_references_become_footnotes_with_citation_links():
    doc = parse_markdown(PAPER_MD)
    assert not doc.references
    assert [f.id for f in doc.footnotes] == ["1", "2"]
    assert "Vaswani" in doc.footnotes[0].html
    # URL in a reference should be linkified
    assert '<a href="https://example.org/skeptic"' in doc.footnotes[1].html
    intro = doc.sections[0].html
    assert 'data-footnote-id="1"' in intro
    assert 'href="#fn-2"' in intro
    # first occurrence carries the backlink anchor; abstract cites [1] first
    assert 'id="fnref-1"' in doc.abstract_html


def test_native_markdown_footnotes_win_over_references():
    md = """# T

## Body

Claim.[^1]

[^1]: Real footnote text.

## References

[1] Someone. Something.
"""
    doc = parse_markdown(md)
    assert [f.id for f in doc.footnotes] == ["1"]
    assert "Real footnote text" in doc.footnotes[0].html
    # bibliography stays a plain reference block, no popup hijacking
    assert doc.references and doc.references[0].id == "1"
    assert "Someone. Something." in doc.references[0].html


def test_pdf_bold_headings_promoted_and_page_rules_stripped():
    md = "\n".join(
        [
            "# Paper Title",
            "",
            "**Abstract**",
            "",
            "Short summary.",
            "",
            "-----",
            "",
            "**1. Introduction**",
            "",
            "Body text.",
            "",
            "**2.1 Details**",
            "",
            "More text.",
        ]
    )
    doc = parse_markdown(md, from_pdf=True)
    assert doc.title == "Paper Title"
    assert doc.abstract_html and "Short summary" in doc.abstract_html
    assert [s.title for s in doc.sections] == ["Introduction", "Details"]
    # "2.1" has one dot, so it nests one level below "1."
    assert [(s.number, s.level) for s in doc.sections] == [("1", 1), ("1.1", 2)]
    # the page rule must not create an <hr> or setext heading
    assert all("<hr" not in s.html for s in doc.sections)


def test_local_images_embedded_as_data_uri(tmp_path):
    png = tmp_path / "fig.png"
    png.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    ))
    md = f"# T\n\n## Fig\n\n![tiny]({png.name})\n"
    doc = parse_markdown(md, base_path=tmp_path)
    assert 'src="data:image/png;base64,' in doc.sections[0].html
    assert 'alt="tiny"' in doc.sections[0].html


def test_images_left_alone_when_embedding_disabled(tmp_path):
    md = "# T\n\n## Fig\n\n![tiny](fig.png)\n"
    doc = parse_markdown(md, base_path=tmp_path, embed_images=False)
    assert 'src="fig.png"' in doc.sections[0].html


def test_tables_get_namuwiki_classes():
    md = "# T\n\n## Data\n\n| a | b |\n| - | - |\n| 1 | 2 |\n"
    doc = parse_markdown(md)
    html = doc.sections[0].html
    assert '<div class="wiki-table-wrap"><table class="wiki-table">' in html


def test_preamble_without_headings_becomes_abstract():
    doc = parse_markdown("Just one paragraph, no headings at all.")
    assert doc.sections == []
    assert "one paragraph" in doc.abstract_html
