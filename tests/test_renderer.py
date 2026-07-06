from datetime import datetime

from namulizer.document import Document, Footnote, Reference, Section
from namulizer.renderer import render_html


def _doc():
    return Document(
        title="Test Paper",
        authors=["Kim", "Lee"],
        abstract_html='<p class="wiki-paragraph">Summary.</p>',
        sections=[
            Section(title="Intro", level=1, number="1", anchor="s-1", html="<p>hello</p>"),
            Section(title="Sub", level=2, number="1.1", anchor="s-1.1", html="<p>deep</p>"),
        ],
        footnotes=[Footnote(id="1", html="A reference entry")],
        references=[Reference(id="1", html="Bib entry")],
    )


def test_render_basic_structure():
    html = render_html(_doc(), now=datetime(2026, 7, 6, 12, 0, 0))
    assert "<title>Test Paper - 나무위키</title>" in html
    assert "theseed-light-mode" in html
    assert "2026-07-06 12:00:00" in html
    assert "Kim, Lee" in html
    assert "Summary." in html
    # TOC entries link to section anchors
    assert 'href="#s-1"' in html and 'href="#s-1.1"' in html
    # numbered, foldable headings
    assert '<h2 class="wiki-heading level-1"' in html
    assert '<h3 class="wiki-heading level-2"' in html
    assert "wiki-section-fold-toggle" in html
    # footnote and reference blocks
    assert 'id="fn-1"' in html
    assert 'id="ref-1"' in html and "Bib entry" in html


def test_dark_mode_and_no_fold():
    html = render_html(_doc(), dark_mode=True, enable_fold=False)
    assert "theseed-dark-mode" in html
    # no fold toggle anchors rendered (similar strings appear in static JS)
    assert 'class="wiki-edit-section wiki-section-fold-toggle"' not in html
    assert 'data-enable-fold="0"' in html


def test_standalone_no_external_resources():
    html = render_html(_doc())
    for needle in ('src="http', 'href="http', "@import", "url(http"):
        assert needle not in html


def test_title_escaped():
    doc = _doc()
    doc.title = "<script>x</script>"
    html = render_html(doc)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html
