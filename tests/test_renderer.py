from namulizer.parser.base import EquationBlock, Footnote, Paper, Paragraph, Section, TableBlock
from namulizer.renderer.html_renderer import HTMLRenderer


def test_renderer_generates_toc_and_footnotes() -> None:
    paper = Paper(
        title="Renderer Test",
        authors=["Alice"],
        date="2025-01-01",
        abstract="Short abstract",
        sections=[
            Section(level=1, title="Intro", content=[Paragraph("Hello [fn:1].")]),
            Section(level=2, title="Math", content=[EquationBlock(latex="E=mc^2", display=True)]),
            Section(
                level=2,
                title="Table",
                content=[TableBlock(headers=["A", "B"], rows=[["1", "2"]], caption="Result")],
            ),
        ],
        footnotes=[Footnote(id=1, content="Footnote body")],
    )

    html = HTMLRenderer().render(paper)

    assert "theseed-light-mode" in html
    assert "wiki-macro-toc" in html
    assert "wiki-footnote-ref" in html
    assert 'data-footnote-id="1"' in html
    assert 'href="#fn-1"' in html
    assert "wiki-macro-footnote" in html
    assert "renderFootnotePopupHtml" in html
    assert "E=mc^2" in html
    assert "wiki-table" in html
    assert "최근 수정 시각" in html
