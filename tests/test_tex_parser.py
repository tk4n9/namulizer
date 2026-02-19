from pathlib import Path

from namulizer.parser.tex_parser import TeXParser


def test_tex_parser_extracts_sections_and_footnotes(tmp_path: Path) -> None:
    tex = r"""
    \title{A Test Paper}
    \author{Alice \and Bob}
    \date{2024-01-01}
    \begin{document}
    \maketitle
    \begin{abstract}
    Abstract text.
    \end{abstract}

    \section{Intro}
    Intro sentence with note\footnote{Important footnote}.

    \subsection{Method}
    Method body.

    \end{document}
    """
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    parser = TeXParser(embed_images=False)
    paper = parser.parse(tex_path)

    assert paper.title == "A Test Paper"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.abstract == "Abstract text."
    assert len(paper.sections) >= 2
    assert paper.sections[0].title in {"개요", "Intro"}
    assert paper.footnotes
    assert paper.footnotes[0].content == "Important footnote"
