"""Tests for arXiv-specific TeX parsing features.

Covers:
- tar.gz archive extraction
- \\input{}/\\include{} resolution for multi-file projects
- Starred environments (figure*, table*, equation*, align*)
- CLI parser selection for archive extensions
"""

from __future__ import annotations

import gzip
import tarfile
from io import BytesIO
from pathlib import Path

from namulizer.parser.tex_parser import TeXParser, _is_tar_archive, _resolve_inputs, _strip_comments


# ---------------------------------------------------------------------------
# tar.gz support
# ---------------------------------------------------------------------------

def _make_tar_gz(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a .tar.gz archive from a dict of {relative_name: content}."""
    archive_path = tmp_path / "paper.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
    return archive_path


def test_tar_gz_single_file(tmp_path: Path) -> None:
    tex = r"""
    \title{Tar Test}
    \author{Alice}
    \begin{document}
    \begin{abstract}A tar abstract.\end{abstract}
    \section{Intro}
    Hello from tar.
    \end{document}
    """
    archive = _make_tar_gz(tmp_path, {"paper.tex": tex})

    parser = TeXParser(embed_images=False)
    paper = parser.parse(archive)

    assert paper.title == "Tar Test"
    assert paper.authors == ["Alice"]
    assert paper.abstract == "A tar abstract."
    assert any(s.title == "Intro" for s in paper.sections)


def test_tar_gz_multi_file_with_input(tmp_path: Path) -> None:
    main_tex = r"""
    \title{Multi-file arXiv Paper}
    \author{Bob \and Carol}
    \begin{document}
    \begin{abstract}An abstract.\end{abstract}
    \input{intro}
    \input{method.tex}
    \end{document}
    """
    intro_tex = r"""
    \section{Introduction}
    This is the introduction.
    """
    method_tex = r"""
    \section{Method}
    This is the method section.
    """
    archive = _make_tar_gz(tmp_path, {
        "main.tex": main_tex,
        "intro.tex": intro_tex,
        "method.tex": method_tex,
    })

    parser = TeXParser(embed_images=False)
    paper = parser.parse(archive)

    assert paper.title == "Multi-file arXiv Paper"
    assert paper.authors == ["Bob", "Carol"]
    section_titles = [s.title for s in paper.sections]
    assert "Introduction" in section_titles
    assert "Method" in section_titles


def test_gzip_single_tex(tmp_path: Path) -> None:
    """A .gz file containing a single TeX file (not tar)."""
    tex = r"""
    \title{Gzip Test}
    \author{Dave}
    \begin{document}
    \section{Results}
    Result content.
    \end{document}
    """
    gz_path = tmp_path / "paper.tex.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(tex.encode("utf-8"))

    parser = TeXParser(embed_images=False)
    paper = parser.parse(gz_path)

    assert paper.title == "Gzip Test"
    assert any(s.title == "Results" for s in paper.sections)


def test_is_tar_archive(tmp_path: Path) -> None:
    # Real tar.gz
    tar_path = _make_tar_gz(tmp_path, {"a.tex": "hello"})
    assert _is_tar_archive(tar_path) is True

    # Plain .tex file
    tex_path = tmp_path / "plain.tex"
    tex_path.write_text("hello")
    assert _is_tar_archive(tex_path) is False

    # .tgz extension
    tgz_path = tmp_path / "paper.tgz"
    with tarfile.open(tgz_path, "w:gz") as tf:
        data = b"test"
        info = tarfile.TarInfo(name="a.tex")
        info.size = len(data)
        tf.addfile(info, BytesIO(data))
    assert _is_tar_archive(tgz_path) is True


# ---------------------------------------------------------------------------
# \input{} / \include{} resolution
# ---------------------------------------------------------------------------

def test_resolve_inputs_basic(tmp_path: Path) -> None:
    (tmp_path / "intro.tex").write_text("Introduction content here.")
    text = r"Before \input{intro} after."

    resolved = _resolve_inputs(text, tmp_path)
    assert "Introduction content here." in resolved
    assert r"\input{intro}" not in resolved
    assert "Before" in resolved
    assert "after." in resolved


def test_resolve_inputs_adds_tex_extension(tmp_path: Path) -> None:
    (tmp_path / "methods.tex").write_text("Methods content.")
    text = r"\input{methods}"

    resolved = _resolve_inputs(text, tmp_path)
    assert "Methods content." in resolved


def test_resolve_inputs_include(tmp_path: Path) -> None:
    (tmp_path / "chapter.tex").write_text("Chapter body text.")
    text = r"\include{chapter}"

    resolved = _resolve_inputs(text, tmp_path)
    assert "Chapter body text." in resolved


def test_resolve_inputs_nested(tmp_path: Path) -> None:
    (tmp_path / "outer.tex").write_text(r"Outer \input{inner}")
    (tmp_path / "inner.tex").write_text("Inner content.")
    text = r"\input{outer}"

    resolved = _resolve_inputs(text, tmp_path)
    assert "Outer" in resolved
    assert "Inner content." in resolved


def test_resolve_inputs_circular_guard(tmp_path: Path) -> None:
    """Circular includes should not cause infinite recursion."""
    (tmp_path / "a.tex").write_text(r"\input{b}")
    (tmp_path / "b.tex").write_text(r"\input{a}")
    text = r"\input{a}"

    # Should complete without hanging.
    resolved = _resolve_inputs(text, tmp_path)
    assert resolved is not None


def test_resolve_inputs_missing_file(tmp_path: Path) -> None:
    """Missing files leave the \\input command in place."""
    text = r"\input{nonexistent}"
    resolved = _resolve_inputs(text, tmp_path)
    assert r"\input{nonexistent}" in resolved


def test_resolve_inputs_strips_comments(tmp_path: Path) -> None:
    (tmp_path / "commented.tex").write_text("Real content. % this is a comment\n")
    text = r"\input{commented}"

    resolved = _resolve_inputs(text, tmp_path)
    assert "Real content." in resolved
    assert "this is a comment" not in resolved


# ---------------------------------------------------------------------------
# Starred environments
# ---------------------------------------------------------------------------

def test_starred_figure(tmp_path: Path) -> None:
    tex = r"""
    \title{Starred Test}
    \author{Eve}
    \begin{document}
    \section{Results}
    Some text.
    \begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{plot.png}
    \caption{A wide figure.}
    \end{figure*}
    More text.
    \end{document}
    """
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text(tex)

    parser = TeXParser(embed_images=False)
    paper = parser.parse(tex_path)

    from namulizer.parser.base import FigureBlock
    figures = [b for s in paper.sections for b in s.content if isinstance(b, FigureBlock)]
    assert len(figures) == 1
    assert figures[0].caption == "A wide figure."


def test_starred_table(tmp_path: Path) -> None:
    tex = r"""
    \title{Table Test}
    \author{Frank}
    \begin{document}
    \section{Results}
    \begin{table*}[h]
    \caption{Wide table.}
    \begin{tabular}{lcc}
    Model & Acc & F1 \\
    Ours & 95.1 & 94.3 \\
    \end{tabular}
    \end{table*}
    \end{document}
    """
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text(tex)

    parser = TeXParser(embed_images=False)
    paper = parser.parse(tex_path)

    from namulizer.parser.base import TableBlock
    tables = [b for s in paper.sections for b in s.content if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].caption == "Wide table."
    assert tables[0].headers == ["Model", "Acc", "F1"]


def test_starred_equation(tmp_path: Path) -> None:
    tex = r"""
    \title{Equation Test}
    \author{Grace}
    \begin{document}
    \section{Theory}
    \begin{equation*}
    E = mc^2
    \end{equation*}
    \begin{align*}
    a &= b + c \\
    d &= e + f
    \end{align*}
    \end{document}
    """
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text(tex)

    parser = TeXParser(embed_images=False)
    paper = parser.parse(tex_path)

    from namulizer.parser.base import EquationBlock
    equations = [b for s in paper.sections for b in s.content if isinstance(b, EquationBlock)]
    assert len(equations) == 2
    assert "E = mc^2" in equations[0].latex
    assert "a" in equations[1].latex


# ---------------------------------------------------------------------------
# CLI parser selection
# ---------------------------------------------------------------------------

def test_cli_selects_tex_parser_for_tar_gz() -> None:
    from namulizer.cli import _select_parser
    from namulizer.parser.tex_parser import TeXParser as TP
    from pathlib import Path

    for ext in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar", ".gz"):
        # _select_parser checks extension only, not file existence.
        path = Path(f"/tmp/fake{ext}")
        parser = _select_parser(path, embed_images=False)
        assert isinstance(parser, TP), f"Failed for extension {ext}"


# ---------------------------------------------------------------------------
# End-to-end: realistic arXiv-style paper
# ---------------------------------------------------------------------------

def test_full_arxiv_paper_tar_gz(tmp_path: Path) -> None:
    """Simulates a realistic arXiv multi-file paper in tar.gz format."""
    main_tex = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}

\title{Attention Is All You Need (Test)}
\author{Ashish Vaswani \and Noam Shazeer \and Niki Parmar}
\date{June 2017}

\begin{document}
\maketitle

\begin{abstract}
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks. We propose a new simple network architecture,
the Transformer, based solely on attention mechanisms.
\end{abstract}

\input{introduction}
\input{model}
\input{experiments}

\end{document}
"""
    intro_tex = r"""
\section{Introduction}
Recurrent neural networks\footnote{Including LSTM and GRU variants.},
long short-term memory and gated recurrent neural networks have been
firmly established as state of the art approaches in sequence modeling.
"""
    model_tex = r"""
\section{Model Architecture}
Most competitive neural sequence transduction models have an
encoder-decoder structure.

\subsection{Encoder and Decoder Stacks}
The encoder maps an input sequence to a sequence of continuous
representations.

\begin{equation*}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{equation*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{transformer.png}
\caption{The Transformer model architecture.}
\end{figure*}

\subsection{Attention}
An attention function can be described as mapping a query and a set of
key-value pairs to an output.

\begin{table*}[h]
\caption{Comparison of different model architectures.}
\begin{tabular}{lcc}
Model & BLEU & Parameters \\
Transformer (base) & 27.3 & 65M \\
Transformer (big) & 28.4 & 213M \\
\end{tabular}
\end{table*}
"""
    experiments_tex = r"""
\section{Experiments}
We trained our models on the standard WMT 2014 English-German dataset.

\begin{align*}
\mathcal{L} &= -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x) \\
P(y_t | y_{<t}, x) &= \text{softmax}(W h_t)
\end{align*}

\subsection{Training}
We used the Adam optimizer with $\beta_1 = 0.9$ and $\beta_2 = 0.98$.
"""
    bib_tex = r"""
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki},
  journal={Advances in neural information processing systems},
  year={2017}
}
"""
    archive = _make_tar_gz(tmp_path, {
        "main.tex": main_tex,
        "introduction.tex": intro_tex,
        "model.tex": model_tex,
        "experiments.tex": experiments_tex,
        "refs.bib": bib_tex,
    })

    parser = TeXParser(embed_images=False)
    paper = parser.parse(archive)

    # Metadata
    assert paper.title == "Attention Is All You Need (Test)"
    assert len(paper.authors) == 3
    assert "Ashish Vaswani" in paper.authors
    assert paper.date == "June 2017"
    assert "Transformer" in paper.abstract

    # Sections from \input{} files
    section_titles = [s.title for s in paper.sections]
    assert "Introduction" in section_titles
    assert "Model Architecture" in section_titles
    assert "Experiments" in section_titles
    assert "Encoder and Decoder Stacks" in section_titles
    assert "Attention" in section_titles
    assert "Training" in section_titles

    # Footnotes from included files
    assert paper.footnotes
    assert "LSTM" in paper.footnotes[0].content

    # Starred environments were parsed
    from namulizer.parser.base import EquationBlock, FigureBlock, TableBlock
    all_blocks = [b for s in paper.sections for b in s.content]
    figures = [b for b in all_blocks if isinstance(b, FigureBlock)]
    tables = [b for b in all_blocks if isinstance(b, TableBlock)]
    equations = [b for b in all_blocks if isinstance(b, EquationBlock)]

    assert len(figures) >= 1
    assert figures[0].caption == "The Transformer model architecture."
    assert len(tables) >= 1
    assert tables[0].caption == "Comparison of different model architectures."
    assert len(equations) >= 2  # equation* + align*

    # References from .bib
    assert paper.references
    assert any("Attention" in r.text for r in paper.references)
