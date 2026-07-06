from pathlib import Path

import pytest
from click.testing import CliRunner

from namulizer.cli import main

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "sample.pdf"

MD = """# CLI Paper

## 1 Introduction

Hello from markdown [1].

## References

[1] Somebody. Some paper. 2020.
"""


def test_md_to_html(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(MD, encoding="utf-8")
    out = tmp_path / "doc.html"
    result = CliRunner().invoke(main, [str(src), "-o", str(out)])
    assert result.exit_code == 0, result.output
    html = out.read_text(encoding="utf-8")
    assert "CLI Paper - 나무위키" in html
    assert "wiki-heading" in html
    assert 'data-footnote-id="1"' in html


def test_default_output_and_title_override(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(MD, encoding="utf-8")
    result = CliRunner().invoke(main, [str(src), "--title", "Custom", "--dark-mode"])
    assert result.exit_code == 0, result.output
    html = (tmp_path / "doc.html").read_text(encoding="utf-8")
    assert "Custom - 나무위키" in html
    assert "theseed-dark-mode" in html


def test_unsupported_suffix(tmp_path):
    src = tmp_path / "doc.docx"
    src.write_text("x", encoding="utf-8")
    result = CliRunner().invoke(main, [str(src)])
    assert result.exit_code != 0
    assert "unsupported input type" in result.output


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not present")
def test_pdf_end_to_end(tmp_path):
    pytest.importorskip("pymupdf4llm")
    out = tmp_path / "paper.html"
    md_dump = tmp_path / "paper.md"
    result = CliRunner().invoke(
        main, [str(SAMPLE_PDF), "-o", str(out), "--dump-md", str(md_dump)]
    )
    assert result.exit_code == 0, result.output
    html = out.read_text(encoding="utf-8")
    assert "나무위키" in html
    assert "wiki-heading" in html  # sections detected
    assert "wiki-macro-toc" in html  # TOC generated
    assert md_dump.exists() and md_dump.stat().st_size > 0
