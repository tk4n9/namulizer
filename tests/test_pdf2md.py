from pathlib import Path

import pytest

from namulizer.pdf2md import _clean_meta_title, _split_authors, pdf_to_markdown

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "sample.pdf"

pymupdf4llm = pytest.importorskip("pymupdf4llm")


def test_meta_title_junk_filter():
    assert _clean_meta_title("") is None
    assert _clean_meta_title("untitled") is None
    assert _clean_meta_title("paper.dvi") is None
    assert _clean_meta_title("arXiv:2401.00001") is None
    assert _clean_meta_title("A Real Title") == "A Real Title"


def test_author_splitting():
    assert _split_authors("A. Kim; B. Lee and C. Park") == ["A. Kim", "B. Lee", "C. Park"]
    assert _split_authors(None) == []


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not present")
def test_sample_pdf_extraction(tmp_path):
    extraction = pdf_to_markdown(SAMPLE_PDF, image_dir=tmp_path)
    assert len(extraction.markdown) > 1000
    # heading structure must survive: this is the whole point of the backend
    assert any(line.startswith("#") for line in extraction.markdown.splitlines())


def test_unknown_engine_rejected():
    with pytest.raises(ValueError):
        pdf_to_markdown(SAMPLE_PDF, engine="nope")
