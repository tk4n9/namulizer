from pathlib import Path

import pytest

from namulizer.parser.pdf_parser import PDFParser, _is_valid_heading_candidate

fitz = pytest.importorskip("fitz")


def test_pdf_parser_basic_flow(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Introduction", fontsize=18)
    page.insert_text((72, 96), "This is a body paragraph for parser testing.", fontsize=11)
    doc.save(pdf_path)
    doc.close()

    parser = PDFParser(embed_images=False)
    paper = parser.parse(pdf_path)

    assert paper.title == "sample"
    assert paper.sections
    assert any(section.title for section in paper.sections)


def test_heading_candidate_rejects_table_like_numbers() -> None:
    body_font = 10.0
    assert _is_valid_heading_candidate("3.2", "Attention", 10.0, body_font)

    # Decimal-heavy rows from tables should not be treated as numbered headings.
    assert not _is_valid_heading_candidate("0.1", "100K 4.92 25.8 65", 10.0, body_font)
    assert not _is_valid_heading_candidate("5.29", "24.9", 10.0, body_font)
    assert not _is_valid_heading_candidate("28.4", "Transformer (big)", 10.0, body_font)
