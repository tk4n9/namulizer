"""PDF -> Markdown extraction backends.

The default backend is pymupdf4llm, which detects headings from font
properties and emits GitHub-flavoured Markdown with tables and images.
Microsoft's markitdown is available as an alternative engine, but its PDF
backend (pdfminer) produces flat text without headings, so section structure
is mostly lost; prefer pymupdf4llm for papers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ENGINES = ("pymupdf4llm", "markitdown")

_META_TITLE_JUNK = re.compile(
    r"^$|^untitled$|^arxiv|\.(dvi|tex|pdf|docx?)$|^microsoft word|^powerpoint", re.I
)


@dataclass
class PdfExtraction:
    markdown: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    image_dir: Path | None = None


def pdf_to_markdown(
    pdf_path: str | Path,
    *,
    engine: str = "pymupdf4llm",
    embed_images: bool = True,
    image_dir: str | Path | None = None,
) -> PdfExtraction:
    """Convert a PDF file into Markdown text plus document metadata.

    ``image_dir`` is only used as a fallback location when the installed
    pymupdf4llm cannot embed images into the Markdown directly.
    """
    pdf_path = Path(pdf_path)
    if engine == "pymupdf4llm":
        return _via_pymupdf4llm(pdf_path, embed_images=embed_images, image_dir=image_dir)
    if engine == "markitdown":
        return _via_markitdown(pdf_path)
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")


def _clean_meta_title(raw: str | None) -> str | None:
    title = (raw or "").strip()
    if not title or _META_TITLE_JUNK.search(title):
        return None
    return title


def _split_authors(raw: str | None) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parts = re.split(r";|,|\band\b|&", text)
    return [p.strip() for p in parts if p.strip()]


def _via_pymupdf4llm(
    pdf_path: Path, *, embed_images: bool, image_dir: str | Path | None
) -> PdfExtraction:
    try:
        import pymupdf
        import pymupdf4llm
    except ImportError as exc:  # pragma: no cover - install-time problem
        raise RuntimeError(
            "pymupdf4llm is required for PDF conversion: pip install pymupdf4llm"
        ) from exc

    doc = pymupdf.open(pdf_path)
    try:
        meta = doc.metadata or {}
        title = _clean_meta_title(meta.get("title"))
        authors = _split_authors(meta.get("author"))

        used_image_dir: Path | None = None
        attempts: list[dict] = []
        if embed_images:
            attempts.append({"embed_images": True})
            if image_dir is not None:
                attempts.append(
                    {"write_images": True, "image_path": str(image_dir), "image_format": "png"}
                )
        elif image_dir is not None:
            attempts.append(
                {"write_images": True, "image_path": str(image_dir), "image_format": "png"}
            )
        attempts.append({})

        markdown = None
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                markdown = pymupdf4llm.to_markdown(doc, show_progress=False, **kwargs)
            except TypeError as exc:
                # Older/newer pymupdf4llm releases accept different kwargs;
                # fall through to the next known-good combination.
                last_error = exc
                continue
            if "write_images" in kwargs:
                used_image_dir = Path(kwargs["image_path"])
            break
        if markdown is None:
            try:
                markdown = pymupdf4llm.to_markdown(doc)
            except TypeError as exc:
                raise RuntimeError(
                    f"pymupdf4llm.to_markdown could not be called: {last_error or exc}"
                ) from exc

        return PdfExtraction(
            markdown=markdown, title=title, authors=authors, image_dir=used_image_dir
        )
    finally:
        doc.close()


def _via_markitdown(pdf_path: Path) -> PdfExtraction:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            "markitdown engine requested but not installed: "
            "pip install 'markitdown[pdf]' (note: its PDF output has no headings)"
        ) from exc

    result = MarkItDown(enable_plugins=False).convert(str(pdf_path))
    markdown = result.text_content or ""
    title = _clean_meta_title(getattr(result, "title", None))
    return PdfExtraction(markdown=markdown, title=title)
