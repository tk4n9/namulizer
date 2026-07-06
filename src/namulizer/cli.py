"""Command line interface: convert PDF/Markdown into NamuWiki-style HTML."""

from __future__ import annotations

import tempfile
from pathlib import Path

import click

from .md_parser import parse_markdown
from .renderer import render_html

_MD_SUFFIXES = {".md", ".markdown", ".mdown", ".txt"}


def convert_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    dark_mode: bool = False,
    enable_fold: bool = True,
    engine: str = "pymupdf4llm",
    embed_images: bool = True,
    dump_md: str | Path | None = None,
) -> Path:
    """Convert ``input_path`` (.pdf or .md) and return the written HTML path."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_suffix(".html")
    suffix = input_path.suffix.lower()

    authors: list[str] = []
    meta_title: str | None = None

    if suffix == ".pdf":
        from .pdf2md import pdf_to_markdown

        with tempfile.TemporaryDirectory(prefix="namulizer-img-") as tmp_dir:
            extraction = pdf_to_markdown(
                input_path,
                engine=engine,
                embed_images=embed_images,
                image_dir=tmp_dir,
            )
            markdown = extraction.markdown
            meta_title = extraction.title
            authors = extraction.authors
            if dump_md:
                Path(dump_md).write_text(markdown, encoding="utf-8")
            document = parse_markdown(
                markdown,
                from_pdf=True,
                base_path=extraction.image_dir or input_path.parent,
                embed_images=embed_images,
            )
    elif suffix in _MD_SUFFIXES:
        markdown = input_path.read_text(encoding="utf-8")
        if dump_md:
            Path(dump_md).write_text(markdown, encoding="utf-8")
        document = parse_markdown(
            markdown,
            from_pdf=False,
            base_path=input_path.parent,
            embed_images=embed_images,
        )
    else:
        raise click.UsageError(
            f"unsupported input type {suffix!r}: expected .pdf or one of "
            + ", ".join(sorted(_MD_SUFFIXES))
        )

    document.title = title or document.title or meta_title or input_path.stem
    if authors and not document.authors:
        document.authors = authors

    html = render_html(document, dark_mode=dark_mode, enable_fold=enable_fold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output HTML path (default: input name with .html).",
)
@click.option("--title", default=None, help="Override the document title.")
@click.option("--dark-mode", is_flag=True, help="Use the dark NamuWiki theme.")
@click.option("--no-fold", is_flag=True, help="Disable section fold toggles.")
@click.option(
    "--engine",
    type=click.Choice(["pymupdf4llm", "markitdown"]),
    default="pymupdf4llm",
    show_default=True,
    help="PDF-to-Markdown backend (markitdown loses heading structure).",
)
@click.option(
    "--embed-images/--no-embed-images",
    default=True,
    show_default=True,
    help="Base64-embed images into the standalone HTML.",
)
@click.option(
    "--dump-md",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also save the intermediate Markdown (debugging).",
)
def main(input_path, output_path, title, dark_mode, no_fold, engine, embed_images, dump_md):
    """Convert INPUT_PATH (.pdf or .md) into a NamuWiki-style HTML page."""
    try:
        result = convert_file(
            input_path,
            output_path,
            title=title,
            dark_mode=dark_mode,
            enable_fold=not no_fold,
            engine=engine,
            embed_images=embed_images,
            dump_md=dump_md,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {result}")


if __name__ == "__main__":
    main()
