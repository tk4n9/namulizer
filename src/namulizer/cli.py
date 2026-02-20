"""namulizer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import click

from namulizer.parser.pdf_parser import PDFParser
from namulizer.parser.tex_parser import TeXParser
from namulizer.renderer.html_renderer import HTMLRenderer


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output HTML path")
@click.option("--title", type=str, default=None, help="Override document title")
@click.option("--dark-mode", is_flag=True, help="Enable dark mode stylesheet")
@click.option("--no-fold", is_flag=True, help="Disable section folding")
@click.option("--embed-images/--no-embed-images", default=True, show_default=True, help="Embed images as base64")
@click.option(
    "--math-engine",
    type=click.Choice(["none", "katex", "mathjax"], case_sensitive=False),
    default="none",
    show_default=True,
    help="Math rendering mode",
)
def main(
    input_path: Path,
    output: Path,
    title: str | None,
    dark_mode: bool,
    no_fold: bool,
    embed_images: bool,
    math_engine: str,
) -> None:
    """Convert an academic paper into a NamuWiki-style self-contained HTML file."""
    parser = _select_parser(input_path, embed_images=embed_images)
    paper = parser.parse(input_path)

    renderer = HTMLRenderer()
    html = renderer.render(
        paper,
        title_override=title,
        dark_mode=dark_mode,
        enable_fold=not no_fold,
        embed_images=embed_images,
        math_engine=math_engine.lower(),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    click.echo(f"Rendered: {output}")


def _select_parser(input_path: Path, *, embed_images: bool):
    lowered = input_path.name.lower()
    if lowered.endswith(".pdf"):
        return PDFParser(embed_images=embed_images)
    _TEX_EXTENSIONS = (".tex", ".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar", ".gz")
    if any(lowered.endswith(ext) for ext in _TEX_EXTENSIONS):
        return TeXParser(embed_images=embed_images)
    raise click.ClickException(
        f"Unsupported input type: {input_path.name} (expected .pdf, .tex, .zip, or .tar.gz containing TeX source)"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
