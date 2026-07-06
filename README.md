# namulizer

`namulizer` converts academic papers (PDF) and Markdown documents into self-contained, NamuWiki-style HTML pages — table of contents, numbered foldable sections, footnote hover popups, dark mode, all in a single file with images embedded.

## How it works

```
input.pdf ──pymupdf4llm──▶ Markdown ──markdown-it-py──▶ Document ──jinja2──▶ namuwiki.html
input.md  ────────────────────────▶
```

1. **PDF → Markdown**: [pymupdf4llm](https://pypi.org/project/pymupdf4llm/) detects headings from font properties and emits Markdown with tables and images. No Docker, no external services (the 0.2.x GROBID/LaTeXML pipeline is gone).
2. **Markdown → Document**: markdown-it-py tokens are folded into a section tree. Extra care is taken with PDF-flavoured Markdown: page-break rules are stripped, bold-only heading lines are promoted, display equations misdetected as headings are demoted, and section nesting is re-derived from embedded numbering ("3.2.1" → depth 3).
3. **NamuWiki touches**: sections are renumbered NamuWiki-style with fold toggles and a 목차 block. An `Abstract` section becomes the page intro. A `References`/`Bibliography` section is converted into 각주, so inline citations like `[3]` become links with hover popups.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Requires Python ≥ 3.10. All dependencies are pure pip installs (`pymupdf4llm`, `markdown-it-py`, `jinja2`, `click`).

## Usage

```bash
namulizer paper.pdf -o paper.html
namulizer notes.md --dark-mode
```

Options:

- `-o, --output PATH`: output HTML (default: input name with `.html`)
- `--title TEXT`: override the document title
- `--dark-mode`: dark NamuWiki theme
- `--no-fold`: disable section fold toggles
- `--embed-images / --no-embed-images`: base64-embed images (default: on)
- `--engine [pymupdf4llm|markitdown]`: PDF backend. Microsoft's [markitdown](https://github.com/microsoft/markitdown) is supported (`pip install 'namulizer[markitdown]'`) but its PDF backend produces flat text without headings, so section structure is mostly lost — the default is recommended for papers.
- `--dump-md PATH`: save the intermediate Markdown for debugging

## Project layout

- `src/namulizer/pdf2md.py`: PDF → Markdown backends
- `src/namulizer/md_parser.py`: Markdown → `Document` (sections, footnotes, references)
- `src/namulizer/renderer.py`: `Document` → HTML
- `src/namulizer/template/namuwiki.html`: standalone template (inlined CSS/JS)
- `samples/`: `sample.pdf` plus `demo_output.html` generated from it

## Development

```bash
.venv/bin/python -m pytest
```

The suite includes an end-to-end test converting `samples/sample.pdf`.

## Known limitations

- Math is rendered as the extracted text glyphs, not re-typeset (PDFs carry no LaTeX source).
- Stray page headers/footers from the PDF can survive as body text.
- TeX input was dropped in 0.3.0 (the LaTeXML dependency was the main source of breakage); compile to PDF first.
