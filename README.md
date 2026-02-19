# namulizer

`namulizer` is a Python CLI tool that converts academic papers (PDF or TeX sources) into self-contained HTML pages styled like NamuWiki.

## Install

```bash
pip install -e .
```

## Usage

```bash
namulizer input.pdf -o output.html
namulizer input.tex.zip -o output.html
```

Options:

- `--title TEXT`: override document title
- `--dark-mode`: enable dark stylesheet variant
- `--no-fold`: disable section fold toggles
- `--embed-images / --no-embed-images`: base64 embed images (default: enabled)
- `--math-engine [none|katex|mathjax]`: math rendering mode

## Project Layout

- `template/namuwiki.html`: standalone inlined CSS/JS template (phase 1 deliverable)
- `src/namulizer/parser`: TeX/PDF parsing to unified IR
- `src/namulizer/renderer`: IR to HTML rendering
- `src/namulizer/cli.py`: command line entrypoint
- `tests/`: parser and renderer tests
