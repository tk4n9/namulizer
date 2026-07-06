# STATUSLOG

## Entry 001
- Timestamp: 2026-02-23 16:23:56 KST
- User prompt/instruction:
  - Asked for a concrete migration blueprint and then instructed: "From now on please maintain STATUSLOG.md as instructed."
- Action/edit done:
  - Produced a decision-complete cutover plan (no code edits in repo during planning turn).
- Reason:
  - Establish full specification before implementation.
- Validation/tests:
  - N/A (planning only).

## Entry 002
- Timestamp: 2026-02-23 16:24:09 KST
- User prompt/instruction:
  - "Implement plan in a new branch"
- Action/edit done:
  - Verified clean worktree and created branch `codex/tei-xml-cutover` from `main`.
  - Created `STATUSLOG.md` and initialized append-only log entries.
- Reason:
  - Required by user and AGENTS instructions before implementation edits.
- Validation/tests:
  - `git status --short` showed clean state before branch creation.
  - `git checkout -b codex/tei-xml-cutover` succeeded.

## Entry 003
- Timestamp: 2026-02-23 16:35:35 KST
- User prompt/instruction:
  - "Implement plan in a new branch"
- Action/edit done:
  - Implemented cutover scaffolding and core architecture:
    - Added canonical IR types:
      - src/namulizer/ir/canonical_document.py
      - src/namulizer/ir/__init__.py
    - Added RelaxNG schema + validator:
      - src/namulizer/ir/schema/namulizer-tei.rng
      - src/namulizer/ir/schema_validate.py
    - Added converter backends and helpers:
      - src/namulizer/converters/common.py
      - src/namulizer/converters/md_markdownit.py
      - src/namulizer/converters/tex_latexml.py
      - src/namulizer/converters/pdf_grobid.py
      - src/namulizer/converters/__init__.py
    - Replaced parser contract and parser implementations:
      - src/namulizer/parser/base.py
      - src/namulizer/parser/md_parser.py
      - src/namulizer/parser/tex_parser.py
      - src/namulizer/parser/pdf_parser.py
      - src/namulizer/parser/__init__.py
    - Replaced renderer input/output path to canonical XML:
      - src/namulizer/renderer/html_renderer.py
    - Updated CLI options and parser invocation contract:
      - src/namulizer/cli.py
    - Updated package version + dependencies and docs:
      - src/namulizer/__init__.py (version 0.1.0 -> 0.2.0)
      - pyproject.toml (added lxml, markdown-it-py, requests; bumped to 0.2.0)
      - README.md (new architecture + external service setup + options)
- Reason:
  - Execute the approved major-cutover plan to canonical TEI-like XML IR and external converter backends.
- Validation/tests:
  - Validation pending (tests being rewritten to new contract before suite run).

## Entry 004
- Timestamp: 2026-02-23 16:38:46 KST
- User prompt/instruction:
  - Continue implementing approved major cutover plan.
- Action/edit done:
  - Replaced legacy test suite with canonical-IR-focused tests:
    - Added tests/test_ir_schema.py
    - Rewrote tests/test_md_parser.py
    - Rewrote tests/test_tex_parser.py
    - Rewrote tests/test_pdf_parser.py
    - Rewrote tests/test_renderer.py
    - Added tests/test_cli.py
    - Added tests/test_e2e_xml_to_html.py
    - Removed tests/test_arxiv_support.py (obsolete old IR contract tests)
  - Added optional import guard for requests in src/namulizer/converters/pdf_grobid.py to prevent import-time failure when PDF backend deps are absent.
- Reason:
  - Align tests with new canonical XML parser contract and ensure non-PDF workflows remain importable when requests is unavailable.
- Validation/tests:
  - pytest -q -> initial failure due missing requests import at module import time.
  - After guard patch: pytest -> 3 passed, 5 skipped.
  - python3 -m compileall src -> success.

## Entry 005
- Timestamp: 2026-02-23 16:39:04 KST
- User prompt/instruction:
  - Complete implementation and report status.
- Action/edit done:
  - Verified working branch is codex/tei-xml-cutover.
  - Confirmed changed/new files for cutover implementation and tests.
  - Re-ran validation checks and collected final status for handoff.
- Reason:
  - Provide a reliable completion snapshot and make continuation straightforward for future AGENTs.
- Validation/tests:
  - pytest -> 3 passed, 5 skipped.
  - python3 -m compileall src -> success.
  - git diff --stat and git status --short captured for final report.

## Entry 006
- Timestamp: 2026-02-23 16:39:42 KST
- User prompt/instruction:
  - Continue implementation to production-ready state.
- Action/edit done:
  - Added src/namulizer/ir/schema/__init__.py so schema directory is an explicit package.
  - Updated pyproject.toml package-data to include namulizer.ir.schema/namulizer-tei.rng in distributions.
- Reason:
  - Ensure canonical schema file is available at runtime after installation, not only in editable/source mode.
- Validation/tests:
  - pytest -q -> pass (3 passed, 5 skipped).

## Entry 007
- Timestamp: 2026-02-23 16:42:11 KST
- User prompt/instruction:
  - Asked for a full workflow to generate HTML from a local PDF and manually review it in Google Chrome.
- Action/edit done:
  - Provided a step-by-step operational workflow (environment setup, GROBID startup, conversion command, Chrome open, and verification checklist).
- Reason:
  - Enable reproducible end-to-end manual validation for the new canonical XML + GROBID PDF path.
- Validation/tests:
  - N/A (instructional guidance turn; no new code/test execution).

## Entry 008
- Timestamp: 2026-02-23 16:45:50 KST
- User prompt/instruction:
  - Reported Docker error: cannot connect to daemon at ~/.docker/run/docker.sock.
- Action/edit done:
  - Diagnosed runtime state with docker version/context checks, process inspection, and application checks.
  - Confirmed Docker CLI is installed but no daemon/runtime is running and Docker Desktop app is not installed in /Applications.
- Reason:
  - Determine root cause before giving remediation steps for PDF->HTML test workflow.
- Validation/tests:
  - docker version failed on both desktop-linux and default contexts.
  - docker context ls showed contexts but no reachable daemon.
  - No Docker-related runtime process found.

## Entry 009
- Timestamp: 2026-07-06 12:46 KST (local machine time)
- User prompt/instruction:
  - "Please build this project from scratch." Previous LLM attempt failed to produce a working PDF -> NamuWiki HTML converter. User suggested Microsoft's PDF-to-Markdown tool (markitdown) or alternatives.
- Action/edit done:
  - Diagnosed root cause of the 0.2.0 failure: the PDF path required GROBID (Docker daemon not available on this machine, see Entry 008) and the TeX path required LaTeXML. Both external services were unusable, so nothing could convert.
  - Rebuilt the pipeline as pure Python (v0.3.0), no external services:
    - `src/namulizer/pdf2md.py`: PDF -> Markdown via pymupdf4llm (default; heading detection from font properties). markitdown supported as optional `--engine markitdown` (its pdfminer backend loses headings, hence not default).
    - `src/namulizer/md_parser.py`: Markdown -> Document via markdown-it-py + mdit-py-plugins. Handles pymupdf4llm quirks: strips page-break dash rules, promotes bold-only heading lines, demotes equation pseudo-headings (title contains "="), removes raw sup/sub/br tags, re-derives section nesting from embedded numbering ("3.2.1" -> depth 3), strips original numbers (NamuWiki renumbers). Abstract section -> page intro; References/Bibliography -> 각주 footnotes with inline [n] citations linked for hover popups; native markdown footnotes ([^1]) take priority, bibliography then stays a plain reference block. Local images embedded as base64 data URIs.
    - `src/namulizer/renderer.py`: Document -> HTML via jinja2 using the existing `src/namulizer/template/namuwiki.html` (kept; only abstract/references blocks changed to accept rendered HTML).
    - `src/namulizer/cli.py`: click CLI (`namulizer input.{pdf,md} -o out.html`, `--title`, `--dark-mode`, `--no-fold`, `--engine`, `--embed-images/--no-embed-images`, `--dump-md`).
    - `src/namulizer/document.py`: Document/Section/Footnote/Reference dataclasses.
  - Deleted the dead 0.2.0 stack: `src/namulizer/converters/`, `src/namulizer/ir/`, `src/namulizer/parser/`, `src/namulizer/renderer/` (package), duplicate root `template/`, and all old tests.
  - New test suite: `tests/test_md_parser.py`, `tests/test_renderer.py`, `tests/test_pdf2md.py`, `tests/test_cli.py` (includes PDF end-to-end on `samples/sample.pdf`).
  - Rewrote `pyproject.toml` (0.3.0; deps: click, jinja2, markdown-it-py, mdit-py-plugins, linkify-it-py, pymupdf4llm; extras: markitdown) and `README.md`.
  - Environment note: both committed venvs (`venv/`, `mba-venv/`) were created on other machines (paths `/Users/gtpv`, `/Users/parangvo`) and are broken here. Created fresh `.venv` with python3.12 (`/Users/tk/.local/bin/python3.12`); `.venv/` is gitignored.
- Reason:
  - Make PDF -> NamuWiki-style HTML actually work on this machine without Docker/LaTeXML.
- Validation/tests:
  - `.venv/bin/python -m pytest` -> 22 passed.
  - `namulizer samples/sample.pdf -o samples/demo_output.html` -> works on the "Attention Is All You Need" paper: correct title, exact paper TOC hierarchy (3.2.1 nesting), 40 reference footnotes with 57 linked inline citations (hover popups), 10 embedded images, abstract lifted to page intro, equation pseudo-heading removed.
  - Root `sample_output.html` regenerated from the new pipeline.
