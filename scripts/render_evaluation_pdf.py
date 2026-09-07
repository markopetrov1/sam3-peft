#!/usr/bin/env python3
"""
Render evaluation_results.md → evaluation_results.pdf via markdown + weasyprint.

Run from repo root:
    python scripts/render_evaluation_pdf.py
"""

from pathlib import Path

import markdown
from weasyprint import HTML, CSS

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_MD = REPO_ROOT / "evaluation_results.md"
OUTPUT_PDF = REPO_ROOT / "evaluation_results.pdf"

CSS_STYLE = """
@page { size: A4; margin: 1.6cm 1.4cm; }
body {
    font-family: "DejaVu Sans", "Helvetica", sans-serif;
    font-size: 10pt;
    color: #111;
    line-height: 1.35;
}
h1 { font-size: 18pt; margin: 0 0 0.4em; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13pt; margin: 1.1em 0 0.3em; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 0.9em 0 0.2em; page-break-after: avoid; }
p  { margin: 0.3em 0; }
hr { border: 0; border-top: 1px solid #bbb; margin: 0.9em 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; background: #f3f3f3; padding: 1px 3px; border-radius: 2px; }
ul { margin: 0.3em 0 0.3em 1.1em; padding: 0; }
li { margin: 0.1em 0; }
table {
    border-collapse: collapse;
    margin: 0.4em 0 0.8em;
    width: 100%;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #bbb;
    padding: 4px 8px;
    text-align: left;
    vertical-align: top;
}
th { background: #f0f0f0; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
strong { font-weight: 700; }
"""


def main():
    md_text = INPUT_MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"

    HTML(string=html_doc, base_url=str(REPO_ROOT)).write_pdf(
        str(OUTPUT_PDF), stylesheets=[CSS(string=CSS_STYLE)]
    )
    print(f"Written: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
