"""Convert report.md to report.pdf. Pure-Python path (markdown + xhtml2pdf),
no external binary (pandoc/wkhtmltopdf) needed."""
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "report.md"
PDF_PATH = ROOT / "report.pdf"

CSS = """
@page {
    size: A4;
    margin: 1.7cm 1.9cm 2.0cm 1.9cm;
    @frame footer_frame {
        -pdf-frame-content: footer_content;
        bottom: 0.8cm; margin-left: 1.9cm; margin-right: 1.9cm; height: 1cm;
    }
}
body {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 9.7pt;
    line-height: 1.32;
    color: #1a1a18;
}
h1 {
    font-size: 16pt;
    margin-bottom: 2pt;
    color: #111;
}
h2 {
    font-size: 11.8pt;
    margin-top: 12pt;
    margin-bottom: 4pt;
    padding-bottom: 2pt;
    border-bottom: 0.75pt solid #c9c8c0;
    color: #111;
}
p { margin: 4pt 0 4pt 0; text-align: justify; }
strong { color: #111; }
ul, ol { margin: 4pt 0 8pt 0; padding-left: 16pt; }
li { margin-bottom: 2pt; }
code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f1f0eb;
    padding: 0 2pt;
}
pre {
    font-family: "Courier New", monospace;
    font-size: 8.6pt;
    background-color: #f1f0eb;
    padding: 6pt 8pt;
    margin: 6pt 0;
}
img {
    max-width: 85%;
    max-height: 11cm;
    margin: 10pt auto;
    display: block;
}
.fig-caption {
    page-break-before: always;
}
.byline {
    color: #52514e;
    font-size: 9.5pt;
    margin-bottom: 10pt;
}
"""

FOOTER = """
<div id="footer_content" style="text-align:center; font-size:8pt; color:#8a8a86;">
    TKH multi-resolution semantic abstraction, report &middot; page <pdf:pagenumber/>
</div>
"""


def convert_link(uri, rel):
    # resolve relative image paths (e.g. outputs/figures/x.png) against repo root
    p = (ROOT / uri).resolve()
    if p.exists():
        return str(p)
    return uri


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    lines = md_text.split("\n")
    # byline (second non-empty line, "Ilia Dzneladze, ...") gets its own class
    body_md = "\n".join(lines)

    html_body = markdown.markdown(body_md, extensions=["extra", "sane_lists"])
    # wrap the byline paragraph (first <p> right after the <h1>) in a styled span
    html_body = html_body.replace(
        "<h1>Multi-resolution semantic abstraction over the TKH hypergraph</h1>\n<p>",
        "<h1>Multi-resolution semantic abstraction over the TKH hypergraph</h1>\n"
        "<p class=\"byline\">", 1,
    )
    # each figure caption starts a fresh page, image follows immediately after
    # on the same page (no break between caption and its own image)
    for n in (1, 2, 3):
        html_body = html_body.replace(
            f"<p><strong>Figure {n}.</strong>",
            f'<p class="fig-caption"><strong>Figure {n}.</strong>', 1,
        )

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
{FOOTER}
{html_body}
</body></html>"""

    with open(PDF_PATH, "wb") as f:
        result = pisa.CreatePDF(full_html, dest=f, link_callback=convert_link)

    if result.err:
        print(f"PDF generation had {result.err} error(s)", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
