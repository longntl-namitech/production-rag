import subprocess
import sys
from pathlib import Path

import pytest

from production_rag.methods.parse_and_chunk.parsers import (
    LEFT_OUT,
    PARSERS,
    SUPPORTED,
    MissingExtra,
    Parsed,
    UnsupportedFormat,
    detect_format,
    parse_file,
    table_to_markdown,
)


def test_detect_format_supported_and_left_out():
    assert detect_format(Path("a/B.PDF")) == "pdf" and detect_format(Path("x.htm")) == "html"
    with pytest.raises(UnsupportedFormat, match="LibreOffice"):
        detect_format(Path("old.doc"))
    with pytest.raises(UnsupportedFormat, match="LibreOffice"):
        detect_format(Path("old.xls"))
    with pytest.raises(UnsupportedFormat, match="OCR"):
        detect_format(Path("scan.png"))
    with pytest.raises(UnsupportedFormat, match="unknown extension"):
        detect_format(Path("a.xyz"))
    assert not set(SUPPORTED) & set(LEFT_OUT)


def test_parsers_keys_match_supported_formats():
    assert set(PARSERS) == set(SUPPORTED.values())


def test_table_to_markdown():
    md = table_to_markdown([["a", "b|c"], [None, "x\ny"], [None, None], ["1"]])
    assert md.splitlines() == ["| a | b\\|c |", "| --- | --- |", "|  | x y |", "| 1 |  |"]
    assert table_to_markdown([[None, ""]]) == ""


def test_markdown_needs_no_extra(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# T\n\nbody")
    assert parse_file(p) == Parsed("# T\n\nbody", "markdown")


def test_missing_library_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "docx", None)  # import docx -> ImportError
    with pytest.raises(MissingExtra, match="uv sync --extra parse"):
        parse_file(tmp_path / "a.docx")


def test_parsers_import_on_light_install():
    code = "import production_rag.methods.parse_and_chunk.parsers"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def _minimal_pdf(text: str | None) -> bytes:
    stream = b"" if text is None else f"BT /F1 18 Tf 72 700 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


@pytest.mark.integration
def test_docx_headings_lists_tables_in_order(tmp_path):
    import docx

    d = docx.Document()
    d.add_heading("Leave", 1)
    d.add_paragraph("Some body text.")
    d.add_paragraph("item", style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    for r, row in enumerate([["k", "v"], ["days", "20"]]):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    d.add_heading("Pay", 2)
    p = tmp_path / "a.docx"
    d.save(p)
    parsed = parse_file(p)
    md = parsed.markdown
    assert parsed.format == "docx"
    assert md.index("# Leave") < md.index("- item") < md.index("| --- |") < md.index("## Pay")
    assert "| days | 20 |" in md


@pytest.mark.integration
def test_pptx_slide_sections_and_table(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    s1.shapes.title.text = "Agenda"
    tbl = s1.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    for r, row in enumerate([["h1", "h2"], ["a", "b"]]):
        for c, val in enumerate(row):
            tbl.cell(r, c).text = val
    s2 = prs.slides.add_slide(prs.slide_layouts[1])  # title and content
    s2.shapes.title.text = ""
    s2.placeholders[1].text_frame.text = "top point"
    sub = s2.placeholders[1].text_frame.add_paragraph()
    sub.text = "nested point"
    sub.level = 1
    p = tmp_path / "a.pptx"
    prs.save(p)
    md = parse_file(p).markdown
    assert "## Agenda" in md
    assert "| h1 | h2 |" in md and "| --- | --- |" in md and "| a | b |" in md
    assert "## Slide 2" in md
    assert "top point" in md and "  - nested point" in md


@pytest.mark.integration
def test_xlsx_sheet_table_drops_empty_rows(tmp_path):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["name", "qty"])
    ws.append([None, None])
    ws.append(["apple", 3])
    p = tmp_path / "a.xlsx"
    wb.save(p)
    md = parse_file(p).markdown
    assert md.startswith("## Sheet")
    assert "| name | qty |" in md and "| apple | 3 |" in md
    assert "|  |  |" not in md


@pytest.mark.integration
def test_html_drops_scripts_keeps_table(tmp_path):
    p = tmp_path / "a.html"
    p.write_text(
        "<html><head><style>.x{}</style></head><body><script>var secret=1;</script>"
        "<h1>Title</h1><p>Hello</p><table><tr><th>a</th><th>b</th></tr>"
        "<tr><td>1</td><td>2</td></tr></table></body></html>"
    )
    md = parse_file(p).markdown
    assert "secret" not in md and ".x{}" not in md
    assert "# Title" in md and "Hello" in md
    assert "| a | b |" in md and "| 1 | 2 |" in md


@pytest.mark.integration
def test_pdf_text_and_no_text_layer(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(_minimal_pdf("Hello parse"))
    parsed = parse_file(p)
    assert "Hello parse" in parsed.markdown and parsed.pages == 1 and parsed.warnings == ()
    q = tmp_path / "blank.pdf"
    q.write_bytes(_minimal_pdf(None))
    blank = parse_file(q)
    assert blank == Parsed("", "pdf", 1, ("no_text",))
