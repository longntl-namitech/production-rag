"""Turn a document file into Markdown. Each parser imports its library only when called."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
}

_LEGACY = "legacy Office binary format: needs LibreOffice or similar to convert first"
_IMAGE = "image: needs OCR (for example Tesseract)"
_LATER = "not in the first release"
LEFT_OUT: dict[str, str] = {
    **dict.fromkeys([".doc", ".ppt", ".xls"], _LEGACY),
    **dict.fromkeys([".png", ".jpg", ".jpeg", ".tif", ".tiff"], _IMAGE),
    **dict.fromkeys([".csv", ".txt", ".odt", ".ods", ".odp", ".epub", ".rtf"], _LATER),
}

_EXTRA_HINT = "{} needs the parse extra: uv sync --extra parse"


class MissingExtra(ImportError):
    """A parser library from the `parse` extra is not installed."""


class UnsupportedFormat(ValueError):
    """The file type is not parsed; the message says why."""


@dataclass(frozen=True)
class Parsed:
    markdown: str
    format: str
    pages: int | None = None
    warnings: tuple[str, ...] = ()


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED:
        return SUPPORTED[suffix]
    if suffix in LEFT_OUT:
        raise UnsupportedFormat(f"{suffix}: {LEFT_OUT[suffix]}")
    raise UnsupportedFormat(f"unknown extension {suffix or '(none)'!r}")


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cells = [
        [("" if c is None else str(c)).replace("\n", " ").replace("|", "\\|") for c in row]
        for row in rows
    ]
    cells = [row for row in cells if any(c.strip() for c in row)]
    if not cells:
        return ""
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    lines = ["| " + " | ".join(row) + " |" for row in cells]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _blocks(parts: list[str]) -> str:
    return "\n\n".join(p for p in parts if p)


# pdfplumber's default (3) merges words in tightly set PDFs ("etal", "ofthe")
PDF_X_TOLERANCE = 1


def _pdf_page_markdown(page) -> str:
    tables = page.find_tables()
    boxes = [t.bbox for t in tables]

    def outside(obj):
        if "x0" not in obj or "top" not in obj:
            return True
        cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)

    text = (page.filter(outside).extract_text(x_tolerance=PDF_X_TOLERANCE) or "").strip()
    cells = (t.extract(x_tolerance=PDF_X_TOLERANCE) for t in tables)
    return _blocks([text, *(table_to_markdown(c) for c in cells)])


def parse_pdf(path: Path) -> Parsed:
    try:
        import pdfplumber
    except ImportError as e:
        raise MissingExtra(_EXTRA_HINT.format("pdf")) from e
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        if not any(page.chars for page in pdf.pages):
            return Parsed("", "pdf", pages, ("no_text",))
        out = [_pdf_page_markdown(page) for page in pdf.pages]
    return Parsed(_blocks(out), "pdf", pages)


def parse_docx(path: Path) -> Parsed:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:
        raise MissingExtra(_EXTRA_HINT.format("docx")) from e
    document = docx.Document(str(path))
    parts = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}tbl"):
            table = Table(child, document)
            parts.append(table_to_markdown([[c.text for c in row.cells] for row in table.rows]))
        elif child.tag.endswith("}p"):
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style is not None else ""
            if style == "Title":
                text = f"# {text}"
            elif m := re.fullmatch(r"Heading (\d+)", style):
                text = f"{'#' * min(int(m.group(1)), 6)} {text}"
            elif style.startswith("List Bullet"):
                text = f"- {text}"
            elif style.startswith("List Number"):
                text = f"1. {text}"
            parts.append(text)
    return Parsed(_blocks(parts), "docx")


def _pptx_shape_blocks(shapes, skip_id) -> list[str]:
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    out = []
    for shape in shapes:
        if shape.shape_id == skip_id:
            continue
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            out.extend(_pptx_shape_blocks(shape.shapes, skip_id))
        elif getattr(shape, "has_table", False) and shape.has_table:
            rows = [[c.text for c in row.cells] for row in shape.table.rows]
            out.append(table_to_markdown(rows))
        elif shape.has_text_frame:
            lines = []
            for para in shape.text_frame.paragraphs:
                text = "".join(run.text for run in para.runs).strip()
                if text:
                    lines.append(f"{'  ' * para.level}- {text}" if para.level > 0 else text)
            out.append("\n".join(lines))
    return out


def parse_pptx(path: Path) -> Parsed:
    try:
        from pptx import Presentation
    except ImportError as e:
        raise MissingExtra(_EXTRA_HINT.format("pptx")) from e
    prs = Presentation(str(path))
    parts = []
    for n, slide in enumerate(prs.slides, 1):
        title = slide.shapes.title
        heading = title.text_frame.text.strip() if title is not None else ""
        parts.append(f"## {heading or f'Slide {n}'}")
        skip_id = title.shape_id if title is not None else None
        parts.extend(_pptx_shape_blocks(slide.shapes, skip_id))
    return Parsed(_blocks(parts), "pptx")


def parse_xlsx(path: Path) -> Parsed:
    try:
        import openpyxl
    except ImportError as e:
        raise MissingExtra(_EXTRA_HINT.format("xlsx")) from e
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    try:
        for ws in wb.worksheets:
            rows = [
                ["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)
            ]
            rows = [row for row in rows if any(v.strip() for v in row)]
            keep = [
                i
                for i in range(max((len(r) for r in rows), default=0))
                if any(i < len(r) and r[i].strip() for r in rows)
            ]
            rows = [[r[i] if i < len(r) else "" for i in keep] for r in rows]
            parts.append(f"## {ws.title}")
            parts.append(table_to_markdown(rows))
    finally:
        wb.close()
    return Parsed(_blocks(parts), "xlsx")


def parse_html(path: Path) -> Parsed:
    try:
        from bs4 import BeautifulSoup
        from markdownify import markdownify
    except ImportError as e:
        raise MissingExtra(_EXTRA_HINT.format("html")) from e
    soup = BeautifulSoup(path.read_bytes().decode("utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    md = markdownify(str(soup), heading_style="ATX")
    return Parsed(re.sub(r"\n{3,}", "\n\n", md).strip(), "html")


def parse_markdown(path: Path) -> Parsed:
    return Parsed(path.read_bytes().decode("utf-8", errors="replace"), "markdown")


PARSERS: dict[str, Callable[[Path], Parsed]] = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "pptx": parse_pptx,
    "xlsx": parse_xlsx,
    "html": parse_html,
    "markdown": parse_markdown,
}


def parse_file(path: Path) -> Parsed:
    return PARSERS[detect_format(path)](path)
