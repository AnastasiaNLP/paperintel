import base64
import logging
import re
import time
from pathlib import Path
from typing import Optional, TypedDict

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 50
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
NULL_THRESHOLD = 0.5
RAGGED_THRESHOLD = 0.3


class PDFMetadata(TypedDict):
    author: Optional[str]
    title: Optional[str]
    subject: Optional[str]
    creation_date: Optional[str]
    producer: Optional[str]


class TableData(TypedDict):
    page: int
    rows: list[list[Optional[str]]]
    needs_vision: bool


class ParsedPDF(TypedDict):
    file_path: str
    page_count: int
    raw_text: str
    text_by_page: dict[int, str]
    arxiv_id: Optional[str]
    arxiv_id_versioned: Optional[str]
    tables: list[TableData]
    metadata: PDFMetadata


def _check_file(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"PDF too large: {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
    return path


def _strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _extract_arxiv_id(text: str) -> tuple[Optional[str], Optional[str]]:
    patterns = [
        r"arXiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
    ]
    text_head = text[:3000]
    for pattern in patterns:
        match = re.search(pattern, text_head, re.IGNORECASE)
        if match:
            versioned = match.group(1)
            return _strip_arxiv_version(versioned), versioned
    return None, None


def _extract_metadata(doc: fitz.Document) -> PDFMetadata:
    raw_meta = doc.metadata or {}
    return {
        "author": raw_meta.get("author"),
        "title": raw_meta.get("title"),
        "subject": raw_meta.get("subject"),
        "creation_date": raw_meta.get("creationDate"),
        "producer": raw_meta.get("producer"),
    }


def _normalize_cell(cell: object) -> Optional[str]:
    """
    None -> missing cell (merged cell)
    ""   -> empty present cell
    str  -> non-empty present cell
    """
    if cell is None:
        return None
    return str(cell).strip()


def _is_structural_cell(cell: Optional[str]) -> bool:
    return cell not in (None, "")


def _structural_cell_count(row: list[Optional[str]]) -> int:
    return sum(1 for cell in row if _is_structural_cell(cell))


def _present_cell_count(row: list[Optional[str]]) -> int:
    return sum(1 for cell in row if cell is not None)


def _empty_cell_count(row: list[Optional[str]]) -> int:
    return sum(1 for cell in row if cell == "")


def _is_meaningful_row(row: list[Optional[str]]) -> bool:
    """
    Keep rows that contain at least one present cell.
    Preserves rows with empty-but-present cells ("")
    while dropping rows that are entirely missing (all None).
    """
    return any(cell is not None for cell in row)


def _has_enough_structure(rows: list[list[Optional[str]]]) -> bool:
    if len(rows) < MIN_TABLE_ROWS:
        return False
    structured_rows = sum(
        1 for row in rows if _structural_cell_count(row) >= MIN_TABLE_COLS
    )
    return structured_rows >= MIN_TABLE_ROWS


def _header_is_suspicious(rows: list[list[Optional[str]]]) -> bool:
    if not rows:
        return True
    first_row = rows[0]
    first_structural = _structural_cell_count(first_row)
    first_present = _present_cell_count(first_row)
    first_empty = _empty_cell_count(first_row)

    if first_present == 0:
        return True
    if first_structural == 0:
        return True
    if first_present > 0 and (first_empty / first_present) > 0.6:
        return True
    if len(first_row) >= MIN_TABLE_COLS and first_structural < max(1, len(first_row) // 2):
        return True
    return False


def _is_complex_table(rows: list[list[Optional[str]]]) -> bool:
    if not rows:
        return True

    total_cells = sum(len(row) for row in rows)
    if total_cells == 0:
        return True

    null_count = sum(1 for row in rows for cell in row if cell is None)
    null_ratio = null_count / total_cells
    if null_ratio > NULL_THRESHOLD:
        return True

    structured_rows = sum(
        1 for row in rows if _structural_cell_count(row) >= MIN_TABLE_COLS
    )
    if structured_rows < MIN_TABLE_ROWS:
        return True

    if _header_is_suspicious(rows):
        return True

    expected_present_cells = max(_present_cell_count(row) for row in rows)
    if expected_present_cells == 0:
        return True

    ragged_rows = sum(
        1 for row in rows
        if _present_cell_count(row) < expected_present_cells * 0.7
    )
    if ragged_rows > len(rows) * RAGGED_THRESHOLD:
        return True

    return False


def _parse_table(table: object, page_num: int) -> Optional[TableData]:
    try:
        raw = table.extract()
    except Exception:
        logger.warning("Failed to extract table on page %d", page_num)
        return None

    if not raw:
        return None

    rows: list[list[Optional[str]]] = []
    for row in raw:
        normalized_row = [_normalize_cell(cell) for cell in row]
        if _is_meaningful_row(normalized_row):
            rows.append(normalized_row)

    if not _has_enough_structure(rows):
        return None

    return {
        "page": page_num,
        "rows": rows,
        "needs_vision": _is_complex_table(rows),
    }


def _extract_tables_from_doc(doc: fitz.Document) -> list[TableData]:
    tables: list[TableData] = []
    for page_num, page in enumerate(doc, start=1):
        try:
            page_tables = page.find_tables()
        except Exception:
            logger.warning("Table extraction failed on page %d", page_num)
            continue
        for table in page_tables:
            parsed = _parse_table(table, page_num)
            if parsed is not None:
                tables.append(parsed)
    return tables


def _extract_text_from_doc(
    doc: fitz.Document,
) -> tuple[str, dict[int, str]]:
    parts: list[str] = []
    text_by_page: dict[int, str] = {}
    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text()
        except Exception:
            logger.warning("Text extraction failed on page %d", page_num)
            text = ""
        text_by_page[page_num] = text
        parts.append(f"--- PAGE {page_num} ---\n{text}")
    return "\n".join(parts), text_by_page


def _extract_page_image_base64(page: fitz.Page) -> str:
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")


def parse_pdf(file_path: str) -> ParsedPDF:
    """
    Full PDF parsing path.
    Returns raw_text с page markers, text_by_page,
    tables, arxiv_id, arxiv_id_versioned, typed metadata.
    """
    t0 = time.perf_counter()
    path = _check_file(file_path)
    logger.info("Parsing PDF: %s", file_path)

    doc = fitz.open(str(path))
    try:
        page_count = len(doc)
        metadata = _extract_metadata(doc)
        raw_text, text_by_page = _extract_text_from_doc(doc)
        tables = _extract_tables_from_doc(doc)
    finally:
        doc.close()

    arxiv_id, arxiv_id_versioned = _extract_arxiv_id(raw_text)
    latency = time.perf_counter() - t0
    logger.info(
        "PDF parsed: %d pages, %d tables, arxiv_id=%s, arxiv_id_versioned=%s, latency=%.2fs",
        page_count, len(tables), arxiv_id, arxiv_id_versioned, latency,
    )

    return {
        "file_path": str(path),
        "page_count": page_count,
        "raw_text": raw_text,
        "text_by_page": text_by_page,
        "arxiv_id": arxiv_id,
        "arxiv_id_versioned": arxiv_id_versioned,
        "tables": tables,
        "metadata": metadata,
    }


def extract_tables(file_path: str) -> list[TableData]:
    """
    Fast path для Benchmark Agent.
    only tables - without raw_text, metadata, arxiv_id.
    """
    t0 = time.perf_counter()
    path = _check_file(file_path)
    logger.info("Extracting tables: %s", file_path)

    doc = fitz.open(str(path))
    try:
        tables = _extract_tables_from_doc(doc)
    finally:
        doc.close()

    latency = time.perf_counter() - t0
    logger.info("Extracted %d tables in %.2fs", len(tables), latency)
    return tables


def get_page_image(file_path: str, page_num: int) -> str:
    """
    Base64 PNG pages for Claude Vision fallback.
    page_num: 1-indexed.
    """
    path = _check_file(file_path)

    doc = fitz.open(str(path))
    try:
        page_count = len(doc)
        if page_num < 1 or page_num > page_count:
            raise ValueError(
                f"Page {page_num} out of range (1..{page_count})"
            )
        img_b64 = _extract_page_image_base64(doc[page_num - 1])
    finally:
        doc.close()

    logger.info("Rendered page %d of %s as PNG", page_num, file_path)
    return img_b64