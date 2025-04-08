import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ExtractionError(Exception):
    pass


@dataclass
class Page:
    number: int
    text: str


def normalize(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_pages(data: bytes) -> list[Page]:
    if not data.startswith(b"%PDF"):
        raise ExtractionError("not a PDF file")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("encrypted PDFs are not supported")
        pages = [Page(i + 1, normalize(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    except PdfReadError as e:
        raise ExtractionError(f"could not read PDF: {e}") from e
    pages = [p for p in pages if p.text]
    if not pages:
        raise ExtractionError("no extractable text (scanned or empty PDF)")
    return pages
