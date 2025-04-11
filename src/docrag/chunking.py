import hashlib
import uuid
from dataclasses import dataclass

from docrag.extract import Page


@dataclass
class Chunk:
    id: str
    doc_id: str
    filename: str
    page: int
    index: int
    text: str


def document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        pieces.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_pages(
    pages: list[Page], doc_id: str, filename: str, size: int, overlap: int
) -> list[Chunk]:
    chunks = []
    for page in pages:
        for text in split_text(page.text, size, overlap):
            index = len(chunks)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{index}"))
            chunks.append(Chunk(chunk_id, doc_id, filename, page.number, index, text))
    return chunks
