from dataclasses import dataclass

from docrag.chunking import Chunk, chunk_pages, document_id
from docrag.embeddings import Embedder
from docrag.extract import extract_pages
from docrag.llm import LLM
from docrag.store import VectorStore


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]]


class Pipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, filename: str, data: bytes) -> tuple[str, int, int]:
        pages = extract_pages(data)
        doc_id = document_id(data)
        chunks = chunk_pages(pages, doc_id, filename, self.chunk_size, self.chunk_overlap)
        self.store.delete_document(doc_id)
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return doc_id, len(pages), len(chunks)

    def retrieve(self, question: str, k: int) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.embed([question])[0], k)

    def ask(self, question: str, k: int) -> Answer:
        hits = self.retrieve(question, k)
        text = self.llm.generate(question, [c for c, _ in hits])
        return Answer(text, hits)
