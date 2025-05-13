from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docrag.config import Settings
from docrag.extract import ExtractionError
from docrag.pipeline import Pipeline

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    filename: str
    page: int
    chunk: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def build_pipeline(settings: Settings) -> Pipeline:
    from qdrant_client import QdrantClient

    from docrag.embeddings import SentenceTransformerEmbedder
    from docrag.llm import make_llm
    from docrag.store import VectorStore

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(QdrantClient(path=settings.qdrant_path), settings.collection, embedder.dim)
    llm = make_llm(settings)
    return Pipeline(embedder, store, llm, settings.chunk_size, settings.chunk_overlap)


def create_app(pipeline: Pipeline | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_pipeline(settings)
        yield

    app = FastAPI(title="docrag", lifespan=lifespan)

    @app.post("/documents", response_model=IngestResponse)
    def upload(file: UploadFile) -> IngestResponse:
        filename = file.filename or "upload.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(415, "only PDF files are supported")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large")
        if not data:
            raise HTTPException(422, "empty file")
        try:
            doc_id, pages, chunks = app.state.pipeline.ingest(filename, data)
        except ExtractionError as e:
            raise HTTPException(422, str(e)) from e
        return IngestResponse(doc_id=doc_id, filename=filename, pages=pages, chunks=chunks)

    @app.get("/documents")
    def list_documents() -> dict[str, str]:
        return app.state.pipeline.store.documents()

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> None:
        app.state.pipeline.store.delete_document(doc_id)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        result = app.state.pipeline.ask(request.question, request.top_k or settings.top_k)
        sources = [
            Source(
                doc_id=c.doc_id,
                filename=c.filename,
                page=c.page,
                chunk=c.index,
                score=score,
                text=c.text,
            )
            for c, score in result.sources
        ]
        return AskResponse(answer=result.text, sources=sources)

    return app


app = create_app()
