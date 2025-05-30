# document-intelligence-rag

Upload PDFs, ask questions, get answers with file and page citations.

PDF text is extracted with pypdf, split into overlapping chunks, embedded with
`all-MiniLM-L6-v2` and stored in Qdrant (local mode, no server needed). Questions
retrieve the top-k chunks, which are passed to an LLM with numbered sources.

## Run

Requires [uv](https://docs.astral.sh/uv/) 0.7.9.

```
uv sync
cp .env.example .env
uv run uvicorn docrag.api:app --reload
```

```
curl -F "file=@report.pdf" localhost:8000/documents
curl -H "content-type: application/json" -d '{"question": "What was the revenue?"}' localhost:8000/ask
curl localhost:8000/documents
curl -X DELETE localhost:8000/documents/<doc_id>
```

The default `LLM_PROVIDER=mock` returns the source sentence that best matches the
question, so everything works offline. For real answers, point it at any
OpenAI-compatible endpoint, e.g. Ollama:

```
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

## Evaluation

`eval/` has three small PDFs and 15 questions with the expected file, page and answer keywords.

```
uv run python -m docrag.evaluate -k 3
```

Reports hit rate and MRR for retrieval, and how often the answer contains the expected
keywords and a citation. With MiniLM and the mock LLM: hit rate 1.0, MRR 1.0, keyword
rate 0.73. The set is tiny, so treat it as a regression check, not a benchmark.

## Tests

```
uv run pytest
```

Tests use a hashing embedder and in-memory Qdrant, so no model download is needed.

## Limitations

- No OCR. Scanned PDFs are rejected with 422.
- Encrypted PDFs are rejected.
- Chunking is character based, not aware of headings or tables.
- Local Qdrant mode allows one process at a time; use a Qdrant server for more.
- Uploads are processed synchronously in the request.
