# Architecture

```
upload -> extract -> chunk -> embed -> Qdrant
question -> embed -> Qdrant top-k -> LLM -> answer + sources
```

| Module          | Job |
|-----------------|-----|
| `extract.py`    | pypdf text per page, whitespace and hyphenation cleanup, errors for non-PDF, corrupt, encrypted and text-less files |
| `chunking.py`   | splits each page into overlapping chunks on word boundaries; document id is a hash of the file bytes, chunk ids are UUIDv5 of `doc_id:index` |
| `embeddings.py` | `Embedder` protocol and a sentence-transformers implementation (normalized vectors) |
| `store.py`      | Qdrant collection with cosine distance; chunk metadata stored as payload |
| `llm.py`        | `LLM` protocol, prompt with numbered sources, OpenAI-compatible HTTP client and an offline mock |
| `pipeline.py`   | `ingest` and `ask`, wiring the pieces above |
| `api.py`        | FastAPI endpoints and building the pipeline from settings |
| `evaluate.py`   | runs the eval set and prints retrieval and answer metrics |

Re-uploading the same file replaces its chunks instead of duplicating them, because ids are
derived from content. Each part sits behind a small interface, so tests swap in a hashing
embedder, in-memory Qdrant and the mock LLM.
