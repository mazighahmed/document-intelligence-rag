import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from docrag.config import Settings
from docrag.embeddings import Embedder
from docrag.llm import LLM
from docrag.pipeline import Pipeline
from docrag.store import VectorStore


def evaluate(
    embedder: Embedder, llm: LLM, docs_dir: Path, questions: list[dict], k: int, settings: Settings
) -> dict:
    store = VectorStore(QdrantClient(":memory:"), "eval", embedder.dim)
    pipeline = Pipeline(embedder, store, llm, settings.chunk_size, settings.chunk_overlap)
    for path in sorted(docs_dir.glob("*.pdf")):
        pipeline.ingest(path.name, path.read_bytes())

    hits, reciprocal_ranks, answered, cited, rows = 0, 0.0, 0, 0, []
    for q in questions:
        answer = pipeline.ask(q["question"], k)
        found = [(c.filename, c.page) for c, _ in answer.sources]
        rank = found.index((q["file"], q["page"])) + 1 if (q["file"], q["page"]) in found else None
        has_keywords = all(word.lower() in answer.text.lower() for word in q["keywords"])
        has_citation = "[" in answer.text and "]" in answer.text
        hits += rank is not None
        reciprocal_ranks += 1 / rank if rank else 0.0
        answered += has_keywords
        cited += has_citation
        rows.append({"question": q["question"], "rank": rank, "keywords_ok": has_keywords})

    n = len(questions)
    return {
        "questions": n,
        "k": k,
        "hit_rate": round(hits / n, 3),
        "mrr": round(reciprocal_ranks / n, 3),
        "answer_keyword_rate": round(answered / n, 3),
        "citation_rate": round(cited / n, 3),
        "details": rows,
    }


def main() -> None:
    from docrag.embeddings import SentenceTransformerEmbedder
    from docrag.llm import make_llm

    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers on eval/")
    parser.add_argument("--eval-dir", default="eval", type=Path)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    settings = Settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    questions = json.loads((args.eval_dir / "questions.json").read_text())
    result = evaluate(
        embedder, make_llm(settings), args.eval_dir / "docs", questions, args.k, settings
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
