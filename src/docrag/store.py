from dataclasses import asdict

from qdrant_client import QdrantClient, models

from docrag.chunking import Chunk


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client = client
        self.collection = collection
        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(id=c.id, vector=v, payload=asdict(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        hits = self.client.query_points(self.collection, query=vector, limit=k).points
        return [(Chunk(**h.payload), h.score) for h in hits]

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        )

    def documents(self) -> dict[str, str]:
        docs = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=["doc_id", "filename"]
            )
            for p in points:
                docs[p.payload["doc_id"]] = p.payload["filename"]
            if offset is None:
                return docs
