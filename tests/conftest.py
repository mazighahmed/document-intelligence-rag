import hashlib
import re

import pytest
from qdrant_client import QdrantClient

from docrag.llm import MockLLM
from docrag.pipeline import Pipeline
from docrag.store import VectorStore


def make_pdf(pages: list[str]) -> bytes:
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    font_id = 3
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = []
    for text in pages:
        lines = [line.replace("(", r"\(").replace(")", r"\)") for line in text.split("\n")]
        ops = " ".join(f"({line}) Tj 0 -14 Td" for line in lines)
        stream = f"BT /F1 11 Tf 50 750 Td {ops} ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return out


class HashEmbedder:
    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors


@pytest.fixture
def pipeline() -> Pipeline:
    embedder = HashEmbedder()
    store = VectorStore(QdrantClient(":memory:"), "test", embedder.dim)
    return Pipeline(embedder, store, MockLLM(), chunk_size=200, chunk_overlap=20)
