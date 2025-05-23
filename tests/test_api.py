import httpx
import pytest
from conftest import make_pdf
from fastapi.testclient import TestClient

from docrag.api import create_app


@pytest.fixture
def client(pipeline):
    with TestClient(create_app(pipeline)) as c:
        yield c


def upload(client, name, data):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_upload_list_ask_delete(client):
    r = upload(client, "facts.pdf", make_pdf(["Water boils at 100 degrees Celsius."]))
    assert r.status_code == 200
    doc_id = r.json()["doc_id"]
    assert client.get("/documents").json() == {doc_id: "facts.pdf"}

    r = client.post("/ask", json={"question": "When does water boil?"})
    source = r.json()["sources"][0]
    assert (source["filename"], source["page"]) == ("facts.pdf", 1)

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == {}


def test_rejects_non_pdf_extension(client):
    assert upload(client, "notes.txt", b"hello").status_code == 415


def test_rejects_corrupt_pdf(client):
    assert upload(client, "bad.pdf", b"%PDF-1.4 garbage").status_code == 422


def test_rejects_empty_file(client):
    assert upload(client, "empty.pdf", b"").status_code == 422


def test_rejects_pdf_without_text(client):
    r = upload(client, "scan.pdf", make_pdf([""]))
    assert r.status_code == 422
    assert "no extractable text" in r.json()["detail"]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 0}).status_code == 422


def test_llm_failure_returns_502(pipeline):
    class BrokenLLM:
        def generate(self, question, sources):
            raise httpx.ConnectError("connection refused")

    pipeline.llm = BrokenLLM()
    with TestClient(create_app(pipeline)) as client:
        r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 502
