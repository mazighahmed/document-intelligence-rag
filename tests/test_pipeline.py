from conftest import make_pdf


def test_ingest_and_ask_returns_cited_source(pipeline):
    pdf = make_pdf(["The capital of France is Paris.", "Water boils at 100 degrees Celsius."])
    doc_id, pages, chunks = pipeline.ingest("facts.pdf", pdf)
    assert (pages, chunks) == (2, 2)

    answer = pipeline.ask("At what temperature does water boil?", k=1)
    chunk, _ = answer.sources[0]
    assert chunk.page == 2
    assert chunk.doc_id == doc_id
    assert "[1]" in answer.text


def test_reingesting_same_file_does_not_duplicate(pipeline):
    pdf = make_pdf(["some content about cats"])
    pipeline.ingest("a.pdf", pdf)
    pipeline.ingest("a.pdf", pdf)
    assert len(pipeline.retrieve("cats", k=10)) == 1


def test_ask_with_empty_index(pipeline):
    answer = pipeline.ask("anything?", k=3)
    assert answer.sources == []
    assert answer.text == "I don't know."
