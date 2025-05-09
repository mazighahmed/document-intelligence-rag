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


def test_mock_llm_picks_matching_sentence():
    from docrag.chunking import Chunk
    from docrag.llm import MockLLM

    sources = [
        Chunk("1", "d", "a.pdf", 1, 0, "Title. Cats sleep a lot."),
        Chunk("2", "d", "a.pdf", 2, 1, "Water boils at 100 degrees."),
    ]
    assert MockLLM().generate("When does water boil?", sources) == "Water boils at 100 degrees. [2]"
