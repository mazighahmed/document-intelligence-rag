import pytest
from conftest import make_pdf

from docrag.extract import ExtractionError, extract_pages, normalize


def test_extracts_text_per_page():
    pages = extract_pages(make_pdf(["first page", "second page"]))
    assert [(p.number, p.text) for p in pages] == [(1, "first page"), (2, "second page")]


def test_skips_blank_pages_but_keeps_numbers():
    pages = extract_pages(make_pdf(["", "only text here"]))
    assert [p.number for p in pages] == [2]


def test_empty_pdf_raises():
    with pytest.raises(ExtractionError, match="no extractable text"):
        extract_pages(make_pdf([""]))


def test_non_pdf_raises():
    with pytest.raises(ExtractionError, match="not a PDF"):
        extract_pages(b"hello")


def test_normalize_joins_hyphenated_words_and_spaces():
    assert normalize("infor-\nmation   retrieval\n\n\n\nnext") == "information retrieval\n\nnext"


def test_truncated_pdf_raises():
    data = make_pdf(["some text"])
    with pytest.raises(ExtractionError, match="could not read"):
        extract_pages(data[: len(data) // 2])
