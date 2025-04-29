import pytest

from docrag.chunking import chunk_pages, document_id, split_text
from docrag.extract import Page


def test_split_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    pieces = split_text(text, 100, 20)
    assert all(len(p) <= 100 for p in pieces)
    assert pieces[0].split()[-1] in pieces[1]


def test_short_text_is_one_chunk():
    assert split_text("short text", 100, 10) == ["short text"]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_long_word_without_spaces_still_splits():
    pieces = split_text("x" * 250, 100, 10)
    assert "".join(pieces).count("x") >= 250


def test_chunks_keep_page_and_stable_ids():
    pages = [Page(1, "alpha " * 50), Page(3, "beta")]
    first = chunk_pages(pages, "doc", "a.pdf", 100, 10)
    second = chunk_pages(pages, "doc", "a.pdf", 100, 10)
    assert [c.id for c in first] == [c.id for c in second]
    assert first[-1].page == 3
    assert [c.index for c in first] == list(range(len(first)))


def test_document_id_depends_on_content():
    assert document_id(b"a") == document_id(b"a")
    assert document_id(b"a") != document_id(b"b")
