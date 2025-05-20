import json

import httpx
import pytest

from docrag.chunking import Chunk
from docrag.config import Settings
from docrag.llm import MockLLM, OpenAICompatibleLLM, build_prompt, make_llm

SOURCES = [Chunk("1", "d", "guide.pdf", 4, 0, "Panels face south.")]


def test_prompt_numbers_sources_with_file_and_page():
    prompt = build_prompt("Which way?", SOURCES)
    assert "[1] (guide.pdf, page 4)\nPanels face south." in prompt
    assert prompt.endswith("Question: Which way?")


def test_openai_client_sends_prompt_and_reads_answer():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "South [1]"}}]})

    llm = OpenAICompatibleLLM("http://llm.test/v1", "some-model", api_key="secret")
    llm.client = httpx.Client(
        base_url="http://llm.test/v1",
        headers=llm.client.headers,
        transport=httpx.MockTransport(handler),
    )
    assert llm.generate("Which way?", SOURCES) == "South [1]"
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "some-model"
    assert seen["auth"] == "Bearer secret"


def test_openai_client_raises_on_http_error():
    llm = OpenAICompatibleLLM("http://llm.test/v1", "m")
    llm.client = httpx.Client(
        base_url="http://llm.test/v1", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    with pytest.raises(httpx.HTTPStatusError):
        llm.generate("q", SOURCES)


def test_make_llm():
    assert isinstance(make_llm(Settings(llm_provider="mock")), MockLLM)
    assert isinstance(make_llm(Settings(llm_provider="openai")), OpenAICompatibleLLM)
    with pytest.raises(ValueError):
        make_llm(Settings(llm_provider="nope"))
