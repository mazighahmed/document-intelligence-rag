import re
from typing import Protocol

import httpx

from docrag.chunking import Chunk
from docrag.config import Settings

SYSTEM_PROMPT = (
    "Answer the question using only the numbered sources. "
    "Cite sources inline like [1] or [2]. "
    "If the sources do not contain the answer, say you don't know."
)


class LLM(Protocol):
    def generate(self, question: str, sources: list[Chunk]) -> str: ...


def build_prompt(question: str, sources: list[Chunk]) -> str:
    blocks = [
        f"[{i}] ({c.filename}, page {c.page})\n{c.text}" for i, c in enumerate(sources, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class MockLLM:
    """Extractive stand-in: returns the source sentence sharing the most words with the question."""

    def generate(self, question: str, sources: list[Chunk]) -> str:
        if not sources:
            return "I don't know."
        asked = words(question)
        candidates = [
            (len(asked & words(sentence)), sentence.strip(), i)
            for i, c in enumerate(sources, start=1)
            for sentence in re.split(r"(?<=[.!?])\s+", c.text)
        ]
        _, sentence, i = max(candidates, key=lambda x: x[0])
        return f"{sentence} [{i}]"


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 60.0):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self.model = model

    def generate(self, question: str, sources: list[Chunk]) -> str:
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, sources)},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def make_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
    if settings.llm_provider == "mock":
        return MockLLM()
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")
