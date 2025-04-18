from typing import Protocol

import httpx

from docrag.chunking import Chunk

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


class MockLLM:
    def generate(self, question: str, sources: list[Chunk]) -> str:
        if not sources:
            return "I don't know."
        first_sentence = sources[0].text.split(". ")[0].rstrip(".")
        return f"{first_sentence}. [1]"


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
