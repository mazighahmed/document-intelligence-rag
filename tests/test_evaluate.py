import json
from pathlib import Path

from conftest import HashEmbedder

from docrag.config import Settings
from docrag.evaluate import evaluate
from docrag.llm import MockLLM

EVAL_DIR = Path(__file__).parent.parent / "eval"


def test_evaluation_runs_on_eval_set():
    questions = json.loads((EVAL_DIR / "questions.json").read_text())
    result = evaluate(HashEmbedder(), MockLLM(), EVAL_DIR / "docs", questions, 3, Settings())
    assert result["questions"] == len(questions)
    assert 0 <= result["mrr"] <= result["hit_rate"] <= 1
    assert result["citation_rate"] == 1.0
