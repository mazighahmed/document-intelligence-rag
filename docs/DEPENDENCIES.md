# Dependencies

Checked against PyPI and GitHub release pages. Cutoff: 2025-05-31.
`exclude-newer` in `pyproject.toml` keeps transitive packages under the same cutoff.

## Runtime

| Package               | Version | Released   |
|-----------------------|---------|------------|
| Python                | 3.13.3  | 2025-04-08 |
| fastapi               | 0.115.12| 2025-03-23 |
| uvicorn               | 0.34.2  | 2025-04-19 |
| pydantic              | 2.11.5  | 2025-05-22 |
| pydantic-settings     | 2.9.1   | 2025-04-18 |
| python-multipart      | 0.0.20  | 2024-12-16 |
| pypdf                 | 5.5.0   | 2025-05-11 |
| sentence-transformers | 4.1.0   | 2025-04-15 |
| torch (CPU wheels)    | 2.7.0   | 2025-04-23 |
| qdrant-client         | 1.14.2  | 2025-04-24 |
| httpx                 | 0.28.1  | 2024-12-06 |

## Dev and tooling

| Tool                     | Version | Released   |
|--------------------------|---------|------------|
| uv                       | 0.7.9   | 2025-05-30 |
| hatchling                | 1.27.0  | 2024-12-15 |
| pytest                   | 8.3.5   | 2025-03-02 |
| ruff                     | 0.11.12 | 2025-05-29 |
| actions/checkout         | v4.2.2  | 2024-10-23 |
| astral-sh/setup-uv       | v6.1.0  | 2025-05-23 |
| python:3.13.3-slim-bookworm | -    | 2025-05-22 |

## Models

| Model                                   | Released |
|-----------------------------------------|----------|
| sentence-transformers/all-MiniLM-L6-v2  | 2021     |
| llama3.2 (via Ollama, optional)         | 2024-09  |
