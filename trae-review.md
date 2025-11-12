## Status

- Uses a modern src layout with PEP 621 metadata and Hatchling; CLI entry points are defined and `src/llm_server` is packaged via wheel target `pyproject.toml:105`.
- Python version is pinned to `3.13` via `.python-version` and `requires-python >=3.13` `pyproject.toml:6`, `.python-version:1`.
- UV integration is present and now aligned for ROCm 7 nightly PyTorch with dependency groups, default dev group, and index/source mapping.

## Changes Made

- Removed an invalid line that would crash imports `src/llm_server/server.py:15`.
- Updated `pyproject.toml` for UV and modern dev workflow:
  - Switched dev tools to the standardized `[dependency-groups]` per PEP 735; added `pytest`, `httpx`, `ruff`, `mypy` `pyproject.toml:51`.
  - Set `[tool.uv] default-groups = ["dev"]` so dev deps install by default `pyproject.toml:73`.
  - Mapped ROCm 7 nightly index and directed `torch` to that source `pyproject.toml:76`, `pyproject.toml:81`.
  - Left `torch` unpinned to allow UV to resolve a suitable ROCm 7 nightly wheel; kept `accelerate` unpinned to match torch compatibility `pyproject.toml:37`, `pyproject.toml:40`.
  - Added lint/type-check configs for `ruff` and `mypy` `pyproject.toml:96`, `pyproject.toml:100`.

## ROCm 7 Nightly Setup

- Custom index: `[[tool.uv.index]] name = "pytorch-rocm" url = "https://download.pytorch.org/whl/nightly/rocm7.0"` `pyproject.toml:76`.
- Source mapping: `[tool.uv.sources] torch = { index = "pytorch-rocm" }` (also `torchvision`, `torchaudio`) `pyproject.toml:81`.
- Ensures UV resolves PyTorch from the ROCm 7 nightly index; if you need prereleases explicitly, include the prerelease flag when syncing.
  - UV installs Python matching `requires-python` and syncs groups/lock in one step [1], [2].

## How To Use (UV + Virtualenv)

- Create and sync the environment (installs Python 3.13 if missing and populates `.venv`):
  - `uv sync`
- Include only runtime dependencies (skip dev tools):
  - `uv sync --no-dev`
- Force refresh and allow prereleases (useful for nightlies on ROCm):
  - `uv sync --upgrade --prerelease`
- Run the server:
  - `uv run llm-server --model meta-llama/Meta-Llama-3-8B-Instruct`
- Run tests and tooling:
  - `uv run pytest`
  - `uv run ruff check src tests`
  - `uv run mypy src`

## Project Review

- Packaging: PEP 621 fields complete; Hatchling backend defined `pyproject.toml:69`.
- Scripts: CLI, benchmarking, profiling entry points registered `pyproject.toml:59`.
- Ignore rules include `.venv`, `uv.lock`, and `.python-version.lock` `./.gitignore:24`, `./.gitignore:30`.
- FastAPI app entry is packaged and importable from `llm_server` `src/llm_server/__init__.py:7`.
- ROCm flags: experimental AOTriton enabled in app import path `src/llm_server/server.py:17`.

## Notes

- Tests import `from main import app` `tests/test_llm_server.py:4`, but there is no `main.py`; consider switching to `from llm_server.server import app` to match the packaged entrypoint.

## Sources

- UV dependency groups and dev install behavior [1] https://stackoverflow.com/questions/78902565/how-do-i-install-python-dev-dependencies-using-uv
- UV project dependency concepts and sync behavior [2] https://docs.astral.sh/uv/concepts/projects/dependencies/
