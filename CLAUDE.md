# CLAUDE.md

## Project Memory
- **Read `MEMORY.md` first** on every task. It records the design decisions and their rationale (data facts, preprocessing rules, model/serving contracts, code conventions, open issues). `readme.md` §5 describes the workflow and file layout.
- When a decision changes, update the matching section in `MEMORY.md` and add a line to its 변경 이력. Don't duplicate what the code or `readme.md` already states.

## Build & Test Commands
- **Preprocess**: `python .py/preprocess.py` (train, ~7 min) → `python .py/preprocess.py --split test`
- **Train + evaluate**: `python .py/train.py --exclude-non-welding --cv 4` (~1 min) → `python .py/train.py --evaluate`
- **Experiments**: `python .py/experiments.py cache --window 60` → `python .py/experiments.py run <name>` (results in `results/`)
- **Run Server**: `uvicorn main:app --reload` (stream replay: `python .py/replay.py`)
- **Run Tests**: `pytest` (synthetic end-to-end, ~1 min; never touches the real data folders)
- **Run Single Test**: `pytest tests/test_pipeline.py -k rule`
- **Lint**: `ruff check .` (config in `pyproject.toml`, line length 120)

## Code Style & Architecture
- **Environment**: Python 3.11+, Windows, CPU-only. pandas / scikit-learn / FastAPI + Pydantic v2.
- **Layout**: scripts in `.py/` (`preprocess.py` → `train.py`; `experiments.py` lab; `rag_mapping.py` handoff), API in root `main.py`, which imports `train` / `preprocess` so that training and serving share one implementation. No database.
- **Type Hints**: Explicit type hints for function parameters and return values in `main.py`; the `.py/` scripts follow their existing style.
- **Validation**: Pydantic v2 schemas for all API request/response models. `AnomalyResult` / `RagHandoff` are contracts: add fields, never remove or change meaning.
- **Paths**: from `PROJECT_ROOT` (`__file__`), never the cwd. Outputs go to `models/`, `results/`, `eda_out/`, never into `train/`, `test/`, `preprocessed/`.

## Token & Output Optimization Rules
- **Concise Responses**: Skip intros, outros, polite chatter, and conversational filler.
- **Diff-focused Output**: When modifying files, show only the changed code snippet or apply edits directly without reprinting entire unchanged files.
- **Targeted Reading**: Read specific referenced files (`@file`) instead of scanning whole directories unless explicitly instructed.
- **Self-Healing Loop**: Run tests (`pytest`), inspect error logs directly, fix the code, and re-run until passing without long verbal explanations.