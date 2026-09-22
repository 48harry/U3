# CLAUDE.md

## Project Memory
- **Read `MEMORY.md` first** on every task. It records the design decisions and their rationale (data facts, preprocessing rules, model/serving contracts, code conventions, open issues). `readme.md` §5 describes the workflow and file layout.
- When a decision changes, update the matching section in `MEMORY.md` and add a line to its 변경 이력. Don't duplicate what the code or `readme.md` already states.

## Build & Test Commands
- **Run Server**: `uvicorn main:app --reload`
- **Run Tests**: `pytest`
- **Run Single Test**: `pytest tests/test_user.py -k test_login`
- **Lint & Format**: `black . && ruff check .`
- **Docker**: `docker-compose up --build`

## Code Style & Architecture
- **Environment**: Python 3.11+ / FastAPI
- **Architecture**: Controller (Router) -> Service -> Repository (SQLAlchemy ORM)
- **Type Hints**: Explicit type hints required for all function parameters and return values.
- **Validation**: Use Pydantic v2 schemas for all request/response models.
- **Async**: Use `async/await` for database I/O and external API calls.

## Token & Output Optimization Rules
- **Concise Responses**: Skip intros, outros, polite chatter, and conversational filler.
- **Diff-focused Output**: When modifying files, show only the changed code snippet or apply edits directly without reprinting entire unchanged files.
- **Targeted Reading**: Read specific referenced files (`@file`) instead of scanning whole directories unless explicitly instructed.
- **Self-Healing Loop**: Run tests (`pytest`), inspect error logs directly, fix the code, and re-run until passing without long verbal explanations.