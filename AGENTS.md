# Agent Guidelines

- Scope: entire repository.
- Prefer small, focused changes; keep GUI signal handlers safe against early invocations (guards are welcome).
- Before committing, run `python -m compileall main.py` to ensure the code compiles.
- Document changes with clear summaries and include the commands you ran in the Testing section.
