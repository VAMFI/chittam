# Contributing to Chittam

Thanks for helping. Chittam is small on purpose, so please keep changes focused.

## Setup

```bash
git clone https://github.com/VAMFI/chittam.git
cd chittam
uv sync
uv run pytest -q                       # all tests must pass
uv run chittam benchmark --synthetic   # Recall@5 1.0, privacy_leakage 0
```

The tests don't need Ollama. They use fake embedders and temporary folders.

## Ground rules

1. **Privacy first.** Every new adapter must call `is_path_excluded()` before opening a file and `redact_secrets()` before storing text. Add a test that proves a sensitive fixture is never ingested.
2. **Read-only access to other tools.** Open agent databases with SQLite `?mode=ro`. Never write to another tool's files, except through the backed-up, reversible `harnesses.py` flow.
3. **Local by default.** New providers that call the network must go through the consent gate (`enabled` + `user_consented`).
4. **No personal data in the repo.** Fixtures use generic names and `tmp_path`, never real home paths.
5. **Keep the routing rule under 150 words.** A test enforces this.

## Where things go

| Change | File(s) |
| :--- | :--- |
| New file/media type | `src/chittam/adapters/files.py` (`register_media_handler`) |
| New agent history adapter | `src/chittam/adapters/<agent>.py`, then register it in `adapters/__init__.py` |
| Ranking / search | `src/chittam/store.py` |
| New MCP tool | `src/chittam/mcp_server.py` |
| New model provider | `src/chittam/providers.py` |
| Agent config wiring | `src/chittam/harnesses.py` |
| Benchmark cases | `src/chittam/benchmark.py` |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit.

## Good first contributions

- Linux `systemd` timer support alongside launchd
- History adapters for Cursor, Gemini CLI, or OpenCode
- An approximate-nearest-neighbour index (USearch/HNSW) for large corpora
- An optional local cross-encoder reranker
- An audio handler using a local Whisper model

## Pull request checklist

- [ ] `uv run pytest -q` passes
- [ ] `uv run chittam benchmark --synthetic` still reports `privacy_leakage: 0`
- [ ] New behaviour has a test
- [ ] Docs updated if user-facing behaviour changed
