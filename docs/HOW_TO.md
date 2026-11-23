# How-to guides

Step-by-step recipes. Every command assumes you are inside the cloned `chittam` folder.

1. [First-time setup](#1-first-time-setup)
2. [Choose what gets indexed (and what never does)](#2-choose-what-gets-indexed-and-what-never-does)
3. [Fix which copy of a project counts as current](#3-fix-which-copy-of-a-project-counts-as-current)
4. [Connect Antigravity, Codex, and Claude Code](#4-connect-antigravity-codex-and-claude-code)
5. [Keep the index fresh (watch + scheduled sync)](#5-keep-the-index-fresh-watch--scheduled-sync)
6. [Record decisions agents should respect](#6-record-decisions-agents-should-respect)
7. [Opt in to cloud models, reranking, or a remote Ollama server](#7-opt-in-to-cloud-models-reranking-or-a-remote-ollama-server)
8. [Add a new media type (audio, video, …)](#8-add-a-new-media-type-audio-video-)
9. [Measure quality, compact storage, and diagnose health](#9-measure-quality-compact-storage-and-diagnose-health)
10. [Upgrading from v0.1 & uninstalling cleanly](#10-upgrading-from-v01--uninstalling-cleanly)

---

## 1. First-time setup

**macOS (Apple Silicon recommended) or Linux**

```bash
# Prerequisites
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv (Python package manager)
# Install Ollama from https://ollama.com, then:
ollama pull embeddinggemma:latest
ollama pull qwen3-vl:2b

# Chittam
git clone https://github.com/VAMFI/chittam.git
cd chittam
uv sync                                 # add `--extra rerank` for optional local ONNX reranking
uv run chittam benchmark --synthetic    # sanity check: expect "Recall@5": 1.0, "privacy_leakage": 0
uv run chittam init                     # writes ~/.config/chittam/config.yaml
```

Before the first `sync`, open `~/.config/chittam/config.yaml` and review `scan_roots` (see §2). Then run:

```bash
uv run chittam sync
uv run chittam doctor
```

> **Tip:** On huge machines, embeddings and image captions are capped per run (`max_embed_chunks_per_sync`, `max_images_per_sync`). Thanks to the v0.2 `stat()` pre-filter, follow-up incremental syncs skip unchanged files in ~2–3 seconds and resume backfilling only pending chunks.

## 2. Choose what gets indexed (and what never does)

All settings live in `~/.config/chittam/config.yaml`. Your file is merged over the defaults in [`config/policy.default.yaml`](../config/policy.default.yaml).

**Pick folders.** These can be developer folders or everyday ones:

```yaml
privacy:
  scan_roots:
    - ~/Projects
    - ~/Code
    - ~/Documents
    - ~/Pictures/Screenshots
    - "~"            # home itself: top-level files only, never subfolders
  home_top_level_files_only: true
```

**Pick media types & embedding tiers:**

```yaml
privacy:
  enabled_media: [code, markdown, text, pdf, image, transcript, decision]
  embed_tiers: [active, reference]                 # add `archive` to vectorize old clones too
  embed_always_media: [image, transcript, pdf, decision]
```

**Block anything sensitive.** Add to the lists. Defaults already cover keys, `.env*`, credentials, and `tax`/`taxes` folders.

```yaml
privacy:
  exclude_path_globs:
    - ~/tax/**
    - ~/Documents/Medical/**
    - ~/Documents/passport.jpg
  exclude_dir_names: [tax, taxes, medical, legal]
  exclude_file_patterns: ["*.env", "*.pem", "*statement*", "*payslip*"]
```

> **Important:** When you override a list in your config, it **replaces** the default list rather than adding to it. Copy the defaults you want to keep from `config/policy.default.yaml`, then append your own entries.

Files that are already indexed but newly excluded are removed on the next `chittam sync`.

**Tune vector size & workload:**

```yaml
local_provider:
  matryoshka_dim: 256              # slice 768-d EmbeddingGemma -> 256-d (~3x smaller vectors)
  vector_dtype: i8                 # i8 (~12x total savings vs f32), or f16 / f32
  rerank_model: null               # set "ms-marco-MiniLM-L-12-v2" to enable FlashRank by default
  rerank_top_k: 20

privacy:
  max_images_per_sync: 15          # new image captions per run
  max_embed_chunks_per_sync: 6000  # new unique chunks per run (null = unlimited)
  max_code_files_per_repo: 250
```

## 3. Fix which copy of a project counts as current

```bash
uv run chittam atlas --refresh           # rescan git repos
uv run chittam atlas --tier active       # list current projects
uv run chittam atlas --family my-app     # every copy of one project
```

Edit `~/.config/chittam/canonical-projects.yaml` to pin a tier:

```yaml
projects:
  my-app--archives:
    tier: active            # this old-looking copy is actually current
    manual_override: true   # keep my choice on future refreshes
    notes: "Main branch lives here now"
```

Then run `uv run chittam sync` so documents pick up the new tier.

## 4. Connect Antigravity, Codex, and Claude Code

```bash
uv run chittam install-harnesses
```

This backs up and edits:

| Agent | Config file |
| :--- | :--- |
| Antigravity | `~/.gemini/config/mcp_config.json` |
| Codex | `~/.codex/config.toml` |
| Claude Code | `~/.claude.json` |

Restart each agent so it loads the new MCP servers. Then paste the contents of [`docs/ROUTING_RULE.md`](ROUTING_RULE.md) into each agent's global instructions (for example `AGENTS.md`, `CLAUDE.md`, or a rule/skill). It's under 150 words, so it costs almost nothing per turn and never changes between turns, which keeps prompt caching intact.

**Using a different MCP client?** Point it at:

```json
{ "command": "/path/to/chittam/.venv/bin/chittam-mcp", "args": [] }
```

**Undo:** `uv run chittam uninstall-harnesses`. Backups stay in `~/.local/share/chittam/backups/`.

## 5. Keep the index fresh (watch + scheduled sync)

**Real-time watcher (macOS FSEvents / Linux inotify):**

```bash
uv run chittam watch                     # re-indexes edited files & chats in milliseconds
uv run chittam watch --debounce-ms 2000 --no-vision
```

**macOS background schedule (periodic safety net):**

```bash
uv run chittam install-launchd    # every 30 minutes
uv run chittam doctor             # verifies launchd & vector parity
tail -f ~/.local/share/chittam/sync.log
```

**Linux** (cron):

```bash
crontab -e
# add:
*/30 * * * * /path/to/chittam/.venv/bin/chittam sync --quiet >> ~/.local/share/chittam/sync.log 2>&1
```

**Manual:** `uv run chittam sync` (add `--no-vision` for a fast text-only pass, or `--refresh-atlas` after moving repos).

Overlapping syncs and watcher runs are safe: a shared lockfile guarantees two writers never fight over SQLite or `vectors.usearch`.

> **macOS note:** If `chittam doctor` or `sync.log` reports `Operation not permitted` on `~/Desktop`, `~/Documents`, or `~/Downloads`, grant Full Disk Access to your terminal / `.venv/bin/python` or remove that folder from `scan_roots`.

## 6. Record decisions agents should respect

```bash
uv run chittam decide my-app storage "Use SQLite locally; Postgres only for cloud sync"
uv run chittam decisions                 # current decisions
```

Recording another decision with the same `my-app` + `storage` pair closes the old one (`valid_to` and `superseded_by` are set) and keeps it in history. Agents can do the same through the `record_decision` MCP tool, and decisions are both keyword- and vector-indexed so `memory_search` finds them reliably.

## 7. Opt in to cloud models, reranking, or a remote Ollama server

**Local cross-encoder reranking (optional `[rerank]` extra):**

```bash
uv sync --extra rerank
uv run chittam search "retry backoff jitter" --rerank
```

To make reranking always-on in MCP searches too, set `local_provider.rerank_model: ms-marco-MiniLM-L-12-v2` in `~/.config/chittam/config.yaml`.

**Cloud or remote Ollama models:** Everything stays local by default. To use a remote model you control, **both** flags must be `true`:

```yaml
external_provider:
  enabled: true
  user_consented: true            # explicit acknowledgement that content leaves this machine
  provider_type: openai_compatible   # or: gemini, remote_ollama
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY        # the key is read from this env var, never stored in config
  embedding_model: text-embedding-3-small
  vision_model: gpt-4o-mini
```

Run with `uv run chittam sync --external`.

- **Remote Ollama** (for example, a GPU box on your LAN): `provider_type: remote_ollama`, `base_url: http://192.168.1.50:11434`, no key needed.
- If either flag is `false`, Chittam raises `ConsentRequiredError` and sends nothing.
- In v0.2, `index_meta` tracks which model source (`local` vs `external`) and dimension built your store, so queries automatically match the index's active model and Matryoshka dimension.

## 8. Add a new media type (audio, video, …)

```python
# my_plugins/audio.py
from pathlib import Path
from chittam.adapters import chunk_markdown_or_text, make_doc_id, make_l0_summary, normalize_iso_timestamp, sha256_bytes
from chittam.adapters.files import register_media_handler
from chittam.models import DocumentRecord, MediaType, RepoTier

def handle_audio(path: Path, ctx) -> DocumentRecord | None:
    text = transcribe_locally(path)                     # e.g. whisper.cpp / mlx-whisper
    doc_id = make_doc_id("file_audio", str(path))
    tier = ctx.repo.tier if ctx.repo else RepoTier.REFERENCE
    updated = normalize_iso_timestamp(None, fallback_mtime=path.stat().st_mtime)
    chunks = chunk_markdown_or_text(doc_id=doc_id, text=text, project_id=None,
                                    canonical_tier=tier, path_or_uri=str(path), updated_at=updated)
    return DocumentRecord(doc_id=doc_id, uri=str(path), source_harness="files",
                          media_type=MediaType.AUDIO, canonical_tier=tier, title=path.name,
                          l0_summary=make_l0_summary(text), sha256=sha256_bytes(path.read_bytes()),
                          updated_at=updated, chunks=chunks)

register_media_handler(MediaType.AUDIO, {".wav", ".mp3", ".m4a"}, handle_audio)
```

Then add `audio` to `privacy.enabled_media`. The privacy gate, `stat()` + Merkle diff, content-deduped embedding, and USearch search all work automatically.

Adding a new **agent harness** (for example Cursor or Gemini CLI) follows the same pattern as `adapters/claude_code.py`: read its history files read-only, return `DocumentRecord`s with `media_type=TRANSCRIPT`, and register the loader in `collect_all_documents()`.

## 9. Measure quality, compact storage, and diagnose health

```bash
uv run chittam doctor                   # checks config, DB/USearch parity, Ollama, TCC, harnesses, launchd
uv run chittam status                   # docs, chunks, vectors, unique hashes, USearch size
uv run chittam compact                  # purge orphan blobs, rebuild USearch, checkpoint WAL, VACUUM
uv run chittam reindex-vectors          # rebuild vectors.usearch offline from SQLite
uv run chittam benchmark --synthetic    # reproducible, no personal data
uv run chittam benchmark                # against your real index
uv run chittam benchmark --rerank       # compare accuracy & latency with cross-encoder reranking
```

For the real-index benchmark, write queries that match *your* files in `~/.config/chittam/benchmark_cases.yaml`:

```yaml
cases:
  - {id: Q1, category: cross_repo, query: "billing webhook retry logic", expect_any: [billing-service]}
  - {id: Q2, category: tier, query: "mobile app login screen", expect_any: [mobile-app], require_non_archive: true}
  - {id: Q3, category: threads, mode: threads, harness: claude, query: "rate limiter design", expect_any: [rate]}
  - {id: Q4, category: image, mode: image, query: "system architecture diagram", expect_any: [architecture]}
  - {id: Q5, category: privacy, mode: privacy, query: "tax return passport"}
```

- `expect_any`: a hit passes if any top-5 result's path, title, project, or snippet contains one of these strings.
- `require_non_archive: true`: the first matching hit must not be an archived copy.
- `mode: privacy`: passes only if no result comes from an excluded path.

## 10. Upgrading from v0.1 & uninstalling cleanly

**Upgrading from v0.1 to v0.2:**
Pull and run `uv sync`, then run `uv run chittam sync && uv run chittam compact`. Your existing 768-d vectors, image caption cache, and decisions migrate **automatically in-place** to Matryoshka `256-d i8` + `vectors.usearch` with zero Ollama calls.

**Removing Chittam completely:**

```bash
uv run chittam uninstall-harnesses
uv run chittam uninstall-launchd
rm -rf ~/.local/share/chittam ~/.config/chittam
# optionally: ollama rm embeddinggemma qwen3-vl:2b
```
