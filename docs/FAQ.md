# FAQ

**Basics**
- [What is Chittam, in one sentence?](#what-is-chittam-in-one-sentence)
- [What is "semantic search" and "hybrid RAG"?](#what-is-semantic-search-and-hybrid-rag)
- [Do I need to be a developer to use it?](#do-i-need-to-be-a-developer-to-use-it)
- [What does the name mean?](#what-does-the-name-mean)

**Privacy**
- [Does any of my data leave my computer?](#does-any-of-my-data-leave-my-computer)
- [How do I make sure a folder is never indexed?](#how-do-i-make-sure-a-folder-is-never-indexed)
- [What if a secret is inside a normal file?](#what-if-a-secret-is-inside-a-normal-file)
- [Can Chittam break my Codex or Claude Code setup?](#can-chittam-break-my-codex-or-claude-code-setup)

**How it compares**
- [Why not just let the agent use grep?](#why-not-just-let-the-agent-use-grep)
- [How is this different from CodeGraph, LEANN, qmd, or claude-mem?](#how-is-this-different-from-codegraph-leann-qmd-or-claude-mem)
- [Why pull-only instead of auto-injecting memory?](#why-pull-only-instead-of-auto-injecting-memory)
- [Why SQLite instead of a vector database?](#why-sqlite-instead-of-a-vector-database)

**Using it**
- [What hardware do I need?](#what-hardware-do-i-need)
- [How fast is it?](#how-fast-is-it)
- [How often should I re-index?](#how-often-should-i-re-index)
- [Can it search images and PDFs? Audio or video?](#can-it-search-images-and-pdfs-audio-or-video)
- [Which agents are supported?](#which-agents-are-supported)
- [Does it work on Windows?](#does-it-work-on-windows)
- [How do I undo everything?](#how-do-i-undo-everything)
- [What are the known limitations?](#what-are-the-known-limitations)

---

## Basics

### What is Chittam, in one sentence?
A local search engine over your projects, documents, screenshots, and past AI-agent conversations, which your coding agents can query through MCP.

### What is "semantic search" and "hybrid RAG"?
- **Keyword search** finds exact words. Searching "login retry" finds files containing those words.
- **Semantic search** finds *meaning*. Each piece of text is turned into a list of numbers (an "embedding") so that similar ideas end up close together. "Where do we retry failed sign-ins?" can then find code that never uses the word "retry".
- **Hybrid** runs both and merges the results, because each catches things the other misses.
- **RAG** (retrieval-augmented generation) just means the AI looks things up before answering, instead of relying on memory alone.

### Do I need to be a developer to use it?
Setup currently uses the terminal. Once it runs, though, it's useful for anyone with lots of documents. Point `scan_roots` at `~/Documents` or a screenshots folder and ask questions in plain language. A friendlier installer is on the roadmap.

### What does the name mean?
*Chittam* comes from the Sanskrit *citta*: the layer of mind where impressions and memories are stored.

## Privacy

### Does any of my data leave my computer?
**Not by default.** Embeddings and image captions come from Ollama running on `127.0.0.1`, and the index is a single SQLite file in `~/.local/share/chittam/`. Cloud models are possible only if you set **both** `external_provider.enabled: true` **and** `external_provider.user_consented: true`. Otherwise the code raises `ConsentRequiredError`. Optional companions such as CodeGraph have their own telemetry settings: run `codegraph telemetry off`.

### How do I make sure a folder is never indexed?
Add it to `privacy.exclude_path_globs` (for example `~/Medical/**`) or add its name to `exclude_dir_names`. Exclusions are checked **before** a file is opened. Folders named `tax` or `taxes`, plus `.ssh`, `.aws`, and `.gnupg`, are blocked by hard-coded guards even if you edit the config. See [HOW_TO §2](HOW_TO.md#2-choose-what-gets-indexed-and-what-never-does).

### What if a secret is inside a normal file?
Indexed text passes through `redact_secrets()`, which replaces GitHub tokens, `sk-…` keys, Google `AIza…` keys, AWS access keys, PEM private keys, and `password=` / `api_key=` style values with `[REDACTED_…]`. This is pattern-based, so treat it as a safety net. Unusual secret formats can slip through. Keep secrets in `.env` files, which are never read.

### Can Chittam break my Codex or Claude Code setup?
It's designed not to. Codex databases are opened **read-only**. Before editing any agent config, Chittam saves a backup, and it re-validates the file afterwards (restoring the backup if parsing fails). `uninstall-harnesses` removes only what it added.

## How it compares

### Why not just let the agent use grep?
For a name you already know in the current repo, `grep` is perfect, and the routing rule tells agents to keep using it. It struggles with:
- Questions without an exact keyword ("how do we handle offline sync?")
- Searching across 100+ repos without reading dozens of files
- Telling the current copy of a project from last year's backup
- Images, scanned PDFs, and past conversations

### How is this different from CodeGraph, LEANN, qmd, or claude-mem?
They're good tools, and Chittam works alongside them:

| | Strength | Gap Chittam fills |
| :--- | :--- | :--- |
| **CodeGraph** | Call graphs inside a repo | Docs, images, agent history, cross-repo ranking. *Installed together.* |
| **LEANN** | Tiny vector indexes | Project tiers, decision history, agent adapters. *Optional export target.* |
| **qmd** | Hybrid search over notes | Code-aware chunking, images, agent threads, tiers |
| **claude-mem** | Automatic session memory | Pull-only (no prompt-cache misses), multi-agent, files and images |

### Why pull-only instead of auto-injecting memory?
LLM providers cache the beginning of your prompt. If a memory tool inserts different context every turn, that cache is invalidated and you pay for the whole prompt again. Chittam adds one short, unchanging rule to the system prompt and lets the agent call a tool only when it needs to.

### Why SQLite instead of a vector database?
One file, no Docker, no server, easy backups, and FTS5 keyword search built in. For personal-scale data (up to about 100k vectors) brute-force cosine similarity with numpy is fast enough. Larger corpora can export to LEANN, and an ANN index is on the roadmap.

## Using it

### What hardware do I need?
Any Apple Silicon Mac, or a Linux machine with roughly 8 GB of free RAM. The default models need about 2.5 GB of disk. The index for about 30k documents was about 1.3 GB.

### How fast is it?
Measured on an M4 Pro with 24 GB RAM:

| Operation | Typical |
| :--- | :--- |
| Search (≈330k chunks, ≈6k vectors) | p50 ≈ 380 ms, p95 ≈ 2.4 s (the first query loads the embedding model) |
| Synthetic benchmark search | under 1 ms per query |
| Embedding | roughly 20 to 60 chunks/s |
| Image caption (`qwen3-vl:2b`) | roughly 1 to 17 s per image |
| First full sync (≈30k documents) | about 2 hours |
| Incremental sync | proportional to what changed |

Your numbers will vary with corpus size and hardware.

### How often should I re-index?
With `install-launchd` (macOS) or a cron job (Linux), every 30 minutes. Unchanged files are skipped by hash, so frequent syncs are cheap.

### Can it search images and PDFs? Audio or video?
- **Images** (PNG, JPG, WEBP): a local vision model writes a description, including visible text and diagram labels. Captions are cached by file hash.
- **PDFs**: text is extracted per page, and pages with almost no text are flagged as possibly scanned.
- **Audio and video**: the data model and a plug-in registry are ready, but no handler ships yet. See [HOW_TO §8](HOW_TO.md#8-add-a-new-media-type-audio-video-).

### Which agents are supported?
Built-in history adapters and one-command wiring exist for **Antigravity**, **Codex**, and **Claude Code**. Any MCP client can use the server by pointing at `.venv/bin/chittam-mcp`. Adapters for other agents' histories are welcome contributions.

### Does it work on Windows?
Not tested. WSL2 should work for the core features, but scheduled sync and agent wiring assume macOS/Linux paths.

### How do I undo everything?
```bash
uv run chittam uninstall-harnesses
uv run chittam uninstall-launchd
rm -rf ~/.local/share/chittam ~/.config/chittam
```

### What are the known limitations?
See [ARCHITECTURE §13](ARCHITECTURE.md#13-known-limits). In short: brute-force vector search, vectors fill in over several syncs on large machines, a slow first sync, no reranker yet, loose notes rank below project files, and macOS may block background access to some folders until you allow it.
