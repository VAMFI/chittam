# Chittam (चित्तम्)

**One local memory for all your AI coding agents.**

Chittam indexes your repositories, notes, PDFs, screenshots, and past agent conversations **on your own machine**, then lets Antigravity, Codex, and Claude Code search all of it through one MCP server. Nothing leaves your computer unless you explicitly opt in.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-stdio-purple.svg)](https://modelcontextprotocol.io)
[![Local-first](https://img.shields.io/badge/privacy-local--first-green.svg)](docs/ARCHITECTURE.md#5-privacy-gate)

> *Chittam* (Sanskrit *citta*) is the part of the mind that stores impressions and memories.

---

## The problem

If you use more than one AI coding agent, you have probably hit these:

| Problem | What it looks like |
| :--- | :--- |
| **Agents forget** | Yesterday you designed something with Claude Code. Today Codex has no idea it happened. |
| **Stale copies win** | You have `my-app`, `my-app-old`, and `my-app-backup`. The agent reads the wrong one. |
| **`grep` is not enough** | "Where did we handle login retries?" has no exact keyword to search for. |
| **Images are invisible** | Architecture diagrams and UI screenshots hold answers no text search can see. |
| **Memory tools leak or bloat** | Auto-injected memory breaks prompt caching, and naive indexers happily index `.env` files. |

## What Chittam does

1. **Scans** the folders you choose, plus your Antigravity, Codex, and Claude Code histories.
2. **Blocks** secrets and private folders *before* reading them (`.env`, keys, tax folders, credential files).
3. **Understands** text, code, PDFs, and images, using local models through [Ollama](https://ollama.com).
4. **Ranks** active projects above old copies by tracking every git repo as `active`, `reference`, or `archive`.
5. **Serves** compact search results over MCP, only when an agent asks, so prompt caches stay intact.

```mermaid
flowchart TD
    A["Your folders: repos, notes, PDFs, images"] --> G
    B["Agent histories: Antigravity, Codex, Claude Code"] --> G
    G{"Privacy gate: skip secrets and private folders"} --> P
    P["Parse and chunk: code, markdown, PDF, image captions"] --> M
    M["Local models via Ollama: text embeddings + vision captions"] --> S
    S[("SQLite index: keyword + vector + project atlas + decisions")] --> Q
    Q["Hybrid search: keyword + meaning, active projects first"] --> MCP
    MCP["Pull-only MCP server: chittam-mcp"] --> H1["Antigravity"]
    MCP --> H2["Codex"]
    MCP --> H3["Claude Code"]
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Built on existing tools, not a rewrite

Chittam doesn't rebuild what already works well. It connects existing tools and fills the gaps between them.

| Tool | What it's great at | Where Chittam fits |
| :--- | :--- | :--- |
| [`ripgrep`](https://github.com/BurntSushi/ripgrep) | Exact matches in the current repo | Agents keep using it for known symbols |
| [CodeGraph](https://github.com/colbymchenry/codegraph) | Call graphs and symbol relationships | Installed alongside Chittam as a second MCP server |
| [LEANN](https://github.com/StarTrail-org/LEANN) | Very small vector indexes | Optional export target (`chittam sync --leann-export`) |
| [qmd](https://github.com/tobi/qmd) | Hybrid search over markdown notes | Same hybrid idea, extended to code, images, and agent threads |
| [claude-mem](https://github.com/thedotmack/claude-mem) | Automatic session memory | Chittam is pull-only, so it never rewrites your prompt |

What Chittam adds: **cross-agent recall**, **active-vs-archived project ranking**, **image and PDF understanding**, **decision history**, and a **strict privacy gate**, in one local index.

---

## Quick start (about 5 minutes)

**Requirements:** macOS or Linux, Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com). Any Apple Silicon Mac or a machine with about 8 GB of free RAM works.

```bash
# 1. Install
git clone https://github.com/VAMFI/chittam.git
cd chittam
uv sync

# 2. Pull the local models (about 2.5 GB total)
ollama pull embeddinggemma:latest   # text embeddings
ollama pull qwen3-vl:2b             # image understanding

# 3. Check everything works, without touching your files
uv run chittam benchmark --synthetic

# 4. Create your config, then edit which folders to scan
uv run chittam init
#    -> ~/.config/chittam/config.yaml   (see docs/HOW_TO.md#2)

# 5. Build the index
uv run chittam sync

# 6. Try a search
uv run chittam search "where do we handle login retries"
```

**Connect your agents** (backs up every config file first):

```bash
uv run chittam install-harnesses   # Antigravity + Codex + Claude Code
uv run chittam install-launchd     # macOS: re-sync every 30 minutes
```

Undo at any time with `uv run chittam uninstall-harnesses` and `uv run chittam uninstall-launchd`.

Optional companions:

```bash
npm install -g @colbymchenry/codegraph        # structural code graph MCP
uv tool install leann-core --with leann       # compact vector index export
```

---

## Commands

| Command | What it does |
| :--- | :--- |
| `chittam init` | Create config, database, and project atlas |
| `chittam atlas [--refresh]` | List discovered git repos and their tier |
| `chittam sync` | Incremental index update (only changed files are re-processed) |
| `chittam search "query"` | Hybrid search (`--media-type image`, `--project`, `--harness`, `--no-archived`, `--json`) |
| `chittam decide <project> <topic> "<statement>"` | Record a decision (replaces the previous one on that topic) |
| `chittam decisions` | List current decisions |
| `chittam status` | Index size, counts, background sync status |
| `chittam install-harnesses` / `uninstall-harnesses` | Wire or unwire MCP in all three agents |
| `chittam install-launchd` / `uninstall-launchd` | Enable or disable scheduled sync (macOS) |
| `chittam benchmark [--synthetic] [--cases FILE]` | Measure retrieval quality and privacy leakage |
| `chittam mcp` | Run the MCP server on stdio |

## MCP tools

| Tool | Use it when an agent needs to... |
| :--- | :--- |
| `memory_search` | Find something by meaning across code, docs, PDFs, images, and threads |
| `memory_get_chunk` | Read the full text around a search hit |
| `project_atlas` | Know which copy of a project is the current one |
| `recall_threads` | Recall what was discussed in earlier Antigravity, Codex, or Claude Code sessions |
| `record_decision` | Save an architectural decision so future sessions respect it |
| `index_status` | Check that the index exists and is fresh |

Agents are told *when* to use these through a short rule: [docs/ROUTING_RULE.md](docs/ROUTING_RULE.md).

## Measured results

Reproducible on any machine with `uv run chittam benchmark --synthetic` (14 fixture documents plus 3 private files that must be blocked):

| Metric | Result |
| :--- | :--- |
| Recall@5 | **1.0** (13/13 queries) |
| MRR | **0.95** |
| Privacy leakage | **0** (3/3 private fixtures blocked at ingest) |

On a real developer machine (M4 Pro, 24 GB) with about 31,000 documents and 330,000 chunks, a personal 15-query set scored Recall@5 1.0 and MRR 0.94, with p50 latency around 380 ms. See [the FAQ](docs/FAQ.md#how-fast-is-it) for honest caveats.

## Documentation

| Doc | For |
| :--- | :--- |
| [Architecture](docs/ARCHITECTURE.md) | How it works and why each piece exists |
| [How-to guides](docs/HOW_TO.md) | Setup, privacy config, agents, decisions, cloud opt-in, extending |
| [FAQ](docs/FAQ.md) | Plain-English answers, including limitations |
| [Routing rule](docs/ROUTING_RULE.md) | The short instruction agents follow |
| [Contributing](CONTRIBUTING.md) | Development workflow |

## License

[MIT](LICENSE) © VAMFI Inc.
