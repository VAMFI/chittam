# Memory & Retrieval Routing Rule (Antigravity / Codex / Claude Code)

Keep system prompts static to preserve prompt caching—never preload memory dumps into context. Pull tools on demand:

1. **Active working directory & known symbols** (`grep`, `ripgrep`, `glob`, direct file reads):
   Use native search first for exact identifiers, file paths, or edits inside the current repository.

2. **CodeGraph** (`codegraph_explore`, `codegraph_node`, call-graph queries):
   Use for structural architecture, callers/callees, and blast-radius analysis inside an indexed repo.

3. **Chittam** (`memory_search`, `project_atlas`, `recall_threads`, `record_decision`):
   Use for cross-repo discovery, conceptual questions, active-vs-archived repo disambiguation (`project_atlas`), screenshots/diagrams/PDFs, prior Antigravity/Codex/Claude threads (`recall_threads`), and persisting architectural choices (`record_decision`).
