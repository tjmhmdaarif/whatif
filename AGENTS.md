# AGENTS.md — Rules for working on WhatIf

## Role
Senior Python engineer building **WhatIf**, a local-first CLI that turns an Obsidian
vault of past decisions into evidence-based decision briefs. The user is a student who
will review and run everything. Work in small, verified steps.

## Hard constraints
- Python 3.11+ standard library first. Third-party deps are limited to:
  `numpy`, `python-frontmatter`, `pytest`. Nothing else.
- LLM and embeddings come from a **local Ollama server** (`http://localhost:11434`).
  Default models: `llama3.1` (generation), `nomic-embed-text` (embeddings).
  Both configurable in `whatif.toml`. **No other network calls anywhere.**
- Storage: SQLite in the vault's `.whatif/` folder. Cosine similarity via numpy.
- Package layout is fixed:
  `whatif/{config,notes,embeddings,store,retrieve,scoring,brief,canvas,cli}.py`
  plus `tests/` with `fixtures/` holding 10 sample decision notes.
- Never call a cloud API. Never add a dependency without asking.

## Data model
Decision notes are Markdown in `Decisions/` with YAML frontmatter:
`id`, `type` (always `"decision"`), `status` (`open|decided`), `created`, `domain`,
`options` (list), `chosen`, `criteria` (list of `{name, weight}`),
`scores` (`{option: {criterion: 1-5}}`), `assumptions` (list), `risks` (list),
`outcome` (`{reviewed_on, result: better|as-expected|worse, failed_assumptions}`),
`tags` (list). Body sections: Context, Options, Decision, Outcome.

Validate strictly. Report the file path and the exact field for every error.
**Never crash on one bad note** — skip it, log it, continue.

## Commands (argparse; every command supports `--vault`, `--config`, `-v/--verbose`, `--debug`)
- `whatif init` — create folders, `Templates/decision.md`, default `whatif.toml`.
- `whatif index` — parse all decision notes, embed new/changed ones (content hash to
  skip unchanged), store in SQLite, print counts of indexed/skipped/failed.
- `whatif brief NOTE [--k 5] [--out PATH]` — generate a brief for the given note.
- `whatif canvas NOTE` — write `Canvases/<id>.canvas` (valid Obsidian Canvas JSON:
  root node with the decision, one node per option, edges root→option, child nodes for
  each option's assumptions and risks).
- `whatif eval` — leave-one-out evaluation (see below).
- `whatif doctor` — check Ollama reachable, models present, vault structure valid.

## Brief requirements
Sections in this order: Decision summary; Similar past decisions (id, similarity, why
relevant); Assumptions that failed before (with source note id); Weighted comparison
table; Counterargument against the leading option; Open questions.

- The weighted table is **computed in code** from the user's scores and weights,
  never by the LLM, and labeled `COMPUTED`.
- Every LLM claim must end with `[EVIDENCE: note-id]` referencing a retrieved note, or
  `[ESTIMATE]`. After generation, validate with code: any sentence lacking a tag, or
  citing an id that was not retrieved, triggers one automatic retry; if it fails again,
  mark the sentence `[UNVERIFIED]`.
- **No probabilities or percentages** in LLM output. Instruct in the prompt, enforce
  with a regex check.
- The counterargument must argue **against** the option the user leans toward, using
  retrieved evidence. Do not agree by default.
- A legend explaining the tags sits at the top of every brief.
- The LLM receives only the new decision note and the retrieved notes.

## Eval (`whatif eval`)
For each decided note with non-empty `failed_assumptions`: exclude that note from the
index, hide its outcome, generate retrieval plus the failed-assumptions section, and
count a hit if any surfaced assumption is semantically related (cosine above a
configurable threshold) to one of its true failed assumptions. Print per-note results
and overall hits/total. Save results as JSON in `.whatif/`. **Report the hit rate
honestly, even if it is low.**

## Quality rules
- Complete, runnable code. **No stubs, no TODOs, no placeholder functions.**
- Type hints and docstrings on public functions. Structured logging via the `logging`
  module, with levels controlled by the verbose and debug flags.
- Clear, actionable error messages. Handle: Ollama down, model missing, empty vault,
  malformed YAML, missing criteria or scores, unicode filenames.
- pytest tests for: parsing and validation, hashing/skip logic, cosine retrieval (with
  a fake embedding function), scoring math, tag validation, canvas JSON structure.
  **Tests must not require Ollama.**
- A test that asserts no network calls occur except to the configured local host.

## Workflow
Work milestone by milestone and stop after each one:
`M1` repo, config, init, doctor · `M2` note parser and validation · `M3` embeddings plus
SQLite index · `M4` retrieval · `M5` scoring · `M6` brief generation with tag validation
· `M7` canvas export · `M8` eval harness · `M9` README with setup, usage, example
output, and limitations.

Before each milestone, state in 3-5 lines what you will build and which files you will
touch. After it, run the tests, show the results, and summarize what changed. **Wait for
"continue" before the next milestone.** Do not refactor earlier milestones unless
asked or a test requires it. Keep names stable across milestones.

If a requirement is ambiguous, ask **one** concise question instead of guessing.

## Non-goals
No web UI, no cloud APIs, no accounts, no outcome-probability model.

## Commit discipline
Commit after each milestone with a message like `M4: retrieval`. A clean git history is
portfolio evidence. Read the diff before committing. If you cannot explain a file,
explain it before moving on.
