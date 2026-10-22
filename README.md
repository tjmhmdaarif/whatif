# WhatIf

**A local-first decision brief generator for Obsidian vaults.**

Write a decision into your vault. WhatIf finds the past decisions that resemble it,
surfaces the assumptions that failed before, computes the weighted comparison from
*your* scores, argues against the option you are leaning toward, and writes a brief
that labels every claim as evidence or estimate.

Everything runs on your machine. No accounts, no cloud calls, no telemetry. The only
network traffic is to a local Ollama server.

```text
vault/              whatif/ (Python)                  Ollama              SQLite
Decisions/ ───────► │ notes.py    parse + validation  │ llama3.1          .whatif/
Templates/          │ store.py    notes + vectors     │ nomic-embed-text  index.sqlite
Briefs/             │ embeddings.py Ollama client     │                   │
Canvases/           │ retrieve.py cosine, top-k       │                   │
.whatif/            │ scoring.py   weighted table     │                   │
                    │ brief.py     prompt + tag check │                   │
                    │ canvas.py    Obsidian Canvas    │                   │
                    │ eval.py      leave-one-out      │                   │
                    │ cli.py       argparse commands  │                   │
                    │ webapp.py    local HTTP UI      │                   │
                    └───────────────┬─────────────────┘
                                    │ 127.0.0.1 only
                             ┌──────▼──────┐
                             │  Ollama     │
                             └─────────────┘
```

The flow for a brief:

```text
new note ──► parse ──► embed ──► retrieve top-k past decisions
                                   │
                                   ├──► failed assumptions from those past notes
                                   │
                                   ├──► weighted comparison table  (code, never the LLM)
                                   │
                                   └──► prompt the LLM with only the new note
                                          and the retrieved notes
                                             │
                                             └──► validate every tag (code)
                                                    ├── pass ──────────► brief
                                                    └── fail ──► retry once
                                                               └── fail ──► [UNVERIFIED]
```

---

## Setup

Requirements: Python 3.11+, and [Ollama](https://ollama.com) for the language and
embedding models.

```bash
# Install WhatIf (numpy, python-frontmatter; pytest for the test suite)
pip install -e ".[dev]"

# Pull the default models
ollama pull llama3.1
ollama pull nomic-embed-text
```

The only hard dependency on the network is the local Ollama server at
`http://localhost:11434`. Both models are configurable in `whatif.toml`.

### Offline demo (no Ollama, no internet)

If you cannot install Ollama, the repo ships a stand-in server that impersonates the
Ollama API with a deterministic hash-based embedder. It is useful for exercising the
pipeline and seeing the output, but its hit rate is **not** a benchmark:

```bash
python scripts/demo.py
```

That starts the stand-in server, indexes the seed vault, and serves the web interface.
It detects a real Ollama already running and leaves it alone.

---

## Quickstart

```bash
whatif init --vault ./vault          # create folders, config, and a note template
whatif doctor --vault ./vault        # check Ollama, models, vault layout
whatif index  --vault ./vault        # parse and embed every decision note
whatif brief  2026-11-01-laptop      # write a decision brief
whatif canvas 2026-11-01-laptop      # write a branching Obsidian Canvas
whatif eval   --vault ./vault        # leave-one-out evaluation
whatif serve  --vault ./vault        # local web interface on :8765
```

Every command accepts `--vault`, `--config`, `-v/--verbose` and `--debug`.

### Commands

| Command | What it does |
|---|---|
| `init` | Create the vault folder structure, `Templates/decision.md`, and a default `whatif.toml`. |
| `doctor` | Health check: Ollama reachable, both models present, vault structure valid. |
| `index` | Parse all decision notes, embed new/changed ones (content hash to skip unchanged), store in SQLite. Prints indexed/skipped/failed counts. |
| `brief NOTE [--k 5] [--out PATH]` | Generate a brief for the given note. Default output: `Briefs/<id>.md`. |
| `canvas NOTE [--out PATH]` | Write `Canvases/<id>.canvas`, valid Obsidian Canvas JSON. |
| `eval [--threshold FLOAT] [--out PATH]` | Leave-one-out evaluation. Saves JSON results to `.whatif/eval-results.json`. |
| `serve [--host HOST] [--port PORT]` | Local web interface (binds to 127.0.0.1 only). |

---

### The local web interface

```bash
whatif serve --vault ./vault --port 8765
```

Open <http://127.0.0.1:8765>. It is a local console over the same pipeline the CLI
uses, with five views per decision: the brief, the branching canvas drawn as an
animated graph, the retrieval, the computed scoring, and the evaluation.

Keyboard: `/` focuses the filter, `j` / `k` move between decisions, `g` cycles tabs.

![The brief view](docs/screenshots/brief.png)

The brief is rendered from the same file that lands in `Briefs/`. Tags are colour
coded, the weighted table is marked `COMPUTED`, and the counterargument names the
option it is arguing against.

![The canvas view](docs/screenshots/canvas.png)

The canvas view draws the same JSON that Obsidian reads, but as an animated graph.
The root node is the decision, each option carries its computed total and criterion
breakdown, and the shared assumption and risk nodes sit below.

![The retrieval view](docs/screenshots/retrieval.png)

Retrieval shows the similarity of every candidate and marks which past failures
clear the threshold a brief would apply, so it is obvious what a brief would and
would not have shown.

![The scoring view](docs/screenshots/scoring.png)

The scoring view compares the options criterion by criterion. The leading option is
highlighted, which makes the trade-off visible.

![The evaluation view](docs/screenshots/evaluation.png)

The evaluation view names the embedder that produced the numbers, because a hit
rate without that context is not a result.

---

## The note format

Decision notes are Markdown in `Decisions/` with YAML frontmatter:

```yaml
---
id: 2026-10-21-laptop-choice
type: decision
status: open            # open | decided
created: 2026-10-21
domain: tools           # study | career | money | tools | other
options: [Refurbished ThinkPad, New ultrabook]
chosen:
criteria:
  - {name: price, weight: 1}
  - {name: battery_life, weight: 5}
  - {name: repairability, weight: 2}
  - {name: performance, weight: 2}
scores:                 # 1-5 per option per criterion; 5 is always the best outcome
  Refurbished ThinkPad: {price: 5, battery_life: 3, repairability: 5, performance: 3}
  New ultrabook: {price: 2, battery_life: 5, repairability: 1, performance: 5}
assumptions: [The discount would still apply at checkout]
risks: [Soldered memory means no upgrades later]
outcome:
  reviewed_on:
  result:               # better | as-expected | worse
  failed_assumptions: []
tags: []
---
## Context

## Options

## Decision

## Outcome
```

**Scoring convention:** 5 always means *the best outcome for you on that criterion*.
Name criteria so the direction is obvious. `price` (5 = cheapest) and
`free_time_left` (5 = most time remaining) are unambiguous; `cost` is not, because
"5 on cost" could mean expensive or good.

The weighted total is `sum(score x weight)`. The maximum possible is
`5 x sum(weights)`.

---

## Example output

```markdown
# Decision brief: 2026-09-14-side-project-build

> **Legend**
> - `[EVIDENCE: note-id]` - a claim supported by a past decision that was retrieved
> - `[ESTIMATE]` - a judgement with no supporting evidence in your notes
> - `[UNVERIFIED]` - a claim that failed validation and could not be confirmed
> - `COMPUTED` - a value calculated in code from your own scores and weights
>
> WhatIf does not predict outcomes. Nothing here is a probability.

## Decision summary

The user is choosing between two options that differ mainly on price and speed [ESTIMATE]

## Similar past decisions

| Note | Similarity | Why it is relevant |
|---|---|---|
| `2026-05-12-tool-stack` | 0.758 | A past decision where the cheaper option lost for reasons that were not about price |

## Assumptions that failed before

- The rest of the team would move to the new tool and keep using it [EVIDENCE: 2026-05-12-tool-stack] (similarity 0.782)

## Weighted comparison

| Option | ecosystem (w3) | hiring_visibility (w3) | learning_value (w2) | speed_to_first_demo (w2) | **Total (COMPUTED)** |
|---|---|---|---|---|---|
| Build in TypeScript | 5 | 5 | 3 | 3 | **42** |
| Build in Python | 4 | 2 | 4 | 4 | **34** |

*COMPUTED: weighted total = sum of (score x weight). Maximum possible is 50 points.*

## Counterargument: against Build in TypeScript

A similar decision did not pay off the last time it was made [EVIDENCE: 2026-05-12-tool-stack]

## Open questions

What is the real budget once the second option is included? [ESTIMATE]
```

**Rules the brief holds itself to:**

1. **The model never computes anything.** Scores, totals, rankings and similarity
   numbers are calculated in code and spliced into the output. The LLM only writes
   prose, and the table is labeled `COMPUTED`.
2. **Every claim is labelled.** Each sentence ends with `[EVIDENCE: note-id]` or
   `[ESTIMATE]`. A claim that cites a note which was not retrieved is rejected.
3. **No probabilities, ever.** No percentages, no chances, no "likely". A brief
   describes what happened, not how likely something is.
4. **The counterargument argues against you.** It targets the option you are
   leaning toward, and it is not allowed to agree with you by default.
5. **Unverifiable claims are marked, not hidden.** If a claim fails validation even
   after a retry, it is marked `[UNVERIFIED]` rather than quietly dropped.

---

## The evaluation

`whatif eval` runs a leave-one-out test. For every decided note that has a failed
assumption:

1. the note is excluded from the index, so it cannot see its own outcome;
2. its outcome is hidden, and it is queried exactly as a brief would query it;
3. the failed assumptions surfaced from the *other* notes are compared against the
   hidden note's true failed assumptions by cosine similarity;
4. a hit is counted when a surfaced assumption is close enough to one of the true ones.

The hit rate is reported honestly, even when it is low. Results are saved as JSON in
`.whatif/eval-results.json`, and the report names the embedding model that produced
the numbers.

```
Hit rate at threshold 0.60: 0/6 = 0%
Hit rate at other thresholds:
  0.40  3/6 = 50%
  0.50  1/6 = 17%
  0.60  0/6 = 0%
  0.70  0/6 = 0%
  0.80  0/6 = 0%
```

The seed data has ten notes, of which six are evaluable (two are still open, and two
decided notes have no failed assumptions). Six notes is too small for a percentage to
mean much — the seed set exists to prove the pipeline works, not to measure retrieval
quality.

### What the number means

A low hit rate is a finding, not a failure. In the seed data, the surfacing filter
compares a candidate failure against the whole new note, which dilutes the signal; a
better filter would compare the candidate failure against the new decision's
assumptions specifically. The evaluation is honest about that gap rather than hiding
it.

---

## Limitations

- **The hit rate depends on the embedder.** Ollama must be installed and running for
  a real number. Without Ollama, use the stand-in server in `scripts/fake_ollama.py`;
  it reports itself as `standin-hash-embed`, not `nomic-embed-text`, and every report
  carries that name.
- **The seed data is invented.** Ten plausible decisions written to exercise the
  pipeline, not a record of real decisions.
- **Only decided notes with failed assumptions are evaluable.** Open notes and
  decided notes without failed assumptions are skipped.
- **Embeddings are per-note, not per-chunk.** A long note produces one vector for its
  whole body, so less relevant paragraphs dilute the signal.
- **Failed assumptions are matched individually.** A past failure phrased completely
  differently from the new assumption will not be found, no matter how similar the
  two situations were.
- **The LLM sees only the retrieved notes.** It cannot use knowledge from outside
  the vault, by design, and it cannot notice that a retrieved note is irrelevant.
- **One note, one embedding model.** Changing the embedder invalidates the index.
  `whatif doctor` and the evaluation both warn when the index and the config disagree.
- **The web interface is single-user.** No authentication, no sessions, no concurrency
  control beyond one index run at a time. It binds to 127.0.0.1 only.

---

## Repository layout

```
whatif/                 the package
  config.py             TOML configuration, vault paths
  notes.py              parsing and strict validation
  embeddings.py         the only module that opens a network connection
  store.py              SQLite: notes, vectors, assumption vectors
  retrieve.py           cosine similarity, top-k, failed assumptions
  scoring.py            the weighted comparison, computed in code
  brief.py              prompt construction and tag validation
  canvas.py             Obsidian Canvas JSON
  eval.py               leave-one-out evaluation
  cli.py                argparse commands
  webapp.py             local HTTP interface
  web/                  the interface: index.html, styles.css, app.js
tests/                  pytest suite
  fixtures/decisions/   10 seed decision notes
scripts/
  fake_ollama.py        stand-in Ollama server for offline work
  demo.py               one-command demo: server + index + web interface
  inspect_*.py          small tools for looking at the output
vault/                  a working vault with the seed notes
docs/                   screenshots and eval-results.json
```

---

## Development

```bash
python -m pytest tests/ -q     # 404 tests, no Ollama required
```

The suite never needs a model. It substitutes a deterministic fake embedder, and a
network guard test asserts that no module outside `whatif/embeddings.py` opens a
connection.

---

## Non-goals

No cloud APIs, no accounts, no outcome-probability model, no multi-user support, no
web UI beyond the local single-user console.
