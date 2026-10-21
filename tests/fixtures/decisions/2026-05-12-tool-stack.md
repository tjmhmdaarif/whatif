---
id: 2026-05-12-tool-stack
type: decision
status: decided
created: 2026-05-12
domain: tools
options: [Obsidian vault, Notion workspace]
chosen: Obsidian vault
criteria:
  - {name: collaboration, weight: 3}
  - {name: data_ownership, weight: 3}
  - {name: setup_cost, weight: 2}
  - {name: offline_availability, weight: 2}
scores:
  Obsidian vault: {collaboration: 1, data_ownership: 5, setup_cost: 4, offline_availability: 5}
  Notion workspace: {collaboration: 5, data_ownership: 2, setup_cost: 4, offline_availability: 2}
assumptions:
  - The rest of the team would move to the new tool and keep using it
  - I would not need to maintain a second copy of shared documents
  - Offline access would not matter for shared work
risks:
  - My lab group is used to Notion already
  - Plain text files need conventions to stay organized
outcome:
  reviewed_on: 2026-09-20
  result: as-expected
  failed_assumptions:
    - The rest of the team would move to the new tool and keep using it
    - I would not need to maintain a second copy of shared documents
tags: [tools, collaboration, study]
---
## Context

Our lab group needs a shared knowledge base for project notes, meeting notes, and
literature summaries. Notion is the current team default. I want a local-first setup
with plain files I can grep and script, but I do not want to fragment the team's
workflow.

## Options

**Obsidian vault**
- Plain markdown files on disk
- Full offline access, scriptable with any tool
- No built-in real-time collaboration
- Requires folder and naming conventions I maintain

**Notion workspace**
- Shared team workspace with comments and databases
- Everyone already knows it
- Requires internet for most use
- Data lives in someone else's database

## Decision

Chose Obsidian for my personal notes and set up a shared folder for the group, keeping
Notion for the team-facing databases. I expected the group to move over.

## Outcome

Reviewed 2026-09-20. Result: as-expected.

The rest of the team did not move to the new tool and kept using it, so I now maintain
two copies of shared documents, exactly the thing I was trying to avoid. Two people
copied my folder structure by hand, which produced three slightly different versions
of the same reading list. Notion stayed the official shared space because the
collaboration features matter more for a lab group than grep does.

The tool itself performed as expected. Plain files are genuinely better for my own
notes, the offline access was useful during two campus outages, and being able to
script over the files paid off when I generated summaries. But I underestimated how
much of the value of a shared tool is that everyone is in the same place, and no
amount of local-first tooling substitutes for that. The compromise of running both was
the right call, but it is more friction than I designed for.
