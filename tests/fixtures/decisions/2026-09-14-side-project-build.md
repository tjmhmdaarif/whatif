---
id: 2026-09-14-side-project-build
type: decision
status: open
created: 2026-09-14
domain: tools
options: [Build in Python, Build in TypeScript]
chosen:
criteria:
  - {name: learning_value, weight: 2}
  - {name: ecosystem, weight: 3}
  - {name: hiring_visibility, weight: 3}
  - {name: speed_to_first_demo, weight: 2}
scores:
  Build in Python: {learning_value: 4, ecosystem: 4, hiring_visibility: 2, speed_to_first_demo: 4}
  Build in TypeScript: {learning_value: 3, ecosystem: 5, hiring_visibility: 5, speed_to_first_demo: 3}
assumptions:
  - I will keep the scope small enough to ship in six weeks
  - The project will be visible enough to matter to recruiters
  - I will not abandon it during exam periods again
risks:
  - I have abandoned two side projects in the last year
  - Job postings ask for React more than any other framework
tags: [tools, career, learning]
---
## Context

I want one substantial side project to show rather than four half-finished demos. The
idea is a small tool for turning a decision journal into searchable briefs. I need to
pick a stack before I start, because my previous projects died from switching stacks
mid-build.

## Options

**Build in Python**
- Fast to write, I am fluent
- Good support for text and embeddings
- Less impressive in a frontend-heavy portfolio
- Can show a CLI, which is what I actually want to build

**Build in TypeScript**
- React and a web UI if I want one
- Heavier tooling and setup
- Slower to get to a working first version
- Matches more job postings

## Decision

Still open. Currently leaning Python because the CLI is the honest shape of this tool
and I want to finish something.

## Outcome

To be filled in. I want to commit before the next study plan cycle.
