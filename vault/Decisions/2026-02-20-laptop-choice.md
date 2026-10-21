---
id: 2026-02-20-laptop-choice
type: decision
status: decided
created: 2026-02-20
domain: tools
options: [Refurbished ThinkPad, New ultrabook]
chosen: New ultrabook
criteria:
  - {name: price, weight: 1}
  - {name: battery_life, weight: 5}
  - {name: repairability, weight: 2}
  - {name: performance, weight: 2}
scores:
  Refurbished ThinkPad: {price: 5, battery_life: 3, repairability: 5, performance: 3}
  New ultrabook: {price: 2, battery_life: 5, repairability: 1, performance: 5}
assumptions:
  - The student discount would still apply at checkout
  - Running local models on the machine would only be for occasional experiments
  - Battery life in the spec sheet reflects real coursework use
risks:
  - Soldered memory means no upgrades later
  - New laptops get heavier every generation
outcome:
  reviewed_on: 2026-10-01
  result: better
  failed_assumptions: []
tags: [hardware, tools, money]
---
## Context

My three-year-old ThinkPad has a failing battery and fans that spin up during every
lecture. I need a machine for coursework, writing, and occasionally running a small
local language model to test prompts offline.

Two candidates: a refurbished ThinkPad T14 for roughly 600 dollars with replaceable
parts, or a new 14-inch ultrabook for roughly 1,400 dollars with much better battery
life and integrated graphics.

## Options

**Refurbished ThinkPad T14**
- Around 600 dollars
- User-replaceable battery and SSD
- Older CPU, integrated graphics only
- Ships with a one-year warranty

**New 14-inch ultrabook**
- Around 1,400 dollars
- 12+ hour real battery life
- Light, thin, quiet
- Soldered memory and storage

## Decision

Chose the new ultrabook. Battery life carried the most weight for me since I use the
machine on campus all day, and the price gap was smaller than I expected once the
student discount applied.

## Outcome

Reviewed 2026-10-01. Result: better.

The discount did apply, which I had assumed might be gone. Battery life is genuinely
around 11 hours under normal use, and the machine is silent. Local model experiments
happen far more often than I planned, and the extra performance headroom is the main
reason this paid off faster than expected.

The assumption that local model work would be occasional was wrong, but in a good
direction, and the older machine could not have handled it. The repairability concern
did materialize: the memory is soldered and the battery replacement is not
user-friendly, so I expect to replace the whole machine rather than just the battery
in a few years. For a laptop I do not plan to keep more than three years, that tradeoff
was fine.
