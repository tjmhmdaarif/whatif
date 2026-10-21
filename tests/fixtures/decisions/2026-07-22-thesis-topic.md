---
id: 2026-07-22-thesis-topic
type: decision
status: open
created: 2026-07-22
domain: study
options: ["Topic A - retrieval quality", "Topic B - deployment cost", "Topic C - hybrid approach"]
chosen:
criteria:
  - {name: research_interest, weight: 3}
  - {name: feasibility, weight: 3}
  - {name: data_availability, weight: 3}
  - {name: career_relevance, weight: 2}
scores:
  "Topic A - retrieval quality": {research_interest: 5, feasibility: 3, data_availability: 4, career_relevance: 4}
  "Topic B - deployment cost": {research_interest: 2, feasibility: 4, data_availability: 3, career_relevance: 3}
  "Topic C - hybrid approach": {research_interest: 4, feasibility: 2, data_availability: 3, career_relevance: 4}
assumptions:
  - The course dataset will still be publicly available next spring
  - My advisor will approve a hybrid scope rather than forcing a single focus
  - Evaluation labels will be ready by the time I need them
risks:
  - Topic C is likely to be too broad for one thesis
  - Advisor availability is limited over the summer
tags: [study, research]
---
## Context

I need to pick a thesis topic before registering for the spring research module. Three
directions are on the table, all around retrieval-augmented generation. I have not
committed to an advisor yet, and the dataset situation is still unclear.

## Options

**Topic A: retrieval quality**
Focus on ranking and chunking quality. I find this the most interesting problem and
the benchmark data is easiest to get.

**Topic B: deployment cost**
Focus on inference cost of the retrieval stack. More applied, less novelty, but the
lab already has the infrastructure.

**Topic C: hybrid approach**
Combine both. Most interesting on paper, least defined.

## Decision

Still deciding. Leaning toward Topic C if my advisor will approve the scope, since it
combines the parts I care about most.

## Outcome

To be filled in. I want to revisit this once the dataset question is settled.
