---
id: 2026-10-01-certification-vs-project
type: decision
status: decided
created: 2026-10-01
domain: career
options: [Professional certification, Open-source contribution]
chosen: Professional certification
criteria:
  - {name: signal_to_recruiters, weight: 3}
  - {name: skill_growth, weight: 3}
  - {name: time_cost, weight: 2}
  - {name: portfolio_evidence, weight: 2}
scores:
  Professional certification: {signal_to_recruiters: 4, skill_growth: 2, time_cost: 4, portfolio_evidence: 1}
  Open-source contribution: {signal_to_recruiters: 2, skill_growth: 5, time_cost: 2, portfolio_evidence: 5}
assumptions:
  - A certificate would pass the resume screen on its own
  - The exam voucher would cover the full cost
  - Recruiters would ask about the project rather than the certificate
risks:
  - Certification has a half-life of a few years
  - Contributions require interview-level repository access
outcome:
  reviewed_on: 2026-10-20
  result: as-expected
  failed_assumptions:
    - A certificate would pass the resume screen on its own
tags: [career, learning]
---
## Context

I have two ways to spend the next two months and I can only do one. A well-known
vendor certification comes with an exam voucher and a recognizable name. Alternatively
I can contribute to a real open-source project I already use.

## Options

**Professional certification**
- About 120 hours of study plus a proctored exam
- Voucher covers the exam fee
- A recognized name on a resume
- Shows effort, not working software

**Open-source contribution**
- Irregular hours, roughly the same total
- Real code reviewed by maintainers
- Public history in a repository
- Uncertain timeline, depends on maintainers

## Decision

Chose the certification. The resume screen argument felt decisive and the voucher made
it cheap. I planned to use the momentum to build a project afterwards.

## Outcome

Reviewed 2026-10-20. Result: as-expected.

A certificate did not pass the resume screen on its own. In the three applications I
sent with it, I got one interview and it was from a recruiter at a company that hires
people with that specific certification, so it functioned as a filter rather than an
argument. In the other two, nobody mentioned it. The exam voucher did cover the full
cost, so that assumption held.

The third assumption was that recruiters would ask about the project rather than the
certificate, but since I did not do the project, there was nothing to ask about. The
skill growth was the real disappointment. I learned enough to pass an exam, which is
not the same as learning to build, and I have no artifact to point at. The half-life
concern is real too. If I had done the open-source route I would have had a public
commit history, which is harder to fake and does not expire.
