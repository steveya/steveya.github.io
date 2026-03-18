<!--
This directory holds shared workflow instructions for the mean-reversion series.
The actual series currently lives in sibling directories such as
`mean-reversion-strategy-1` through `mean-reversion-strategy-6`.
-->

# Mean-Reversion Strategy Workflow

## Purpose

Use this area for shared instructions governing the full mean-reversion strategy series.

## Scope

This file is the local runbook for series-level work on `mean-reversion-strategy-*`.

## What Counts as Progress

- stronger expected-performance modeling
- more realistic assumptions and trading frictions
- better uncertainty treatment
- clearer mapping from assumptions to strategy implications

## Evidence Expectations

- Keep assumptions explicit.
- Show how costs, frictions, or sampling choices affect the result.
- Be explicit about sensitivity and uncertainty rather than presenting a single point estimate as definitive.

## Workflow Expectations

- Treat the series as one workflow, not as isolated posts.
- Locate the current frontier before editing.
- Update `codex-workflows/mean-reversion-strategy-log.md` after a substantive branch.
- Keep post-local artifacts near the relevant post directory when execution artifacts are needed.

## Deliverables

- Updated narrative or draft notebook for the series
- Clear statement of what changed in the strategy-performance framework
- Updated `codex-workflows/mean-reversion-strategy-log.md`
