<!--
Assumption: the volatility forecasting series currently lives in sibling post
directories `volatility-forecasts-1` through `volatility-forecasts-9` rather
than under a shared parent folder. This directory exists only to hold shared
workflow instructions for the series and is not intended to be rendered as a
published post.
-->

# Volatility Forecast Series Workflow

## Purpose

Use this area for shared instructions that govern the full volatility forecasting series inside `steveya.github.io`.

## Scope

This file is the local runbook for the volatility forecasting series workflow. It complements the repo-level blog instructions and should be used as the local reference when editing volatility-series drafts or planning new volatility-series posts.

The canonical workflow log for this series is `codex-workflows/volatility-forecasts-log.md`.

## Audience

- Quant readers who expect explicit model definitions and careful comparisons
- Trading-oriented readers who care about forecast usefulness, calibration, and robustness
- Research readers who expect methodological honesty, baselines, and diagnostic evidence

## Series Ladder

- `volatility-forecasts-1`: STES baseline
- `volatility-forecasts-2` and `volatility-forecasts-3`: XGBSTES development
- `volatility-forecasts-4`: benchmark STES and XGBSTES against `GARCH(1,1)` and frame the series around QLIKE-aware evaluation
- `volatility-forecasts-5`: develop `PGARCH-L` and `XGB-g-PGARCH`
- `volatility-forecasts-6`: test PGARCH-family models against an expanded feature set
- `volatility-forecasts-7` through `volatility-forecasts-9`: prior-iteration downstream branches including feature expansion, architecture generalization, and VolGRU work

Important note:

- The later draft directories currently include some part-label drift. Treat them as downstream branches in the same series even where draft titles do not align perfectly with folder numbering.
- Use the current frontier, the canonical log, and the owning code paths to determine continuity rather than relying on numbering alone.

## What Counts as Incremental Improvement

Incremental improvement in this series means continuity with the full series frontier, not just a local win for one model family.

- stronger out-of-sample evidence against the current frontier
- better positioning against `GARCH(1,1)` and the relevant current family baseline
- cleaner diagnostics, calibration, or significance evidence
- a more interpretable structural extension that survives validation
- a useful negative result that changes the next branch of the series
- clearer reproducibility and cleaner code-to-narrative linkage

## What to do

- Write for readers who can follow statistical and modeling detail but still need the argument organized cleanly.
- Locate the current series rung, current family baseline, and current frontier before editing.
- Compare serious candidates against `GARCH(1,1)` at a minimum.
- When relevant, also compare against the active family baseline such as `STES`, `XGBSTES`, `PGARCH-L`, `XGB-g-PGARCH`, `XGBPGARCHModel`, or a `VolGRU` variant.
- Keep the code-to-narrative link explicit. Name the owning module, test, example script, or notebook path in `volatility-forecast` when describing a result.
- Update `codex-workflows/volatility-forecasts-log.md` whenever a branch materially advances, matches, or gets pruned.
- Treat PGARCH as one important phase in the series, not as the top-level framing of the workflow.
- Anchor every major claim in evidence:
  - metric tables
  - diagnostics
  - plots with interpretable captions
  - clear train/validation/test protocol
- Update the working notebook or report alongside the narrative when results materially change.

## Writeup Structure

- Start with the research question and why the next model generalization is warranted.
- State the series rung, baseline family, and candidate clearly before showing results.
- Describe the data window, target variable, split protocol, and loss function before interpreting metric tables.
- Present results in a stable order:
  - baseline anchors
  - candidate variants
  - diagnostics and significance checks
  - interpretation
  - conclusion and next step
- End by stating whether the new candidate is better, tied, inconclusive, or pruned.

## Claims and Evidence

- Do not make dominance claims from a single favorable metric if calibration or robustness moves the other way.
- If a candidate wins on RMSE but loses on QLIKE or calibration, say that explicitly.
- If a result is only directional and not statistically convincing, say that explicitly.
- If a run was incomplete, unstable, or skipped for cost or runtime reasons, say that explicitly.
- If a result matters because it reroutes the series rather than because it "wins," say that explicitly.

## Notebook and Report Expectations

- Keep draft notebooks in the post-local draft workflow used by this repo.
- Make notebook cells readable enough that the final post can be reproduced or refreshed later.
- Keep paths to sibling-repo code explicit and easy to update.
- Prefer saving lightweight rendered artifacts, tables, and summaries over copying large code blocks into the notebook.

## Failed Experiment Reporting

- Summarize failed experiments honestly.
- Say what changed, what benchmark held up, and what signal was missing.
- Distinguish between:
  - no improvement
  - unstable improvement
  - overfit improvement
  - implementation blocked or incomplete
- A failed branch should still leave a useful lesson for the next iteration.

## Workflow Expectations

- For STES and XGBSTES work, maintain continuity with `volatility-forecasts-1` through `volatility-forecasts-4`.
- For PGARCH-family work, maintain continuity with `volatility-forecasts-5` and `volatility-forecasts-6`.
- For VolGRU or broader structural relaxations, maintain continuity with the later downstream branches in `volatility-forecasts-7` through `volatility-forecasts-9`.
- If the next step crosses families, state explicitly why the series is moving from one family to another.

## What not to do

- Do not present a serious candidate without a `GARCH(1,1)` baseline.
- Do not overstate weak improvements or ignore contradictory diagnostics.
- Do not paste canonical source code from `volatility-forecast` into the blog repo.
- Do not hide negative results if they are part of the series logic.
- Do not present PGARCH as the whole series when it is only one phase of it.

## Validation

- Re-check that all reported metrics and figure labels match the current notebook outputs.
- Confirm that post text, table labels, and figure captions use the same model names and loss definitions.
- Ensure links or references back to `volatility-forecast` point to real paths.
- Confirm that the narrative correctly places the work inside the current series rung and does not ignore later frontier material.
- If a rendered post exists, preview it before treating the update as final.

## Deliverables

- Updated narrative or draft notebook for the volatility series.
- Clear comparison against `GARCH(1,1)` and the relevant current family baseline.
- Honest summary of failed directions when they informed the decision.
- Updated `codex-workflows/volatility-forecasts-log.md`.
- Explicit mapping from the blog narrative back to the owning code path in `volatility-forecast`.
