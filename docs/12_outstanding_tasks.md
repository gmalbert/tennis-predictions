# Outstanding Tasks & Documentation Audit

> Status: Completed. This audit document tracks what was reviewed and what remains to be aligned in the docs.
>
## Review Summary

Reviewed `docs/01_roadmap.md` through `docs/11_bzzoiro_api.md`.

The docs cover:
- roadmap, data acquisition, feature engineering, modeling, scheduling, odds integration, UI layout, general considerations, data strategy, Copilot guidance, and the Bzzoiro API.
- concrete pipeline guidance for daily data updates, feature rebuilds, and model training.
- implementation notes and recommended architecture.

## Individual file status

| File | Status | Notes |
|---|---|---|
| `01_roadmap.md` | Needs update | References `setup_data.py`, `tennis_predictor.py`, and schedule/odds scripts not present in repo. |
| `02_data_acquisition.md` | Completed | Describes data sources and update flow accurately. |
| `03_features.md` | Completed | Good feature and pipeline coverage; no obvious missing docs. |
| `04_modeling.md` | Completed | Model descriptions align with repo intentions, but script names may need alignment. |
| `05_scheduling.md` | Needs update | Good scheduler coverage, but mentions `tennis_schedule.py` which is absent. |
| `06_odds_integration.md` | Needs update | Strong concept, but actual implementation is in `bzzoiro_api.py` / `matchstat_api.py` not `odds_api.py`. |
| `07_layout.md` | Needs review | UI page list is useful, but should be verified against the actual Streamlit app. |
| `08_general_considerations.md` | Needs update | Deployment notes are solid, but file/script references like `setup_data.py` and `tennis_predictor.py` are outdated. |
| `09_data_strategy.md` | Completed | Effective source strategy and scraper status; may need minor alignment with current repo state. |
| `10_copilot_instructions.md` | Completed | Clear coding rules and repo-specific conventions. |
| `11_bzzoiro_api.md` | Completed | Strong integration guidance; matches available repo code. |

## Completed from the docs review

- Confirmed that the repository has the main data pipeline scripts: `update_tml_data.py`, `ingest_tennis_data_co_uk.py`, `enrich_with_odds.py`, `features.py`, `train.py`, `predict.py`, `predictions.py`, `bzzoiro_api.py`, and `matchstat_api.py`.
- Verified that the `features.py` bug fix was applied and compiled successfully.
- Confirmed that the docs are broadly aligned with the repository's purpose and architecture.
- Noted that the docs are missing a dedicated task tracker; this file fills that gap.

## Key outstanding tasks

### High priority

1. [ ] Resolve documentation/reality mismatch for missing script names.
   - `docs/01_roadmap.md` and `docs/08_general_considerations.md` refer to `setup_data.py`, but that file does not exist in the repository.
   - `docs/05_scheduling.md` refers to `tennis_schedule.py`, which is also not present.
   - `docs/06_odds_integration.md` refers to `odds_api.py`, but the repo has `bzzoiro_api.py` and `matchstat_api.py` instead.
   - `docs/04_modeling.md` and other docs mention `tennis_predictor.py`, which is missing.

2. [ ] Audit and/or implement the full Streamlit page set described in `docs/07_layout.md`.
   - The app should verify whether all 7 pages exist and work as described.
   - If pages are missing, add them or update docs to reflect the actual UI.

3. [ ] Shift scheduling/live-score source guidance to the current project stack.
   - Docs recommend Flashscore + ESPN + Matchstat, but the repo also includes the Bzzoiro integration.
   - Confirm that Flashscore parsing is not currently integrated into the live app path, despite the parser module existing.
   - Decide whether Bzzoiro should be documented as the primary live source and Matchstat as fallback.

### Medium priority

4. [ ] Add WTA support documentation and implementation if the roadmap intends it.
   - `docs/11_bzzoiro_api.md` highlights WTA as a major gap/opportunity.
   - The current repo appears focused on ATP; if WTA is intended, this should be explicit.

5. [ ] Align the data strategy docs with current file and source availability.
   - Confirm whether 2025 Sackmann data is still pending and document precisely how the repo handles it.
   - Clarify which scraped sources are actually in use versus experimental only.

6. [ ] Update `README.md` and docs index to reference this new outstanding-tasks tracking file.
   - Add a link to `docs/12_outstanding_tasks.md` from the README or docs landing page.

### Low priority

7. [ ] Add explicit test coverage documentation for core pipeline components.
   - Schedule parsing, odds matching, value-bet logic, and feature engineering are good candidates.

8. [ ] Clean up docs that reference deprecated or unimplemented features.
   - Remove or revise any “future enhancement” text that is no longer current.

## Notes

This document is intended as the canonical audit record for the repo’s docs-review status.
It should be updated as outstanding items are completed or reprioritized.
