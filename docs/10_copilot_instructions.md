# Copilot Coding Guidelines

This document explains the coding practices that GitHub Copilot (the AI
assistant) should follow when editing or adding code in this repository.
Human developers should also review and follow these conventions.

## General Rules

- **Search before you patch.** When you modify a pattern (e.g. `use_container_width`),
  look for all occurrences in the repository and update them consistently.
- **Respect existing formatting.** The codebase uses 4-space indentation, single
  quotes for strings, and blank lines around top-level sections.
- **Always run `python -m py_compile file.py`** after editing a Python file to
  catch syntax errors before committing.

## UI-specific guidelines

- **Use `width="stretch"` in `st.dataframe` calls.**
  `use_container_width` is deprecated; search for it and replace globally.
  A comment near the top of `predictions.py` reminds Copilot of this rule.
- **Cache data-loading functions** with `@st.cache_data` to avoid expensive repeats.
- **Avoid hard-coding port numbers** when launching Streamlit; use the default.

## Data pipeline guidelines

- **Only pull 2020+ matches for features.** Older years remain on disk for seeding
  but should not be loaded into `features.py` filters.
- **Maintain separation of concerns**: data acquisition (`update_tml_data.py`),
  enrichment (`enrich_with_odds.py`), and feature engineering (`features.py`) are
  independent scripts.
- **Use the GitHub Action** to run the daily update; do not add manual cron jobs.

## Caching and Secrets

- **Never commit API keys.** `ODDS_API_KEY` and `RAPIDAPI_KEY` belong in `.env`
  or `.streamlit/secrets.toml` (both are gitignored).
- **Cache responses under `cache/`** and add `cache/` to `.gitignore` if not
  already present.  Use descriptive subdirectories (e.g. `matchstat/`).

## Documentation

- **Always update `README.md`** when new user-visible features or scripts are added.
- **Add or modify `.md` files** in the `docs/` folder rather than editing the main
  README for deep technical details.

By following these instructions, Copilot will produce consistent, maintainable
code while avoiding regressions and styling drift.
