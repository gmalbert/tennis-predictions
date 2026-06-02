# Tennis Predictions — Model Suggested Enhancements

## Priority 1: Match Prediction Model

### Surface-Specific Ensemble
- Train separate models for hard, clay, grass, and indoor hard surfaces (4 models).
- Blend with a meta-classifier that weights based on the match surface.
- Expected improvement: ~3% accuracy gain on clay and grass vs. a single all-surface model.

### Head-to-Head Encoding
- Add `h2h_win_rate` (on this surface only) and `h2h_last3_result` (streak in last 3 meetings).

### Tournament Stage Features
- Players in Grand Slam final weeks face cumulative fatigue. Add `days_of_consecutive_match_play` and `round_number_in_tournament`.

## Priority 2: Strokes Gained Equivalent

### Service Features
- `first_serve_pct_l10`, `first_serve_win_pct_l10`, `ace_rate_l10` from ATP/WTA stats.

### Return Features
- `return_win_pct_l10`, `break_pts_converted_l10`.

### Pressure Points
- `tiebreak_win_pct_l10`: Players who dominate tiebreaks outperform their overall win % on fast surfaces.

## Priority 3: Ranking Dynamics

### Ranking Momentum
- Add `rank_change_l52w` (52-week ranking change). A player rising 100 spots performs better than their current rank suggests.

### Ranking Mismatch
- When a player is ranked significantly higher but the xElo is close, flag as uncertain: confidence discount.

## Priority 4: Calibration

- Track prediction accuracy by surface and tournament tier.
- Apply isotonic regression to probability outputs.
- Surface the calibration curve on the Model Performance tab.
