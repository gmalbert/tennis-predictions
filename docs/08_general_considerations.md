# General Considerations

## 1. Deployment

### Streamlit Cloud (Recommended)

1. Push repo to GitHub.
2. Go to https://share.streamlit.io → **New App**.
3. Set **Main file path** to `predictions.py`.
4. Add secrets in the Streamlit Cloud dashboard (e.g., `ODDS_API_KEY`).
5. On first run, `setup_data.py` will clone the ATP data automatically.

**Requirements for Streamlit Cloud:**
- `requirements.txt` at repo root
- `predictions.py` as the entry point
- Git must be available (it is on Streamlit Cloud)
- Max 1 GB RAM on free tier
- Auto-sleeps after inactivity

### Docker (Alternative)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download ATP data at build time
RUN python setup_data.py

EXPOSE 8501

CMD ["streamlit", "run", "predictions.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Railway / Render

Both support Docker-based deployments. Add a `railway.json` or `render.yaml`:

```json
{
    "$schema": "https://railway.app/railway.schema.json",
    "build": { "builder": "DOCKERFILE" },
    "deploy": {
        "startCommand": "streamlit run predictions.py --server.port=$PORT --server.address=0.0.0.0"
    }
}
```

---

## 2. Performance

### Caching Strategy

| Level | Mechanism | TTL | What |
|---|---|---|---|
| Model & ELO | `@st.cache_resource` | App lifetime | Heavy .pkl load, done once |
| Live matches | `@st.cache_data(ttl=120)` | 2 min | Flashscore / ESPN requests |
| Odds | Internal `dict` cache | 5 min | The Odds API responses |
| Player lists | Module-level global | App lifetime | Sorted name list for dropdowns |

### Bottlenecks & Mitigations

| Bottleneck | Impact | Mitigation |
|---|---|---|
| Model training | 5–15 min on full dataset | Train offline; deploy .pkl |
| Flashscore request | ~1 s per fetch | 2-min cache |
| Feature engineering loop | ~2 min for 200K+ matches | Only run during training |
| Player name matching | O(n) substring fallback | Pre-built O(1) lookup dicts |

---

## 3. Data Freshness

| Data | Update Frequency | Method |
|---|---|---|
| ATP match results | Weekly | `git pull` in `tennis_atp/` |
| ELO ratings | Weekly | Retrain model after data update |
| Live schedule | Every 2 min | Flashscore cached fetch |
| Betting odds | Every 5 min | Odds API cached fetch |

### Weekly Retrain Script

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Pulling latest ATP data ==="
cd tennis_atp && git pull origin master && cd ..

echo "=== Retraining model ==="
python tennis_predictor.py

echo "=== Done $(date) ==="
```

Cron: `0 3 * * 1 /path/to/update_data.sh >> /var/log/tennis-retrain.log 2>&1`

---

## 4. Model Accuracy Monitoring

Track prediction accuracy over time to detect model drift:

- **Prediction Backlog page** shows daily accuracy by confidence tier
- Log predictions + outcomes to CSV/SQLite for long-term analysis
- Alert if rolling 7-day accuracy drops below 60%

```python
import csv
from datetime import datetime


def log_prediction(match, predicted_winner, confidence, actual_winner, filepath="prediction_log.csv"):
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            match.player1_name,
            match.player2_name,
            match.tournament,
            match.surface,
            predicted_winner,
            confidence,
            actual_winner,
            predicted_winner == actual_winner,
        ])
```

---

## 5. Legal & Ethical Considerations

### Data Licensing

| Source | License | Notes |
|---|---|---|
| Jeff Sackmann `tennis_atp` | CC BY-NC-SA 4.0 | Non-commercial only; must attribute |
| Flashscore | No official API; internal feed | Use responsibly; cache aggressively; respect ToS |
| ESPN API | Unofficial public endpoint | No guaranteed availability |
| The Odds API | Commercial API with free tier | Respect rate limits |

### Responsible Gambling

- **Always display a disclaimer** on betting-related pages
- Never present predictions as guaranteed outcomes
- Include language like: *"For informational and entertainment purposes only. Past performance does not guarantee future results."*
- Consider adding links to responsible gambling resources

### Privacy

- No user accounts or personal data are collected in this design
- If you add user features later, comply with GDPR / applicable privacy laws

---

## 6. Future Enhancements

| Enhancement | Difficulty | Impact |
|---|---|---|
| WTA data integration | Low | Double the coverage |
| Serve stats as features | Medium | +1-2% accuracy potential |
| Fatigue/scheduling features | Medium | Better predictions for back-to-back matches |
| Real-time ELO updates | Medium | Update ratings as matches finish (via Flashscore) |
| User accounts + bet tracking | High | Personalized experience |
| Ensemble model (ELO + XGBoost weighted) | Medium | More robust predictions |
| SMS/email alerts for high-value bets | Medium | User engagement |
| Historical accuracy dashboard | Low | Long-term performance tracking |
| Mobile-responsive CSS | Low | Better mobile experience |

---

## 7. Testing

### Unit Tests

```python
# tests/test_elo.py
def test_elo_initial_rating():
    elo = EloRatingSystem()
    assert elo.get_rating(999) == 1500

def test_elo_winner_gains():
    elo = EloRatingSystem()
    elo.update_ratings(1, 2, "Hard", "2025-01-01")
    assert elo.get_rating(1) > 1500
    assert elo.get_rating(2) < 1500

def test_elo_expected_score_equal():
    elo = EloRatingSystem()
    assert abs(elo.expected_score(1500, 1500) - 0.5) < 0.001
```

### Integration Tests

```python
# tests/test_schedule.py
def test_flashscore_returns_list():
    matches = get_todays_matches()
    assert isinstance(matches, list)

def test_player_matching():
    names = {1: "Jannik Sinner", 2: "Carlos Alcaraz"}
    assert match_player_to_database("Sinner J.", names) == 1
    assert match_player_to_database("Alcaraz C.", names) == 2
```

---

## 8. `.gitignore`

```gitignore
# Data (cloned at runtime)
tennis_atp/

# Models (generated by training)
*.pkl
models/

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Secrets
.streamlit/secrets.toml
```
