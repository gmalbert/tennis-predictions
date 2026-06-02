# Tennis Predictions — 6-Month Feature Roadmap

## Month 1: Tournament Hub

- **This week's draws** — All ATP/WTA tournaments with draws, seeds, and model win probabilities.
- **Surface badge** — Hard / Clay / Grass / Indoor indicator on every match card.
- **Seeding mismatch flag** — Highlight when the model strongly disagrees with the seeding.
- **Live draw tracker** — Update draw bracket as results come in.

## Month 2: Player Pages

- **Player profile** — Ranking history, surface splits, recent results, tournament history.
- **Surface specialist badge** — "Clay Court Specialist" / "Grass Specialist" based on surface win% delta.
- **H2H comparison tool** — Direct H2H record between any two players, filtered by surface.

## Month 3: Betting Tools

- **Value finder** — Filter matches with model edge > 3% vs. DraftKings moneyline.
- **Surface arbitrage** — Highlight players whose DraftKings price doesn't reflect their surface-specific edge.
- **Set handicap analysis** — Model probability of winning in straight sets vs. DraftKings −1.5 sets market.

## Month 4: Grand Slam Mode

- **Grand Slam draw simulator** — Monte Carlo simulation of each quarter of the bracket; champion probabilities.
- **Fatigue tracker** — Days of consecutive play and projected energy levels through the draw.
- **Historical champion profile** — What does a typical Grand Slam champion's stats look like coming in?

## Month 5: Rankings & Analytics

- **ATP/WTA ranking tables** — Integrated live rankings with model Elo comparison.
- **Rising star tracker** — Players who have improved their Elo 200+ points in the last 52 weeks.
- **Tour accuracy report** — Model accuracy by ATP vs. WTA and by tournament category.

## Month 6: Automation

- **Monday email** — Weekly value bets for the upcoming ATP/WTA tournaments.
- **Draw alert** — Email when a value-bet player faces a favourable draw path.
- **GitHub Actions** — Nightly `ingest_tennis_data_co_uk.py` and odds refresh.
