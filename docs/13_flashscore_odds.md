# 13 — Flashscore Odds Integration

## Summary

`flashscore_odds.py` fetches **pre-match decimal odds** for all currently
scheduled ATP/WTA singles matches from Flashscore.  
It uses two plain HTTP endpoints — **no headless browser required**.

Discovered via systematic XHR interception with Playwright while investigating
the Flashscore website during April 2026.

---

## Endpoints

### 1. Live match feed

```
GET https://www.flashscore.com/x/feed/f_2_0_2_en-gb_1
Required header: x-fsign: SW9D1eZo
```

Returns all live + scheduled matches for all sports in a proprietary
pipe-delimited format (~600 KB for tennis on a busy match day).

**Response format** — sections separated by `~`, fields by `¬` (0xAC),
key/value by `÷` (0xF7):

```
ZA÷ATP - SINGLES: Monte Carlo (Monaco), clay¬…~
AA÷tIZL63YR¬AB÷1¬AE÷Bautista-Agut R.¬AF÷Berrettini M.¬WU÷bautista-agut-roberto¬PX÷riOJC5jb¬…~
```

**Key field codes for tennis singles:**

| Field | Meaning |
|-------|---------|
| `ZA`  | Tournament header — `"ATP - SINGLES: Monte Carlo (Monaco), clay"` |
| `AA`  | **Match ID** — used as `eventId` in the odds API |
| `AE`  | Player 1 name (HOME side) |
| `AF`  | Player 2 name (AWAY side) |
| `WU`  | Player 1 URL slug (e.g. `bautista-agut-roberto`) |
| `WV`  | Player 2 URL slug |
| `PX`  | Player 1 public URL ID (e.g. `riOJC5jb`) |
| `PY`  | Player 2 public URL ID |
| `AD`  | Match start timestamp (Unix UTC) |
| `AB`  | Status: `1` = live or scheduled, `3` = finished |
| `CA`/`CB` | WTA/ATP ranking |
| `GRA`/`GRB` | Current game scores (non-zero = match in progress) |
| `OA`/`OB` | Player photo filenames (not odds!) |
| `FH`/`FK` | Short player name display labels |
| `FJ`/`FL` | Doubles partner names (only in doubles sections) |

The feed also covers doubles, challengers, and other circuits. Filter for
sections where `ZA` contains both `"singles"` and a circuit name (`"atp"` /
`"wta"`).

### 2. Odds GraphQL API (persisted query `oce`)

```
GET https://global.ds.lsapp.eu/odds/pq_graphql
Params: _hash=oce, eventId=<AA>, projectId=2, geoIpCode=<ISO2>, geoIpSubdivisionCode=<ISO>
```

Returns `findOddsByEventId` JSON with all available bet types for the match.

**Query parameters:**

| Param | Required | Value | Notes |
|-------|----------|-------|-------|
| `_hash` | Yes | `oce` | Persisted query identifier |
| `eventId` | Yes | `AA` feed value | 8-char alphanumeric match ID |
| `projectId` | Yes | `2` | Flashscore project ID |
| `geoIpCode` | Yes | `US`, `GB`, `DE`, … | Determines which bookmakers are shown |
| `geoIpSubdivisionCode` | No | `USNH`, … | Required for some regions (e.g. US states) |

**Bookmakers shown by geoIp** (examples):
- `US` + `USNH` → bet365.us, BetMGM.us, FanDuel
- `GB` → Bet365, William Hill, Betfair, Unibet (more coverage)
- `DE` → bwin, Unibet, bet365.de

**Response structure:**

```json
{
  "data": {
    "findOddsByEventId": {
      "eventId": "KpvU6IJj",
      "settings": {
        "bookmakers": [
          { "bookmaker": { "id": 549, "name": "bet365.us" } }
        ]
      },
      "odds": [
        {
          "bookmakerId": 549,
          "bettingType": "HOME_AWAY",
          "bettingScope": "FULL_TIME",
          "odds": [
            { "eventParticipantId": "UcOyEyUN", "value": "3.50", "opening": "5.50" },
            { "eventParticipantId": "0tF78Fio", "value": "1.30", "opening": "1.14" }
          ]
        }
      ]
    }
  }
}
```

**Bet-type filtering for match winner odds:**
```python
bettingType == "HOME_AWAY" and bettingScope == "FULL_TIME"
```

Other bet types available in the response: `OVER_UNDER`, `ASIAN_HANDICAP`,
`CORRECT_SCORE`; scopes: `FIRST_SET`, `FULL_TIME`.

**Player ordering:**
- `odds[0]` = HOME participant = `AE` (player 1 in the live feed)
- `odds[1]` = AWAY participant = `AF` (player 2 in the live feed)

This ordering is consistent across all bookmakers in the same response.

---

## Match page URL formula

For building direct Flashscore match URLs (e.g. to scrape deeper data or link
out to the match page):

```python
segs = sorted([f"{wu}-{px}", f"{wv}-{py}"])
url  = f"https://www.flashscore.com/match/tennis/{segs[0]}/{segs[1]}/"
```

Where `wu`/`wv` = player URL slugs and `px`/`py` = player URL IDs
(both from the live feed section).

The two player path segments are sorted **alphabetically**, not by home/away.

---

## Implementation (`flashscore_odds.py`)

### Quick usage

```python
from flashscore_odds import fetch_flashscore_odds

rows = fetch_flashscore_odds()
# Returns list[dict], one row per (match × bookmaker)

# Filter to a specific tournament
mc_rows = fetch_flashscore_odds(tourney_filter="monte carlo")

# European bookmakers (more coverage than US)
eu_rows = fetch_flashscore_odds(geo_code="GB", geo_sub="")
```

### Row schema

```python
{
    "event_id":       str,   # AA from live feed (e.g. "tIZL63YR")
    "p1":             str,   # player 1 name (HOME)
    "p2":             str,   # player 2 name (AWAY)
    "wu":             str,   # player 1 URL slug
    "wv":             str,   # player 2 URL slug
    "tourney":        str,   # full ZA field (e.g. "ATP - SINGLES: Monte Carlo (Monaco), clay")
    "surface":        str,   # e.g. "clay", "hard", "grass", "clay (indoor)"
    "ts_start":       int,   # Unix UTC timestamp of scheduled match start
    "bookmaker_id":   int,   # numeric Flashscore bookmaker ID
    "bookmaker_name": str,   # human name (e.g. "bet365.us")
    "p1_odds":        float, # current decimal odds for player 1
    "p2_odds":        float, # current decimal odds for player 2
    "p1_opening":     float, # opening line odds for player 1
    "p2_opening":     float, # opening line odds for player 2
}
```

### API reference

| Function | Description |
|----------|-------------|
| `fetch_flashscore_odds(...)` | Main entry point — returns all rows |
| `fetch_live_singles_matches(...)` | Feed parser only — returns match metadata |
| `fetch_match_odds(event_id, ...)` | Odds API call for one match |

---

## Limitations & notes

- **GeoIP and bookmakers**: The odds API returns different bookmakers depending
  on the `geoIpCode` parameter. US params (default) show US-licensed books.
  Change `GEO_IP_CODE` / `GEO_IP_SUB` at the top of the module for other regions.

- **Live matches**: When a match is in progress, `df_od_1_{AA}` (the internal
  feed variant) returns `"0"`. The `oce` GraphQL endpoint tested here still
  returns odds for in-progress matches during the day they were scheduled.

- **Availability window**: Odds typically appear 1–7 days before the match and
  disappear after the match concludes. Matches more than a few days out may not
  be in the live feed yet.

- **Rate limits**: No documented API rate limit was observed, but the module
  applies a 0.3 s delay between odds requests by default.

- **No authentication**: Neither endpoint requires an API key or cookies.
  The `x-fsign: SW9D1eZo` header is required on the live feed and is a
  static value embedded in the Flashscore frontend JavaScript.

- **`_hash=oce` stability**: The persisted query hash `oce` was confirmed
  valid in April 2026. If Flashscore updates their frontend, this hash may
  change. In that case, intercept the `global.ds.lsapp.eu/odds/pq_graphql`
  XHR from a Playwright session to find the new hash.

---

## Cross-repo reuse

Copy `flashscore_odds.py` into any project. The only dependency is `requests`.

```
pip install requests
```

The module has zero coupling to the rest of this codebase — no imports from
other local files, no framework-specific code.

---

## Discovery method

The endpoints were found by:

1. Fetching the live feed (`f_2_0_2_en-gb_1`) via direct `requests` to map
   all feed field codes.
2. Loading a match page with Playwright, clicking the "Odds" tab, and
   intercepting **all** XHR/fetch responses (not just flashscore.com — also
   `*.lsapp.eu`).
3. Identifying `global.ds.lsapp.eu/odds/pq_graphql?_hash=oce` as the odds
   source, with `eventId` = the `AA` field from the live feed.
4. Confirming the endpoint works with plain `requests` (no session cookies,
   no browser required).
5. Mapping the JSON structure to isolate `bettingType=HOME_AWAY` +
   `bettingScope=FULL_TIME` as the correct match-winner filter.

Reference repos consulted during research:
- [M3MONs/FlashscoreScraper](https://github.com/M3MONs/FlashscoreScraper) —
  confirmed CSS selectors for the odds DOM fallback approach
- [gustavofariaa/FlashscoreScraping](https://github.com/gustavofariaa/FlashscoreScraping)

---

## Status

- [x] Live feed parsing (ATP/WTA singles filter)
- [x] Odds API — match winner decimal odds per bookmaker
- [x] Opening odds available alongside current odds
- [x] Works without Playwright (plain `requests`)
- [x] Configurable geoIp for regional bookmaker sets
- [ ] Integration with `predictions.py` as supplementary odds source
- [ ] Caching layer (TTL ≈ 60 s for odds, 30 s for live feed)
- [ ] Playwright DOM fallback if `oce` hash changes
