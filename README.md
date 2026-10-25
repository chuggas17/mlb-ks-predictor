# Sports Prop Lab — MLB + NFL

One Streamlit app with NFL game projections and all five requested prop markets:
- MLB pitcher strikeouts
- NFL rushing attempts
- NFL rushing yards
- NFL receptions
- NFL receiving yards

## Update your existing app from iPhone

1. Download and unzip `sports-prop-lab-complete.zip` in Files.
2. In `chuggas17/mlb-ks-predictor`, upload **app.py**, **requirements.txt**, and **README.md** into the repository root, replacing the old versions.
3. Commit the changes. You do not need to upload the tests to run the app.
4. Keep Streamlit's branch as **main** and entrypoint as **app.py**. Reboot the existing app if the update does not appear.
5. Open the sidebar (top-left on mobile) to switch between NFL, MLB, comparisons, and the tracker.

Old `model.py` and `mlb_api.py` files may remain. This edition does not import them. No API key, sportsbook login, or second app is required.

## Included

- NFL game/week/team/player selectors, with search across teams for trades.
- Live public NFL schedules and prior weekly player data; MLB schedules/probable pitchers and earlier start logs.
- Opponent-by-position NFL comparisons with a conservative, optional adjustment.
- Recent-game averages, over/under historical hit rates, game-log tables and charts.
- 30,000-draw simulations, mean/median, 10th–90th percentile ranges, and outcome distributions.
- Manual workload and yardage-efficiency scenarios.
- American odds, fair odds, no-vig comparison, expected value, and integer-line push handling.
- Alternate-line probabilities and “at least” thresholds using the same simulation draws.
- Saved comparison board, CSV exports, manual bet tracker and bankroll/exposure calculations.
- JSON tracker backup and restore with duplicate-ID merging.
- NFL walk-forward historical diagnostic using earlier games only.
- Manual NFL CSV upload and a blank template if a data feed is unavailable.
- Feed refresh control, missing-data errors and stale-history warnings.

## Run on a computer

Python **3.10 or newer**:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Tests:

```sh
python -m unittest discover -s tests -v
```

## What is still manual

Sportsbook lines/prices, final active/inactive and starting-QB confirmation, replacement quality, unusual role changes, and bet settlement remain manual. NFL game projections now load weekly injury reports, prior snap participation and kickoff weather automatically when those feeds are available. Player props check current-week injury status and block players listed Out, Injured Reserve or Suspended. No bets are placed. Team labels use the player's latest prior stat row, so a recently traded or inactive player may need manual checking. NFL feeds can lag and undergo stat corrections. Missing player rows are never manufactured as zero-stat games.

The app does not automatically connect to Bet365, scrape odds, claim calibrated probabilities, calculate parlay correlation, or provide guaranteed income. These models are transparent experimental baselines. A favorable estimated EV is not proof of a betting edge.

## Data and calculations

NFL: [nflverse weekly player statistics](https://nflreadr.nflverse.com/reference/load_player_stats.html), via the `stats_player` release's `stats_player_week_YEAR.csv`; [nflverse schedules](https://github.com/nflverse/nfldata/blob/master/data/games.csv). [Update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

MLB: public [MLB Stats API](https://statsapi.mlb.com/api/v1). Source availability, licensing and upstream schema changes are outside this app's control.

NFL counts use weighted Poisson/negative-binomial workload. Receptions use simulated targets with a beta catch-rate posterior. Yardage uses simulated opportunities and shrunk game-level efficiency, with opportunity-scaled noise. NFL opponent factors compare per-game production allowed to a position with the league baseline, shrink with ten league-average games, and cap at ±20%; they are not adjusted for opponent strength.

MLB uses recency-weighted workload and a beta-binomial rate with a fixed illustrative prior plus opponent K-rate adjustment. Workload scenarios affect batters faced separately from historical strikeout rate.

The historical diagnostic does not include current role overrides, opponent factors or historical prices. It tests a fixed user-selected line; it is not an earnings backtest and is subject to selection bias and later source corrections. Five prior games are required. Data on or after the target date is excluded, including same-day earlier games.

## Storage

Predictions, boards and bets are kept in Streamlit **session state only**. Export before closing, refreshing the session or rebooting. The tracker JSON backup can restore bets; the comparison board is included for reference but is not imported. CSV exports escape formula-like text for safer spreadsheet opening. No personal or sportsbook account data is needed.

## Package layout

- `app.py`: complete application, data adapters, models and UI in one file.
- `requirements.txt`: all dependencies.
- `README.md`: setup, features and limitations.
- `tests/test_models.py`, `tests/test_ui.py`: automated model, data-filtering, pricing, UI and tracker checks.
- `VALIDATION.md`: verification performed for this version.

## Version 5: NFL pregame context

Open **NFL games** for the automatically selected upcoming slate. Select a matchup to fetch current-week injuries, historical snap participation and kickoff weather. Every source failure or uncertain location/roof is disclosed. Missing data is not substituted with last week's report or a guessed home-city forecast.

### Injuries and lineup uncertainty
- Current-week injury designations and practice status from nflverse; source URL, retrieval time and publication time when supplied.
- Reports more than 72 hours old (by supplied publication timestamp) are rejected. Missing timestamps are flagged, not represented as fresh verification.
- Reports are applied only to target dates from today through the next eight days; historical games never receive current injury information.
- Offense, defense, QB and specialist effects are scaled by weighted prior snap share across up to 20 covered games. Persistent absences reduce historical role weight; unmatched players get no invented starter role and are flagged for manual review.
- Editable absence assumptions: Out 1.0, Doubtful 0.75, Questionable 0.25. These are explicit scenario assumptions, not medical or empirically calibrated participation probabilities. Practice-only reports do not imply that a player is active.
- Full-snap illustrative impacts: QB 6 points; WR 1.3; RB 0.8; TE 0.7; tackle 0.9; other offensive line 0.6–0.7; defensive positions 0.6–0.9; specialists 0.1–0.4. Historical snap share scales these values. Each unavailable offensive player reduces own scoring; defensive absence increases opposing scoring. Joint net shifts are capped at -12/+8 points per team. All assumptions can be edited or disabled.
- Injured/QB uncertainty is simulated with independent Bernoulli absence scenarios. Real availability, injury impacts and replacement quality can be correlated and are not learned by this model.
- Late scratches, official final inactive lists, suspensions/IR omissions, new starters and replacement quality still require confirmation. Current depth charts are not automatically inferred from previous playing time.
- NFL props are conditional on playing the specified role. A DNP is not modeled as a winning under. Out/IR/suspended players with matching current report entries are blocked; doubtful/questionable players are flagged. Automatic team-score injury/weather effects do not silently alter every player prop or transfer touches to backups.

### Weather, venue and rest
ESPN's event scoreboard identifies the actual venue and kickoff, including international and neutral games. Open-Meteo geocoding requires an unambiguous city/country/state match. The four-hour window beginning at kickoff uses hourly temperature, sustained wind, gusts and precipitation. It is a city-level forecast, not a stadium instrument reading. Missing, out-of-range or ambiguous forecasts produce a warning with zero automatic weather effect.

A closed roof/fixed dome disables weather effects. An unknown or retractable roof requires choosing open/closed before any weather effect is applied. The outdoor heuristic reduces each team's scoring by 0.06 points per mph above 12 mph, up to 1.2 points for precipitation (0.12 per mm), and 0.02 points per degree F below 20 F; total reduction capped at 3.5 points per team. Gusts are displayed for review but not double-counted. These are unvalidated scenario coefficients.

Rest uses days since each team's last completed game: 0.12 points per day above/below seven, capped at -0.5/+0.6 points. Gaps over 21 days do not receive a rest bonus. Travel distance, jet lag, coaching changes and game script do not receive invented automatic coefficients; use explicit manual scenarios when appropriate.

### Scoring model and outputs
The pregame baseline jointly fits team offense and opponent defense with ridge regression and a home/neutral venue term. This incorporates opponent strength; it uses completed games strictly before the target date from the prior two years, exponential 120-day recency weighting and regularization toward league-average scoring. Five prior games per team and 20 league games are required. Paired empirical fitted residuals provide outcome variation. Hyperparameters are fixed illustrative choices, not calibrated against sportsbook prices.

30,000 outcomes produce projected scores, winner probabilities, home spread, total, outcome ranges, cover/over/under/push probabilities and manual-price EV. Scores are approximate nonnegative integers; impossible one-point scores are removed. Overtime is not resolved, and tied draws remain explicitly unresolved. These win probabilities are not settlement-ready moneyline probabilities.

Baseline and context-adjusted results appear side by side, along with every applied point shift. Game simulation CSV and a JSON source/input snapshot can be exported. Results persist through ordinary widget reruns and are labeled with their original inputs and times; recalculate after changing settings. Refresh fetches new context, but does not retroactively rewrite a saved calculation.

### Sources and installation
- [nflverse injury reports](https://nflreadr.nflverse.com/reference/load_injuries.html)
- [nflverse snap counts](https://nflreadr.nflverse.com/reference/load_snap_counts.html)
- [nflverse schedules](https://nflreadr.nflverse.com/reference/load_schedules.html)
- [Official NFL injury reports for verification](https://www.nfl.com/injuries/)
- [ESPN NFL scoreboard](https://www.espn.com/nfl/scoreboard)
- [Open-Meteo forecast API and attribution](https://open-meteo.com/en/docs) — free endpoint is intended for non-commercial use; review provider terms before offering this app commercially.

No API keys or additional modules are required. Replace **app.py**, **requirements.txt**, **README.md** in your existing repository and reboot Streamlit. Keep **main → app.py**. This ZIP does not update the hosted app automatically.
