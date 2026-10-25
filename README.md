# Sports Prop Lab — MLB + NFL

One Streamlit app with all five requested markets:
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

Sportsbook lines/prices, player availability, active roster confirmation, injuries, workload/role changes, game-script/weather scenarios, and bet settlement. No bets are placed. Team labels use the player's latest prior stat row, so a recently traded or inactive player may need manual checking. NFL feeds can lag and undergo stat corrections. Missing player rows are never manufactured as zero-stat games.

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

## Version 4: NFL pre-game games
Open **NFL games**. The next published unplayed week loads automatically, including postseason. Choose a matchup and enter your sportsbook home spread (negative = favored), total and odds. Existing props remain available.

Outputs: 30,000 approximate score simulations, projected scores, win probabilities, unresolved ties, model home spread and total, 10th–90th percentile outcome ranges, cover/over/under/push probabilities, EV and CSV export. Ranges describe outcomes, not confidence in model accuracy. Known kickoff times are filtered; verify unknown kickoff times manually.

Model: up to 20 prior games per team within two years; eight-game recency half-life; six-game shrinkage toward league scoring; descriptive venue advantage and neutral-site handling. Paired empirical score deviations supply variability. Only completed games before the matchup date are used. Scores are rounded nonnegative integers with impossible one-point scores removed. This is not a drive-level simulation, does not resolve overtime, and ties remain unresolved. Win estimates are not settlement-ready moneyline probabilities. No strength-of-schedule adjustment or automatic personnel/weather feed. Manual scoring adjustments are scenarios. Model probabilities are uncalibrated and profitability is unproven.

Install: replace app.py, requirements.txt and README.md in your repository. Keep Streamlit on main → app.py and reboot. No new modules or API keys required. This package does not update your hosted app automatically.

Schedule documentation: https://nflreadr.nflverse.com/reference/load_schedules.html
