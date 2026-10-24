# MLB K Edge Lab — Complete Edition

This edition combines the MLB data-fetching functions, simulation model, and Streamlit UI into **one `app.py`**, so you won't get mismatched imports from different versions of `model.py` or `mlb_api.py`.

## Features
- MLB schedule and announced probable starters
- Prior pitcher start history (no future-game leakage)
- Opponent full-season and recent-30-game K-rate blending
- Recency-weighted pitcher workload and beta-binomial strikeout simulation
- Model over/under probability, 10th/90th percentiles, expected Ks and batters faced
- Manual American odds, break-even probability, estimated EV and example stake
- Alternate-line probability table and CSV export
- Mobile-friendly Streamlit interface

## Run locally
Requires Python 3.10 or newer.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit hosting
Use repository `chuggas17/mlb-ks-predictor`, branch `main`, entrypoint `app.py`.
The complete edition runs from one file and does not import the legacy
`model.py` or `mlb_api.py`. Existing deployments can keep the same configuration.

## Checks
```sh
python -m unittest discover -s tests -v
```
Tests use fixed data and mocked API responses; they do not validate live MLB
availability or the model's betting accuracy.

## Important limitations
This is an experimental, unvalidated model, not a proven betting edge. The 22% league strikeout baseline and 50% matchup adjustment are heuristic. It does not yet incorporate confirmed batting lineups, handedness, injuries, park factors, umpire or weather; nor does it offer automatic odds feeds or a statistically rigorous historical backtest. Data depends on MLB Stats API availability. CSV downloads are manual; no cloud persistence.
