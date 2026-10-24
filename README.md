# MLB K Edge Lab — iPhone-friendly Streamlit prototype

A working **starter application** that reads free public MLB Stats API game schedules, probable starters, and pitcher game logs, estimates strikeout probabilities, and evaluates manually entered American odds. It does **not** place bets or scrape sportsbooks.

## Publish entirely from your iPhone (GitHub + Streamlit Community Cloud)

1. Create a free account at https://github.com and sign in using Safari.
2. Create a **new repository** named `mlb-ks-predictor` (public, since the free Streamlit Community Cloud deployment workflow may require a public repo). Do not add secrets or betting account credentials.
3. Open this ZIP from the Files app and unzip it. In your new GitHub repository, choose **Add file → Upload files** (Safari's **Request Desktop Website** option may help). Upload the files from inside the unzipped `mlb-ks-predictor` folder, keeping `app.py`, `mlb_api.py`, `model.py`, and `requirements.txt` in the repository root. Commit the upload.
4. Open https://share.streamlit.io/ in Safari, sign in with GitHub, and choose **Create app / Deploy app**. Choose your repository, branch `main`, and entry point `app.py`.
5. Deploy. After dependencies install, copy the hosted `*.streamlit.app` URL and choose **Share → Add to Home Screen** on your iPhone.
6. Select a game date, pick a probable starter, enter a strikeout line, bet side, American odds, and stake. Tap **Calculate prediction**. Export each result as CSV if you want to keep it.

The Streamlit Cloud interface may change; look for the equivalent repository/branch/main-file deployment controls. The free hosted app may go to sleep after inactivity and restart on visit. The MLB API is public but not guaranteed always available.

## Local testing (optional, if you ever use a computer)

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m unittest discover -s tests -v
```

## Model

The model uses the latest 20 prior starts (minimum five) from the selected and previous season. It resamples historical batters faced with recency weights, estimates the pitcher's strikeout rate with an illustrative 22% prior and 80 pseudo-batters faced, and runs 30,000 beta-binomial simulations. `P(over line)` and `P(under line)` are compared with sportsbook break-even probabilities. Odds EV assumes a non-push half-integer line.

**Important:** This is a transparent baseline and has **not** been calibrated, verified against current API responses, or backtested. It does not account for confirmed opposing lineups, weather, park, injuries, bullpen, or pitch limits. Positive modeled EV does not imply actual profitability. Do not risk money based on this prototype.

## Next development milestones

1. Add a historical walk-forward evaluation and probability calibration report.
2. Add opponent projected lineup strikeout adjustment and workload projection from pitch counts.
3. Save every pregame prediction with timestamp to a persistent database.
4. Add a licensed sportsbook odds feed for automatic line comparisons.
5. Add alerting only after robust out-of-sample validation.

## Data sources

- MLB Stats API: https://statsapi.mlb.com/api/v1/schedule?sportId=1
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
- GitHub: https://github.com
