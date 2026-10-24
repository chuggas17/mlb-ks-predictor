from datetime import date
import numpy as np
import pandas as pd
import streamlit as st


# DATA AND PREDICTION FUNCTIONS (all bundled in one file)
"""Public MLB Stats API data fetching. No API key required."""
import requests

BASE = 'https://statsapi.mlb.com/api/v1'

@st.cache_data(ttl=1800, show_spinner=False)
def get_json(path: str, params: dict | None = None) -> dict:
    response = requests.get(f'{BASE}/{path}', params=params, timeout=18,
                            headers={'User-Agent': 'MLB-Ks-Research-Dashboard/1.0'})
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=1800, show_spinner=False)
def get_games(day: str) -> list[dict]:
    data = get_json('schedule', {'sportId': 1, 'date': day, 'hydrate': 'probablePitcher'})
    games = []
    for d in data.get('dates', []):
        for g in d.get('games', []):
            away = g['teams']['away']['team']['name']
            home = g['teams']['home']['team']['name']
            for side, opponent in [('away', home), ('home', away)]:
                pitcher = g['teams'][side].get('probablePitcher')
                if pitcher and pitcher.get('id'):
                    games.append({'game': f'{away} @ {home}', 'team': g['teams'][side]['team']['name'],
                                  'opponent': opponent, 'opponent_id': g['teams']['home' if side == 'away' else 'away']['team']['id'], 'pitcher': pitcher['fullName'],
                                  'pitcher_id': pitcher['id'], 'game_id': g['gamePk'],
                                  'status': g.get('status', {}).get('detailedState', 'Unknown')})
    return games

@st.cache_data(ttl=3600, show_spinner=False)
def get_pitcher_logs(pitcher_id: int, target_date: str) -> list[dict]:
    year = int(target_date[:4])
    games = []
    for season in (year - 1, year):
        data = get_json(f'people/{pitcher_id}/stats',
                        {'stats': 'gameLog', 'group': 'pitching', 'season': season, 'sportIds': 1})
        for group in data.get('stats', []):
            for entry in group.get('splits', []):
                game_date = entry.get('date', '')
                s = entry.get('stat', {})
                if not game_date or game_date >= target_date or not s.get('gamesStarted', 0):
                    continue
                bf = s.get('battersFaced')
                so = s.get('strikeOuts')
                try:
                    if bf is not None and so is not None:
                        games.append({'date': game_date, 'bf': int(bf), 'so': int(so)})
                except (ValueError, TypeError):
                    pass
    # Deduplicate if API returns multiple rows for the same game.
    unique = {g['date']: g for g in games}
    return sorted(unique.values(), key=lambda x: x['date'])


@st.cache_data(ttl=3600, show_spinner=False)
def get_opponent_k_rate(team_id: int, target_date: str) -> dict:
    """Completed team games strictly before target date, current season only."""
    year = int(target_date[:4])
    data = get_json(f'teams/{team_id}/stats', {
        'stats': 'gameLog', 'group': 'hitting', 'season': year, 'sportIds': 1})
    strikeouts = plate_appearances = 0
    games = 0
    recent_games = []
    for group in data.get('stats', []):
        for entry in group.get('splits', []):
            if not entry.get('date') or entry['date'] >= target_date:
                continue
            stat = entry.get('stat', {})
            try:
                k = int(stat['strikeOuts'])
                pa = int(stat['plateAppearances'])
            except (KeyError, ValueError, TypeError):
                continue
            if pa > 0 and 0 <= k <= pa:
                strikeouts += k
                plate_appearances += pa
                games += 1
                recent_games.append((entry.get('date', ''), k, pa))
    if games < 10 or plate_appearances < 300:
        raise ValueError('Opponent needs at least 10 prior games and 300 plate appearances this season')
    recent_games.sort(key=lambda x: x[0])
    last = recent_games[-30:]
    recent_k = sum(x[1] for x in last)
    recent_pa = sum(x[2] for x in last)
    season_rate = strikeouts / plate_appearances
    recent_rate = recent_k / recent_pa if recent_pa else season_rate
    # Blend recent 30-game rate with season baseline using 1000 PA of prior weight.
    # This is a heuristic, not a trained coefficient.
    blended = (recent_k + 1000 * season_rate) / (recent_pa + 1000)
    return {'k_rate': blended, 'season_k_rate': season_rate,
            'recent_k_rate': recent_rate, 'recent_games': len(last),
            'games': games, 'pa': plate_appearances}

"""Transparent baseline for pitcher strikeout props; not a validated betting edge."""


def american_to_decimal(odds: int) -> float:
    if odds == 0 or abs(odds) < 100:
        raise ValueError('American odds must be <= -100 or >= +100')
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def analyze_odds(probability: float, odds: int, stake: float) -> dict:
    dec = american_to_decimal(odds)
    if not 0 <= probability <= 1 or stake <= 0:
        raise ValueError('Invalid probability or stake')
    implied = 1 / dec
    ev = probability * (dec - 1) - (1 - probability)
    full_kelly = max(0, (probability * dec - 1) / (dec - 1))
    return {'implied': implied, 'edge': probability - implied, 'roi': ev,
            'expected_profit': stake * ev, 'profit_if_win': stake * (dec - 1),
            'quarter_kelly': full_kelly / 4}


def predict(games: list[dict], line: float, simulations: int = 30000, seed: int = 42, opponent_k_rate: float | None = None, league_k_rate: float = 0.22) -> dict:
    """Simulate BF and K using empirical workload + beta-binomial rate.

    Historical game logs must precede the target game. Workload is sampled from
    a recency-weighted empirical distribution; strikeout rate shrinks to 22%.
    This ignores lineup, umpire, injuries and pitch limits; use for research.
    """
    valid = [g for g in games if g['bf'] > 0 and 0 <= g['so'] <= g['bf']]
    if len(valid) < 5:
        raise ValueError('At least five prior starts with BF and SO are needed')
    valid = sorted(valid, key=lambda g: g['date'])
    # Last 20 starts, weighted toward recent starts (newest has highest weight).
    recent = valid[-20:]
    weights = np.linspace(0.5, 1.5, len(recent))
    weights = weights / weights.sum()
    bf = np.array([g['bf'] for g in recent], dtype=int)
    so = np.array([g['so'] for g in recent], dtype=int)
    # Shrink toward broad, illustrative 22% prior with 80 BF pseudo-sample.
    alpha = 0.22 * 80 + float(np.dot(weights, so)) * len(recent)
    beta = 0.78 * 80 + float(np.dot(weights, bf - so)) * len(recent)
    rng = np.random.default_rng(seed)
    workloads = rng.choice(bf, size=simulations, p=weights)
    rates = rng.beta(alpha, beta, size=simulations)
    # Log-odds matchup adjustment, shrunk 50% to limit overreaction.
    # League baseline is an illustrative constant, not live season-calibrated.
    if opponent_k_rate is not None:
        if not (0 < opponent_k_rate < 1 and 0 < league_k_rate < 1):
            raise ValueError('Invalid opponent or league K rate')
        from scipy.special import expit, logit
        rates = expit(logit(rates) + 0.5 * (logit(opponent_k_rate) - logit(league_k_rate)))
    strikeouts = rng.binomial(workloads, rates)
    over = float(np.mean(strikeouts > line))
    return {'over': over, 'under': float(np.mean(strikeouts < line)),
            'push': float(np.mean(strikeouts == line)),
            'mean': float(np.mean(strikeouts)), 'median': float(np.median(strikeouts)),
            'starts': len(valid), 'recent_starts': len(recent),
            'projected_bf': float(np.mean(workloads)),
            'k_rate': float(alpha / (alpha + beta)),
            'p10': float(np.quantile(strikeouts, 0.10)),
            'p90': float(np.quantile(strikeouts, 0.90)),
            'probability_mc_se': float(np.sqrt(over * (1 - over) / simulations))}

def main():
    st.set_page_config(page_title='MLB K Edge Lab', page_icon='⚾', layout='centered')
    st.title('⚾ MLB K Edge Lab — Complete Edition')
    st.caption('One-file edition · Free MLB data · Opponent season/recent K% · Simulations · Manual odds · Alternate lines · CSV export')
    st.warning('Experimental baseline, NOT a proven betting edge. Uses blended opponent season and recent-30-game K%, but not confirmed lineup, weather, injuries, or pitch limits. Do not treat estimated EV as reliable profit.')

    with st.sidebar:
        st.header('Game selection')
        game_day = st.date_input('Game date', value=date.today())
        st.caption('Select a future date with announced probable starters. During the offseason, choose a past date to explore historical predictions.')
        st.caption('Historical predictions use only earlier game logs, but past-day sportsbook odds are not provided.')

    day = game_day.isoformat()
    try:
        starters = get_games(day)
    except Exception as exc:
        st.error(f'Could not load MLB schedule: {exc}')
        st.stop()

    if not starters:
        st.info('No announced probable starters found for this date. Try another date or check back after starters are announced.')
        st.stop()

    labels = [f"{p['pitcher']} — {p['game']}" for p in starters]
    choice = st.selectbox('Select starting pitcher', range(len(starters)), format_func=lambda i: labels[i])
    p = starters[choice]
    st.caption(f"Opponent: {p['opponent']} · Game status: {p['status']}")

    c1, c2 = st.columns(2)
    with c1:
        line = st.number_input('Strikeout line', min_value=0.5, max_value=20.5, value=5.5, step=1.0)
    with c2:
        market = st.selectbox('Bet side', ['Over', 'Under'])
    c3, c4 = st.columns(2)
    with c3:
        odds = st.number_input('American odds (manual)', value=-115, step=5,
                               help='Enter Bet365 odds, e.g. -115 or +120')
    with c4:
        stake = st.number_input('Example stake ($)', min_value=0.01, value=10.0, step=1.0)

    if st.button('Calculate prediction', type='primary', use_container_width=True):
        if line % 1 != 0.5:
            st.error('Use a half-strikeout line such as 5.5 so the odds calculation has no push outcome.')
            st.stop()
        if -99 <= odds <= 99:
            st.error('Use American odds of -100 or lower, or +100 or higher.')
            st.stop()
        try:
            logs = get_pitcher_logs(p['pitcher_id'], day)
            try:
                opponent_stats = get_opponent_k_rate(p['opponent_id'], day)
                opponent_rate = opponent_stats['k_rate']
                adjustment_note = (f"{p['opponent']} K%: season {opponent_stats['season_k_rate']:.1%}; "
                                   f"last {opponent_stats['recent_games']} games {opponent_stats['recent_k_rate']:.1%}; "
                                   f"blended rate {opponent_rate:.1%} ({opponent_stats['pa']} season PA).")
            except (ValueError, KeyError, requests.RequestException) as exc:
                opponent_rate = None
                adjustment_note = f'Opponent adjustment unavailable ({exc}); using pitcher-only baseline.'
            result = predict(logs, line, opponent_k_rate=opponent_rate)
            prob = result['over'] if market == 'Over' else result['under']
            pricing = analyze_odds(prob, int(odds), float(stake))
        except ValueError as exc:
            st.warning(f'Insufficient data: {exc}')
            st.stop()
        except Exception as exc:
            st.error(f'Unable to calculate: {exc}')
            st.stop()

        st.subheader(f"{p['pitcher']} · {market} {line} Ks")
        a, b = st.columns(2)
        a.metric('Model probability', f'{prob:.1%}')
        b.metric('Sportsbook break-even', f"{pricing['implied']:.1%}")
        a.metric('Estimated edge', f"{pricing['edge']:+.1%}")
        b.metric('Estimated EV / $1', f"{pricing['roi']:+.1%}")

        st.subheader('Alternate strikeout lines (model estimates)')
        alt_rows = []
        for candidate in [2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]:
            alt_result = predict(logs, candidate, opponent_k_rate=opponent_rate)
            alt_rows.append({'Line':candidate, 'Over probability':f"{alt_result['over']:.1%}",
                             'Under probability':f"{alt_result['under']:.1%}"})
        st.dataframe(pd.DataFrame(alt_rows), hide_index=True, use_container_width=True)

        st.write(adjustment_note)
        st.write(f"**Expected strikeouts:** {result['mean']:.2f} · **Projected batters faced:** {result['projected_bf']:.1f}")
        st.write(f"**Simulated strikeout range (10th–90th percentile):** {result['p10']:.0f}–{result['p90']:.0f}")
        st.write(f"**Historical starts available:** {result['starts']} · **Starts used:** {result['recent_starts']}")
        st.write(f"**Expected P/L on ${stake:,.2f}:** ${pricing['expected_profit']:+.2f} (long-run mathematical estimate only)")
        if pricing['edge'] > 0:
            st.info('Model indicates positive EV, but the model is unvalidated and can be systematically wrong. Paper trade first.')
        else:
            st.info('No positive estimated edge at these odds.')
        if result['push'] > 0:
            st.caption(f"Push probability: {result['push']:.1%}. EV calculation above assumes a two-outcome half-strikeout market; use half-integer lines.")
        st.caption('Monte Carlo sampling error is not model uncertainty; the predicted edge can still be wrong.')
        st.caption('Method: recency-weighted workload resampling and beta-binomial K-rate simulation (30,000 draws). A fixed 22% league-style prior is illustrative, not season-calibrated.')
        st.caption('Opponent K% blends season and last 30 games, then applies a conservative log-odds factor, but confirmed lineup, ballpark, weather, pitch count, and availability are not modeled. Do not use it for real-money betting without independent validation.')
        record = pd.DataFrame([{'date': day, 'pitcher': p['pitcher'], 'opponent': p['opponent'],
                                 'market': market, 'line': line, 'odds': odds,
                                 'model_probability': round(prob, 5), 'estimated_roi': round(pricing['roi'], 5),
                                 'historical_starts': result['starts']}])
        st.download_button('Download this prediction (CSV)', data=record.to_csv(index=False),
                           file_name=f"mlb_ks_{day}_{p['pitcher_id']}.csv", mime='text/csv',
                           use_container_width=True)

    st.divider()
    st.subheader('Model limitations')
    st.markdown('''- This first version calculates **one pitcher/line at a time** and does not automatically scrape Bet365.
    - It does **not** persist prediction history server-side; use the CSV download to retain records.
    - Free Streamlit hosting can sleep when idle; predictions run when you open the dashboard.
    - Game logs are fetched for the selected year and preceding year. Predictions for older dates use only prior logs from those seasons.
    - This is a **baseline prototype**, not a calibrated, backtested production model.''')


if __name__ == "__main__":
    main()
