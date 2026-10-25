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
    if not np.isfinite(odds) or odds == 0 or abs(odds) < 100:
        raise ValueError('American odds must be <= -100 or >= +100')
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


# MULTI-SPORT WORKSPACE
from datetime import datetime, timedelta, timezone
from io import BytesIO
from zoneinfo import ZoneInfo
import json
import uuid

VERSION = '4.0.0'
MARKETS = {'Rushing attempts': 'carries', 'Rushing yards': 'rushing_yards',
           'Receptions': 'receptions', 'Receiving yards': 'receiving_yards'}
STATS_URL = 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv'
SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
STAT_COLS = ['carries', 'rushing_yards', 'targets', 'receptions', 'receiving_yards']


def local_today():
    return datetime.now(ZoneInfo('America/Chicago')).date()


@st.cache_data(ttl=1800, show_spinner=False)
def download_csv(url):
    response = requests.get(url, timeout=(10, 45), headers={'User-Agent': 'Sports-Prop-Lab/3.0'})
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content), low_memory=False)


def nfl_schedule():
    df = download_csv(SCHEDULE_URL).copy()
    needed = {'game_id', 'season', 'week', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score'}
    if not needed.issubset(df.columns):
        raise ValueError('NFL schedule schema changed. Use manual CSV mode while the feed is unavailable.')
    df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce').dt.normalize()
    return df.dropna(subset=['gameday'])


def normalize_nfl_stats(df, schedule):
    df = df.copy()
    if 'team' not in df and 'recent_team' in df:
        df = df.rename(columns={'recent_team': 'team'})
    if 'player_display_name' not in df and 'player_name' in df:
        df['player_display_name'] = df['player_name']
    required = {'player_id', 'player_display_name', 'position', 'team', 'season', 'week'} | set(STAT_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'NFL stats missing columns: {", ".join(sorted(missing))}')
    if 'game_id' not in df:
        long = pd.concat([schedule.assign(team=schedule[side]) for side in ['home_team', 'away_team']])
        df = df.merge(long[['game_id', 'season', 'week', 'team']], on=['season', 'week', 'team'], validate='many_to_one')
    meta = schedule[['game_id', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates('game_id')
    df = df.drop(columns=['gameday', 'home_team', 'away_team', 'home_score', 'away_score'], errors='ignore')
    df = df.merge(meta, on='game_id', how='inner', validate='many_to_one')
    # Require final scores and exclude all same-day and future rows at selection time.
    df = df.dropna(subset=['home_score', 'away_score', 'player_id'])
    df['opponent_team'] = np.where(df['team'] == df['home_team'], df['away_team'], df['home_team'])
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['player_id'] = df['player_id'].astype(str)
    df['player_display_name'] = df['player_display_name'].fillna(df['player_id'])
    return df.drop_duplicates(['game_id', 'player_id']).sort_values(['gameday', 'game_id'])


def load_nfl_data(season, schedule):
    frames, notes = [], []
    for year in [season - 1, season]:
        try:
            frames.append(normalize_nfl_stats(download_csv(STATS_URL.format(season=year)), schedule))
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            notes.append(f'{year} player stats unavailable: {exc}')
    if not frames:
        raise ValueError('No NFL player data could be loaded. Retry or use your own game-log CSV.')
    return pd.concat(frames, ignore_index=True).drop_duplicates(['player_id', 'game_id']), notes


def prior_nfl_rows(df, target_date):
    return df[df['gameday'] < pd.Timestamp(target_date)].sort_values(['gameday', 'game_id']).copy()


def clean_history(frame):
    required = {'date'} | set(STAT_COLS)
    if not required.issubset(frame.columns):
        raise ValueError('CSV needs: date, carries, rushing_yards, targets, receptions, receiving_yards.')
    df = frame.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['date'] + STAT_COLS].isna().any().any():
        raise ValueError('Dates and all five stat columns must be filled with valid values; missing is not zero.')
    count_cols = ['carries', 'targets', 'receptions']
    if (df[count_cols] < 0).any().any() or (df[count_cols] % 1 != 0).any().any():
        raise ValueError('Attempts, targets and receptions must be nonnegative whole numbers.')
    if (df['receptions'] > df['targets']).any():
        raise ValueError('Receptions cannot exceed targets.')
    if ((df.carries == 0) & (df.rushing_yards != 0)).any() or ((df.receptions == 0) & (df.receiving_yards != 0)).any():
        raise ValueError('Yards without a carry/reception require corrected game-log data.')
    if df['date'].duplicated().any():
        raise ValueError('Use one row per player game; duplicate dates found.')
    return df.sort_values('date').reset_index(drop=True)


def player_history(df, player_id, target_date):
    rows = prior_nfl_rows(df[df['player_id'] == player_id], target_date)
    return clean_history(rows.rename(columns={'gameday': 'date'})[['date'] + STAT_COLS])


def opponent_context(df, opponent, position, metric, target_date, season):
    rows = prior_nfl_rows(df, target_date)
    rows = rows[rows.season == season]
    # All offense games provide the denominator, including zero production at the selected position.
    offense = rows[rows.position.isin(['QB', 'RB', 'FB', 'WR', 'TE'])]
    games = offense[['game_id', 'opponent_team']].drop_duplicates()
    pos = ['RB', 'FB'] if position in ['RB', 'FB'] else [position]
    totals = offense[offense.position.isin(pos)].groupby(['game_id', 'opponent_team'])[metric].sum(min_count=1).reset_index()
    totals = games.merge(totals, on=['game_id', 'opponent_team'], how='left').fillna({metric: 0})
    selected = totals[totals.opponent_team == opponent]
    league = float(totals[metric].mean()) if len(totals) else 0
    if len(selected) < 3 or league <= 0:
        return {'factor': 1.0, 'games': len(selected), 'note': 'Insufficient current-season opponent history; neutral adjustment.'}
    allowed = float(selected[metric].mean())
    # Ten league-average games of shrinkage, then cap the illustrative adjustment.
    factor = float(np.clip((selected[metric].sum() + 10 * league) / ((len(selected) + 10) * league), 0.8, 1.2))
    return {'factor': factor, 'games': len(selected), 'allowed': allowed, 'league': league,
            'note': f'{opponent} allowed {allowed:.1f} {metric.replace("_", " ")} per game to {position}; league {league:.1f}, {len(selected)} games.'}


def count_samples(values, weights, multiplier, rng, n):
    mean = float(np.dot(weights, values)) * multiplier
    if mean <= 0:
        return np.zeros(n, dtype=int)
    variance = float(np.dot(weights, (values - np.dot(weights, values)) ** 2)) * multiplier ** 2
    if variance > mean + 1e-6:
        shape = mean * mean / (variance - mean)
        return rng.negative_binomial(shape, shape / (shape + mean), n)
    return rng.poisson(mean, n)


def simulate_nfl(history, metric, window=12, half_life=5., workload=1., efficiency=1., defense=1., simulations=30000, seed=42):
    if metric not in MARKETS.values():
        raise ValueError('Unsupported NFL market')
    if not (5 <= window <= 30 and half_life > 0 and 0 < workload <= 3 and 0 < efficiency <= 3 and 0.5 <= defense <= 1.5):
        raise ValueError('Invalid model settings')
    if not isinstance(simulations, int) or simulations < 100:
        raise ValueError('At least 100 simulations required')
    df = clean_history(history).tail(window)
    if len(df) < 5:
        raise ValueError('At least five valid prior game logs are required. Use a CSV for missing history.')
    weights = 0.5 ** (np.arange(len(df) - 1, -1, -1) / half_life)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    if metric in ['carries', 'rushing_yards']:
        count = count_samples(df.carries.to_numpy(), weights, workload * (defense if metric == 'carries' else 1), rng, simulations)
        opportunities = df.carries.to_numpy(float)
        yards = df.rushing_yards.to_numpy(float)
        volume_label = 'carries'
    else:
        targets = count_samples(df.targets.to_numpy(), weights, workload * (defense if metric == 'receptions' else 1), rng, simulations)
        # Beta posterior catch rate, with a weak 65% / 10-target illustrative prior.
        alpha = 6.5 + float(np.dot(weights, df.receptions)) * len(df)
        beta = 3.5 + float(np.dot(weights, df.targets - df.receptions)) * len(df)
        count = rng.binomial(targets, rng.beta(alpha, beta, simulations))
        opportunities = df.receptions.to_numpy(float)
        yards = df.receiving_yards.to_numpy(float)
        volume_label = 'receptions'
    if metric in ['carries', 'receptions']:
        draws = count
    else:
        mask = opportunities > 0
        if not mask.any():
            raise ValueError('No prior opportunities for this yardage market.')
        yw = weights[mask] / weights[mask].sum()
        average = float(np.dot(weights[mask], yards[mask]) / np.dot(weights[mask], opportunities[mask]))
        per_game = yards[mask] / opportunities[mask]
        # Shrink noisy small-workload game efficiencies toward pooled player efficiency.
        smoothed = (yards[mask] + 5 * average) / (opportunities[mask] + 5)
        sampled = rng.choice(smoothed, simulations, p=yw)
        spread = max(float(np.sqrt(np.dot(yw, (per_game - average) ** 2))), 1.0)
        # Game-level efficiency variation + opportunity-scaled noise; zero opportunity means zero yards.
        draws = np.rint(count * sampled * efficiency * defense + rng.normal(0, spread * np.sqrt(count), simulations)).astype(int)
        draws[count == 0] = 0
    return {'samples': draws, 'projected_volume': float(np.mean(count)), 'volume_label': volume_label,
            'games': len(df), 'effective_games': float(1 / np.sum(weights ** 2)),
            'last_game': df.date.max().date().isoformat(), 'history': df}


def summarize_samples(draws, line):
    if not np.isfinite(line):
        raise ValueError('Line must be finite')
    x = np.asarray(draws)
    return {'over': float(np.mean(x > line)), 'under': float(np.mean(x < line)),
            'push': float(np.mean(x == line)), 'mean': float(np.mean(x)),
            'median': float(np.median(x)), 'p10': float(np.quantile(x, .1)), 'p90': float(np.quantile(x, .9))}


def price_market(win, push, odds, stake):
    dec = american_to_decimal(odds)
    if not np.isfinite([win, push, stake]).all() or win < 0 or push < 0 or win + push > 1 + 1e-8 or stake < 0:
        raise ValueError('Invalid probability or stake')
    lose = max(0., 1 - win - push)
    ev = win * (dec - 1) - lose
    conditional = win / (1 - push) if push < 1 else None
    fair_dec = 1 / conditional if conditional and conditional < 1 else None
    fair = None if fair_dec is None else (100 * (fair_dec - 1) if fair_dec >= 2 else -100 / (fair_dec - 1))
    return {'roi': ev, 'expected_profit': ev * stake, 'break_even': 1 / dec,
            'conditional_win': conditional, 'edge': None if conditional is None else conditional - 1 / dec,
            'fair_odds': fair, 'profit_if_win': stake * (dec - 1)}


def no_vig(over_odds, under_odds):
    a, b = 1 / american_to_decimal(over_odds), 1 / american_to_decimal(under_odds)
    return a / (a + b), b / (a + b)


def walk_forward(history, metric, line, window=12, half_life=5.):
    history = clean_history(history)
    records = []
    # Pure baseline backtest: no present-day role, opponent, or efficiency overrides.
    for i in range(5, len(history)):
        out = simulate_nfl(history.iloc[:i], metric, window=window, half_life=half_life, simulations=5000, seed=42+i)
        result = summarize_samples(out['samples'], line)
        actual = float(history.iloc[i][metric])
        records.append({'date': history.iloc[i]['date'].date().isoformat(), 'actual': actual,
                        'predicted_mean': result['mean'], 'over_probability': result['over'],
                        'push_probability': result['push'], 'over_hit': float(actual > line),
                        'absolute_error': abs(actual-result['mean'])})
    return pd.DataFrame(records)


def csv_bytes(df):
    # Avoid spreadsheet formulas in player names / user-entered notes.
    out = df.copy()
    for col in out.select_dtypes(include=['object', 'string']):
        out[col] = out[col].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v)
    return out.to_csv(index=False).encode('utf-8')


def state_init():
    for name, default in [('board', []), ('bets', []), ('results', {})]:
        if name not in st.session_state:
            st.session_state[name] = default


def add_board(record):
    base = {k: v for k, v in record.items() if k not in ['samples', 'history']}
    base['saved_at'] = datetime.now(timezone.utc).isoformat()
    base['id'] = uuid.uuid4().hex[:12]
    st.session_state.board.append(base)


def market_inputs(prefix, default_line):
    a, b, c = st.columns(3)
    line = a.number_input('Prop line', min_value=0.0, max_value=500.0, value=float(default_line), step=.5, key=prefix+'line')
    side = b.selectbox('Bet side', ['Over', 'Under'], key=prefix+'side')
    odds = c.number_input('American odds', value=-110, step=5, key=prefix+'odds')
    a, b = st.columns(2)
    stake = a.number_input('Example stake ($)', min_value=0.0, value=5.0, step=1.0, key=prefix+'stake')
    other_odds = b.number_input('Opposite side odds', value=-110, step=5, key=prefix+'other')
    st.caption('Enter current sportsbook prices manually. Opposite-side odds are used only for the no-vig comparison.')
    return {'line': line, 'side': side, 'odds': odds, 'stake': stake, 'other_odds': other_odds}


def display_projection(bundle, prefix):
    draws, record = bundle['samples'], bundle['record']
    summary = summarize_samples(draws, record['line'])
    win = summary[record['side'].lower()]
    price = price_market(win, summary['push'], record['odds'], record['stake'])
    st.subheader(f"{record['player']} · {record['side']} {record['line']:g} {record['market']}")
    st.caption(f"Saved calculation for {record['date']} · {record['source']} · inputs below belong to this result")
    a, b, c = st.columns(3)
    a.metric('Estimated win probability', f'{win:.1%}')
    b.metric('Projected average', f"{summary['mean']:.1f}")
    c.metric('Estimated EV / $1', f"{price['roi']:+.1%}")
    a.metric('10th–90th percentile', f"{summary['p10']:.0f}–{summary['p90']:.0f}")
    b.metric('Push probability', f"{summary['push']:.1%}")
    c.metric('Example expected P/L', f"${price['expected_profit']:+.2f}")
    st.caption(f"Odds {record['odds']:+g} · stake ${record['stake']:.2f} · break-even excluding pushes {price['break_even']:.1%}. Percentile range is a model interval, not an accuracy guarantee.")
    if price['fair_odds'] is not None:
        st.write(f"**Model fair odds:** {price['fair_odds']:+.0f}")
    try:
        over_odds, under_odds = (record['odds'], record['other_odds']) if record['side'] == 'Over' else (record['other_odds'], record['odds'])
        fair_o, fair_u = no_vig(over_odds, under_odds)
        st.caption(f'Market no-vig probability, excluding pushes: Over {fair_o:.1%} · Under {fair_u:.1%}. Removing vig does not identify the true probability.')
    except ValueError:
        st.caption('No-vig comparison unavailable: enter valid odds for the opposite side.')
    for note in bundle.get('notes', []):
        st.info(note)
    tabs = st.tabs(['Alternate lines', 'Distribution', 'Recent games', 'Save / export'])
    with tabs[0]:
        yardage = 'yards' in record['market'].lower()
        step = 10 if yardage else 1
        center = np.floor(record['line']) + .5
        lines = sorted(set(max(.5, center+i*step) for i in range(-4, 5)))
        rows = []
        for line in lines:
            r = summarize_samples(draws, line)
            rows.append({'Line': line, 'Over %': round(100*r['over'], 1), 'Under %': round(100*r['under'], 1),
                         'At least': int(np.floor(line)+1), 'At least %': round(100*r['over'], 1)})
        alt = pd.DataFrame(rows)
        st.dataframe(alt, hide_index=True, width='stretch')
        st.download_button('Download alternate lines', csv_bytes(alt), 'alternate-lines.csv', 'text/csv', key=prefix+'altcsv')
    with tabs[1]:
        bins = min(35, max(5, int(np.ptp(draws))+1))
        counts, edges = np.histogram(draws, bins=bins)
        chart = pd.DataFrame({'Outcome': (edges[:-1]+edges[1:])/2, 'Probability': counts/len(draws)}).set_index('Outcome')
        st.bar_chart(chart)
        st.caption('30,000 simulations. More simulations reduce random sampling noise, not model bias.')
    with tabs[2]:
        hist = bundle['history'].copy()
        st.dataframe(hist.sort_values('date', ascending=False), hide_index=True, width='stretch')
        metric = record.get('metric', 'so')
        if metric in hist:
            st.line_chart(hist.set_index('date')[[metric]])
            rows = []
            for n in [5, 10, 20]:
                h = hist.tail(n)
                rows.append({'Window': f'Last {len(h)}', 'Average': h[metric].mean(),
                             'Over hit %': (h[metric] > record['line']).mean()*100,
                             'Under hit %': (h[metric] < record['line']).mean()*100})
            st.dataframe(pd.DataFrame(rows).drop_duplicates(), hide_index=True)
            st.caption('Historical hit rates describe this sample; they are not calibrated forecast probabilities.')
    with tabs[3]:
        export = {**record, **summary, 'win_probability': win, 'estimated_roi': price['roi'], 'model_version': VERSION}
        st.download_button('Download prediction CSV', csv_bytes(pd.DataFrame([export])), 'prediction.csv', 'text/csv', key=prefix+'csv')
        if st.button('Add to comparison board', key=prefix+'save'):
            add_board(export)
            st.success('Added. Open Comparison board in the sidebar. Export it before ending this session.')
        st.json({k: v for k, v in record.items() if k not in ['player_id']}, expanded=False)


def mlb_page():
    st.header('⚾ MLB strikeouts')
    day = st.date_input('Game date', value=local_today(), key='mlb_date').isoformat()
    try:
        starters = get_games(day)
    except (requests.RequestException, ValueError, KeyError) as exc:
        st.error(f'MLB schedule could not load: {exc}')
        return
    if not starters:
        st.info('No announced probable starters for this date. Choose another date or try after starters are announced.')
        return
    p = starters[st.selectbox('Pitcher', range(len(starters)), format_func=lambda i: f"{starters[i]['pitcher']} — {starters[i]['game']}")]
    st.caption(f"{p['game']} · {p['status']} · Confirm starter and pitch limit before using a projection.")
    inputs = market_inputs('mlb_', 5.5)
    workload = st.slider('Expected workload adjustment (%)', 50, 150, 100, 5, key='mlb_workload', help='Manual scenario for pitch limits or role changes. Not an automatic injury feed.') / 100
    if st.button('Calculate MLB projection', type='primary'):
        try:
            american_to_decimal(inputs['odds'])
            logs = get_pitcher_logs(p['pitcher_id'], day)
            opp_rate, notes = None, []
            try:
                opp = get_opponent_k_rate(p['opponent_id'], day)
                opp_rate = opp['k_rate']
                notes.append(f"Opponent blended K rate {opp_rate:.1%}; season {opp['season_k_rate']:.1%}; recent {opp['recent_k_rate']:.1%}.")
            except (requests.RequestException, ValueError, KeyError) as exc:
                notes.append(f'Opponent adjustment unavailable; pitcher-only baseline. {exc}')
            # Keep K rate fixed: sample original model, then adjust only the simulated workload in predict helper.
            result = predict_mlb_workload(logs, inputs['line'], workload, opp_rate)
            notes.append(f"{result['recent_starts']} starts used; projected batters faced {result['projected_bf']:.1f}. Workload scenario {workload:.0%}.")
            record = dict(sport='MLB', player=p['pitcher'], date=day, market='Strikeouts', metric='so',
                          opponent=p['opponent'], source='MLB Stats API', workload=workload, **inputs)
            st.session_state.results['mlb'] = {'samples': result['samples'], 'history': pd.DataFrame(logs), 'record': record, 'notes': notes}
        except (requests.RequestException, ValueError, KeyError) as exc:
            st.error(f'Could not calculate: {exc}')
    if 'mlb' in st.session_state.results:
        display_projection(st.session_state.results['mlb'], 'mlb_result_')


def predict_mlb_workload(logs, line, workload=1., opponent_rate=None):
    # Same underlying beta-binomial model, with workload separate from historical K rate.
    valid = sorted([g for g in logs if g['bf'] > 0 and 0 <= g['so'] <= g['bf']], key=lambda g: g['date'])[-20:]
    if len(valid) < 5:
        raise ValueError('At least five prior starts required.')
    weights = np.linspace(.5, 1.5, len(valid)); weights /= weights.sum()
    bf = np.array([g['bf'] for g in valid]); so = np.array([g['so'] for g in valid])
    alpha = .22*80 + np.dot(weights, so)*len(valid)
    beta = .78*80 + np.dot(weights, bf-so)*len(valid)
    rng = np.random.default_rng(42)
    volumes = np.rint(rng.choice(bf, 30000, p=weights)*workload).astype(int)
    rates = rng.beta(alpha, beta, 30000)
    if opponent_rate is not None:
        from scipy.special import expit, logit
        rates = expit(logit(rates)+.5*(logit(opponent_rate)-logit(.22)))
    draws = rng.binomial(volumes, rates)
    return {**summarize_samples(draws, line), 'samples': draws, 'recent_starts': len(valid), 'projected_bf': float(volumes.mean())}


def manual_nfl():
    st.caption('Upload one player’s game log. Keep genuine zero-stat appearances; do not enter missing games or DNPs as zero.')
    template = pd.DataFrame(columns=['date']+STAT_COLS)
    st.download_button('Download blank game-log template', csv_bytes(template), 'nfl-game-log-template.csv', 'text/csv')
    upload = st.file_uploader('Player game-log CSV', type='csv', key='manual_nfl_csv')
    name = st.text_input('Player name', value='My player')
    target = st.date_input('Prediction date', value=local_today(), key='manual_target')
    if upload is None:
        st.info('Choose Live NFL data for automatic player stats, or upload a CSV here.')
        return None
    try:
        history = clean_history(pd.read_csv(upload))
        history = history[history.date < pd.Timestamp(target)]
        return dict(history=history, player=name, date=target.isoformat(), opponent='Manual / not applied',
                    source='User CSV', context=None, position=None, season=None, team='Manual', notes=[])
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return None


def live_nfl():
    today = local_today()
    default_season = today.year if today.month >= 3 else today.year-1
    season = int(st.number_input('NFL season', min_value=2000, max_value=today.year+1, value=default_season, step=1))
    try:
        with st.spinner('Loading NFL schedule and player history…'):
            schedule = nfl_schedule()
            season_games = schedule[schedule.season == season].copy()
            if season_games.empty:
                st.info('No schedule published for this season. Choose an available season or manual CSV mode.')
                return None
            stats, notes = load_nfl_data(season, schedule)
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        st.error(f'NFL data unavailable: {exc}')
        st.info('Use Refresh data, try another season, or switch to manual CSV. No invented replacement stats are used.')
        return None
    for note in notes:
        st.warning(note)
    weeks = sorted(season_games.week.astype(int).unique())
    upcoming = season_games[season_games.gameday >= pd.Timestamp(today)].sort_values('gameday')
    default_week = int(upcoming.iloc[0].week) if len(upcoming) else weeks[-1]
    week = st.selectbox('Week (postseason follows regular season)', weeks, index=weeks.index(default_week))
    games = season_games[season_games.week == week].sort_values(['gameday', 'game_id']).reset_index(drop=True)
    idx = st.selectbox('Game', range(len(games)), format_func=lambda i: f"{games.iloc[i].away_team} @ {games.iloc[i].home_team} · {games.iloc[i].gameday:%b %d}")
    game = games.iloc[idx]
    target = game.gameday.date().isoformat()
    team = st.selectbox('Player’s team for this game', [game.away_team, game.home_team])
    opponent = game.home_team if team == game.away_team else game.away_team
    history_before = prior_nfl_rows(stats, target)
    if history_before.empty:
        st.info('No player stats available before this game. Try a later week or manual CSV.')
        return None
    newest = history_before.gameday.max().date()
    st.caption(f'Game date: {target} · newest eligible stat date: {newest} · fetched/cached for up to 30 minutes.')
    if (game.gameday.date()-newest).days > 16:
        st.warning('History is old relative to this game. Check missing games, offseason changes, and player availability.')
    if pd.notna(game.home_score):
        st.info('Historical mode: this game has a recorded result. Its player stats and same-day games are excluded from the projection.')
    latest = history_before.sort_values(['gameday', 'game_id']).groupby('player_id', as_index=False).tail(1)
    latest = latest[latest.position.isin(['QB', 'RB', 'FB', 'WR', 'TE'])]
    all_players = st.checkbox('Search all players (for trades or missing team assignments)', value=False)
    candidates = latest if all_players else latest[latest.team == team]
    candidates = candidates.sort_values('player_display_name').reset_index(drop=True)
    st.caption('Team assignments come from the latest prior stat row, not a verified active roster. Confirm trades, injuries, scratches, and starting roles yourself.')
    if candidates.empty:
        st.info('No historical players found for that team. Enable Search all players or use manual CSV.')
        return None
    player_idx = st.selectbox('Player', range(len(candidates)), format_func=lambda i: f"{candidates.iloc[i].player_display_name} · {candidates.iloc[i].position} · last team {candidates.iloc[i].team}")
    player = candidates.iloc[player_idx]
    try:
        history = player_history(stats, player.player_id, target)
    except ValueError as exc:
        st.error(f'Player game logs need review: {exc}')
        return None
    return dict(history=history, player=player.player_display_name, date=target, opponent=opponent,
                team=team, source='nflverse weekly player stats', context=stats, position=player.position,
                season=season, notes=notes, game_id=game.game_id)


def nfl_page():
    st.header('🏈 NFL player props')
    source = st.radio('Data source', ['Live NFL data', 'Manual game-log CSV'], horizontal=True)
    selected = live_nfl() if source == 'Live NFL data' else manual_nfl()
    if selected is None:
        return
    metric_name = st.selectbox('Market', list(MARKETS))
    metric = MARKETS[metric_name]
    history = selected['history']
    if len(history) < 5:
        st.warning(f'Only {len(history)} valid prior games. Five are required; rookies and new roles may not have enough history.')
        return
    recent = history.tail(12)
    a, b, c = st.columns(3)
    a.metric('Prior games', len(history))
    b.metric('Last 5 average', f'{history.tail(5)[metric].mean():.1f}')
    c.metric('Last 12 average', f'{recent[metric].mean():.1f}')
    default_line = max(.5, float(np.floor(recent[metric].mean())+.5))
    inputs = market_inputs('nfl_'+metric+'_', default_line)
    with st.expander('Model settings & workload scenarios', expanded=False):
        window = st.slider('Recent games used', 5, 25, 12)
        half_life = st.slider('Recency half-life (games)', 2, 12, 5)
        workload = st.slider('Workload vs historical role (%)', 25, 200, 100, 5) / 100
        efficiency = st.slider('Yards per opportunity adjustment (%)', 50, 150, 100, 5, disabled='yards' not in metric) / 100
        use_defense = st.checkbox('Apply opponent / position adjustment', value=True, disabled=selected['context'] is None)
        st.caption('Workload changes carries or targets. Efficiency changes yardage only. These manual scenarios can represent role changes, injuries, game script or weather; none is automatically verified. Defense adjustment is heuristic, capped at ±20%.')
    context = {'factor': 1., 'note': 'Neutral opponent adjustment.', 'games': 0}
    if selected['context'] is not None and use_defense:
        context = opponent_context(selected['context'], selected['opponent'], selected['position'], metric, selected['date'], selected['season'])
    st.caption(context['note'])
    if st.button('Calculate NFL projection', type='primary'):
        try:
            american_to_decimal(inputs['odds'])
            result = simulate_nfl(history, metric, window, half_life, workload, efficiency, context['factor'])
            notes = [f"{result['games']} games used; recency-weighted effective sample {result['effective_games']:.1f}. Last player game {result['last_game']}.",
                     f"Projected {result['volume_label']}: {result['projected_volume']:.1f}. Workload {workload:.0%}; efficiency {efficiency:.0%}; opponent factor {context['factor']:.2f}.",
                     context['note']]
            if (pd.Timestamp(selected['date'])-pd.Timestamp(result['last_game'])).days > 21:
                notes.append('This player’s latest game is over 21 days before the target. Verify injury, roster and role changes.')
            record = {k: selected[k] for k in ['player', 'date', 'opponent', 'team', 'source']}
            record.update(dict(sport='NFL', market=metric_name, metric=metric, window=window, half_life=half_life,
                               workload=workload, efficiency=efficiency, defense_factor=context['factor'], **inputs))
            st.session_state.results['nfl'] = {'samples': result['samples'], 'history': history, 'record': record, 'notes': notes}
        except ValueError as exc:
            st.error(str(exc))
    if 'nfl' in st.session_state.results:
        display_projection(st.session_state.results['nfl'], 'nfl_result_')
    with st.expander('Historical check — walk-forward baseline'):
        st.caption('Tests one selected fixed line against later historical games. Each prediction uses earlier games only. Excludes today’s manual adjustments and opponent factor. This is an exploratory diagnostic, not proof of profitability; there are no historical sportsbook prices.')
        if st.button('Run historical check'):
            with st.spinner('Testing earlier-game-only predictions…'):
                test = walk_forward(history, metric, inputs['line'], window, half_life)
            if test.empty:
                st.info('Need at least six games to hold out one result.')
            else:
                st.metric('Held-out games', len(test))
                st.metric('Mean absolute error', f"{test.absolute_error.mean():.2f}")
                st.metric('Over-event Brier score (lower is better)', f"{((test.over_probability-test.over_hit)**2).mean():.3f}")
                st.caption('Brier score evaluates the event strictly over the line; pushes count as not-over for this diagnostic only.')
                st.dataframe(test, hide_index=True, width='stretch')
                st.download_button('Download historical check', csv_bytes(test), 'historical-check.csv', 'text/csv')


def comparison_page():
    st.header('Comparison board')
    st.caption('Saved calculations remain in this browser session only. Different player markets can be correlated; this board does not calculate parlay probabilities.')
    board = st.session_state.board
    if board:
        df = pd.DataFrame(board)
        sort = st.selectbox('Sort by', ['estimated_roi', 'win_probability', 'saved_at'])
        shown = df.sort_values(sort, ascending=False)
        cols = ['sport', 'player', 'date', 'market', 'side', 'line', 'odds', 'win_probability', 'push', 'estimated_roi']
        st.dataframe(shown[cols], hide_index=True, width='stretch')
        st.caption('Probabilities and ROI are decimals: 0.60 = 60%. A high estimated ROI may reflect model error.')
        st.download_button('Export comparison board CSV', csv_bytes(df), 'comparison-board.csv', 'text/csv')
        i = st.selectbox('Saved calculation to track', range(len(board)), format_func=lambda i: f"{board[i]['player']} {board[i]['side']} {board[i]['line']} {board[i]['market']}")
        if st.button('Copy selected to bet tracker'):
            r = board[i]
            st.session_state.bets.append({'id': uuid.uuid4().hex[:12], 'date': r['date'], 'sport': r['sport'],
                'player': r['player'], 'market': r['market'], 'side': r['side'], 'line': r['line'],
                'odds': r['odds'], 'stake': r['stake'], 'status': 'Pending', 'notes': 'Copied projection; not a placed bet'})
            st.success('Added as Pending. This app does not place bets.')
        if st.button('Clear comparison board'):
            st.session_state.board = []
            st.rerun()
    else:
        st.info('Calculate a projection, then choose Save / export → Add to comparison board.')


def bet_profit(row):
    if row['status'] == 'Win':
        return float(row['stake']) * (american_to_decimal(float(row['odds']))-1)
    if row['status'] == 'Loss':
        return -float(row['stake'])
    return 0.


def validate_bets(rows):
    if not isinstance(rows, list) or len(rows) > 5000:
        raise ValueError('Bet records must be a list with at most 5,000 items.')
    out = []
    required = {'date', 'sport', 'player', 'market', 'side', 'line', 'odds', 'stake', 'status'}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError('Invalid bet record format.')
        r = {k: row[k] for k in required}
        if r['status'] not in ['Pending', 'Win', 'Loss', 'Push', 'Void'] or r['side'] not in ['Over', 'Under']:
            raise ValueError('Invalid bet status or side.')
        for k in ['line', 'odds', 'stake']:
            r[k] = float(r[k])
        american_to_decimal(r['odds'])
        if not np.isfinite([r['line'], r['odds'], r['stake']]).all() or r['stake'] < 0:
            raise ValueError('Invalid bet numbers.')
        r['id'] = str(row.get('id', uuid.uuid4().hex[:12]))[:80]
        r['notes'] = str(row.get('notes', ''))[:1000]
        out.append(r)
    return out


def tracker_page():
    st.header('Bet tracker & bankroll')
    st.caption('Manual tracker. Nothing is sent to a sportsbook. Download a backup before closing: free hosting and browser sessions do not guarantee storage.')
    start = st.number_input('Starting bankroll ($)', min_value=0., value=250., step=10.)
    unit = st.slider('Planning unit (% of bankroll)', .25, 2., .5, .25)
    with st.expander('Add a bet manually'):
        with st.form('new_bet'):
            day = st.date_input('Bet date', local_today())
            sport = st.selectbox('Sport', ['NFL', 'MLB'])
            player = st.text_input('Player')
            market = st.selectbox('Prop', ['Strikeouts']+list(MARKETS))
            side = st.selectbox('Side', ['Over', 'Under'])
            line = st.number_input('Line', min_value=0., value=5.5, step=.5)
            odds = st.number_input('Odds', value=-110, step=5)
            stake = st.number_input('Stake ($)', min_value=0., value=5.)
            if st.form_submit_button('Add pending bet'):
                try:
                    american_to_decimal(odds)
                    if not player.strip():
                        raise ValueError('Enter a player name.')
                    st.session_state.bets.append(dict(id=uuid.uuid4().hex[:12], date=day.isoformat(), sport=sport,
                        player=player, market=market, side=side, line=line, odds=odds, stake=stake, status='Pending', notes=''))
                    st.success('Added to this session.')
                except ValueError as exc:
                    st.error(str(exc))
    bets = st.session_state.bets
    profit = sum(bet_profit(r) for r in bets)
    pending = sum(r['stake'] for r in bets if r['status'] == 'Pending')
    risked = sum(r['stake'] for r in bets if r['status'] in ['Win', 'Loss'])
    a, b, c = st.columns(3)
    a.metric('Settled P/L', f'${profit:+.2f}')
    b.metric('Pending exposure', f'${pending:.2f}')
    c.metric('Available after pending', f'${start+profit-pending:.2f}')
    st.caption(f'Planning unit: ${max(0,start+profit)*unit/100:.2f}. This is a size calculator, not a stake recommendation.')
    if risked:
        st.write(f'**Settled ROI (wins/losses):** {profit/risked:+.1%}')
    if bets:
        view = pd.DataFrame(bets); view['pnl'] = [bet_profit(r) for r in bets]
        st.dataframe(view.drop(columns='id'), hide_index=True, width='stretch')
        idx = st.selectbox('Bet to settle', range(len(bets)), format_func=lambda i: f"{bets[i]['date']} · {bets[i]['player']} · {bets[i]['market']} · {bets[i]['status']}")
        status = st.selectbox('Result', ['Pending', 'Win', 'Loss', 'Push', 'Void'])
        if st.button('Update result'):
            st.session_state.bets[idx]['status'] = status
            st.rerun()
        st.download_button('Export tracker CSV', csv_bytes(view), 'bet-tracker.csv', 'text/csv')
    backup = {'version': VERSION, 'bets': bets, 'board': st.session_state.board}
    st.download_button('Download session backup (JSON)', json.dumps(backup, default=str, indent=2), 'sports-prop-backup.json', 'application/json')
    upload = st.file_uploader('Restore tracker from JSON backup', type='json')
    st.caption('Restore merges bet records by ID. Board exports are retained in the backup for reference but are not re-imported.')
    if upload is not None and st.button('Restore bet records'):
        try:
            if upload.size > 5_000_000:
                raise ValueError('Backup is too large (maximum 5 MB).')
            data = json.load(upload)
            restored = validate_bets(data.get('bets'))
            merged = {r['id']: r for r in bets}
            merged.update({r['id']: r for r in restored})
            st.session_state.bets = list(merged.values())
            st.success(f'Restored {len(restored)} records. Reopen the tracker to refresh totals.')
        except (ValueError, TypeError, AttributeError) as exc:
            st.error(f'Cannot restore backup: {exc}')


def guide_page():
    st.header('How to use Sports Prop Lab')
    st.markdown('''
1. Select **MLB strikeouts** or **NFL props** in the sidebar.
2. Choose the game, player, market, line, and sportsbook odds.
3. Confirm the player's role and availability. Adjust workload only when you have a reason.
4. Calculate, inspect recent games and alternate lines, and save candidates to the comparison board.
5. Track any bets manually and export a backup before leaving.

**What is automatic:** MLB schedule and probable pitchers; prior MLB logs and opponent strikeout rates; NFL schedule and weekly player logs; NFL opponent-by-position comparisons; simulations, odds math, charts and exports.

**What is manual:** sportsbook prices, injury/active status, role and workload changes, weather/game-script scenarios, bet entry and settlement. No accounts or API keys are required. No bet is placed by this app.

**NFL model:** exponentially weighted recent games; Poisson or negative-binomial workload depending on sample variance; receptions from targets and a beta-binomial catch rate; yards from simulated opportunities and shrunk historical per-opportunity efficiency. Games with zero stats are kept when supplied by the source. Missing rows are not assumed to be zero or a DNP. Count and yardage distributions are heuristic and uncalibrated. NFL yardage can be negative.

**Opponent factor:** current-season per-game production allowed to a position, shrunk with ten league-average games and capped at 0.8–1.2. It is descriptive and not adjusted for strength of schedule, personnel, or game script. The historical check intentionally uses the baseline without this factor.

**MLB model:** recency-weighted prior starts, beta-binomial strikeout rate with a fixed 22% / 80-batter prior; optional blended opponent factor and manual workload adjustment. Pitch limits, confirmed batting lineups and handedness splits are not automatically modeled.

**Probability and price:** Over means strictly above the line; Under strictly below it. Integer lines can push. EV = win probability × net payout − loss probability; pushes return the stake. “At least 5” is the over-4.5 event. Fair odds and no-vig probabilities are conditional on no push.

**Historical checks:** no target-day/future rows are used. However, published datasets may contain later stat corrections, and choosing players/lines after viewing results introduces selection bias. The check is not a verified historical betting strategy or return simulation.

**Limits:** recent samples are small; usage can change abruptly; rookies need five prior valid games. The app does not infer active rosters, guarantee fresh source data, auto-grade bets, calculate correlated parlays, or persist data across sessions. Model edge is an estimate, not evidence of profit.

**Data:** [MLB Stats API](https://statsapi.mlb.com/api/v1) · [nflverse player statistics](https://nflreadr.nflverse.com/reference/load_player_stats.html) · [nflverse update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html) · [NFL schedules](https://github.com/nflverse/nfldata).
''')


def upcoming_nfl_slate(schedule, today, now=None):
    """Next unplayed week; never silently substitute an old season."""
    s = schedule.copy()
    if 'game_type' in s:
        s = s[s.game_type.isin(['REG', 'WC', 'DIV', 'CON', 'SB'])]
    pending = s[(s.gameday >= pd.Timestamp(today)) & s.home_score.isna() & s.away_score.isna()].sort_values('gameday')
    if pending.empty:
        return pending
    if now is not None and 'gametime' in pending:
        kickoff = pd.to_datetime(pending.gameday.dt.strftime('%Y-%m-%d')+' '+pending.gametime.fillna(''), errors='coerce').dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
        pending = pending[kickoff.isna() | (kickoff > pd.Timestamp(now))]
        if pending.empty:
            return pending
    first = pending.iloc[0]
    return pending[(pending.season == first.season) & (pending.week == first.week)].reset_index(drop=True)


def simulate_nfl_game(schedule, game, home_adjust=0., away_adjust=0., n=30000, seed=42):
    """Shrunk recent scoring baseline, paired empirical score residuals; heuristic, not calibrated."""
    cutoff = pd.Timestamp(game.gameday)
    h = schedule[(schedule.gameday < cutoff) & (schedule.gameday >= cutoff-pd.Timedelta(days=730))].copy()
    if 'game_type' in h:
        h = h[h.game_type.isin(['REG', 'WC', 'DIV', 'CON', 'SB'])]
    h = h.dropna(subset=['home_score', 'away_score']).sort_values('gameday')
    h = h[(h.home_score >= 0) & (h.away_score >= 0)]
    if len(h) < 20:
        raise ValueError('At least 20 completed league games before kickoff date are required.')
    league = float(pd.concat([h.home_score, h.away_score]).mean())
    # Remove venue effect before computing team attack/defense ratings.
    neutral = h.get('location', pd.Series('Home', index=h.index)).eq('Neutral')
    home_edge = float(np.clip((h.loc[~neutral, 'home_score']-h.loc[~neutral, 'away_score']).mean(), 0, 4)) if (~neutral).any() else 0.
    venue = np.where(neutral, 0., home_edge/2)
    long = pd.concat([
        pd.DataFrame({'team':h.home_team, 'date':h.gameday, 'scored':h.home_score-venue, 'allowed':h.away_score+venue}),
        pd.DataFrame({'team':h.away_team, 'date':h.gameday, 'scored':h.away_score+venue, 'allowed':h.home_score-venue})])
    def rating(team):
        rows = long[long.team == team].sort_values('date').tail(20)
        if len(rows) < 5:
            raise ValueError(f'{team}: fewer than five prior completed games; no reliable baseline.')
        w = np.exp2(-np.arange(len(rows)-1, -1, -1)/8.)
        return ((float(np.dot(rows.scored, w))+6*league)/(w.sum()+6),
                (float(np.dot(rows.allowed, w))+6*league)/(w.sum()+6), len(rows))
    ho, hd, hn = rating(game.home_team)
    ao, ad, an = rating(game.away_team)
    edge = 0 if game.get('location', 'Home') == 'Neutral' else home_edge/2
    hm = float(np.clip(ho+ad-league+edge+home_adjust, 3, 55))
    am = float(np.clip(ao+hd-league-edge+away_adjust, 3, 55))
    rng = np.random.default_rng(seed)
    # Sampling paired deviations retains empirical within-game score correlation.
    residual = h[['home_score','away_score']].to_numpy(dtype=float)
    residual -= residual.mean(axis=0)
    draws = residual[rng.integers(0,len(residual),n)] + np.array([hm,am])
    draws = np.maximum(0, np.rint(draws)).astype(int)
    # A score of one is impossible. These are approximate full-game scores, not drive simulations.
    draws[draws == 1] = 0
    return dict(home=draws[:,0], away=draws[:,1], history=h,
                home_games=hn, away_games=an, home_edge=home_edge,
                expected_home=hm, expected_away=am)


def game_market_rows(home, away, home_line, total_line):
    margin, total = home-away, home+away
    events = [('Home spread',margin+home_line),('Away spread',-margin-home_line),
              ('Over total',total-total_line),('Under total',total_line-total)]
    return [dict(Market=name, Win=float(np.mean(x>0)), Push=float(np.mean(x==0)), Loss=float(np.mean(x<0))) for name,x in events]


def nfl_games_page():
    st.header('NFL pre-game simulator')
    st.caption('Automatic upcoming slate · 30,000 simulated outcomes per game · no live updates')
    try:
        schedule = nfl_schedule()
        slate = upcoming_nfl_slate(schedule, local_today(), datetime.now(ZoneInfo("America/Chicago")))
    except Exception as exc:
        st.error(f'NFL schedule unavailable: {exc}')
        return
    if slate.empty:
        st.info('No upcoming unplayed games in the published schedule. Refresh later; old games are not substituted.')
        return
    st.write(f"Season {int(slate.iloc[0].season)} · Week {int(slate.iloc[0].week)}")
    st.caption(f"Schedule checked {datetime.now(ZoneInfo('America/Chicago')):%b %d, %I:%M %p} Central; feed cached up to 30 minutes. Confirm kickoff and postponements.")
    st.dataframe(slate[['gameday','away_team','home_team']], hide_index=True)
    idx = st.selectbox('NFL matchup', range(len(slate)), format_func=lambda i: f'{slate.iloc[i].away_team} @ {slate.iloc[i].home_team}')
    game = slate.iloc[idx]
    # Key controls by game so an old matchup's settings cannot silently carry over.
    key = 'game_'+str(game.game_id)
    st.caption('Enter your sportsbook lines. Negative home spread means the home team is favored.')
    a,b = st.columns(2)
    spread = a.number_input('Home team spread', value=-3., step=.5, key=key+'spread')
    total = b.number_input('Game total', min_value=0., value=44.5, step=.5, key=key+'total')
    odds = st.number_input('Odds for selected market', value=-110, step=5, key=key+'odds')
    market = st.selectbox('Game market', ['Home spread','Away spread','Over total','Under total'],key=key+'market')
    with st.expander('Injury / weather scenario adjustments'):
        st.caption('Manual points added to each team’s scoring mean. No automatic injury or weather feed.')
        ha = st.number_input('Home scoring adjustment',min_value=-15.,max_value=15.,value=0.,step=.5,key=key+'ha')
        aa = st.number_input('Away scoring adjustment',min_value=-15.,max_value=15.,value=0.,step=.5,key=key+'aa')
    st.warning('Baseline model, not a proven edge. Scores are approximate; strength of schedule, current personnel and drive-level football scoring are not modeled. Tied simulations remain unresolved, so win probabilities are not sportsbook moneyline probabilities.')
    if st.button('Simulate NFL game'):
        try:
            american_to_decimal(odds)
            r = simulate_nfl_game(schedule,game,ha,aa)
            home,away = r['home'],r['away']
            margin,totals = home-away,home+away
            hp,ap,tp = np.mean(margin>0),np.mean(margin<0),np.mean(margin==0)
            a,b,c = st.columns(3)
            a.metric('Projected home score', f'{home.mean():.1f}')
            b.metric('Projected away score', f'{away.mean():.1f}')
            c.metric('Projected total', f'{totals.mean():.1f}')
            st.write(f"**Projected winner:** {game.home_team if hp>ap else game.away_team if ap>hp else 'Even matchup'} · **Model home spread:** {-margin.mean():+.1f}")
            st.write(f'{game.home_team} win: {hp:.1%} · {game.away_team} win: {ap:.1%} · Unresolved tie: {tp:.1%}')
            st.caption('Projected scores are simulated averages, not a predicted exact score. Spread equals negative average home margin.')
            ranges = pd.DataFrame([{'Outcome':name,'10th percentile':np.quantile(x,.1),'Median':np.median(x),'90th percentile':np.quantile(x,.9)} for name,x in [('Home score',home),('Away score',away),('Home margin',margin),('Total points',totals)]])
            st.dataframe(ranges, hide_index=True)
            rows = game_market_rows(home,away,spread,total)
            st.dataframe(pd.DataFrame(rows),hide_index=True)
            chosen = next(row for row in rows if row['Market']==market)
            ev = chosen['Win']*(american_to_decimal(odds)-1)-chosen['Loss']
            st.metric('Estimated return per $1 risked',f'${ev:+.3f}')
            st.caption('Pushes return the stake. Positive model EV does not establish a real betting advantage.')
            st.bar_chart(pd.Series(totals).value_counts().sort_index().rename('Simulations'))
            st.caption(f"Training: {len(r['history'])} completed games through {r['history'].gameday.max():%Y-%m-%d}; home team {r['home_games']} recent games; away team {r['away_games']}. Only games before the matchup date are used.")
            export = pd.DataFrame({'home_score':home,'away_score':away,'home_margin':margin,'total':totals})
            st.download_button('Download game simulations', csv_bytes(export),f'{game.game_id}-simulations.csv','text/csv',on_click='ignore')
        except (ValueError,TypeError,KeyError) as exc:
            st.error(str(exc))


def main():
    st.set_page_config(page_title='Sports Prop Lab | MLB + NFL', page_icon='📊', layout='centered')
    state_init()
    st.title('Sports Prop Lab')
    st.caption('MLB + NFL · five prop markets + NFL game simulations · one workspace')
    st.warning('Research estimates, not proven betting edges. Confirm player availability and sportsbook rules. Prices are entered manually.')
    with st.sidebar:
        st.header('Your workspace')
        page = st.radio('Open', ['NFL props', 'NFL games', 'MLB strikeouts', 'Comparison board', 'Bet tracker', 'Guide & sources'])
        st.caption(f'Version {VERSION} · America/Chicago dates')
        if st.button('Refresh data'):
            st.cache_data.clear()
            st.rerun()
        st.caption('Refresh clears cached feeds. Saved calculations keep their original inputs.')
    {'NFL games': nfl_games_page, 'NFL props': nfl_page, 'MLB strikeouts': mlb_page, 'Comparison board': comparison_page,
     'Bet tracker': tracker_page, 'Guide & sources': guide_page}[page]()


if __name__ == '__main__':
    main()
