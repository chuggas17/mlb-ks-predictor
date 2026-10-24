"""Public MLB Stats API data fetching. No API key required."""
from datetime import date
import requests
import streamlit as st

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
            if entry.get('date', '') >= target_date:
                continue
            stat = entry.get('stat', {})
            try:
                k = int(stat['strikeOuts'])
                pa = int(stat['plateAppearances'])
            except (KeyError, ValueError, TypeError):
                continue
            if pa > 0:
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
