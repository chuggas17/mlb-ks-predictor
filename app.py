from datetime import date
import pandas as pd
import streamlit as st
from mlb_api import get_games, get_pitcher_logs, get_opponent_k_rate
from model import analyze_odds, predict

st.set_page_config(page_title='MLB K Edge Lab', page_icon='⚾', layout='centered')
st.title('⚾ MLB K Edge Lab')
st.caption('Free MLB data · Pitcher strikeout probability research · Manual sportsbook odds')
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
        except (ValueError, KeyError) as exc:
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
