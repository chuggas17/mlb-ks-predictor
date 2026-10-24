"""Transparent baseline for pitcher strikeout props; not a validated betting edge."""
import numpy as np


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


def predict(games: list[dict], line: float, simulations: int = 30000, seed: int = 42) -> dict:
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
    strikeouts = rng.binomial(workloads, rates)
    over = float(np.mean(strikeouts > line))
    return {'over': over, 'under': float(np.mean(strikeouts < line)),
            'push': float(np.mean(strikeouts == line)),
            'mean': float(np.mean(strikeouts)), 'median': float(np.median(strikeouts)),
            'starts': len(valid), 'recent_starts': len(recent),
            'projected_bf': float(np.mean(workloads)),
            'k_rate': float(alpha / (alpha + beta))}
