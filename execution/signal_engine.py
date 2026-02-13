"""
Signal Engine - Evaluates VuManChu buy/sell conditions and BTC confluence.
Translates Pine Script signal logic (lines 377-407) into Python.
"""

from datetime import datetime, timezone

import pandas as pd
from execution.indicators import compute_all_indicators


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------

def evaluate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Given an OHLCV DataFrame, compute indicators and return signal columns.

    Buy signal (Green circle, line 378):
        wtCross and wtCrossUp and wtOversold

    Sell signal (Red circle, line 390):
        wtCross and wtCrossDown and wtOverbought

    Buy divergence (line 380-383):
        wtBullDiv or wtBullDiv2 or stochBullDiv or rsiBullDiv

    Sell divergence (line 392-395):
        wtBearDiv or wtBearDiv2 or stochBearDiv or rsiBearDiv

    Gold buy (lines 401-407):
        (wtBullDiv or rsiBullDiv) and wtLow_prev <= -75 and wt2 > -75
        and wtLow_prev - wt2 <= -5 and lastRsi < 30
    """
    ind = compute_all_indicators(df)

    # Small dot signals: any WT cross (Pine lines 515, 521)
    ind['buy_dot'] = ind['wt_cross'] & ind['wt_cross_up']
    ind['sell_dot'] = ind['wt_cross'] & ind['wt_cross_down']

    # Big circle signals: WT cross at OB/OS levels (Pine lines 378, 390)
    ind['buy_signal'] = ind['buy_dot'] & ind['wt_oversold']
    ind['sell_signal'] = ind['sell_dot'] & ind['wt_overbought']

    # Divergence buy/sell — WaveTrend divergences only (no RSI/Stochastic)
    ind['buy_div'] = ind['wt_bull_div'] | ind['wt_bull_div2']
    ind['sell_div'] = ind['wt_bear_div'] | ind['wt_bear_div2']

    # Gold buy (simplified - checks extreme oversold + WT divergence + low RSI)
    os_level3 = -75
    ind['gold_buy'] = (
        ind['wt_bull_div'] &
        (ind['wt2'].shift(1) <= os_level3) &
        (ind['wt2'] > os_level3) &
        (ind['rsi'] < 30)
    )

    # Combined: any buy/sell = dot signals + big circles + divergences
    ind['any_buy'] = ind['buy_dot'] | ind['buy_div'] | ind['gold_buy']
    ind['any_sell'] = ind['sell_dot'] | ind['sell_div']

    return ind


def _find_last_signal(ind: pd.DataFrame, col: str) -> dict | None:
    """Walk backwards through the DataFrame to find the last row where col is True.
    Returns dict with date (ISO string), signal type, and the row's price."""
    mask = ind[col]
    if not mask.any():
        return None
    last_idx = ind.index[mask][-1]
    row = ind.loc[last_idx]
    ts = str(last_idx)

    # Determine specific signal type (most significant first)
    if col == 'any_buy':
        if row.get('gold_buy', False):
            sig_type = 'STRONG BUY'
        elif row.get('buy_signal', False):
            sig_type = 'BUY'
        elif row.get('buy_div', False):
            sig_type = 'BUY (Div)'
        else:
            sig_type = 'BUY'
    elif col == 'any_sell':
        if row.get('sell_signal', False):
            sig_type = 'SELL'
        elif row.get('sell_div', False):
            sig_type = 'SELL (Div)'
        else:
            sig_type = 'SELL'
    else:
        sig_type = col

    return {
        'date': ts,
        'type': sig_type,
        'price': round(float(row['close']), 8),
    }


def get_current_signal(df: pd.DataFrame) -> dict:
    """Evaluate signals and return a summary dict for the dashboard table view.
    Returns the single most recent signal (buy or sell) with its timestamp."""
    if len(df) < 30:
        return {'error': 'Not enough data for indicator calculation'}

    ind = evaluate_signals(df)
    latest = ind.iloc[-1]

    # Find last buy and last sell across all history
    last_buy = _find_last_signal(ind, 'any_buy')
    last_sell = _find_last_signal(ind, 'any_sell')

    # Determine which was most recent — that's the latest signal
    latest_signal = None
    signal_strength = 0

    if last_buy and last_sell:
        if last_buy['date'] >= last_sell['date']:
            latest_signal = last_buy
            signal_strength = 1
        else:
            latest_signal = last_sell
            signal_strength = -1
    elif last_buy:
        latest_signal = last_buy
        signal_strength = 1
    elif last_sell:
        latest_signal = last_sell
        signal_strength = -1

    return {
        'price': round(float(latest['close']), 8),
        'signal_strength': signal_strength,
        'latest_signal': latest_signal,
        'mfi': round(float(latest['mfi']), 2),
    }


# ---------------------------------------------------------------------------
# BTC Confluence
# ---------------------------------------------------------------------------

def evaluate_with_confluence(usdt_df: pd.DataFrame,
                             btc_df: pd.DataFrame) -> dict:
    """Evaluate signals on the USDT pair and the BTC pair, then combine.

    Confluence: if both the USDT pair and the BTC pair agree on direction,
    the signal is confirmed. If they disagree, it's flagged as conflicting.
    """
    usdt_signal = get_current_signal(usdt_df)
    btc_signal = get_current_signal(btc_df)

    if 'error' in usdt_signal or 'error' in btc_signal:
        return {
            'usdt_pair': usdt_signal,
            'btc_pair': btc_signal,
            'confluence': 'INSUFFICIENT DATA',
            'confluence_confirmed': False,
        }

    usdt_dir = usdt_signal['signal_strength']
    btc_dir = btc_signal['signal_strength']

    # Both latest signals agree = confirmed
    if usdt_dir > 0 and btc_dir > 0:
        confluence = 'CONFIRMED BUY'
    elif usdt_dir < 0 and btc_dir < 0:
        confluence = 'CONFIRMED SELL'
    elif usdt_dir > 0 and btc_dir < 0:
        confluence = 'CONFLICTING'
    elif usdt_dir < 0 and btc_dir > 0:
        confluence = 'CONFLICTING'
    elif usdt_dir == 0 and btc_dir == 0:
        confluence = 'NO SIGNAL'
    else:
        confluence = 'PARTIAL'

    return {
        'usdt_pair': usdt_signal,
        'btc_pair': btc_signal,
        'confluence': confluence,
    }


# ---------------------------------------------------------------------------
# Buy Score (0-100%)
# ---------------------------------------------------------------------------

# Weights for each signal component (must sum to 100)
_SCORE_WEIGHTS = {
    '1d_usdt':  50,   # Primary buy trigger — dominant weight
    '1d_btc':   15,   # Daily BTC confluence
    '1w_usdt':  15,   # Weekly trend alignment
    '1w_btc':   10,   # Weekly BTC confluence
    '1d_mfi':   10,   # Supplementary money flow
}

# Signal type multipliers
_SIGNAL_MULT = {
    'STRONG BUY': 1.0,
    'BUY':        1.0,
    'BUY (Div)':  1.0,
}
# Anything else (SELL, SELL (Div), None) = 0.0

# Recency multipliers
_RECENCY_TODAY     = 1.0
_RECENCY_YESTERDAY = 0.75
_RECENCY_OLDER     = 0.4


def _recency_multiplier(date_str: str | None) -> float:
    """Compute recency multiplier based on signal date vs today (UTC).
    Binance candle timestamps are tz-naive but represent UTC midnight."""
    if not date_str:
        return 0.0
    try:
        sig_date = pd.Timestamp(date_str).tz_localize(None).normalize()
        now_utc = datetime.now(timezone.utc)
        today = pd.Timestamp(now_utc.year, now_utc.month, now_utc.day)
        days_ago = (today - sig_date).days
        if days_ago <= 0:
            return _RECENCY_TODAY
        elif days_ago == 1:
            return _RECENCY_YESTERDAY
        else:
            return _RECENCY_OLDER
    except Exception:
        return _RECENCY_OLDER


def _score_signal(signal_data: dict | None, weight: int,
                   apply_recency: bool = True) -> tuple[float, str, str]:
    """Score a single buy/sell signal component.

    Args:
        signal_data: The usdt_pair or btc_pair dict from the API response.
                     Expected to have 'latest_signal' with 'type' and 'date'.
        weight: Maximum points for this component.
        apply_recency: Whether to apply recency multiplier (False for weekly signals).

    Returns:
        (points, signal_type, recency_label)
    """
    if not signal_data or 'error' in signal_data:
        return 0.0, 'N/A', ''

    latest = signal_data.get('latest_signal')
    if not latest:
        return 0.0, 'No signal', ''

    sig_type = latest.get('type', '')
    sig_date = latest.get('date', '')

    # Signal type multiplier (0 for sells)
    type_mult = _SIGNAL_MULT.get(sig_type, 0.0)

    if apply_recency:
        # Recency multiplier (daily signals only)
        rec_mult = _recency_multiplier(sig_date)
        points = weight * type_mult * rec_mult

        # Recency label for breakdown display
        if rec_mult >= _RECENCY_TODAY:
            rec_label = 'TODAY'
        elif rec_mult >= _RECENCY_YESTERDAY:
            rec_label = 'YESTERDAY'
        else:
            rec_label = 'OLDER'
    else:
        # No recency — buy = full points, sell = 0
        points = weight * type_mult
        rec_label = ''

    return points, sig_type, rec_label


def compute_buy_score(daily: dict, weekly: dict) -> dict:
    """Compute a buy score (0-100%) for an asset based on 5 signal components.

    Args:
        daily: The 'daily' dict from _process_symbol(), with 'usdt_pair' and 'btc_pair'.
        weekly: The 'weekly' dict from _process_symbol(), with 'usdt_pair' and 'btc_pair'.

    Returns:
        dict with 'score' (0-100 int) and 'breakdown' (list of component details).
    """
    d_usdt = daily.get('usdt_pair')
    d_btc = daily.get('btc_pair')
    w_usdt = weekly.get('usdt_pair')
    w_btc = weekly.get('btc_pair')

    # --- Override rule: 1D USDT is SELL → score is 0 ---
    if d_usdt and d_usdt.get('latest_signal'):
        sig_type = d_usdt['latest_signal'].get('type', '')
        if 'SELL' in sig_type.upper():
            return {
                'score': 0,
                'breakdown': [
                    {'name': '1D Signal', 'points': 0, 'max': _SCORE_WEIGHTS['1d_usdt'],
                     'detail': f'{sig_type} (override: 0%)'},
                    {'name': '1D BTC', 'points': 0, 'max': _SCORE_WEIGHTS['1d_btc'], 'detail': '--'},
                    {'name': '1W Signal', 'points': 0, 'max': _SCORE_WEIGHTS['1w_usdt'], 'detail': '--'},
                    {'name': '1W BTC', 'points': 0, 'max': _SCORE_WEIGHTS['1w_btc'], 'detail': '--'},
                    {'name': '1D MFI', 'points': 0, 'max': _SCORE_WEIGHTS['1d_mfi'], 'detail': '--'},
                ],
            }

    # --- Score each component ---
    breakdown = []

    # 1. 1D USDT Signal (weight: 50)
    pts, stype, rec = _score_signal(d_usdt, _SCORE_WEIGHTS['1d_usdt'])
    breakdown.append({
        'name': '1D Signal', 'points': round(pts, 1),
        'max': _SCORE_WEIGHTS['1d_usdt'],
        'detail': f'{stype} {rec}' if stype != 'No signal' else 'No signal',
    })

    # 2. 1D BTC Pair (weight: 15)
    pts2, stype2, rec2 = _score_signal(d_btc, _SCORE_WEIGHTS['1d_btc'])
    breakdown.append({
        'name': '1D BTC', 'points': round(pts2, 1),
        'max': _SCORE_WEIGHTS['1d_btc'],
        'detail': f'{stype2} {rec2}' if stype2 not in ('No signal', 'N/A') else stype2,
    })

    # 3. 1W USDT Signal (weight: 15) — no recency multiplier for weekly
    pts3, stype3, rec3 = _score_signal(w_usdt, _SCORE_WEIGHTS['1w_usdt'], apply_recency=False)
    breakdown.append({
        'name': '1W Signal', 'points': round(pts3, 1),
        'max': _SCORE_WEIGHTS['1w_usdt'],
        'detail': stype3 if stype3 != 'No signal' else 'No signal',
    })

    # 4. 1W BTC Pair (weight: 10) — no recency multiplier for weekly
    pts4, stype4, rec4 = _score_signal(w_btc, _SCORE_WEIGHTS['1w_btc'], apply_recency=False)
    breakdown.append({
        'name': '1W BTC', 'points': round(pts4, 1),
        'max': _SCORE_WEIGHTS['1w_btc'],
        'detail': stype4 if stype4 not in ('No signal', 'N/A') else stype4,
    })

    # 5. 1D MFI (weight: 10) — binary: bullish=10, bearish=0
    mfi_val = d_usdt.get('mfi', 0) if d_usdt else 0
    mfi_pts = _SCORE_WEIGHTS['1d_mfi'] if mfi_val and mfi_val > 0 else 0
    breakdown.append({
        'name': '1D MFI', 'points': mfi_pts,
        'max': _SCORE_WEIGHTS['1d_mfi'],
        'detail': 'BULLISH' if mfi_pts > 0 else 'BEARISH',
    })

    total = sum(item['points'] for item in breakdown)
    score = round(total)

    return {
        'score': min(score, 100),  # Cap at 100 just in case
        'breakdown': breakdown,
    }
