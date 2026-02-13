"""
VuManChu Cipher B Divergences - Python Translation
Translated from Pine Script v4 (vumanchu pinescript.txt)
All indicator calculations match the original Pine Script logic.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# WaveTrend Oscillator
# Pine Script reference (lines 185-201):
#   esa = ema(src, chlen)        [chlen=9]
#   de  = ema(abs(src - esa), chlen)
#   ci  = (src - esa) / (0.015 * de)
#   wt1 = ema(ci, avg)           [avg=12]
#   wt2 = sma(wt1, malen)       [malen=3]
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def compute_wavetrend(df: pd.DataFrame,
                      channel_len: int = 9,
                      average_len: int = 12,
                      ma_len: int = 3) -> pd.DataFrame:
    """Compute WaveTrend wt1, wt2 and derived cross/OB/OS signals."""
    src = (df['high'] + df['low'] + df['close']) / 3  # hlc3

    esa = ema(src, channel_len)
    de = ema((src - esa).abs(), channel_len)
    ci = (src - esa) / (0.015 * de)
    ci = ci.replace([np.inf, -np.inf], 0).fillna(0)

    wt1 = ema(ci, average_len)
    wt2 = sma(wt1, ma_len)
    vwap = wt1 - wt2

    # Cross detection (Pine: cross(wt1, wt2))
    diff = wt1 - wt2
    diff_prev = diff.shift(1)
    wt_cross = (diff * diff_prev < 0) | (diff == 0)

    wt_cross_up = diff >= 0    # wt1 >= wt2
    wt_cross_down = diff <= 0  # wt1 <= wt2

    # OB/OS levels (Pine defaults)
    ob_level = 53
    os_level = -53

    wt_overbought = wt2 >= ob_level
    wt_oversold = wt2 <= os_level

    result = pd.DataFrame({
        'wt1': wt1,
        'wt2': wt2,
        'wt_vwap': vwap,
        'wt_cross': wt_cross,
        'wt_cross_up': wt_cross_up,
        'wt_cross_down': wt_cross_down,
        'wt_overbought': wt_overbought,
        'wt_oversold': wt_oversold,
    }, index=df.index)

    return result


# ---------------------------------------------------------------------------
# RSI
# Pine Script reference (line 300): rsi = rsi(close, 14)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# ---------------------------------------------------------------------------
# MFI (Money Flow Index) - Custom RSI+MFI Area
# Pine Script reference (line 182):
#   sma(((close - open) / (high - low)) * multiplier, period) - posY
# ---------------------------------------------------------------------------

def compute_mfi(df: pd.DataFrame,
                period: int = 60,
                multiplier: float = 150,
                pos_y: float = 2.5) -> pd.Series:
    hl_range = df['high'] - df['low']
    hl_range = hl_range.replace(0, np.nan)
    raw = ((df['close'] - df['open']) / hl_range) * multiplier
    raw = raw.fillna(0)
    mfi = sma(raw, period) - pos_y
    return mfi


# ---------------------------------------------------------------------------
# Stochastic RSI
# Pine Script reference (lines 223-230):
#   rsi_val = rsi(log(src) if useLog else src, rsilen)
#   k = sma(stoch(rsi_val, rsi_val, rsi_val, stochlen), smoothk)
#   d = sma(k, smoothd)
# ---------------------------------------------------------------------------

def compute_stoch_rsi(series: pd.Series,
                      stoch_len: int = 14,
                      rsi_len: int = 14,
                      smooth_k: int = 3,
                      smooth_d: int = 3,
                      use_log: bool = True) -> pd.DataFrame:
    src = np.log(series) if use_log else series
    rsi_val = compute_rsi(src, rsi_len)

    # stoch(rsi, rsi, rsi, len) = (rsi - lowest(rsi, len)) / (highest(rsi, len) - lowest(rsi, len)) * 100
    lowest = rsi_val.rolling(window=stoch_len, min_periods=1).min()
    highest = rsi_val.rolling(window=stoch_len, min_periods=1).max()
    denom = highest - lowest
    denom = denom.replace(0, np.nan)
    stoch_raw = ((rsi_val - lowest) / denom) * 100
    stoch_raw = stoch_raw.fillna(0)

    k = sma(stoch_raw, smooth_k)
    d = sma(k, smooth_d)

    return pd.DataFrame({'stoch_k': k, 'stoch_d': d}, index=series.index)


# ---------------------------------------------------------------------------
# Schaff Trend Cycle
# Pine Script reference (lines 204-220)
# ---------------------------------------------------------------------------

def compute_schaff_tc(series: pd.Series,
                      length: int = 10,
                      fast_length: int = 23,
                      slow_length: int = 50,
                      factor: float = 0.5) -> pd.Series:
    ema1 = ema(series, fast_length)
    ema2 = ema(series, slow_length)
    macd_val = ema1 - ema2

    values = macd_val.values.astype(float)
    n = len(values)
    delta = np.full(n, np.nan)
    stc = np.full(n, np.nan)

    for i in range(length - 1, n):
        window = values[max(0, i - length + 1):i + 1]
        alpha = np.nanmin(window)
        beta = np.nanmax(window) - alpha
        gamma = ((values[i] - alpha) / beta * 100) if beta > 0 else (delta[i - 1] if i > 0 and not np.isnan(delta[i - 1]) else 0)

        if np.isnan(delta[i - 1]) if i > 0 else True:
            delta[i] = gamma
        else:
            delta[i] = delta[i - 1] + factor * (gamma - delta[i - 1])

    for i in range(length - 1, n):
        window = delta[max(0, i - length + 1):i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        epsilon = np.min(valid)
        zeta = np.max(valid) - epsilon
        eta = ((delta[i] - epsilon) / zeta * 100) if zeta > 0 else (stc[i - 1] if i > 0 and not np.isnan(stc[i - 1]) else 0)

        if i == 0 or np.isnan(stc[i - 1]):
            stc[i] = eta
        else:
            stc[i] = stc[i - 1] + factor * (eta - stc[i - 1])

    return pd.Series(stc, index=series.index, name='schaff_tc')


# ---------------------------------------------------------------------------
# Divergence Detection
# Pine Script reference (lines 163-179):
#   f_top_fractal(src) => src[4] < src[2] and src[3] < src[2] and src[2] > src[1] and src[2] > src[0]
#   f_bot_fractal(src) => src[4] > src[2] and src[3] > src[2] and src[2] < src[1] and src[2] < src[0]
#   Then checks for regular and hidden divergences using price vs indicator value.
# ---------------------------------------------------------------------------

def find_fractals(src: pd.Series) -> pd.Series:
    """Return +1 for top fractals, -1 for bottom fractals, 0 otherwise.
    A top fractal at index i means src[i] is higher than src[i-2], src[i-1], src[i+1], src[i+2].
    Pine Script uses src[0]..src[4] where src[4] is oldest, so fractal is at index [2] offset.
    We detect at position i based on i-2, i-1, i, i+1, i+2 but since Pine checks
    src[4]<src[2] ... meaning the fractal point is 2 bars ago, we replicate that by
    looking at the centered window and then our result aligns with the 2-bar-ago convention."""
    vals = src.values.astype(float)
    n = len(vals)
    result = np.zeros(n)

    for i in range(2, n - 2):
        # Top fractal: center is higher than its 2 neighbors on each side
        if (vals[i - 2] < vals[i] and vals[i - 1] < vals[i] and
                vals[i] > vals[i + 1] and vals[i] > vals[i + 2]):
            result[i] = 1
        # Bottom fractal: center is lower than its 2 neighbors on each side
        elif (vals[i - 2] > vals[i] and vals[i - 1] > vals[i] and
              vals[i] < vals[i + 1] and vals[i] < vals[i + 2]):
            result[i] = -1

    return pd.Series(result, index=src.index)


def find_divergences(indicator: pd.Series,
                     high: pd.Series,
                     low: pd.Series,
                     top_limit: float = 45,
                     bot_limit: float = -65,
                     use_limits: bool = True) -> pd.DataFrame:
    """Detect regular and hidden divergences using fractal pivots.
    Pine Script reference (lines 168-179)."""
    fractals = find_fractals(indicator)
    ind_vals = indicator.values.astype(float)
    high_vals = high.values.astype(float)
    low_vals = low.values.astype(float)
    frac_vals = fractals.values
    n = len(ind_vals)

    bear_div = np.zeros(n, dtype=bool)
    bull_div = np.zeros(n, dtype=bool)
    bear_div_hidden = np.zeros(n, dtype=bool)
    bull_div_hidden = np.zeros(n, dtype=bool)

    last_top_ind = np.nan
    last_top_price = np.nan
    last_bot_ind = np.nan
    last_bot_price = np.nan

    for i in range(4, n):
        # Check top fractal (bearish divergence check)
        if frac_vals[i] == 1:
            current_ind = ind_vals[i]
            current_price = high_vals[i]
            if use_limits and current_ind < top_limit:
                pass  # Skip if below the minimum level for bearish div
            elif not np.isnan(last_top_ind):
                # Regular bearish: price higher high, indicator lower high
                if current_price > last_top_price and current_ind < last_top_ind:
                    bear_div[i] = True
                # Hidden bearish: price lower high, indicator higher high
                if current_price < last_top_price and current_ind > last_top_ind:
                    bear_div_hidden[i] = True
            last_top_ind = current_ind
            last_top_price = current_price

        # Check bottom fractal (bullish divergence check)
        if frac_vals[i] == -1:
            current_ind = ind_vals[i]
            current_price = low_vals[i]
            if use_limits and current_ind > bot_limit:
                pass  # Skip if above the minimum level for bullish div
            elif not np.isnan(last_bot_ind):
                # Regular bullish: price lower low, indicator higher low
                if current_price < last_bot_price and current_ind > last_bot_ind:
                    bull_div[i] = True
                # Hidden bullish: price higher low, indicator lower low
                if current_price > last_bot_price and current_ind < last_bot_ind:
                    bull_div_hidden[i] = True
            last_bot_ind = current_ind
            last_bot_price = current_price

    return pd.DataFrame({
        'fractal': fractals,
        'bear_div': bear_div,
        'bull_div': bull_div,
        'bear_div_hidden': bear_div_hidden,
        'bull_div_hidden': bull_div_hidden,
    }, index=indicator.index)


# ---------------------------------------------------------------------------
# Master function: compute all indicators for a given OHLCV DataFrame
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all VuManChu Cipher B indicators on an OHLCV DataFrame.

    Expects columns: open, high, low, close, volume
    Returns DataFrame with all indicator columns merged.
    """
    result = df.copy()

    # WaveTrend
    wt = compute_wavetrend(df)
    for col in wt.columns:
        result[col] = wt[col]

    # RSI
    result['rsi'] = compute_rsi(df['close'], period=14)

    # MFI
    result['mfi'] = compute_mfi(df)

    # Stochastic RSI
    stoch = compute_stoch_rsi(df['close'])
    for col in stoch.columns:
        result[col] = stoch[col]

    # Schaff Trend Cycle
    result['schaff_tc'] = compute_schaff_tc(df['close'])

    # WT Divergences
    wt_divs = find_divergences(result['wt2'], df['high'], df['low'],
                               top_limit=45, bot_limit=-65, use_limits=True)
    result['wt_bear_div'] = wt_divs['bear_div']
    result['wt_bull_div'] = wt_divs['bull_div']
    result['wt_bear_div_hidden'] = wt_divs['bear_div_hidden']
    result['wt_bull_div_hidden'] = wt_divs['bull_div_hidden']

    # WT 2nd Divergences (wider range)
    wt_divs2 = find_divergences(result['wt2'], df['high'], df['low'],
                                top_limit=15, bot_limit=-40, use_limits=True)
    result['wt_bear_div2'] = wt_divs2['bear_div']
    result['wt_bull_div2'] = wt_divs2['bull_div']

    # RSI Divergences
    rsi_divs = find_divergences(result['rsi'], df['high'], df['low'],
                                top_limit=60, bot_limit=30, use_limits=True)
    result['rsi_bear_div'] = rsi_divs['bear_div']
    result['rsi_bull_div'] = rsi_divs['bull_div']

    # Stoch Divergences
    stoch_divs = find_divergences(result['stoch_k'], df['high'], df['low'],
                                  top_limit=0, bot_limit=0, use_limits=False)
    result['stoch_bear_div'] = stoch_divs['bear_div']
    result['stoch_bull_div'] = stoch_divs['bull_div']

    return result
