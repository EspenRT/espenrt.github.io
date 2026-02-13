"""
Binance API Client - Fetches OHLCV candlestick data with in-memory caching.
Uses the public Binance API (no API key required for market data).
Connection pooling via requests.Session reduces DNS lookups.
Retry with exponential backoff handles transient DNS / network errors.
"""

import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# In-memory cache: key -> (timestamp, DataFrame)
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 300  # 5 minutes - daily/weekly candles don't change fast


BINANCE_KLINES_URL = 'https://data-api.binance.vision/api/v3/klines'

TIMEFRAME_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h',
    '1d': '1d', '1w': '1w', '1M': '1M',
}

# Persistent session with connection pooling and automatic retries.
# pool_connections/pool_maxsize=20 keeps TCP connections alive across requests,
# avoiding repeated DNS lookups that overwhelm Windows resolver under load.
# Retry handles transient DNS/network failures with exponential backoff.
_session = requests.Session()
_retry_strategy = Retry(
    total=3,                       # up to 3 retries per request
    backoff_factor=0.5,            # 0.5s, 1s, 2s between retries
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
_adapter = HTTPAdapter(
    max_retries=_retry_strategy,
    pool_connections=20,           # keep 20 host connections pooled
    pool_maxsize=20,               # max 20 concurrent connections per host
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def fetch_ohlcv(symbol: str,
                interval: str = '1d',
                limit: int = 200) -> pd.DataFrame:
    """Fetch OHLCV data from Binance.

    Args:
        symbol: Trading pair symbol, e.g. 'ETHUSDT'
        interval: Candle interval, e.g. '1d', '4h', '1h'
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Indexed by datetime.
    """
    cache_key = f'{symbol}_{interval}_{limit}'

    # Check cache
    if cache_key in _cache:
        cached_time, cached_df = _cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            return cached_df

    bi_interval = TIMEFRAME_MAP.get(interval, interval)

    params = {
        'symbol': symbol.upper(),
        'interval': bi_interval,
        'limit': limit,
    }

    # Use session for connection pooling + automatic retry on transient errors
    resp = _session.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Cache result
    _cache[cache_key] = (time.time(), df)

    return df


def get_btc_pair(usdt_symbol: str) -> str | None:
    """Convert a USDT pair to its BTC equivalent.

    E.g. 'ETHUSDT' -> 'ETHBTC', 'BTCUSDT' -> None (no BTC/BTC pair).
    """
    if usdt_symbol.upper() == 'BTCUSDT':
        return None

    base = usdt_symbol.upper().replace('USDT', '')
    return f'{base}BTC'


def validate_symbol(symbol: str) -> bool:
    """Check if a symbol exists on Binance by fetching 1 candle."""
    try:
        params = {'symbol': symbol.upper(), 'interval': '1d', 'limit': 1}
        resp = _session.get(BINANCE_KLINES_URL, params=params, timeout=5)
        return resp.status_code == 200 and len(resp.json()) > 0
    except Exception:
        return False


def is_cached(symbol: str, interval: str = '1d', limit: int = 200) -> bool:
    """Check if OHLCV data is in cache and not expired."""
    cache_key = f'{symbol}_{interval}_{limit}'
    if cache_key in _cache:
        cached_time, _ = _cache[cache_key]
        return time.time() - cached_time < CACHE_TTL
    return False


def clear_cache():
    """Clear the in-memory cache."""
    _cache.clear()
