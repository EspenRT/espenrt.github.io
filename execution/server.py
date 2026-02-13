"""
Flask Server - Serves the dashboard and API endpoints.
Run with: python -m execution.server
"""

import json
import os
import sys
import threading
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from flask import Flask, jsonify, send_from_directory

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from execution.binance_data import fetch_ohlcv, get_btc_pair, is_cached
from execution.signal_engine import get_current_signal, evaluate_with_confluence, compute_buy_score

app = Flask(__name__,
            static_folder=os.path.join(PROJECT_ROOT, 'static'),
            static_url_path='/static')

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.json')

# Thread pool for parallel Binance fetches
# 20 workers matches connection pool size — avoids overwhelming DNS resolver
_executor = ThreadPoolExecutor(max_workers=20)

# Signal-level cache: avoids recomputing indicators when OHLCV data hasn't changed
_signal_cache: dict[str, tuple[float, dict]] = {}
SIGNAL_CACHE_TTL = 300  # 5 minutes — matches OHLCV cache

# Response-level cache: stores the serialized JSON to avoid re-serialization
_response_cache: tuple[float, str] | None = None
RESPONSE_CACHE_TTL = 300  # 5 minutes

# Warmup synchronization: prevents duplicate work when user opens browser during warmup
_warmup_done = threading.Event()


def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/config')
def api_config():
    """Return current configuration."""
    config = load_config()
    return jsonify(config)


def _prefetch_uncached(pairs: list[str]):
    """Pre-fetch only OHLCV data that isn't already cached.
    Skips symbols whose signal cache is still warm (no OHLCV needed at all).
    For symbols needing recomputation, only fetches uncached OHLCV intervals."""
    fetch_tasks = []
    for symbol in pairs:
        # Skip entirely if signal cache is warm — we won't need OHLCV data
        if symbol in _signal_cache:
            cached_time, _ = _signal_cache[symbol]
            if _time.time() - cached_time < SIGNAL_CACHE_TTL:
                continue

        btc_symbol = get_btc_pair(symbol)
        for tf in ('1d', '1w'):
            if not is_cached(symbol, tf, 200):
                fetch_tasks.append((symbol, tf))
            if btc_symbol and not is_cached(btc_symbol, tf, 200):
                fetch_tasks.append((btc_symbol, tf))

    if not fetch_tasks:
        return  # Everything cached, nothing to fetch

    futures = {
        _executor.submit(fetch_ohlcv, sym, interval=tf, limit=200): (sym, tf)
        for sym, tf in fetch_tasks
    }

    for future in as_completed(futures):
        try:
            future.result()
        except Exception:
            pass  # Individual errors handled later in _compute_pair_signals


def _is_stale(df, timeframe: str) -> bool:
    """Check if the last candle in a DataFrame is too old (pair likely delisted).
    For daily: stale if last candle > 3 days ago.
    For weekly: stale if last candle > 14 days ago."""
    if df is None or df.empty:
        return True
    last_date = df.index[-1]
    now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    days_old = (now - last_date).days
    max_age = 14 if timeframe == '1w' else 3
    return days_old > max_age


def _compute_pair_signals(symbol: str, timeframe: str) -> dict:
    """Compute USDT + BTC pair signals for a single symbol at a given timeframe.
    Assumes OHLCV data is already cached from _prefetch_uncached.
    Detects delisted/stale BTC pairs and marks them accordingly."""
    usdt_df = fetch_ohlcv(symbol, interval=timeframe, limit=200)
    btc_symbol = get_btc_pair(symbol)

    if btc_symbol:
        try:
            btc_df = fetch_ohlcv(btc_symbol, interval=timeframe, limit=200)

            # Check if BTC pair data is stale (delisted pair)
            if _is_stale(btc_df, timeframe):
                usdt_signal = get_current_signal(usdt_df)
                result = {
                    'usdt_pair': usdt_signal,
                    'btc_pair': {'error': f'BTC pair delisted (last data: {btc_df.index[-1].date()})'},
                    'btc_symbol': btc_symbol,
                    'confluence': 'BTC PAIR DELISTED',
                }
            else:
                result = evaluate_with_confluence(usdt_df, btc_df)
                result['btc_symbol'] = btc_symbol
        except Exception as e:
            usdt_signal = get_current_signal(usdt_df)
            result = {
                'usdt_pair': usdt_signal,
                'btc_pair': {'error': f'BTC pair unavailable: {str(e)}'},
                'btc_symbol': btc_symbol,
                'confluence': 'BTC PAIR UNAVAILABLE',
            }
    else:
        usdt_signal = get_current_signal(usdt_df)
        result = {
            'usdt_pair': usdt_signal,
            'btc_pair': None,
            'btc_symbol': None,
            'confluence': 'N/A (BTC itself)',
        }

    return result


def _process_symbol(symbol: str) -> dict:
    """Process a single symbol: compute daily + weekly signals.
    Uses signal cache to avoid redundant indicator computation."""
    if symbol in _signal_cache:
        cached_time, cached_result = _signal_cache[symbol]
        if _time.time() - cached_time < SIGNAL_CACHE_TTL:
            return cached_result

    daily = _compute_pair_signals(symbol, '1d')
    weekly = _compute_pair_signals(symbol, '1w')
    price = daily.get('usdt_pair', {}).get('price')

    # Compute buy score (0-100%)
    score_data = compute_buy_score(daily, weekly)

    result = {
        'symbol': symbol,
        'price': price,
        'score': score_data['score'],
        'score_breakdown': score_data['breakdown'],
        'daily': daily,
        'weekly': weekly,
    }
    _signal_cache[symbol] = (_time.time(), result)
    return result


def _build_signals_response(pairs: list[str]) -> str:
    """Fetch data, compute signals, and return serialized JSON response.
    Shared by both the warmup thread and the API endpoint."""
    global _response_cache

    _prefetch_uncached(pairs)

    results = []
    for symbol in pairs:
        try:
            results.append(_process_symbol(symbol))
        except Exception as e:
            results.append({
                'symbol': symbol,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })

    response_data = json.dumps({'pairs': results})
    _response_cache = (_time.time(), response_data)
    return response_data


@app.route('/api/signals')
def api_signals():
    """Compute and return signals for all configured pairs on 1D and 1W.
    Uses 3-level caching: response → signal → OHLCV.
    If warmup is running, waits for it instead of duplicating work."""

    # If warmup is still running, wait for it to finish
    if not _warmup_done.is_set():
        _warmup_done.wait(timeout=300)

    # Level 1: Return cached JSON response if available
    if _response_cache:
        cached_time, cached_json = _response_cache
        if _time.time() - cached_time < RESPONSE_CACHE_TTL:
            return app.response_class(
                response=cached_json,
                status=200,
                mimetype='application/json'
            )

    # Cache miss — recompute everything
    config = load_config()
    pairs = config.get('pairs', [])
    response_data = _build_signals_response(pairs)

    return app.response_class(
        response=response_data,
        status=200,
        mimetype='application/json'
    )


@app.route('/api/signal/<symbol>')
def api_single_signal(symbol: str):
    """Compute signal for a single pair."""
    config = load_config()
    timeframe = config.get('timeframe', '1d')

    try:
        df = fetch_ohlcv(symbol.upper(), interval=timeframe, limit=200)
        signal = get_current_signal(df)
        signal['symbol'] = symbol.upper()
        return jsonify(signal)
    except Exception as e:
        return jsonify({'error': str(e), 'symbol': symbol.upper()}), 500


@app.route('/health')
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'warmup_complete': _warmup_done.is_set(),
        'cache_entries': len(_signal_cache),
    })


def _warmup():
    """Pre-warm all caches on server startup so the first page load is fast.
    Runs the same logic as api_signals directly (no HTTP overhead)."""
    try:
        print('[WARMUP] Pre-loading signal data from Binance...')
        t0 = _time.time()

        config = load_config()
        pairs = config.get('pairs', [])
        _build_signals_response(pairs)

        elapsed = _time.time() - t0
        print(f'[WARMUP] Done in {elapsed:.1f}s — dashboard ready!')
    except Exception as e:
        print(f'[WARMUP] Failed: {e}')
    finally:
        _warmup_done.set()


# Start warmup at module level — works under both gunicorn and direct execution.
# For direct execution with debug reloader, only start in the child process.
# For gunicorn (no WERKZEUG_RUN_MAIN env var, no debug reloader), always start.
_is_reloader_parent = os.environ.get('WERKZEUG_RUN_MAIN') is None and os.environ.get('GUNICORN_ARBITER') is None
_is_debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true')

if not (_is_debug and _is_reloader_parent):
    threading.Thread(target=_warmup, daemon=True).start()


if __name__ == '__main__':
    config = load_config()
    port = int(os.environ.get('PORT', config.get('port', 5000)))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true')

    print(f'Starting KryptoRadar on http://localhost:{port}')
    print(f'Pairs: {len(config.get("pairs", []))} configured')
    print(f'Debug: {debug}')

    app.run(host='0.0.0.0', port=port, debug=debug)
