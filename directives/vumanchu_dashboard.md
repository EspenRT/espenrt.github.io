# VuManChu Cipher B Trading Dashboard

## Goal
Provide real-time buy/sell signal monitoring for crypto pairs based on the VuManChu Cipher B indicator, with BTC pair confluence confirmation.

## How to Run
```
cd "C:\Users\Espen\Desktop\CLAUDE\Trading dashboard"
python -m execution.server
```
Open http://localhost:5000 in your browser.

## Configuration
Edit `config.json` to change:
- `pairs`: List of Binance USDT pair symbols (e.g. `["ETHUSDT", "SOLUSDT"]`)
- `timeframe`: Candle interval (`"1d"`, `"4h"`, `"1h"`, etc.)
- `port`: Server port (default 5000)

## Signal Logic (from VuManChu Pine Script)

### Buy Signals
| Signal | Condition | Badge |
|--------|-----------|-------|
| Strong Buy (Gold) | WT divergence + extreme oversold (WT2 was <= -75) + RSI < 30 | STRONG BUY |
| Buy (Green circle) | WaveTrend cross up while WT2 <= -53 (oversold) | BUY |
| Buy (Small dot) | WaveTrend cross up (any zone) | BUY |
| Buy Divergence | Bullish divergence on WT, Stoch RSI, or RSI | BUY (Div) |

### Sell Signals
| Signal | Condition | Badge |
|--------|-----------|-------|
| Sell (Red circle) | WaveTrend cross down while WT2 >= 53 (overbought) | SELL |
| Sell (Small dot) | WaveTrend cross down (any zone) | SELL |
| Sell Divergence | Bearish divergence on WT, Stoch RSI, or RSI | SELL (Div) |

**Important**: The dashboard tracks ALL WaveTrend cross signals (small dots), not just the big circle signals that require OB/OS zones. This was a critical fix — without small dot signals, many valid crosses were missed.

### BTC Confluence
For each USDT pair, the same indicators are computed on the /BTC pair. If both agree on direction, it shows "CONFIRMED BUY/SELL". If they disagree, "CONFLICTING".

### MFI (Daily)
Money Flow Index is shown for USDT pairs on the daily timeframe. MFI > 0 = BULLISH, MFI < 0 = BEARISH.

## Buy Score (0-100%)
Each asset gets a composite buy score based on 5 weighted signal components. Table is sorted by score descending (best buys at top).

**Override**: If 1D USDT Signal is SELL → Score = 0% (no matter what).

| Signal | Weight | Description |
|--------|--------|-------------|
| 1D USDT Signal | 50 | Primary buy trigger (dominant) |
| 1D BTC Pair | 15 | Daily BTC confluence |
| 1W USDT Signal | 15 | Weekly trend alignment |
| 1W BTC Pair | 10 | Weekly BTC confluence |
| 1D MFI | 10 | Money flow (binary: bullish=10, bearish=0) |

For signals 1-4: `points = weight × signal_mult × recency_mult`
- Signal multiplier: STRONG BUY/BUY=1.0, BUY(Div)=0.8, SELL/none=0.0
- Recency: TODAY=1.0, YESTERDAY=0.7, OLDER=0.5

Score color coding: 80-100% bright green, 60-79% green, 40-59% yellow, 20-39% orange, 0-19% red.

## Dashboard Columns
| Column | Description |
|--------|-------------|
| Asset | Symbol + current price |
| Score | Buy score 0-100%, color coded |
| 1D Signal | Latest daily USDT pair signal with date badge |
| 1D BTC Pair | Latest daily BTC pair signal with date badge |
| 1D MFI | Daily MFI for USDT pair (BULLISH/BEARISH) |
| 1W Signal | Latest weekly USDT pair signal with date badge |
| 1W BTC Pair | Latest weekly BTC pair signal with date badge |
| Confluence | Daily USDT+BTC agreement (CONFIRMED/CONFLICTING) |

### Date Badges
- **TODAY** (green): Signal fired on today's candle
- **YESTERDAY** (orange): Signal fired yesterday
- **OLDER** (gray): Signal fired earlier, shows the date

## Indicators Computed
- **WaveTrend (WT1, WT2)**: Main oscillator. Channel=9, Average=12, MA=3
- **RSI**: 14-period RSI on close
- **MFI**: Custom Money Flow Index (period=60, multiplier=150)
- **Stochastic RSI**: Log-based, K smooth=3, D smooth=3
- **Schaff Trend Cycle**: Fast=23, Slow=50, Length=10
- **Divergences**: 5-bar fractal pivot detection on WT, RSI, and StochRSI

## Files
| File | Purpose |
|------|---------|
| `execution/indicators.py` | All indicator calculations (translated from Pine Script) |
| `execution/signal_engine.py` | Buy/sell signal evaluation + BTC confluence |
| `execution/binance_data.py` | Binance API client with 5-minute cache |
| `execution/server.py` | Flask web server with parallel fetching + warmup |
| `static/index.html` | Dashboard frontend (dark theme table) |
| `config.json` | User configuration (50 pairs) |

## Performance Architecture
The server uses a 3-level caching strategy:

1. **Response cache (L1)**: Serialized JSON response, 5-minute TTL. Returns in <10ms.
2. **Signal cache (L2)**: Per-symbol computed signals, 5-minute TTL. Avoids re-running indicator computation.
3. **OHLCV cache (L3)**: Per-symbol/timeframe Binance data, 5-minute TTL. Avoids re-fetching from API.

### Parallel Fetching
- `ThreadPoolExecutor` with 40 workers fetches all OHLCV data in parallel (I/O bound)
- Only fetches data that isn't already cached (smart prefetch)
- ~200 API calls completed in ~5 parallel batches

### Startup Warmup
- Background thread pre-loads all data when the server starts
- If user opens browser during warmup, the request waits for warmup to finish (avoids duplicate work)
- After warmup completes, first page load is instant

### Performance Numbers (50 pairs)
| Scenario | Time |
|----------|------|
| Cold start (first ever load) | ~30-60s (depends on Binance latency) |
| First load after server start (warmup) | Waits for warmup, then instant |
| Subsequent refreshes | 5-10ms |
| Auto-refresh interval | 60 seconds |

## Edge Cases & Learnings
- Binance rate limit: 1200 requests/minute. 50 pairs × 4 fetches = 200 requests per refresh cycle. With 5-minute cache, that's well under the limit.
- Some altcoins may not have a BTC pair on Binance — the dashboard gracefully falls back to USDT-only signal.
- Divergence detection requires ~50+ candles of history to produce reliable fractals. We fetch 200 candles.
- `localhost` DNS resolution on Windows can add 2s latency per request (IPv6 fallback). The frontend uses relative URLs so this isn't an issue for the browser, but testing with `requests` library should use `127.0.0.1`.
- Flask debug mode reloader clears all in-memory caches on file changes — the warmup runs again.
- `pip` on this system must be called via `python -m pip`.
