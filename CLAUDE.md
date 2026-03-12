# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-file Python trading bot (`bot.py`) that monitors BTCUSDT on Binance across multiple timeframes and sends Wyckoff-based signals (Absorption and Exhaustion) to a Telegram channel.

## Environment Setup

Requires a `.env` file with:
```
BINANCE_API_KEY=
BINANCE_API_SECRET=
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
```

## Commands

```bash
# Local development (activate venv first)
source .venv/bin/activate
pip install -r requirements.txt
python bot.py

# Tests
pytest tests/ -v

# Docker
docker build -t wyckoff_bot .
docker run --env-file .env wyckoff_bot
```

## Architecture

All logic lives in `bot.py`. The main loop runs every 10 seconds and iterates over `TIMEFRAMES = ['30m', '1h', '4h']`.

**Data flow:** `get_data()` → fetches 500 Binance klines, computes RSI/EMA50/EMA200/VolumeSMA → `check_signals()` → `send_telegram_message()`

**Multi-timeframe trend gate:** The main loop caches 4h data (5-min TTL) and calls `_get_trend()` to get the 4h trend each iteration. For 30m and 1h, signals only fire if the local trend matches the 4h trend (`htf_trend` gate). The 4h timeframe has no gate (it IS the HTF). This eliminates counter-trend noise.

**Signal logic in `check_signals(df, interval, htf_trend=None)`:**
- **HTF gate**: If `htf_trend` is provided and disagrees with local trend → return None immediately.
- **Absorption**: High volume + narrow spread + **bearish candle** (VSA: smart money absorbs sellers) + close in upper 40%+ of candle range + BULLISH trend. Bearish absorption is the mirror.
- **Exhaustion**: Confirmed swing low/high via `find_swing_lows`/`find_swing_highs` (scipy) + RSI divergence compared at the actual prior swing index.
- **Every signal includes**: Entry Zone (candle body), Stop Loss (nearest swing ± 0.5×ATR), T1/T2 from structural swing levels (actual R:R shown, e.g. `2.3R — Swing High`). T2 omitted if no second structural level found.
- **Quality filters**: Signal skipped if `risk > 3 × ATR` (too wide) or structural T1 gives < 1.0R (too close).

**Cooldown**: 300s for all timeframes (tracked via `last_signal_time` dict).

**Helper functions (pure, testable):**
- `close_position(high, low, close)` — returns 0.0–1.0 where close sits in the candle range
- `find_swing_lows(df, distance=5, atr_mult=0.5)` — scipy find_peaks on inverted lows, ATR-based prominence
- `find_swing_highs(df, distance=5, atr_mult=0.5)` — scipy find_peaks on highs
- `_get_trend(df)` — returns "BULLISH" / "BEARISH" / "NEUTRAL" from EMA50/EMA200; returns "NEUTRAL" for None or short df
- `calculate_rr(direction, lower_entry, upper_entry, stop, atr, t1=None, t2=None)` — returns entry/stop/T1/T2 dict with R:R values, or None

## Dependencies

- `python-binance` — Binance REST API client
- `pandas` — all indicator calculations (RSI, EMA, rolling stats)
- `requests` — Telegram HTTP calls
- `python-dotenv` — `.env` loading
- `scipy` — swing high/low detection via `scipy.signal.find_peaks`
- `pytest` — test suite

The `.venv` uses Python 3.14; Dockerfile uses Python 3.11-slim for deployment.

## Testing

Tests live in `tests/test_bot.py`. `tests/conftest.py` patches the required env vars so `bot.py` can be imported without a real `.env`.

When writing new tests that call `check_signals()` or other signal logic:
- Use `make_indicators(n=300)` for a full indicator DataFrame
- For swing detection tests, **set absolute prices on a flat baseline** rather than relative offsets from random data — `find_peaks` prominence depends on surrounding bars, so random drift can make injected valleys undetectable
- Always use unconditional assertions (`assert signal is not None`) in positive-path tests; conditional gates (`if signal is not None`) make tests pass trivially when logic is broken

## Project Structure

```
bot.py              # all production logic
requirements.txt
Dockerfile
tests/
  conftest.py       # env var patching for test imports
  test_bot.py       # unit tests for helpers + signal logic
docs/plans/         # design docs and implementation plans
```
