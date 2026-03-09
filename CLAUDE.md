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

# Docker
docker build -t wyckoff_bot .
docker run --env-file .env wyckoff_bot
```

## Architecture

All logic lives in `bot.py`. The main loop runs every 10 seconds and iterates over `TIMEFRAMES = ['1m', '5m', '30m', '1h', '4h']`.

**Data flow:** `get_data()` → fetches 500 Binance klines, computes RSI/EMA50/EMA200/VolumeSMA → `check_signals()` → `send_telegram_message()`

**Signal logic in `check_signals()`:**
- **Absorption**: High volume + narrow spread candle, aligned with EMA trend (bullish if above EMA200 with EMA50 > EMA200, bearish if below)
- **Exhaustion**: Price makes new high/low while RSI diverges (Selling/Buying Climax), requires high volume

**Cooldown**: 60s for `1m` TF, 300s for all others (tracked via `last_signal_time` dict).

## Dependencies

- `python-binance` — Binance REST API client
- `pandas` — all indicator calculations (RSI, EMA, rolling stats)
- `requests` — Telegram HTTP calls
- `python-dotenv` — `.env` loading

The `.venv` uses Python 3.14; Dockerfile uses Python 3.11-slim for deployment.
