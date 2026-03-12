# Multi-Timeframe Trend Gate Design

**Date:** 2026-03-12
**Status:** Approved

## Problem

The bot currently runs Absorption and Exhaustion signals across `['1m', '5m', '30m', '1h', '4h']` independently. A 1m bullish signal can fire while the 4h trend is bearish — structurally low quality. Wyckoff methodology is designed to identify institutional accumulation/distribution over hours, not minutes. 1m/5m are noise at that scale.

## Solution

1. **Remove 1m and 5m timeframes** — wrong tool for Wyckoff/VSA analysis
2. **Add a 4h trend gate** — 30m and 1h signals only fire when their direction aligns with the 4h trend
3. **Cache 4h data** — re-fetch every 5 minutes instead of every 10 seconds

---

## Section 1: Timeframes

```python
TIMEFRAMES = ['30m', '1h', '4h']
```

1m and 5m removed entirely. Research confirms Wyckoff signals on crypto are unreliable below 30m due to noise and liquidity shuffling.

---

## Section 2: 4h Trend Gate

`check_signals` gains one optional parameter:

```python
def check_signals(df, interval, htf_trend=None):
```

At the top of the function, after computing local `trend`, add:

```python
if htf_trend and htf_trend != "NEUTRAL" and trend != htf_trend:
    return None
```

**Gate is strict:** local trend must exactly match `htf_trend`. If either is NEUTRAL, no signal fires.

The 4h timeframe passes `htf_trend=None` (it is the HTF — no gate applied).

---

## Section 3: 4h Data Caching

In `main()`, the 4h df is cached with a 5-minute TTL:

```python
htf_cache = {'df': None, 'ts': 0}

while True:
    if time.time() - htf_cache['ts'] >= 300:
        htf_cache['df'] = get_data(client, SYMBOL, '4h')
        htf_cache['ts'] = time.time()

    htf_trend = _get_trend(htf_cache['df'])

    for tf in TIMEFRAMES:
        df = get_data(client, SYMBOL, tf)
        gate = htf_trend if tf != '4h' else None
        signal = check_signals(df, tf, htf_trend=gate)
        ...
```

---

## Section 4: `_get_trend` Helper

Small pure function, extractable from existing `check_signals` logic:

```python
def _get_trend(df):
    if df is None or len(df) < 200:
        return "NEUTRAL"
    current = df.iloc[-1]
    if current['close'] > current['ema_200'] and current['ema_50'] > current['ema_200']:
        return "BULLISH"
    if current['close'] < current['ema_200'] and current['ema_50'] < current['ema_200']:
        return "BEARISH"
    return "NEUTRAL"
```

Same EMA50/EMA200 logic already in `check_signals` — no new indicators.

---

## Files to Modify

| File | Change |
|---|---|
| `bot.py` | Change `TIMEFRAMES`; add `_get_trend`; add `htf_trend` param to `check_signals`; update `main` loop with cache |
| `tests/test_bot.py` | Add tests for `_get_trend`; update `check_signals` call sites to pass `htf_trend` |

No new dependencies. No new files.

---

## Expected Outcome

- ~40–50% fewer signals
- Signals only fire when 30m/1h structure agrees with 4h trend
- No counter-trend noise from short timeframes
