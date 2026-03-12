# Multi-Timeframe Trend Gate Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove noisy 1m/5m timeframes and add a 4h trend gate so 30m/1h signals only fire when the 4h trend agrees.

**Architecture:** A new pure helper `_get_trend(df)` extracts the trend string from any dataframe. `check_signals` gains an optional `htf_trend` param — if it disagrees with the local trend, the function returns None immediately. The main loop caches 4h data (5-min TTL) and passes the gate to 30m/1h signal checks.

**Tech Stack:** Python, pandas (already installed), pytest

---

## Reference

Design doc: `docs/plans/2026-03-12-mtf-trend-gate-design.md`

Key rules:
- `TIMEFRAMES = ['30m', '1h', '4h']` — 1m and 5m removed
- Gate is strict: local trend must exactly match `htf_trend`
- NEUTRAL on either side → no signal
- 4h passes `htf_trend=None` — no gate applied to itself
- 4h data cached with 5-minute TTL in `main()`

---

## Chunk 1: `_get_trend` helper

### Task 1: Add `_get_trend` pure helper + tests

**Files:**
- Modify: `bot.py` — add `_get_trend` function after `find_swing_highs`
- Modify: `tests/test_bot.py` — add `_get_trend` tests

---

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_bot.py` (after the existing `find_swing_highs` tests):

```python
def test_get_trend_bullish():
    from bot import _get_trend
    df = make_indicators()
    # make_indicators sets ema_200 = close - 500, ema_50 = close - 200
    # → close > ema_200 and ema_50 > ema_200 → BULLISH
    assert _get_trend(df) == "BULLISH"


def test_get_trend_bearish():
    from bot import _get_trend
    df = make_indicators()
    df['ema_200'] = df['close'] + 500   # price below ema_200
    df['ema_50']  = df['close'] + 200   # ema_50 below ema_200
    assert _get_trend(df) == "BEARISH"


def test_get_trend_neutral():
    from bot import _get_trend
    df = make_indicators()
    df['ema_200'] = df['close'] + 1     # price just below ema_200
    df['ema_50']  = df['close'] - 100
    assert _get_trend(df) == "NEUTRAL"


def test_get_trend_returns_neutral_on_none():
    from bot import _get_trend
    assert _get_trend(None) == "NEUTRAL"


def test_get_trend_returns_neutral_on_short_df():
    from bot import _get_trend
    df = make_indicators(50)   # only 50 rows, < 200 required
    assert _get_trend(df) == "NEUTRAL"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && pytest tests/test_bot.py -k "get_trend" -v
```

Expected: 5 FAILED with `ImportError: cannot import name '_get_trend'`

- [ ] **Step 3: Add `_get_trend` to bot.py**

Insert this function in `bot.py` immediately after the `find_swing_highs` function (before `calculate_rr`):

```python
def _get_trend(df):
    """Extract trend direction from a dataframe using EMA50/EMA200.

    Returns 'BULLISH', 'BEARISH', or 'NEUTRAL'.
    Used by main() to get the 4h HTF trend for gating lower timeframe signals.
    """
    if df is None or len(df) < 200:
        return "NEUTRAL"
    current = df.iloc[-1]
    if current['close'] > current['ema_200'] and current['ema_50'] > current['ema_200']:
        return "BULLISH"
    if current['close'] < current['ema_200'] and current['ema_50'] < current['ema_200']:
        return "BEARISH"
    return "NEUTRAL"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && pytest tests/test_bot.py -k "get_trend" -v
```

Expected: 5 PASSED

- [ ] **Step 5: Run full suite to catch regressions**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && git add bot.py tests/test_bot.py && git commit -m "feat: add _get_trend helper for HTF trend extraction"
```

---

## Chunk 2: `htf_trend` gate in `check_signals`

### Task 2: Add `htf_trend` parameter to `check_signals` + tests

**Files:**
- Modify: `bot.py` — `check_signals` signature + gate logic
- Modify: `tests/test_bot.py` — add gate tests

---

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_bot.py`:

```python
def test_check_signals_blocked_when_htf_trend_disagrees():
    """Signal returns None when local trend is BULLISH but htf_trend is BEARISH."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators()
    # make_indicators produces a BULLISH local trend (ema setup)
    # Passing htf_trend='BEARISH' must block all signals regardless
    signal = b.check_signals(df, '30m', htf_trend='BEARISH')
    assert signal is None


def test_check_signals_not_blocked_when_htf_trend_agrees():
    """Gate passes when htf_trend matches local trend — same result as no gate."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators()
    signal_gated   = b.check_signals(df, '30m', htf_trend='BULLISH')
    signal_ungated = b.check_signals(df, '30m', htf_trend=None)
    assert signal_gated == signal_ungated


def test_check_signals_no_gate_when_htf_trend_is_none():
    """htf_trend=None means no gate — used for 4h itself."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators()
    signal = b.check_signals(df, '4h', htf_trend=None)
    assert signal is None or isinstance(signal, str)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && pytest tests/test_bot.py -k "htf_trend or blocked or gate" -v
```

Expected: failures — `check_signals` doesn't accept `htf_trend` yet

- [ ] **Step 3: Update `check_signals` signature and add gate**

In `bot.py`, change the function signature from:
```python
def check_signals(df, interval):
```
to:
```python
def check_signals(df, interval, htf_trend=None):
```

Then, immediately after the local `trend` is computed (after the `elif` block that sets `trend = "BEARISH"`), add the gate:

```python
    # HTF trend gate: if a higher-timeframe trend is provided, skip signals
    # that go against it. NEUTRAL on either side = no signal.
    if htf_trend and htf_trend != "NEUTRAL" and trend != htf_trend:
        return None
```

The full trend block should look like this after the change:

```python
    # Trend Filter
    trend = "NEUTRAL"
    if current['close'] > current['ema_200'] and current['ema_50'] > current['ema_200']:
        trend = "BULLISH"
    elif current['close'] < current['ema_200'] and current['ema_50'] < current['ema_200']:
        trend = "BEARISH"

    # HTF trend gate: if a higher-timeframe trend is provided, skip signals
    # that go against it. NEUTRAL on either side = no signal.
    if htf_trend and htf_trend != "NEUTRAL" and trend != htf_trend:
        return None
```

- [ ] **Step 4: Run gate tests**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && pytest tests/test_bot.py -k "htf_trend or blocked or gate" -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run full suite**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && pytest tests/ -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && git add bot.py tests/test_bot.py && git commit -m "feat: check_signals accepts htf_trend gate parameter"
```

---

## Chunk 3: Update `main` loop

### Task 3: Change TIMEFRAMES, add 4h cache, pass htf_trend

**Depends on:** Task 1 (`_get_trend` must exist in `bot.py` before this task runs)

**Files:**
- Modify: `bot.py` — `TIMEFRAMES` constant, `main()` function

No new tests — `main()` is integration code that depends on a live Binance client.

---

- [ ] **Step 1: Update `TIMEFRAMES` constant**

In `bot.py`, change line 22 from:
```python
TIMEFRAMES = ['1m', '5m', '30m', '1h', '4h']
```
to:
```python
TIMEFRAMES = ['30m', '1h', '4h']
```

- [ ] **Step 2: Replace `main()` body**

Replace the entire `main()` function with:

```python
def main():
    client = Client(API_KEY, API_SECRET)
    print(f"Bot started for {SYMBOL} on {TIMEFRAMES}...")

    last_signal_time = dict.fromkeys(TIMEFRAMES, 0)

    # 4h data cache — re-fetched every 5 minutes
    htf_cache = {'df': None, 'ts': 0}

    while True:
        try:
            # Refresh 4h HTF context if cache is stale
            if time.time() - htf_cache['ts'] >= 300:
                htf_cache['df'] = get_data(client, SYMBOL, '4h')
                htf_cache['ts'] = time.time()

            htf_trend = _get_trend(htf_cache['df'])

            for tf in TIMEFRAMES:
                df = get_data(client, SYMBOL, tf)
                # 4h has no gate (it IS the HTF); 30m and 1h are gated by 4h trend
                gate = None if tf == '4h' else htf_trend
                signal = check_signals(df, tf, htf_trend=gate)
                current_time = time.time()

                if signal and (current_time - last_signal_time[tf] > 300):
                    print(f"Signal found on {tf}!")
                    send_telegram_message(signal)
                    last_signal_time[tf] = current_time

            time.sleep(10)

        except KeyboardInterrupt:
            print("Stopping bot...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)
```

Note: cooldown is now a flat 300s for all timeframes (1m is gone, its special 60s case removed).

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && pytest tests/ -v --tb=short
```

Expected: all PASSED

- [ ] **Step 4: Verify clean import**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && source .venv/bin/activate && python -c "import tests.conftest; import bot; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 5: Commit**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && git add bot.py && git commit -m "feat: drop 1m/5m, add 4h trend gate with cache to main loop"
```

- [ ] **Step 6: Push**

```bash
cd /Users/rubenadisuryo/Developer/project/wyckoff_bot && git push
```

---

## Summary of Changes

| File | What changes |
|---|---|
| `bot.py` — `TIMEFRAMES` | `['1m', '5m', '30m', '1h', '4h']` → `['30m', '1h', '4h']` |
| `bot.py` — `_get_trend` | New pure helper, extracts BULLISH/BEARISH/NEUTRAL from any df |
| `bot.py` — `check_signals` | New `htf_trend=None` param; gate at top of trend block |
| `bot.py` — `main` | 4h cache (5-min TTL); passes `htf_trend` gate per TF; flat 300s cooldown |
| `tests/test_bot.py` | 5 tests for `_get_trend`; 3 tests for `htf_trend` gate |
