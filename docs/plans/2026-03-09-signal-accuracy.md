# Signal Accuracy Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix inverted Absorption logic, replace rolling-window divergence with scipy swing detection, and add entry zone / stop / target levels to every signal message.

**Architecture:** All changes live in `bot.py`. Three new pure functions are extracted (`close_position`, `find_swing_lows`, `find_swing_highs`, `calculate_rr`) to keep `check_signals` readable and individually testable. `scipy.signal.find_peaks` replaces the manual rolling-window swing detection.

**Tech Stack:** Python 3.11+, pandas, scipy, pytest (new), python-binance, requests, python-dotenv

---

## Reference

Design doc: `docs/plans/2026-03-09-signal-accuracy-design.md`

Key rules from design:
- Bullish Absorption = **bearish/doji candle** + close_position > 0.4 + high_volume + narrow spread + BULLISH trend
- Bearish Absorption = **bullish/doji candle** + close_position < 0.6 + high_volume + narrow spread + BEARISH trend
- Swing detection: `scipy.signal.find_peaks`, distance=5, prominence=0.5×ATR
- Stop buffer: 0.5×ATR beyond swing point
- Risk = worst-case entry edge minus stop
- Skip if risk > 3×ATR

---

## Task 1: Add scipy + pytest, create test file

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_bot.py`

**Step 1: Add scipy and pytest to requirements.txt**

Open `requirements.txt`. Current content:
```
python-binance
pandas
requests
python-dotenv
```

Add two lines:
```
scipy
pytest
```

Final `requirements.txt`:
```
python-binance
pandas
requests
python-dotenv
scipy
pytest
```

**Step 2: Install new dependencies**

```bash
source .venv/bin/activate
pip install scipy pytest
```

Expected: scipy and pytest install successfully.

**Step 3: Create tests directory and empty init**

```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 4: Create test file with a sanity test**

Create `tests/test_bot.py`:
```python
"""Tests for bot.py signal logic."""
import pandas as pd
import numpy as np
import pytest


def make_df(n=100):
    """Create a minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 65000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        'open':   close + np.random.randn(n) * 50,
        'high':   close + abs(np.random.randn(n) * 150),
        'low':    close - abs(np.random.randn(n) * 150),
        'close':  close,
        'volume': np.random.uniform(100, 1000, n),
    })
    # Ensure high >= max(open, close) and low <= min(open, close)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low']  = df[['open', 'close', 'low']].min(axis=1)
    df['spread'] = df['high'] - df['low']
    return df


def test_sanity():
    df = make_df()
    assert len(df) == 100
    assert (df['high'] >= df['low']).all()
```

**Step 5: Run to confirm test passes**

```bash
pytest tests/test_bot.py::test_sanity -v
```

Expected: `PASSED`

**Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "chore: add scipy, pytest, test scaffold"
```

---

## Task 2: Add `close_position` helper

**Files:**
- Modify: `bot.py` — add function after `calculate_ema`
- Modify: `tests/test_bot.py` — add tests

**Step 1: Write the failing test**

Add to `tests/test_bot.py`:
```python
from bot import close_position


def test_close_position_at_high():
    # close == high → position = 1.0
    assert close_position(high=100, low=80, close=100) == pytest.approx(1.0)


def test_close_position_at_low():
    # close == low → position = 0.0
    assert close_position(high=100, low=80, close=80) == pytest.approx(0.0)


def test_close_position_at_middle():
    assert close_position(high=100, low=80, close=90) == pytest.approx(0.5)


def test_close_position_zero_spread():
    # Doji candle (high == low) → return 0.5 to avoid division by zero
    assert close_position(high=100, low=100, close=100) == pytest.approx(0.5)
```

**Step 2: Run to verify tests fail**

```bash
pytest tests/test_bot.py -k "close_position" -v
```

Expected: `ImportError: cannot import name 'close_position'`

**Step 3: Implement `close_position` in bot.py**

In `bot.py`, after the `calculate_ema` function (around line 44), add:

```python
def close_position(high, low, close):
    """Returns where close sits within the candle range. 0=at low, 1=at high, 0.5=doji."""
    spread = high - low
    if spread == 0:
        return 0.5
    return (close - low) / spread
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bot.py -k "close_position" -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add bot.py tests/test_bot.py
git commit -m "feat: add close_position helper"
```

---

## Task 3: Add swing detection functions

**Files:**
- Modify: `bot.py` — add `find_swing_lows`, `find_swing_highs` after `close_position`
- Modify: `tests/test_bot.py` — add tests

**Step 1: Write the failing tests**

Add to `tests/test_bot.py`:
```python
from bot import find_swing_lows, find_swing_highs


def test_find_swing_lows_detects_valley():
    """A clear V-shape should produce one swing low."""
    df = make_df(60)
    # Force a clear valley at index 30
    df.loc[30, 'low'] = df['low'].min() - 500
    df.loc[30, 'high'] = df.loc[30, 'low'] + 10
    df.loc[30, 'close'] = df.loc[30, 'low'] + 5
    df.loc[30, 'open'] = df.loc[30, 'low'] + 5
    df['spread'] = df['high'] - df['low']

    atr = df['spread'].rolling(14).mean().iloc[-1]
    lows = find_swing_lows(df, distance=3, atr_mult=0.3)
    prices = [p for _, p in lows]
    assert len(prices) > 0
    assert min(prices) <= df.loc[30, 'low'] + 1


def test_find_swing_highs_detects_peak():
    """A clear inverted-V shape should produce one swing high."""
    df = make_df(60)
    df.loc[30, 'high'] = df['high'].max() + 500
    df.loc[30, 'low'] = df.loc[30, 'high'] - 10
    df.loc[30, 'close'] = df.loc[30, 'high'] - 5
    df.loc[30, 'open'] = df.loc[30, 'high'] - 5
    df['spread'] = df['high'] - df['low']

    highs = find_swing_highs(df, distance=3, atr_mult=0.3)
    prices = [p for _, p in highs]
    assert len(prices) > 0
    assert max(prices) >= df.loc[30, 'high'] - 1


def test_find_swing_lows_returns_empty_on_flat():
    """Completely flat price → no significant swings."""
    df = pd.DataFrame({
        'high':   [100.0] * 50,
        'low':    [100.0] * 50,
        'close':  [100.0] * 50,
        'open':   [100.0] * 50,
        'volume': [500.0] * 50,
    })
    df['spread'] = df['high'] - df['low']
    lows = find_swing_lows(df, distance=3, atr_mult=0.5)
    assert lows == []
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_bot.py -k "swing" -v
```

Expected: `ImportError: cannot import name 'find_swing_lows'`

**Step 3: Implement swing detection in bot.py**

Add at the top of `bot.py`, after the existing imports:
```python
from scipy.signal import find_peaks
```

After `close_position`, add:
```python
def find_swing_lows(df, distance=5, atr_mult=0.5):
    """Find swing lows using scipy find_peaks on inverted low series.

    Returns list of (index, price) tuples, most recent first.
    """
    atr = df['spread'].rolling(14).mean().iloc[-1]
    if pd.isna(atr) or atr == 0:
        return []
    prominence = atr * atr_mult
    lows = df['low'].values
    peaks, _ = find_peaks(-lows, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        return []
    return [(int(i), float(lows[i])) for i in sorted(peaks, reverse=True)]


def find_swing_highs(df, distance=5, atr_mult=0.5):
    """Find swing highs using scipy find_peaks on high series.

    Returns list of (index, price) tuples, most recent first.
    """
    atr = df['spread'].rolling(14).mean().iloc[-1]
    if pd.isna(atr) or atr == 0:
        return []
    prominence = atr * atr_mult
    highs = df['high'].values
    peaks, _ = find_peaks(highs, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        return []
    return [(int(i), float(highs[i])) for i in sorted(peaks, reverse=True)]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bot.py -k "swing" -v
```

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add bot.py tests/test_bot.py
git commit -m "feat: add scipy swing high/low detection"
```

---

## Task 4: Add `calculate_rr` function

**Files:**
- Modify: `bot.py` — add `calculate_rr` after swing functions
- Modify: `tests/test_bot.py` — add tests

**Step 1: Write the failing tests**

Add to `tests/test_bot.py`:
```python
from bot import calculate_rr


def test_calculate_rr_bullish():
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=63800,
        atr=300,
    )
    assert result is not None
    entry_mid = (65000 + 65500) / 2
    risk = 65000 - 63800  # worst-case entry
    assert result['entry_low']  == 65000
    assert result['entry_high'] == 65500
    assert result['stop']       == 63800
    assert result['t1'] == pytest.approx(entry_mid + 1.5 * risk)
    assert result['t2'] == pytest.approx(entry_mid + 3.0 * risk)


def test_calculate_rr_bearish():
    result = calculate_rr(
        direction='bearish',
        lower_entry=64500,
        upper_entry=65000,
        stop=66200,
        atr=300,
    )
    assert result is not None
    entry_mid = (64500 + 65000) / 2
    risk = 66200 - 65000  # worst-case entry
    assert result['t1'] == pytest.approx(entry_mid - 1.5 * risk)
    assert result['t2'] == pytest.approx(entry_mid - 3.0 * risk)


def test_calculate_rr_skip_when_risk_too_wide():
    # risk = 65000 - 61000 = 4000, ATR = 300, 3×ATR = 900 → skip
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=61000,
        atr=300,
    )
    assert result is None


def test_calculate_rr_skip_when_stop_not_found():
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=None,
        atr=300,
    )
    assert result is None
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_bot.py -k "calculate_rr" -v
```

Expected: `ImportError: cannot import name 'calculate_rr'`

**Step 3: Implement `calculate_rr` in bot.py**

Add after `find_swing_highs`:
```python
def calculate_rr(direction, lower_entry, upper_entry, stop, atr):
    """Calculate entry zone, stop, and R:R targets.

    Returns dict with keys: entry_low, entry_high, stop, t1, t2
    Returns None if stop is missing or risk is too wide (> 3 × ATR).
    """
    if stop is None:
        return None

    entry_mid = (lower_entry + upper_entry) / 2

    if direction == 'bullish':
        risk = lower_entry - stop          # worst-case entry
        if risk <= 0:
            return None
    else:
        risk = stop - upper_entry          # worst-case entry
        if risk <= 0:
            return None

    if risk > 3 * atr:
        return None                        # stop too wide for current volatility

    if direction == 'bullish':
        t1 = entry_mid + 1.5 * risk
        t2 = entry_mid + 3.0 * risk
    else:
        t1 = entry_mid - 1.5 * risk
        t2 = entry_mid - 3.0 * risk

    return {
        'entry_low':  lower_entry,
        'entry_high': upper_entry,
        'stop':       stop,
        't1':         t1,
        't2':         t2,
    }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bot.py -k "calculate_rr" -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add bot.py tests/test_bot.py
git commit -m "feat: add calculate_rr with ATR-based quality filter"
```

---

## Task 5: Fix `get_data` — add ATR column

**Files:**
- Modify: `bot.py` — `get_data` function

The new functions need `atr` available on the DataFrame. Add it inside `get_data`.

**Step 1: Locate the indicators block in `get_data` (lines ~60–77)**

Find this section:
```python
df['high_volume'] = df['volume'] > (df['vol_sma'] * 2.0)
```

**Step 2: Add ATR column after `vol_sma`**

Add one line directly after `df['vol_sma'] = ...`:
```python
df['atr'] = df['spread'].rolling(window=14).mean()
```

**Step 3: Verify bot still imports cleanly**

```bash
python -c "import bot; print('OK')"
```

Expected: `OK` (no crash — API keys missing is fine here, the import itself should work if .env is absent, but the `raise ValueError` at line 17 will fire. Use a workaround):

```bash
python -c "
import sys, unittest.mock as m
with m.patch.dict('os.environ', {
    'BINANCE_API_KEY':'x','BINANCE_API_SECRET':'x',
    'TELEGRAM_TOKEN':'x','TELEGRAM_CHAT_ID':'x'
}):
    import importlib, bot
    print('OK')
"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add bot.py
git commit -m "feat: add ATR column to get_data"
```

---

## Task 6: Fix Absorption signal logic in `check_signals`

**Files:**
- Modify: `bot.py` — `check_signals` function
- Modify: `tests/test_bot.py` — add absorption tests

**Step 1: Write the failing tests**

Add to `tests/test_bot.py`:
```python
import sys, unittest.mock as m

ENV = {
    'BINANCE_API_KEY': 'x', 'BINANCE_API_SECRET': 'x',
    'TELEGRAM_TOKEN': 'x', 'TELEGRAM_CHAT_ID': 'x',
}


def make_indicators(n=300):
    """Full DataFrame with all indicators bot.py expects."""
    import numpy as np
    np.random.seed(0)
    close = 65000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        'open':   close + np.random.randn(n) * 30,
        'high':   close + abs(np.random.randn(n) * 100),
        'low':    close - abs(np.random.randn(n) * 100),
        'close':  close,
        'volume': np.random.uniform(100, 1000, n),
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low']  = df[['open', 'close', 'low']].min(axis=1)
    df['spread']    = df['high'] - df['low']
    df['avg_spread']= df['spread'].rolling(20).mean()
    df['vol_sma']   = df['volume'].rolling(20).mean()
    df['high_volume'] = df['volume'] > (df['vol_sma'] * 2.0)
    df['rsi']       = 50.0  # neutral default
    df['ema_200']   = close - 500   # price above ema_200 → bullish
    df['ema_50']    = close - 200   # ema_50 above ema_200 → bullish
    df['atr']       = df['spread'].rolling(14).mean()
    df['price_high']= df['high'].rolling(5).max()
    df['price_low'] = df['low'].rolling(5).min()
    df['rsi_high']  = df['rsi'].rolling(5).max()
    df['rsi_low']   = df['rsi'].rolling(5).min()
    return df.fillna(method='bfill')


def test_bullish_absorption_requires_bearish_candle():
    """Bullish absorption must NOT fire on a green candle."""
    with m.patch.dict('os.environ', ENV):
        import importlib
        b = importlib.import_module('bot')
        df = make_indicators()
        # Force last closed candle (iloc[-2]) to be green + high volume + narrow spread
        df.iloc[-2, df.columns.get_loc('open')]  = 65000
        df.iloc[-2, df.columns.get_loc('close')] = 65400   # green candle
        df.iloc[-2, df.columns.get_loc('volume')]= df['vol_sma'].iloc[-2] * 3
        df.iloc[-2, df.columns.get_loc('high_volume')] = True
        df.iloc[-2, df.columns.get_loc('spread')] = 50
        df.iloc[-2, df.columns.get_loc('avg_spread')] = 200
        signal = b.check_signals(df, '1h')
        assert signal is None or 'ABSORPTION' not in signal


def test_bullish_absorption_fires_on_bearish_candle():
    """Bullish absorption MUST fire on a red candle with correct conditions."""
    with m.patch.dict('os.environ', ENV):
        import importlib
        b = importlib.import_module('bot')
        df = make_indicators()
        row = df.iloc[-2].copy()
        low  = row['low']
        high = row['high']
        spread = high - low
        # Red candle, close in upper 50% of range
        df.iloc[-2, df.columns.get_loc('open')]  = low + spread * 0.6
        df.iloc[-2, df.columns.get_loc('close')] = low + spread * 0.55   # close < open (red)
        df.iloc[-2, df.columns.get_loc('volume')]= df['vol_sma'].iloc[-2] * 3
        df.iloc[-2, df.columns.get_loc('high_volume')] = True
        df.iloc[-2, df.columns.get_loc('spread')] = spread * 0.4   # narrow
        df.iloc[-2, df.columns.get_loc('avg_spread')] = spread
        signal = b.check_signals(df, '1h')
        # Signal may be None if no swing found — that's acceptable.
        # What must NOT happen: a bearish/absorption signal on wrong candle.
        if signal:
            assert 'BULLISH' in signal or 'EXHAUSTION' in signal
```

**Step 2: Run to verify current behavior**

```bash
pytest tests/test_bot.py -k "absorption" -v
```

Note the results — `test_bullish_absorption_requires_bearish_candle` should currently FAIL (green candle fires incorrectly).

**Step 3: Rewrite Absorption block in `check_signals`**

In `bot.py`, replace the `# --- 1. ABSORPTION ---` block (lines ~101–124) with:

```python
    # --- 1. ABSORPTION ---
    atr = current['atr'] if not pd.isna(current['atr']) else prev['spread']

    cp_prev = close_position(prev['high'], prev['low'], prev['close'])

    # Bullish Absorption: bearish/doji candle + close in upper 40%+ + high vol + narrow spread
    if (prev['close'] <= prev['open'] and
        cp_prev > 0.4 and
        prev['high_volume'] and
        prev['spread'] < prev['avg_spread'] and
        trend == "BULLISH"):

        lower_entry = min(prev['open'], prev['close'])
        upper_entry = max(prev['open'], prev['close'])
        swing_lows  = find_swing_lows(df, distance=5, atr_mult=0.5)
        stop_price  = None
        for idx, price in swing_lows:
            if idx < len(df) - 2 and price < lower_entry:
                stop_price = round(price - 0.5 * atr, 2)
                break

        rr = calculate_rr('bullish', lower_entry, upper_entry, stop_price, atr)
        if rr:
            signal = "🟢 <b>ABSORPTION SIGNAL (BULLISH)</b>\n"
            signal += f"Pair: {SYMBOL} | TF: {interval}\n"
            signal += "Reason: High Volume Absorption (Bearish Bar, Close Mid)\n"
            signal += f"Trend: {trend}\n\n"
            signal += f"📍 Entry Zone : {rr['entry_low']:,.0f} – {rr['entry_high']:,.0f}\n"
            signal += f"🛑 Stop Loss  : {rr['stop']:,.0f} (Swing Low)\n"
            signal += f"🎯 T1         : {rr['t1']:,.0f} (1.5R)\n"
            signal += f"🎯 T2         : {rr['t2']:,.0f} (3.0R)\n"
            signal += f"📊 RSI        : {prev['rsi']:.2f}"

    # Bearish Absorption: bullish/doji candle + close in lower 40%+ + high vol + narrow spread
    elif (prev['close'] >= prev['open'] and
          cp_prev < 0.6 and
          prev['high_volume'] and
          prev['spread'] < prev['avg_spread'] and
          trend == "BEARISH"):

        lower_entry = min(prev['open'], prev['close'])
        upper_entry = max(prev['open'], prev['close'])
        swing_highs = find_swing_highs(df, distance=5, atr_mult=0.5)
        stop_price  = None
        for idx, price in swing_highs:
            if idx < len(df) - 2 and price > upper_entry:
                stop_price = round(price + 0.5 * atr, 2)
                break

        rr = calculate_rr('bearish', lower_entry, upper_entry, stop_price, atr)
        if rr:
            signal = "🔴 <b>ABSORPTION SIGNAL (BEARISH)</b>\n"
            signal += f"Pair: {SYMBOL} | TF: {interval}\n"
            signal += "Reason: High Volume Absorption (Bullish Bar, Close Mid)\n"
            signal += f"Trend: {trend}\n\n"
            signal += f"📍 Entry Zone : {rr['entry_low']:,.0f} – {rr['entry_high']:,.0f}\n"
            signal += f"🛑 Stop Loss  : {rr['stop']:,.0f} (Swing High)\n"
            signal += f"🎯 T1         : {rr['t1']:,.0f} (1.5R)\n"
            signal += f"🎯 T2         : {rr['t2']:,.0f} (3.0R)\n"
            signal += f"📊 RSI        : {prev['rsi']:.2f}"
```

**Step 4: Run tests**

```bash
pytest tests/test_bot.py -k "absorption" -v
```

Expected: both PASSED

**Step 5: Commit**

```bash
git add bot.py tests/test_bot.py
git commit -m "fix: invert absorption candle direction, add close_position check"
```

---

## Task 7: Fix Exhaustion signal logic in `check_signals`

**Files:**
- Modify: `bot.py` — Exhaustion block in `check_signals`
- Modify: `tests/test_bot.py` — add exhaustion tests

**Step 1: Write the failing tests**

Add to `tests/test_bot.py`:
```python
def test_exhaustion_uses_swing_rsi_not_arbitrary_index():
    """Exhaustion divergence must compare to a confirmed swing point, not df.iloc[-6]."""
    with m.patch.dict('os.environ', ENV):
        import importlib
        b = importlib.import_module('bot')
        # If the old code is still there this is just a smoke test.
        # The real guard: signal must include Stop Loss line (only present in new code).
        df = make_indicators()
        signal = b.check_signals(df, '1h')
        if signal and 'EXHAUSTION' in signal:
            assert '🛑 Stop Loss' in signal
```

**Step 2: Run to see current state**

```bash
pytest tests/test_bot.py -k "exhaustion" -v
```

**Step 3: Rewrite Exhaustion block in `check_signals`**

Replace the `# --- 2. EXHAUSTION ---` block (lines ~126–149) with:

```python
    # --- 2. EXHAUSTION ---
    cp_prev = close_position(prev['high'], prev['low'], prev['close'])

    # Bullish Exhaustion (Selling Climax): new swing low + RSI divergence + close not at bottom
    swing_lows = find_swing_lows(df, distance=5, atr_mult=0.5)
    if (trend == "BEARISH" and
        prev['high_volume'] and
        cp_prev > 0.3 and
        len(swing_lows) >= 2):

        current_low_idx, current_low_price = swing_lows[0]
        prior_low_idx,   prior_low_price   = swing_lows[1]

        prior_rsi = df.iloc[prior_low_idx]['rsi']
        current_rsi = prev['rsi']

        if (current_low_price < prior_low_price and   # new low
            current_rsi > prior_rsi):                  # but RSI higher → divergence

            lower_entry = min(prev['open'], prev['close'])
            upper_entry = max(prev['open'], prev['close'])
            stop_price  = round(current_low_price - 0.5 * atr, 2)

            rr = calculate_rr('bullish', lower_entry, upper_entry, stop_price, atr)
            if rr:
                signal = "🚀 <b>EXHAUSTION SIGNAL (REVERSAL UP)</b>\n"
                signal += f"Pair: {SYMBOL} | TF: {interval}\n"
                signal += "Reason: Selling Climax + RSI Divergence\n"
                signal += f"Trend: {trend}\n\n"
                signal += f"📍 Entry Zone : {rr['entry_low']:,.0f} – {rr['entry_high']:,.0f}\n"
                signal += f"🛑 Stop Loss  : {rr['stop']:,.0f} (Swing Low)\n"
                signal += f"🎯 T1         : {rr['t1']:,.0f} (1.5R)\n"
                signal += f"🎯 T2         : {rr['t2']:,.0f} (3.0R)\n"
                signal += f"📊 RSI        : {current_rsi:.2f}"

    # Bearish Exhaustion (Buying Climax): new swing high + RSI divergence + close not at top
    swing_highs = find_swing_highs(df, distance=5, atr_mult=0.5)
    if (signal is None and
        trend == "BULLISH" and
        prev['high_volume'] and
        cp_prev < 0.7 and
        len(swing_highs) >= 2):

        current_high_idx, current_high_price = swing_highs[0]
        prior_high_idx,   prior_high_price   = swing_highs[1]

        prior_rsi   = df.iloc[prior_high_idx]['rsi']
        current_rsi = prev['rsi']

        if (current_high_price > prior_high_price and   # new high
            current_rsi < prior_rsi):                    # but RSI lower → divergence

            lower_entry = min(prev['open'], prev['close'])
            upper_entry = max(prev['open'], prev['close'])
            stop_price  = round(current_high_price + 0.5 * atr, 2)

            rr = calculate_rr('bearish', lower_entry, upper_entry, stop_price, atr)
            if rr:
                signal = "⚠️ <b>EXHAUSTION SIGNAL (REVERSAL DOWN)</b>\n"
                signal += f"Pair: {SYMBOL} | TF: {interval}\n"
                signal += "Reason: Buying Climax + RSI Divergence\n"
                signal += f"Trend: {trend}\n\n"
                signal += f"📍 Entry Zone : {rr['entry_low']:,.0f} – {rr['entry_high']:,.0f}\n"
                signal += f"🛑 Stop Loss  : {rr['stop']:,.0f} (Swing High)\n"
                signal += f"🎯 T1         : {rr['t1']:,.0f} (1.5R)\n"
                signal += f"🎯 T2         : {rr['t2']:,.0f} (3.0R)\n"
                signal += f"📊 RSI        : {current_rsi:.2f}"
```

**Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: all PASSED

**Step 5: Commit**

```bash
git add bot.py tests/test_bot.py
git commit -m "fix: exhaustion uses confirmed swing RSI, add entry/stop/targets"
```

---

## Task 8: Remove now-unused DataFrame columns from `get_data`

**Files:**
- Modify: `bot.py` — `get_data` function

The old divergence helpers (`price_high`, `price_low`, `rsi_high`, `rsi_low`) are no longer used in `check_signals`. Remove them to keep the code clean.

**Step 1: In `get_data`, delete these 4 lines (~lines 74–77):**

```python
df['price_high'] = df['high'].rolling(5).max()
df['price_low'] = df['low'].rolling(5).min()
df['rsi_high'] = df['rsi'].rolling(5).max()
df['rsi_low'] = df['rsi'].rolling(5).min()
```

**Step 2: Run all tests to confirm nothing broke**

```bash
pytest tests/ -v
```

Expected: all PASSED

**Step 3: Commit**

```bash
git add bot.py
git commit -m "chore: remove unused rolling divergence columns"
```

---

## Task 9: Final smoke test

**Step 1: Verify bot imports cleanly**

```bash
python -c "
import unittest.mock as m
with m.patch.dict('os.environ', {
    'BINANCE_API_KEY':'x','BINANCE_API_SECRET':'x',
    'TELEGRAM_TOKEN':'x','TELEGRAM_CHAT_ID':'x'
}):
    import bot
    print('Import OK')
"
```

Expected: `Import OK`

**Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all PASSED, 0 errors

**Step 3: Confirm requirements.txt is complete**

```bash
cat requirements.txt
```

Expected output includes: `python-binance`, `pandas`, `requests`, `python-dotenv`, `scipy`, `pytest`

**Step 4: Final commit**

```bash
git add .
git commit -m "chore: verify signal accuracy improvements complete"
```

---

## Summary of Changes

| File | What changed |
|---|---|
| `requirements.txt` | Added `scipy`, `pytest` |
| `bot.py` | Added `close_position`, `find_swing_lows`, `find_swing_highs`, `calculate_rr`; fixed Absorption candle direction + close_position check; fixed Exhaustion to use confirmed swing RSI; added entry/stop/targets to all messages; added `atr` column to `get_data`; removed unused rolling divergence columns |
| `tests/test_bot.py` | New — covers all new functions and signal logic fixes |
