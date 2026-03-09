# Signal Accuracy Improvement Design

**Date:** 2026-03-09
**Status:** Approved

## Problem

Signals fire correctly but produce poor risk/reward setups because:
1. Absorption candle direction is inverted (green candle for bullish, red for bearish — the opposite of VSA logic)
2. Close position within the candle is not checked, allowing full-bearish candles to trigger bullish signals
3. Exhaustion divergence uses an arbitrary row index (`df.iloc[-6]`) instead of actual confirmed swing points
4. No entry zone, stop loss, or targets included in the message — no way to evaluate R:R before acting

## Solution Overview

Option B: Fix signal logic (Absorption direction + close position + proper swing detection) and add entry/stop/target output to every signal message.

---

## Section 1 — Signal Logic Fix

### Absorption (critical inversion fix)

VSA principle: *"The power of a market is shown in a bearish candlestick, while weakness is shown in a bullish candlestick."* Substantial buying by smart money appears as a **bearish candle with a volume spike** — sellers are absorbed, price cannot fall.

**Bullish Absorption (was: `close > open`):**
```
close <= open                          # bearish/doji candle — smart money absorbs sellers
close_position > 0.4                   # close in upper 40%+ of range — buyers defend
high_volume                            # > 2x vol_sma
spread < avg_spread                    # narrow — effort without result
trend == BULLISH
```

**Bearish Absorption (was: `close < open`):**
```
close >= open                          # bullish/doji candle — smart money absorbs buyers
close_position < 0.6                   # close in lower 40%+ of range — sellers defend
high_volume
spread < avg_spread
trend == BEARISH
```

Where:
```python
close_position = (close - low) / spread   # 0 = close at low, 1 = close at high
```

### Exhaustion (divergence reference fix)

Direction logic is correct. Replace `df.iloc[-6]` with actual confirmed swing point RSI (see Section 2).

**Bullish Exhaustion (Selling Climax):**
```
trend == BEARISH
high_volume
price makes new swing low (scipy detection)
RSI > RSI at prior confirmed swing low    # bullish divergence
close_position > 0.3                      # close not at absolute bottom of climax candle
```

**Bearish Exhaustion (Buying Climax):**
```
trend == BULLISH
high_volume
price makes new swing high (scipy detection)
RSI < RSI at prior confirmed swing high   # bearish divergence
close_position < 0.7                      # close not at absolute top of climax candle
```

---

## Section 2 — Swing Detection Algorithm

Replace the rolling-window approximation with `scipy.signal.find_peaks`.

**Why scipy over manual loop:**
- `prominence` parameter measures structural significance, not just "lowest in N bars"
- ATR-based prominence scales automatically per timeframe
- Inverted series trick detects lows as peaks

**Parameters:**
```
distance  = 5          # minimum candles between swing points
prominence = 0.5 × ATR(14)   # structural significance filter — scales with volatility
lookback  = 50         # scan last 50 candles
```

**Stop buffer:**
```
bullish stop = swing_low  - (0.5 × ATR)
bearish stop = swing_high + (0.5 × ATR)
```
0.5× ATR (increased from a naive 0.25) accounts for BTC's tendency to spike through swing lows before reversing (Spring/Shakeout behavior).

**Divergence:** compare signal candle's RSI vs RSI value at the most recently confirmed swing low (bullish) or swing high (bearish) — replaces `df.iloc[-6]['rsi_low']`.

**Skip condition:** if no swing point found within lookback window, skip the signal entirely.

---

## Section 3 — Entry Zone, Stop & Target Calculation

### Entry Zone
Candle body of the signal candle:
```python
lower_entry = min(open, close)
upper_entry = max(open, close)
```
Produces natural "65,000 – 65,500" format in the message.

### Stop Loss
```python
bullish_stop = swing_low  - (0.5 × ATR)
bearish_stop = swing_high + (0.5 × ATR)
```

### Risk (worst-case entry)
Calculated from the edge of the entry zone closest to stop — guarantees stated R:R holds even at the worst fill:
```python
bullish_risk = lower_entry - stop
bearish_risk = stop - upper_entry
```

### Targets (from entry midpoint)
```python
entry_mid = (lower_entry + upper_entry) / 2

bullish_T1 = entry_mid + (1.5 × risk)
bullish_T2 = entry_mid + (3.0 × risk)

bearish_T1 = entry_mid - (1.5 × risk)
bearish_T2 = entry_mid - (3.0 × risk)
```

### Quality Filter (ATR-based, replaces fixed 3%)
```python
if risk > 3 × ATR:
    skip signal    # stop structurally too wide for current volatility
```
ATR-based filter is timeframe-adaptive: a 3% stop on 1m is enormous; the same 3% on 4h may be reasonable. ATR scales correctly for both.

---

## Message Format

```
🟢 ABSORPTION SIGNAL (BULLISH)
Pair: BTCUSDT | TF: 1h
Reason: High Volume Absorption (Bearish Bar, Close Mid)
Trend: BULLISH

📍 Entry Zone : 65,000 – 65,500
🛑 Stop Loss  : 63,800 (Swing Low)
🎯 T1         : 66,950 (1.5R)
🎯 T2         : 68,900 (3.0R)
📊 RSI        : 52.30
```

---

## Dependencies

Add `scipy` to `requirements.txt` and `Dockerfile`.

---

## Files to Modify

| File | Change |
|---|---|
| `bot.py` | Signal logic, swing detection, R:R calculation, message format |
| `requirements.txt` | Add `scipy` |
| `Dockerfile` | Add `scipy` (already handled by requirements.txt if pip install -r is used) |
