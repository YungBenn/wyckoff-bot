# Structural Targets Design

**Date:** 2026-03-12
**Status:** Approved

## Problem

Current T1 and T2 targets are fixed R:R multiples (1.5R and 3R) — synthetic numbers derived from math, not market structure. A trader receiving the signal cannot verify these levels on their chart. Wyckoff methodology is inherently structural: targets should come from levels the market has already validated.

## Solution

Replace fixed R:R targets with structural swing levels detected by the existing `find_swing_highs` / `find_swing_lows` functions. The signal only fires when a real structural target exists above (bullish) or below (bearish) the entry zone at a minimum 1.0R distance.

---

## Section 1: How Structural Targets Are Found

**Bullish signals:**
- Use `find_swing_highs` filtered to swings **above `upper_entry`**
- Sort ascending by price
- T1 = closest swing high above entry
- T2 = next swing high above T1 (omitted from message if not found)

**Bearish signals:**
- Use `find_swing_lows` filtered to swings **below `lower_entry`**
- Sort descending by price
- T1 = closest swing low below entry
- T2 = next swing low below T1 (omitted if not found)

**Skip conditions:**
- No swing found above/below entry → skip signal
- T1 gives < 1.0R → skip signal (structure too close)

**Message format:**
```
🎯 T1 : 66,500 (2.3R) — Swing High
🎯 T2 : 68,200 (4.1R) — Swing High   ← omitted if no T2 found
```

---

## Section 2: Changes to `calculate_rr`

**Revised signature:**
```python
calculate_rr(direction, lower_entry, upper_entry, stop, atr, t1=None, t2=None)
```

**Logic:**
```
if t1 is None → return None
entry_mid = (lower_entry + upper_entry) / 2
risk = lower_entry - stop           (bullish)
     = stop - upper_entry           (bearish)
if risk > 3 × ATR → return None    (stop too wide, unchanged)
t1_rr = (t1 - entry_mid) / risk    (bullish)
       = (entry_mid - t1) / risk   (bearish)
if t1_rr < 1.0 → return None      (structure too close)
t2_rr = computed if t2 provided, else None
```

**Return dict:**
```python
{
    'entry_low':  lower_entry,
    'entry_high': upper_entry,
    'stop':       stop,
    't1':         t1,
    't1_rr':      t1_rr,
    't2':         t2,         # None if not found
    't2_rr':      t2_rr,      # None if not found
}
```

**Edge cases:**

| Case | Handling |
|---|---|
| Swing high found but below entry | Skip, look for next one |
| T1 and T2 same index | Use only T1 |
| Swing detection returns empty | t1=None → no signal |
| No T2 found | Signal fires with T1 only |

---

## Section 3: Changes in `check_signals`

Before calling `calculate_rr`, `check_signals` must:

1. Find swing highs/lows for stop (unchanged)
2. Find structural targets:
   ```python
   # Bullish: find swing highs above upper_entry
   targets = [(idx, p) for idx, p in find_swing_highs(df)
              if idx < len(df) - 2 and p > upper_entry]
   targets.sort(key=lambda x: x[1])   # ascending by price
   t1 = targets[0][1] if len(targets) >= 1 else None
   t2 = targets[1][1] if len(targets) >= 2 else None
   ```
3. Pass `t1` and `t2` into `calculate_rr`

---

## Files to Modify

| File | Change |
|---|---|
| `bot.py` | Revise `calculate_rr` signature + logic; update Absorption + Exhaustion blocks in `check_signals` to find structural targets and pass them in |
| `tests/test_bot.py` | Update `calculate_rr` tests; update signal tests to inject structural targets |

No new dependencies. No new files.
