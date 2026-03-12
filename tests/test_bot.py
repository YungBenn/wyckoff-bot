"""Tests for bot.py signal logic."""
import pandas as pd
import numpy as np
import pytest
from bot import close_position, find_swing_lows, find_swing_highs, calculate_rr

ENV = {
    'BINANCE_API_KEY': 'x', 'BINANCE_API_SECRET': 'x',
    'TELEGRAM_TOKEN': 'x', 'TELEGRAM_CHAT_ID': 'x',
}


def make_df(n=100):
    """Create a minimal OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    close = 65000 + np.cumsum(rng.standard_normal(n) * 100)
    df = pd.DataFrame({
        'open':   close + rng.standard_normal(n) * 50,
        'high':   close + abs(rng.standard_normal(n) * 150),
        'low':    close - abs(rng.standard_normal(n) * 150),
        'close':  close,
        'volume': rng.uniform(100, 1000, n),
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


def test_close_position_at_high():
    assert close_position(high=100, low=80, close=100) == pytest.approx(1.0)


def test_close_position_at_low():
    assert close_position(high=100, low=80, close=80) == pytest.approx(0.0)


def test_close_position_at_middle():
    assert close_position(high=100, low=80, close=90) == pytest.approx(0.5)


def test_close_position_zero_spread():
    assert close_position(high=100, low=100, close=100) == pytest.approx(0.5)


def test_find_swing_lows_detects_valley():
    """A clear V-shape should produce one swing low."""
    df = make_df(60)
    # Force a clear valley at index 30
    df.loc[30, 'low'] = df['low'].min() - 500
    df.loc[30, 'high'] = df.loc[30, 'low'] + 10
    df.loc[30, 'close'] = df.loc[30, 'low'] + 5
    df.loc[30, 'open'] = df.loc[30, 'low'] + 5
    df['spread'] = df['high'] - df['low']

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


def test_find_swing_highs_returns_empty_on_flat():
    """Completely flat price → no significant swing highs."""
    df = pd.DataFrame({
        'high':   [100.0] * 50,
        'low':    [100.0] * 50,
        'close':  [100.0] * 50,
        'open':   [100.0] * 50,
        'volume': [500.0] * 50,
    })
    df['spread'] = df['high'] - df['low']
    highs = find_swing_highs(df, distance=3, atr_mult=0.5)
    assert highs == []


def test_calculate_rr_bullish_structural_target():
    # T1 swing high above entry gives 2.3R → signal fires
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=63800,
        atr=500,
        t1=67900,   # (67900 - 65250) / (65000 - 63800) = 2166/1200 ≈ 1.8R
        t2=69500,
    )
    assert result is not None
    entry_mid = (65000 + 65500) / 2   # 65250
    risk = 65000 - 63800              # 1200
    t1_rr = (67900 - entry_mid) / risk
    t2_rr = (69500 - entry_mid) / risk
    assert result['entry_low']  == 65000
    assert result['entry_high'] == 65500
    assert result['stop']       == 63800
    assert result['t1']         == 67900
    assert result['t1_rr']      == pytest.approx(t1_rr)
    assert result['t2']         == 69500
    assert result['t2_rr']      == pytest.approx(t2_rr)


def test_calculate_rr_bearish_structural_target():
    result = calculate_rr(
        direction='bearish',
        lower_entry=64500,
        upper_entry=65000,
        stop=66200,
        atr=500,
        t1=63200,   # below lower_entry
        t2=61500,
    )
    assert result is not None
    entry_mid = (64500 + 65000) / 2   # 64750
    risk = 66200 - 65000              # 1200
    t1_rr = (entry_mid - 63200) / risk
    t2_rr = (entry_mid - 61500) / risk
    assert result['t1_rr'] == pytest.approx(t1_rr)
    assert result['t2_rr'] == pytest.approx(t2_rr)


def test_calculate_rr_skip_when_no_t1():
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=63800,
        atr=500,
        t1=None,
    )
    assert result is None


def test_calculate_rr_skip_when_t1_too_close():
    # T1 at 65800 → (65800 - 65250) / 1200 = 0.46R < 1.0 → skip
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=63800,
        atr=500,
        t1=65800,
    )
    assert result is None


def test_calculate_rr_fires_with_t1_only_no_t2():
    # T2 not provided → signal fires, t2 fields are None
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=63800,
        atr=500,
        t1=67900,
        t2=None,
    )
    assert result is not None
    assert result['t2']    is None
    assert result['t2_rr'] is None


def test_calculate_rr_skip_when_risk_too_wide():
    # risk = 65000 - 61000 = 4000, 3×ATR = 900 → skip
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=61000,
        atr=300,
        t1=67900,
    )
    assert result is None


def test_calculate_rr_skip_when_stop_is_none():
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=None,
        atr=500,
        t1=67900,
    )
    assert result is None


def make_indicators(n=300):
    """Full DataFrame with all indicators bot.py expects."""
    rng = np.random.default_rng(0)
    close = 65000 + np.cumsum(rng.standard_normal(n) * 100)
    df = pd.DataFrame({
        'open':   close + rng.standard_normal(n) * 30,
        'high':   close + abs(rng.standard_normal(n) * 100),
        'low':    close - abs(rng.standard_normal(n) * 100),
        'close':  close,
        'volume': rng.uniform(100, 1000, n),
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low']  = df[['open', 'close', 'low']].min(axis=1)
    df['spread']     = df['high'] - df['low']
    df['avg_spread'] = df['spread'].rolling(20).mean()
    df['vol_sma']    = df['volume'].rolling(20).mean()
    df['high_volume'] = df['volume'] > (df['vol_sma'] * 2.0)
    df['rsi']        = 50.0
    df['ema_200']    = close - 500   # price above ema_200 → bullish
    df['ema_50']     = close - 200   # ema_50 above ema_200 → bullish
    df['atr']        = df['spread'].rolling(14).mean()
    df['price_high'] = df['high'].rolling(5).max()
    df['price_low']  = df['low'].rolling(5).min()
    df['rsi_high']   = df['rsi'].rolling(5).max()
    df['rsi_low']    = df['rsi'].rolling(5).min()
    return df.ffill().bfill()


def test_bullish_absorption_must_not_fire_on_green_candle():
    """Bullish absorption must NOT fire on a green candle (old bug)."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators()
    # Force signal candle (iloc[-2]) to be a GREEN (bullish) candle
    idx = len(df) - 2
    low  = df.iloc[idx]['low']
    high = df.iloc[idx]['high']
    spread = high - low
    df.iloc[idx, df.columns.get_loc('open')]  = low + spread * 0.3
    df.iloc[idx, df.columns.get_loc('close')] = low + spread * 0.7
    df.iloc[idx, df.columns.get_loc('volume')] = df['vol_sma'].iloc[idx] * 3
    df.iloc[idx, df.columns.get_loc('high_volume')] = True
    df.iloc[idx, df.columns.get_loc('spread')] = spread * 0.4
    df.iloc[idx, df.columns.get_loc('avg_spread')] = spread
    signal = b.check_signals(df, '1h')
    # A green candle must NOT produce a bullish absorption signal
    assert signal is None or 'ABSORPTION SIGNAL (BULLISH)' not in signal


def test_bullish_absorption_can_fire_on_red_candle_in_upper_range():
    """Bullish absorption fires on a red candle and includes structural T1."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators()
    idx = len(df) - 2
    low  = df.iloc[idx]['low']
    high = df.iloc[idx]['high']
    spread = high - low

    # Red candle (close < open), close in upper 55% of range
    df.iloc[idx, df.columns.get_loc('open')]  = low + spread * 0.65
    df.iloc[idx, df.columns.get_loc('close')] = low + spread * 0.55
    df.iloc[idx, df.columns.get_loc('volume')] = df['vol_sma'].iloc[idx] * 3
    df.iloc[idx, df.columns.get_loc('high_volume')] = True
    df.iloc[idx, df.columns.get_loc('spread')] = spread * 0.3
    df.iloc[idx, df.columns.get_loc('avg_spread')] = spread

    # Inject a clear swing low for stop (20 bars back)
    lower_entry = min(df.iloc[idx]['open'], df.iloc[idx]['close'])
    swing_idx = idx - 20
    df.iloc[swing_idx, df.columns.get_loc('low')] = lower_entry - 800
    for offset in range(-4, 5):
        if offset != 0 and 0 <= swing_idx + offset < len(df):
            df.iloc[swing_idx + offset, df.columns.get_loc('low')] = lower_entry - 200

    # Inject a clear swing high for T1 (10 bars back, above entry upper bound)
    upper_entry = max(df.iloc[idx]['open'], df.iloc[idx]['close'])
    t1_idx = idx - 10
    t1_price = upper_entry + 1500   # well above entry, gives > 1.0R
    df.iloc[t1_idx, df.columns.get_loc('high')] = t1_price
    for offset in range(-4, 5):
        if offset != 0 and 0 <= t1_idx + offset < len(df):
            df.iloc[t1_idx + offset, df.columns.get_loc('high')] = upper_entry + 300

    signal = b.check_signals(df, '1h')
    assert signal is not None, "Expected a bullish absorption signal but got None"
    assert 'ABSORPTION SIGNAL (BULLISH)' in signal
    assert '📍 Entry Zone' in signal
    assert '🛑 Stop Loss'  in signal
    assert '🎯 T1'         in signal


def test_exhaustion_signal_includes_rr_levels():
    """Exhaustion signal must include entry/stop/targets when it fires."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators(300)

    baseline = 64000.0
    df['low']    = baseline
    df['high']   = baseline + 300.0
    df['open']   = baseline + 150.0
    df['close']  = baseline + 150.0
    df['spread'] = 300.0
    df['avg_spread'] = 300.0
    df['atr']    = 300.0
    df['vol_sma'] = 500.0
    df['high_volume'] = False
    df['rsi'] = 50.0

    df['ema_200'] = baseline + 650.0
    df['ema_50']  = baseline + 450.0

    idx = len(df) - 2

    df.iloc[idx, df.columns.get_loc('open')]       = baseline + 120.0
    df.iloc[idx, df.columns.get_loc('close')]      = baseline + 160.0
    df.iloc[idx, df.columns.get_loc('volume')]     = 1500.0
    df.iloc[idx, df.columns.get_loc('high_volume')] = True

    # Prior swing low (40 bars back)
    prior_idx = idx - 40
    df.iloc[prior_idx, df.columns.get_loc('low')] = 63600.0
    df.iloc[prior_idx, df.columns.get_loc('rsi')] = 30.0

    # Current swing low (20 bars back) — new lower low + RSI divergence
    current_sw_idx = idx - 20
    df.iloc[current_sw_idx, df.columns.get_loc('low')] = 63400.0
    df.iloc[current_sw_idx, df.columns.get_loc('rsi')] = 50.0

    # Structural T1: swing high above entry zone (10 bars back)
    upper_entry = max(df.iloc[idx]['open'], df.iloc[idx]['close'])  # baseline + 160
    t1_idx = idx - 10
    df.iloc[t1_idx, df.columns.get_loc('high')] = upper_entry + 1500.0  # clear target above entry
    for offset in range(-4, 5):
        if offset != 0 and 0 <= t1_idx + offset < len(df):
            df.iloc[t1_idx + offset, df.columns.get_loc('high')] = upper_entry + 300.0

    signal = b.check_signals(df, '1h')
    assert signal is not None, "Expected an exhaustion signal but got None"
    assert 'EXHAUSTION' in signal, f"Expected EXHAUSTION in signal, got: {signal}"
    assert '📍 Entry Zone' in signal
    assert '🛑 Stop Loss'  in signal
    assert '🎯 T1'         in signal


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
    df['ema_200'] = df['close'] + 1     # price below ema_200
    df['ema_50']  = df['close'] + 100   # ema_50 above ema_200 → misaligned
    assert _get_trend(df) == "NEUTRAL"


def test_get_trend_returns_neutral_on_none():
    from bot import _get_trend
    assert _get_trend(None) == "NEUTRAL"


def test_get_trend_returns_neutral_on_short_df():
    from bot import _get_trend
    df = make_indicators(50)   # only 50 rows, < 200 required
    assert _get_trend(df) == "NEUTRAL"


def test_check_signals_blocked_when_htf_trend_disagrees():
    """Signal returns None when local trend is BULLISH but htf_trend is BEARISH."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators()
    # make_indicators produces BULLISH local trend; passing BEARISH must block all signals
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
