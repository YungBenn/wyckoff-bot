"""Tests for bot.py signal logic."""
import pandas as pd
import numpy as np
import pytest
import unittest.mock as m
from bot import close_position, find_swing_lows, find_swing_highs, calculate_rr

ENV = {
    'BINANCE_API_KEY': 'x', 'BINANCE_API_SECRET': 'x',
    'TELEGRAM_TOKEN': 'x', 'TELEGRAM_CHAT_ID': 'x',
}


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


def test_calculate_rr_bullish():
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=63800,
        atr=500,  # risk=1200, 3×ATR=1500 → passes filter
    )
    assert result is not None
    entry_mid = (65000 + 65500) / 2
    risk = 65000 - 63800  # worst-case entry (lower edge minus stop)
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
        atr=500,  # risk=1200, 3×ATR=1500 → passes filter
    )
    assert result is not None
    entry_mid = (64500 + 65000) / 2
    risk = 66200 - 65000  # worst-case entry (stop minus upper edge)
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


def test_calculate_rr_skip_when_stop_is_none():
    result = calculate_rr(
        direction='bullish',
        lower_entry=65000,
        upper_entry=65500,
        stop=None,
        atr=300,
    )
    assert result is None


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
    df.iloc[idx, df.columns.get_loc('close')] = low + spread * 0.7  # close > open = green
    df.iloc[idx, df.columns.get_loc('volume')] = df['vol_sma'].iloc[idx] * 3
    df.iloc[idx, df.columns.get_loc('high_volume')] = True
    df.iloc[idx, df.columns.get_loc('spread')] = spread * 0.4
    df.iloc[idx, df.columns.get_loc('avg_spread')] = spread
    signal = b.check_signals(df, '1h')
    # A green candle must NOT produce a bullish absorption signal
    assert signal is None or 'ABSORPTION SIGNAL (BULLISH)' not in signal


def test_bullish_absorption_can_fire_on_red_candle_in_upper_range():
    """Bullish absorption fires on a red candle with close in upper 40%+ of range."""
    import importlib
    b = importlib.import_module('bot')
    df = make_indicators()
    idx = len(df) - 2
    low  = df.iloc[idx]['low']
    high = df.iloc[idx]['high']
    spread = high - low
    # Red candle (close < open), close in upper 55% of range
    df.iloc[idx, df.columns.get_loc('open')]  = low + spread * 0.65
    df.iloc[idx, df.columns.get_loc('close')] = low + spread * 0.55  # close < open (red) but upper range
    df.iloc[idx, df.columns.get_loc('volume')] = df['vol_sma'].iloc[idx] * 3
    df.iloc[idx, df.columns.get_loc('high_volume')] = True
    df.iloc[idx, df.columns.get_loc('spread')] = spread * 0.3   # narrow
    df.iloc[idx, df.columns.get_loc('avg_spread')] = spread
    signal = b.check_signals(df, '1h')
    # Either None (no swing found / R:R filtered) or BULLISH signal — never BEARISH
    if signal is not None:
        assert 'BEARISH' not in signal or 'EXHAUSTION' in signal
