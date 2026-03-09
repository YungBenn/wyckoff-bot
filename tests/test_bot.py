"""Tests for bot.py signal logic."""
import pandas as pd
import numpy as np
import pytest
from bot import close_position, find_swing_lows, find_swing_highs, calculate_rr


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
