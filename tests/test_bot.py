"""Tests for bot.py signal logic."""
import pandas as pd
import numpy as np
import pytest
from bot import close_position, find_swing_lows, find_swing_highs


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
