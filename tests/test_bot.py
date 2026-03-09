"""Tests for bot.py signal logic."""
import pandas as pd
import numpy as np
import pytest
from bot import close_position


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
