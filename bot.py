import time
import os
import pandas as pd
import requests
from binance.client import Client
from dotenv import load_dotenv
from scipy.signal import find_peaks

# ================= LOAD ENVIRONMENT VARIABLES =================
load_dotenv()  # Load variables from .env file

API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Check if keys are loaded
if not all([API_KEY, API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
    raise ValueError("Missing keys in .env file! Please check your configuration.")

SYMBOL = 'BTCUSDT'
TIMEFRAMES = ['1m', '5m', '30m', '1h', '4h'] 
LIMIT = 500

# ================= TELEGRAM FUNCTION =================
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ================= INDICATOR LOGIC (PURE PANDAS) =================
def calculate_rsi(series, period=14):
    """Calculates RSI using pure Pandas."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    """Calculates EMA using pure Pandas."""
    return series.ewm(span=period, adjust=False).mean()

def close_position(high, low, close):
    """Returns where close sits within the candle range. 0=at low, 1=at high, 0.5=doji."""
    spread = high - low
    if spread == 0:
        return 0.5
    return (close - low) / spread

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


def get_data(client, symbol, interval):
    """Fetches candles and computes indicators."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=LIMIT)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Indicators: RSI, EMA, Volume SMA
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['ema_200'] = calculate_ema(df['close'], 200)
        df['ema_50'] = calculate_ema(df['close'], 50)
        df['vol_sma'] = df['volume'].rolling(window=20).mean()

        # Volume Profile Logic
        df['high_volume'] = df['volume'] > (df['vol_sma'] * 2.0)

        # Spread Analysis
        df['spread'] = df['high'] - df['low']
        df['avg_spread'] = df['spread'].rolling(window=20).mean()
        df['atr'] = df['spread'].rolling(window=14).mean()

        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def check_signals(df, interval):
    """Analyzes the dataframe for Absorption and Exhaustion."""
    if df is None or len(df) < 200:
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Trend Filter
    trend = "NEUTRAL"
    if current['close'] > current['ema_200'] and current['ema_50'] > current['ema_200']:
        trend = "BULLISH"
    elif current['close'] < current['ema_200'] and current['ema_50'] < current['ema_200']:
        trend = "BEARISH"

    signal = None
    
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
        for sw_idx, price in swing_lows:
            if sw_idx < len(df) - 2 and price < lower_entry:
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
        for sw_idx, price in swing_highs:
            if sw_idx < len(df) - 2 and price > upper_entry:
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

    # --- 2. EXHAUSTION ---
    cp_prev = close_position(prev['high'], prev['low'], prev['close'])

    # Bullish Exhaustion (Selling Climax): new swing low + RSI divergence + close not at bottom
    swing_lows = find_swing_lows(df, distance=5, atr_mult=0.5)
    if (signal is None and
            trend == "BEARISH" and
            prev['high_volume'] and
            cp_prev > 0.3 and
            len(swing_lows) >= 2):

        _, current_sw_price = swing_lows[0]
        prior_sw_idx, prior_sw_price = swing_lows[1]

        prior_rsi   = df.iloc[prior_sw_idx]['rsi']
        current_rsi = prev['rsi']

        if (current_sw_price < prior_sw_price and    # new lower low
                current_rsi > prior_rsi):             # but RSI higher → bullish divergence

            lower_entry = min(prev['open'], prev['close'])
            upper_entry = max(prev['open'], prev['close'])
            stop_price  = round(current_sw_price - 0.5 * atr, 2)

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

        _, current_sw_price = swing_highs[0]
        prior_sw_idx, prior_sw_price = swing_highs[1]

        prior_rsi   = df.iloc[prior_sw_idx]['rsi']
        current_rsi = prev['rsi']

        if (current_sw_price > prior_sw_price and    # new higher high
                current_rsi < prior_rsi):             # but RSI lower → bearish divergence

            lower_entry = min(prev['open'], prev['close'])
            upper_entry = max(prev['open'], prev['close'])
            stop_price  = round(current_sw_price + 0.5 * atr, 2)

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

    return signal

# ================= MAIN LOOP =================
def main():
    client = Client(API_KEY, API_SECRET)
    print(f"Bot started for {SYMBOL} on {TIMEFRAMES}...")
    
    # Track last signal time per timeframe to avoid spamming
    # 1m needs a shorter cooldown (e.g., 1 min), others can be longer (5 min)
    last_signal_time = dict.fromkeys(TIMEFRAMES, 0)
    
    while True:
        try:
            for tf in TIMEFRAMES:
                df = get_data(client, SYMBOL, tf)
                signal = check_signals(df, tf)
                current_time = time.time()
                
                # Dynamic cooldown: 1m TF = 60s cooldown, others = 300s
                cooldown = 60 if tf == '1m' else 300
                
                if signal and (current_time - last_signal_time[tf] > cooldown):
                    print(f"Signal found on {tf}!")
                    send_telegram_message(signal)
                    last_signal_time[tf] = current_time
            
            # Check every 10 seconds for faster response on 1m timeframe
            time.sleep(10)

        except KeyboardInterrupt:
            print("Stopping bot...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()