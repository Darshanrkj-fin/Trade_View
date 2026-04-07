"""
indicators.py - Technical Indicators Module
Includes classic indicators + advanced institutional-grade indicators:
  - Liquidity Sweep Detection
  - Order Flow Analysis
  - Anchored VWAP
  - Volume Profile
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


# ═══════════════════════════════════════════════
# CLASSIC INDICATORS
# ═══════════════════════════════════════════════

def calculate_indicators(data):
    """Calculate all technical indicators (classic + advanced)."""
    if data.empty or 'Close' not in data:
        return {}

    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # ── Classic Indicators ──
    indicators = {
        'MA50': SMAIndicator(close=close, window=50).sma_indicator().iloc[-1],
        'MA100': SMAIndicator(close=close, window=100).sma_indicator().iloc[-1],
        'MA200': SMAIndicator(close=close, window=200).sma_indicator().iloc[-1],
        'RSI': RSIIndicator(close=close, window=14).rsi().iloc[-1],
        'MACD': MACD(close=close).macd().iloc[-1],
        'Bollinger': BollingerBands(close=close).bollinger_mavg().iloc[-1],
    }

    # ── Advanced Indicators ──
    try:
        # Anchored VWAP (from start of data or last 60 bars)
        avwap = calculate_anchored_vwap(data)
        indicators['AVWAP'] = avwap['current_avwap']
        indicators['AVWAP_deviation'] = avwap['deviation_pct']
        indicators['AVWAP_signal'] = avwap['signal']
        indicators['avwap_series'] = avwap['series']
    except Exception:
        indicators['AVWAP'] = close.iloc[-1]
        indicators['AVWAP_deviation'] = 0.0
        indicators['AVWAP_signal'] = 'Neutral'
        indicators['avwap_series'] = pd.Series(dtype=float)

    try:
        # Volume Profile
        vp = calculate_volume_profile(data)
        indicators['VP_POC'] = vp['poc_price']
        indicators['VP_VAH'] = vp['vah']
        indicators['VP_VAL'] = vp['val']
        indicators['VP_signal'] = vp['signal']
        indicators['vp_profile'] = vp['profile']
    except Exception:
        indicators['VP_POC'] = close.iloc[-1]
        indicators['VP_VAH'] = close.iloc[-1]
        indicators['VP_VAL'] = close.iloc[-1]
        indicators['VP_signal'] = 'Neutral'
        indicators['vp_profile'] = pd.DataFrame()

    try:
        # Liquidity Sweep
        ls = detect_liquidity_sweeps(data)
        indicators['LS_recent_sweep'] = ls['recent_sweep']
        indicators['LS_sweep_type'] = ls['sweep_type']
        indicators['LS_sweep_level'] = ls['sweep_level']
        indicators['LS_signal'] = ls['signal']
        indicators['ls_sweeps'] = ls['all_sweeps']
    except Exception:
        indicators['LS_recent_sweep'] = False
        indicators['LS_sweep_type'] = 'None'
        indicators['LS_sweep_level'] = 0.0
        indicators['LS_signal'] = 'No Sweep'
        indicators['ls_sweeps'] = pd.DataFrame()

    try:
        # Order Flow
        of = analyze_order_flow(data)
        indicators['OF_pressure'] = of['pressure']
        indicators['OF_delta'] = of['cumulative_delta']
        indicators['OF_intensity'] = of['intensity']
        indicators['OF_signal'] = of['signal']
        indicators['of_delta_series'] = of['delta_series']
    except Exception:
        indicators['OF_pressure'] = 'Neutral'
        indicators['OF_delta'] = 0.0
        indicators['OF_intensity'] = 0.0
        indicators['OF_signal'] = 'Neutral'
        indicators['of_delta_series'] = pd.Series(dtype=float)

    return indicators


# ═══════════════════════════════════════════════
# ANCHORED VWAP
# ═══════════════════════════════════════════════

def calculate_anchored_vwap(data, anchor_bars=60):
    """
    Anchored VWAP — Volume-Weighted Average Price from an anchor point.
    Uses the last `anchor_bars` of data as the anchor period.
    
    Institutional traders use AVWAP to identify fair value and mean-reversion levels.
    Price above AVWAP = bullish bias, below = bearish.
    """
    df = data.tail(anchor_bars).copy()

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * df['Volume']).cumsum()
    cum_vol = df['Volume'].cumsum()

    avwap_series = cum_tp_vol / cum_vol
    avwap_series = avwap_series.replace([np.inf, -np.inf], np.nan).ffill()

    current_avwap = avwap_series.iloc[-1]
    current_price = df['Close'].iloc[-1]

    deviation_pct = ((current_price - current_avwap) / current_avwap) * 100

    if deviation_pct > 2:
        signal = 'Above VWAP (Bullish)'
    elif deviation_pct < -2:
        signal = 'Below VWAP (Bearish)'
    else:
        signal = 'Near VWAP (Fair Value)'

    return {
        'current_avwap': round(current_avwap, 2),
        'deviation_pct': round(deviation_pct, 2),
        'signal': signal,
        'series': avwap_series,
    }


# ═══════════════════════════════════════════════
# VOLUME PROFILE
# ═══════════════════════════════════════════════

def calculate_volume_profile(data, num_bins=30, lookback=60):
    """
    Volume Profile — distribution of volume across price levels.
    
    Key levels:
    - POC (Point of Control): Price level with the most traded volume
    - VAH (Value Area High): Upper bound of the 70% volume zone
    - VAL (Value Area Low): Lower bound of the 70% volume zone
    
    Price at POC = equilibrium. Break above VAH = bullish. Break below VAL = bearish.
    """
    df = data.tail(lookback).copy()

    price_min = df['Low'].min()
    price_max = df['High'].max()

    # Create price bins
    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Distribute volume across price bins for each bar
    vol_profile = np.zeros(num_bins)
    for _, row in df.iterrows():
        bar_low = row['Low']
        bar_high = row['High']
        bar_vol = row['Volume']

        # Find which bins this bar spans
        for j in range(num_bins):
            bin_low = bins[j]
            bin_high = bins[j + 1]

            # Overlap between bar range and bin
            overlap_low = max(bar_low, bin_low)
            overlap_high = min(bar_high, bin_high)

            if overlap_high > overlap_low:
                bar_range = max(bar_high - bar_low, 0.01)
                overlap_pct = (overlap_high - overlap_low) / bar_range
                vol_profile[j] += bar_vol * overlap_pct

    # POC = bin with max volume
    poc_idx = np.argmax(vol_profile)
    poc_price = bin_centers[poc_idx]

    # Value Area (70% of total volume around POC)
    total_vol = vol_profile.sum()
    target_vol = total_vol * 0.70

    va_vol = vol_profile[poc_idx]
    low_idx = poc_idx
    high_idx = poc_idx

    while va_vol < target_vol and (low_idx > 0 or high_idx < num_bins - 1):
        expand_low = vol_profile[low_idx - 1] if low_idx > 0 else 0
        expand_high = vol_profile[high_idx + 1] if high_idx < num_bins - 1 else 0

        if expand_low >= expand_high and low_idx > 0:
            low_idx -= 1
            va_vol += vol_profile[low_idx]
        elif high_idx < num_bins - 1:
            high_idx += 1
            va_vol += vol_profile[high_idx]
        else:
            low_idx -= 1
            va_vol += vol_profile[low_idx]

    vah = bin_centers[high_idx]
    val_price = bin_centers[low_idx]

    current_price = df['Close'].iloc[-1]
    if current_price > vah:
        signal = 'Above Value Area (Breakout)'
    elif current_price < val_price:
        signal = 'Below Value Area (Breakdown)'
    elif abs(current_price - poc_price) / poc_price < 0.01:
        signal = 'At POC (Equilibrium)'
    else:
        signal = 'Inside Value Area'

    profile_df = pd.DataFrame({
        'price': bin_centers,
        'volume': vol_profile
    })

    return {
        'poc_price': round(poc_price, 2),
        'vah': round(vah, 2),
        'val': round(val_price, 2),
        'signal': signal,
        'profile': profile_df,
    }


# ═══════════════════════════════════════════════
# LIQUIDITY SWEEP DETECTION
# ═══════════════════════════════════════════════

def detect_liquidity_sweeps(data, lookback=20, sweep_window=5):
    """
    Liquidity Sweep — detects when price sweeps past a recent swing high/low
    and then reverses, indicating stop-hunt / liquidity grab by smart money.
    
    Bullish sweep: Price dips below recent swing low then reverses up (buy signal)
    Bearish sweep: Price spikes above recent swing high then reverses down (sell signal)
    """
    df = data.copy()
    sweeps = []

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    # Identify swing highs and swing lows
    for i in range(lookback, len(df) - sweep_window):
        # Swing High: highest point in lookback window
        lookback_high = np.max(high[i - lookback:i])
        lookback_high_idx = i - lookback + np.argmax(high[i - lookback:i])

        # Swing Low: lowest point in lookback window
        lookback_low = np.min(low[i - lookback:i])
        lookback_low_idx = i - lookback + np.argmin(low[i - lookback:i])

        # Bearish Liquidity Sweep: price exceeds swing high then reverses down
        if high[i] > lookback_high and close[i] < lookback_high:
            # Check if price reversed in the next few bars
            future_closes = close[i + 1:i + sweep_window + 1] if i + 1 < len(close) else []
            if len(future_closes) > 0 and np.mean(future_closes) < close[i]:
                sweeps.append({
                    'date': df.index[i],
                    'type': 'Bearish Sweep',
                    'level': round(lookback_high, 2),
                    'price': round(high[i], 2),
                })

        # Bullish Liquidity Sweep: price dips below swing low then reverses up
        if low[i] < lookback_low and close[i] > lookback_low:
            future_closes = close[i + 1:i + sweep_window + 1] if i + 1 < len(close) else []
            if len(future_closes) > 0 and np.mean(future_closes) > close[i]:
                sweeps.append({
                    'date': df.index[i],
                    'type': 'Bullish Sweep',
                    'level': round(lookback_low, 2),
                    'price': round(low[i], 2),
                })

    sweeps_df = pd.DataFrame(sweeps) if sweeps else pd.DataFrame(columns=['date', 'type', 'level', 'price'])

    # Check last 5 bars for a recent sweep
    recent_sweep = False
    sweep_type = 'None'
    sweep_level = 0.0
    signal = 'No Sweep'

    if not sweeps_df.empty:
        last_5_dates = df.index[-5:]
        recent = sweeps_df[sweeps_df['date'].isin(last_5_dates)]
        if not recent.empty:
            last_sweep = recent.iloc[-1]
            recent_sweep = True
            sweep_type = last_sweep['type']
            sweep_level = last_sweep['level']
            if 'Bullish' in sweep_type:
                signal = f'Bullish Sweep at ₹{sweep_level} (Buy Signal)'
            else:
                signal = f'Bearish Sweep at ₹{sweep_level} (Sell Signal)'

    return {
        'recent_sweep': recent_sweep,
        'sweep_type': sweep_type,
        'sweep_level': sweep_level,
        'signal': signal,
        'all_sweeps': sweeps_df,
    }


# ═══════════════════════════════════════════════
# ORDER FLOW ANALYSIS
# ═══════════════════════════════════════════════

def analyze_order_flow(data, lookback=30):
    """
    Order Flow Analysis — estimates buying vs selling pressure using
    price action and volume.
    
    Uses the Close-to-Range ratio to estimate what % of each bar's volume
    was buying vs selling (approximation without tick data):
      Buy Volume  = Volume × (Close - Low) / (High - Low)
      Sell Volume = Volume × (High - Close) / (High - Low)
      Delta       = Buy Volume - Sell Volume
    
    Cumulative Delta rising = buying pressure dominant
    Cumulative Delta falling = selling pressure dominant
    """
    df = data.tail(lookback).copy()

    bar_range = df['High'] - df['Low']
    bar_range = bar_range.replace(0, 0.01)  # Avoid division by zero

    buy_ratio = (df['Close'] - df['Low']) / bar_range
    sell_ratio = (df['High'] - df['Close']) / bar_range

    buy_volume = df['Volume'] * buy_ratio
    sell_volume = df['Volume'] * sell_ratio

    delta = buy_volume - sell_volume
    cumulative_delta = delta.cumsum()

    # Buying/selling intensity (last 5 bars vs previous 25)
    recent_delta = delta.tail(5).sum()
    total_delta = delta.sum()
    recent_vol = df['Volume'].tail(5).sum()
    total_vol = df['Volume'].sum()

    # Intensity: recent order flow relative to average
    avg_delta_per_bar = total_delta / len(delta) if len(delta) > 0 else 0
    recent_avg = recent_delta / 5 if recent_delta != 0 else 0
    intensity = (recent_avg / abs(avg_delta_per_bar)) if avg_delta_per_bar != 0 else 0
    intensity = round(min(max(intensity, -3), 3), 2)  # Clamp to [-3, 3]

    # Determine pressure direction
    cum_delta_val = cumulative_delta.iloc[-1]
    if cum_delta_val > 0 and recent_delta > 0:
        pressure = 'Strong Buying'
        signal = 'Buyers in control (Bullish)'
    elif cum_delta_val > 0 and recent_delta <= 0:
        pressure = 'Buying Weakening'
        signal = 'Buying fading, watch for reversal'
    elif cum_delta_val < 0 and recent_delta < 0:
        pressure = 'Strong Selling'
        signal = 'Sellers in control (Bearish)'
    elif cum_delta_val < 0 and recent_delta >= 0:
        pressure = 'Selling Weakening'
        signal = 'Selling fading, possible reversal up'
    else:
        pressure = 'Neutral'
        signal = 'Balanced order flow'

    return {
        'pressure': pressure,
        'cumulative_delta': round(cum_delta_val, 0),
        'intensity': intensity,
        'signal': signal,
        'delta_series': cumulative_delta,
    }