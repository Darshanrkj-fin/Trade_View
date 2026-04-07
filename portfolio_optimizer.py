"""
portfolio_optimizer.py - Advanced Portfolio Optimization Module
Implements a 4-step quantitative finance pipeline:
  1. Momentum Ranking: Heuristic scoring for stock selection
  2. Markowitz MVO: Mean-Variance Optimization for weights
  3. CAPM: Capital Asset Pricing Model for expected returns vs risk
  4. Sharpe Ratio: Risk-adjusted performance measurement
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data import fetch_stock_data
import logging

logger = logging.getLogger(__name__)

# TensorFlow is optional. Portfolio ranking currently uses a heuristic score.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Portfolio ranking will use the heuristic momentum scorer.")


# ═══════════════════════════════════════════════
# 1. MOMENTUM RANKING
# ═══════════════════════════════════════════════

def extract_features(data):
    """Extract momentum and volatility features for ranking."""
    if len(data) < 50:
        return np.zeros(5)
        
    close = data['Close']
    returns = close.pct_change().dropna()
    
    # Features: 1M Return, 3M Return, Volatility, Price/MA50 ratio, Max drawdown
    ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0
    ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else 0
    volatility = returns.tail(63).std() * np.sqrt(252) if len(returns) > 63 else 0
    ma50 = close.tail(50).mean()
    p_ma50 = (close.iloc[-1] / ma50 - 1) if ma50 > 0 else 0
    rolling_peak = close.cummax()
    drawdowns = ((close / rolling_peak) - 1).fillna(0)
    max_drawdown = abs(drawdowns.min())

    return np.array([ret_1m, ret_3m, volatility, p_ma50, max_drawdown])

def heuristic_rank(features):
    """Risk-aware heuristic ranking score."""
    ret_1m, ret_3m, vol, p_ma50, drawdown = features

    # Balance medium-term trend and recent strength while penalizing risk more meaningfully.
    score = (
        (ret_1m * 0.20)
        + (ret_3m * 0.25)
        + (p_ma50 * 0.20)
        - (vol * 0.15)
        - (drawdown * 0.20)
    )
    return score


def legacy_heuristic_rank(features):
    """Previous momentum-heavy heuristic retained for ranking comparisons."""
    ret_1m, ret_3m, vol, p_ma50, _drawdown = features
    return (ret_1m * 0.4) + (ret_3m * 0.3) + (p_ma50 * 0.4) - (vol * 0.1)


def explain_feature_profile(features):
    """Return user-facing context for the heuristic momentum score."""
    ret_1m, ret_3m, vol, p_ma50, drawdown = [float(value) for value in features]
    summary = []
    if ret_1m > 0:
        summary.append("positive 1M momentum")
    if ret_3m > 0:
        summary.append("positive 3M momentum")
    if p_ma50 > 0:
        summary.append("trading above 50D average")
    if vol > 0.35:
        summary.append("elevated volatility")
    if drawdown > 0.20:
        summary.append("deep recent drawdown")
    if not summary:
        summary.append("mixed momentum profile")

    return {
        "features": {
            "return_1m": round(ret_1m, 4),
            "return_3m": round(ret_3m, 4),
            "volatility": round(vol, 4),
            "price_vs_ma50": round(p_ma50, 4),
            "max_drawdown": round(drawdown, 4),
        },
        "summary": ", ".join(summary),
    }

def get_ranknet_scores(tickers, historical_data_dict):
    """Score each stock using the current heuristic ranking model."""
    scores = {}
    feature_context = {}
    
    if TF_AVAILABLE and len(tickers) > 1:
        # Placeholder for a future learned scoring network.
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(5,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        # We intentionally keep the heuristic path active because this network is
        # untrained and would otherwise introduce random noise.
        pass
        
    # Apply heuristic scoring (simulating RankNet output)
    for ticker in tickers:
        df = historical_data_dict.get(ticker)
        if df is not None and not df.empty:
            feats = extract_features(df)
            scores[ticker] = heuristic_rank(feats)
            feature_context[ticker] = explain_feature_profile(feats)
        else:
            scores[ticker] = -999
            feature_context[ticker] = {
                "features": {
                "return_1m": 0.0,
                "return_3m": 0.0,
                "volatility": 0.0,
                "price_vs_ma50": 0.0,
                "max_drawdown": 0.0,
                },
                "summary": "insufficient historical data",
            }
            
    # Normalize scores between 0 and 1
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score > min_score:
        for t in scores:
            scores[t] = (scores[t] - min_score) / (max_score - min_score)
    else:
        for t in scores:
            scores[t] = 0.5
            
    return scores, feature_context


def build_ranking_benchmark(tickers, historical_data_dict, current_scores):
    """Compare the current ranking heuristic against the prior momentum-heavy formula."""
    legacy_raw = {}
    current_raw = {}

    for ticker in tickers:
        df = historical_data_dict.get(ticker)
        if df is None or df.empty:
            legacy_raw[ticker] = -999.0
            current_raw[ticker] = -999.0
            continue
        feats = extract_features(df)
        legacy_raw[ticker] = legacy_heuristic_rank(feats)
        current_raw[ticker] = heuristic_rank(feats)

    def normalize(values):
        min_score = min(values.values())
        max_score = max(values.values())
        if max_score > min_score:
            return {k: round((v - min_score) / (max_score - min_score), 4) for k, v in values.items()}
        return {k: 0.5 for k in values}

    legacy_scores = normalize(legacy_raw)
    rank_shift = {
        ticker: round(current_scores.get(ticker, 0.0) - legacy_scores.get(ticker, 0.0), 4)
        for ticker in tickers
    }
    most_improved = max(rank_shift, key=rank_shift.get) if rank_shift else None
    most_penalized = min(rank_shift, key=rank_shift.get) if rank_shift else None

    return {
        "current_scores": {ticker: round(current_scores.get(ticker, 0.0), 4) for ticker in tickers},
        "legacy_scores": legacy_scores,
        "rank_shift": rank_shift,
        "most_improved_asset": most_improved,
        "most_penalized_asset": most_penalized,
        "summary": (
            "Current ranking adds stronger penalties for volatility and drawdown."
            if tickers else
            "No ranking comparison available."
        ),
    }


def build_ranking_validation(tickers, historical_data_dict, lookback_days=126, forward_days=21, max_windows=6):
    """Compare old vs new ranking by realized forward returns on rolling windows."""
    valid_series = {
        ticker: historical_data_dict[ticker]["Close"].dropna().reset_index(drop=True)
        for ticker in tickers
        if ticker in historical_data_dict and historical_data_dict[ticker] is not None and not historical_data_dict[ticker].empty
    }
    if not valid_series:
        return {
            "windows_evaluated": 0,
            "current_avg_forward_return": None,
            "legacy_avg_forward_return": None,
            "current_forward_sharpe": None,
            "legacy_forward_sharpe": None,
            "current_positive_rate": None,
            "legacy_positive_rate": None,
            "win_rate": None,
            "outperformance_spread": None,
            "better_method": "insufficient_data",
            "summary": "Insufficient historical data for ranking validation.",
        }

    min_len = min(len(series) for series in valid_series.values())
    if min_len < lookback_days + forward_days:
        return {
            "windows_evaluated": 0,
            "current_avg_forward_return": None,
            "legacy_avg_forward_return": None,
            "current_forward_sharpe": None,
            "legacy_forward_sharpe": None,
            "current_positive_rate": None,
            "legacy_positive_rate": None,
            "win_rate": None,
            "outperformance_spread": None,
            "better_method": "insufficient_data",
            "summary": "Insufficient historical data for ranking validation.",
        }

    current_returns = []
    legacy_returns = []
    wins = 0
    windows = 0

    for end_idx in range(lookback_days, min_len - forward_days + 1, forward_days):
        if windows >= max_windows:
            break

        current_best = None
        legacy_best = None
        current_best_score = -np.inf
        legacy_best_score = -np.inf

        for ticker, series in valid_series.items():
            history_slice = series.iloc[:end_idx]
            future_slice = series.iloc[end_idx:end_idx + forward_days]
            if len(history_slice) < 50 or len(future_slice) < forward_days:
                continue

            feature_frame = pd.DataFrame({"Close": history_slice})
            features = extract_features(feature_frame)
            current_score = heuristic_rank(features)
            legacy_score = legacy_heuristic_rank(features)
            future_return = float((future_slice.iloc[-1] / future_slice.iloc[0]) - 1)

            if current_score > current_best_score:
                current_best_score = current_score
                current_best = future_return
            if legacy_score > legacy_best_score:
                legacy_best_score = legacy_score
                legacy_best = future_return

        if current_best is None or legacy_best is None:
            continue

        current_returns.append(current_best)
        legacy_returns.append(legacy_best)
        if current_best > legacy_best:
            wins += 1
        windows += 1

    if not windows:
        return {
            "windows_evaluated": 0,
            "current_avg_forward_return": None,
            "legacy_avg_forward_return": None,
            "current_forward_sharpe": None,
            "legacy_forward_sharpe": None,
            "current_positive_rate": None,
            "legacy_positive_rate": None,
            "win_rate": None,
            "outperformance_spread": None,
            "better_method": "insufficient_data",
            "summary": "Insufficient historical data for ranking validation.",
        }

    current_avg = float(np.mean(current_returns) * 100)
    legacy_avg = float(np.mean(legacy_returns) * 100)
    win_rate = float((wins / windows) * 100)
    current_std = float(np.std(current_returns, ddof=0))
    legacy_std = float(np.std(legacy_returns, ddof=0))
    current_sharpe = (np.mean(current_returns) / current_std) if current_std > 0 else 0.0
    legacy_sharpe = (np.mean(legacy_returns) / legacy_std) if legacy_std > 0 else 0.0
    current_positive_rate = float((sum(1 for value in current_returns if value > 0) / windows) * 100)
    legacy_positive_rate = float((sum(1 for value in legacy_returns if value > 0) / windows) * 100)
    outperformance_spread = current_avg - legacy_avg
    better_method = "current" if current_avg >= legacy_avg else "legacy"

    if better_method == "current":
        summary = "Current heuristic delivered stronger forward picks than the legacy ranking."
    else:
        summary = "Legacy heuristic delivered stronger forward picks than the current ranking."

    return {
        "windows_evaluated": windows,
        "current_avg_forward_return": round(current_avg, 2),
        "legacy_avg_forward_return": round(legacy_avg, 2),
        "current_forward_sharpe": round(float(current_sharpe), 2),
        "legacy_forward_sharpe": round(float(legacy_sharpe), 2),
        "current_positive_rate": round(current_positive_rate, 2),
        "legacy_positive_rate": round(legacy_positive_rate, 2),
        "win_rate": round(win_rate, 2),
        "outperformance_spread": round(outperformance_spread, 2),
        "better_method": better_method,
        "summary": summary,
    }


# ═══════════════════════════════════════════════
# 2. MARKOWITZ MEAN-VARIANCE OPTIMIZATION
# ═══════════════════════════════════════════════

def get_markowitz_weights(returns_df, risk_free_rate=0.07):
    """
    Find optimal weights maximizing the Sharpe Ratio.
    Returns: dict of {ticker: optimal_weight}
    """
    tickers = returns_df.columns
    num_assets = len(tickers)
    
    # Calculate annualized mean returns and covariance matrix
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    # Objective function: Minimize negative Sharpe Ratio
    def negative_sharpe(weights):
        p_ret = np.sum(mean_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (p_ret - risk_free_rate) / p_vol
        return -sharpe
        
    # Constraints & Bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights sum to 1
    bounds = tuple((0.01, 1.0) for _ in range(num_assets)) # Min 1% allocation to each
    
    # Initial guess: equal weighting
    init_guess = num_assets * [1. / num_assets]
    
    # Optimize
    opt_result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    weights = opt_result.x
    return {ticker: round(weights[i], 4) for i, ticker in enumerate(tickers)}, mean_returns, cov_matrix, weights


# ═══════════════════════════════════════════════
# 3. CAPM (Capital Asset Pricing Model)
# ═══════════════════════════════════════════════

def calculate_capm(returns_df, market_returns, risk_free_rate=0.07):
    """
    Calculate Beta and CAPM expected return for each asset.
    E(R_i) = R_f + Beta * (E(R_m) - R_f)
    """
    capm_results = {}
    
    # Ensure indices align
    aligned_data = pd.concat([returns_df, market_returns], axis=1, join='inner').dropna()
    if aligned_data.empty:
        return {}
        
    mkt_ret = aligned_data.iloc[:, -1]
    mkt_var = mkt_ret.var()
    ann_mkt_ret = mkt_ret.mean() * 252
    
    for ticker in returns_df.columns:
        asset_ret = aligned_data[ticker]
        covariance = asset_ret.cov(mkt_ret)
        beta = covariance / mkt_var if mkt_var != 0 else 1.0
        
        # CAPM Formula
        expected_ret = risk_free_rate + beta * (ann_mkt_ret - risk_free_rate)
        
        capm_results[ticker] = {
            'beta': round(beta, 2),
            'expected_return': expected_ret,
            'market_premium': ann_mkt_ret - risk_free_rate
        }
        
    return capm_results


# ═══════════════════════════════════════════════
# 4. SHARPE RATIO & ORCHESTRATION
# ═══════════════════════════════════════════════

def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.07):
    """Calculate Return, Volatility, and Sharpe Ratio of the portfolio."""
    w = np.array(weights)
    p_ret = np.sum(mean_returns * w)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 0 else 0
    return p_ret, p_vol, p_sharpe

def generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.07, num_points=50):
    """Generate points for the Efficient Frontier curve."""
    num_assets = len(mean_returns)
    if num_assets == 0:
        return [], []

    mean_returns_arr = np.asarray(mean_returns, dtype=float)
    cov_matrix_arr = np.asarray(cov_matrix, dtype=float)
    bounds = tuple((0, 1) for _ in range(num_assets))
    base_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    init_guess = num_assets * [1. / num_assets]

    # First find the minimum-volatility feasible portfolio so the frontier starts on the true left edge.
    min_vol_result = minimize(
        lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix_arr, w))),
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=base_constraints,
    )

    if not min_vol_result.success:
        return [], []

    min_return = float(np.sum(mean_returns_arr * min_vol_result.x))
    max_return = float(mean_returns_arr.max())
    target_returns = np.linspace(min_return, max_return, num_points)
    frontier_points = []

    for tr in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, target=tr: np.sum(mean_returns_arr * x) - target}
        )

        res = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix_arr, w))),
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:
            frontier_points.append((float(res.fun), float(tr)))

    frontier_points.sort(key=lambda point: point[0])
    frontier_vols = [point[0] for point in frontier_points]
    frontier_rets = [point[1] for point in frontier_points]

    return frontier_vols, frontier_rets

def run_portfolio_optimization(portfolio_list):
    """
    Main orchestration function.
    portfolio_list: list of dicts [{'ticker': 'RELIANCE.NS', 'qty': 100, 'buy_price': 2500}, ...]
    """
    if not portfolio_list:
        raise ValueError("Portfolio is empty")
        
    tickers = [item['ticker'] for item in portfolio_list]
    risk_free_rate = 0.0709  # Fixed India 10Y Bond Yield approx
    
    # 1. Fetch Data
    hist_data = {}
    closes = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, period="1y")
        if not df.empty:
            hist_data[ticker] = df
            closes[ticker] = df['Close']
            
    if not hist_data:
        raise ValueError("Could not fetch data for any of the provided tickers.")
        
    # Market data for CAPM
    mkt = fetch_stock_data("^NSEI", period="1y")
    mkt_returns = mkt['Close'].pct_change().dropna() if not mkt.empty else pd.Series(dtype=float)
        
    price_df = pd.DataFrame(closes).dropna()
    returns_df = price_df.pct_change().dropna()
    
    # 2. Deep RankNet (Score Assets)
    valid_tickers = list(price_df.columns)
    ranknet_scores, ranking_context = get_ranknet_scores(valid_tickers, hist_data)
    ranking_benchmark = build_ranking_benchmark(valid_tickers, hist_data, ranknet_scores)
    ranking_validation = build_ranking_validation(valid_tickers, hist_data)
    
    # 3. Markowitz MVO (Optimal Weights)
    optimal_weights_dict, mean_returns, cov_matrix, opt_w_array = get_markowitz_weights(returns_df, risk_free_rate)
    
    # 4. CAPM (Beta & Required Returns)
    capm_metrics = calculate_capm(returns_df, mkt_returns, risk_free_rate)
     
    # 5. Portfolio Performance (Sharpe Ratio)
    p_ret, p_vol, p_sharpe = calculate_portfolio_metrics(opt_w_array, mean_returns, cov_matrix, risk_free_rate)
    
    # 6. Current Portfolio Stats (Unoptimized / Equal Weight)
    total_invested = sum([item['qty'] * item['buy_price'] for item in portfolio_list])
    current_value = 0
    current_weights_arr = []
    
    for item in portfolio_list:
        t = item['ticker']
        if t in price_df.columns:
            val = item['qty'] * price_df[t].iloc[-1]
            current_value += val
            
    if current_value > 0:
        for t in valid_tickers:
            match = next((i for i in portfolio_list if i['ticker'] == t), None)
            if match:
                w = (match['qty'] * price_df[t].iloc[-1]) / current_value
                current_weights_arr.append(w)
            else:
                current_weights_arr.append(0)
    else:
        current_weights_arr = [1.0/len(valid_tickers)] * len(valid_tickers)
        
    c_ret, c_vol, c_sharpe = calculate_portfolio_metrics(current_weights_arr, mean_returns, cov_matrix, risk_free_rate)
    
    # 7. Generate Trade Suggestions (Buy/Sell/Hold)
    trade_suggestions = {}
    total_turnover = 0.0
    for t in valid_tickers:
        opt_w = optimal_weights_dict.get(t, 0)
        cur_w = current_weights_arr[valid_tickers.index(t)]
        
        diff_w = opt_w - cur_w
        target_value = current_value * opt_w
        current_alloc_val = current_value * cur_w
        action_val = target_value - current_alloc_val
        
        # Determine strictness of hold (e.g. within 2% of optimal is HOLD)
        if abs(diff_w) < 0.02:
            action = "HOLD"
        elif diff_w > 0:
            action = "BUY"
        else:
            action = "SELL"

        total_turnover += abs(diff_w)
            
        trade_suggestions[t] = {
            'action': action,
            'change_percent': diff_w * 100,
            'amount_inr': action_val
        }

    # 9. Comparative analysis: Avg buy vs Real Market price
    stock_comparison = {}
    for item in portfolio_list:
        ticker = item['ticker']
        if ticker in price_df.columns:
            cur_price = price_df[ticker].iloc[-1]
            buy_price = item['buy_price']
            qty = item['qty']
            
            total_cost = qty * buy_price
            cur_val = qty * cur_price
            gain_amt = cur_val - total_cost
            gain_pct = (gain_amt / total_cost * 100) if total_cost > 0 else 0
            
            stock_comparison[ticker] = {
                'qty': qty,
                'avg_buy_price': buy_price,
                'current_price': cur_price,
                'total_cost': total_cost,
                'current_value': cur_val,
                'gain_loss_amt': gain_amt,
                'gain_loss_pct': gain_pct
            }

    # 10. Generate Efficient Frontier for plotting
    ef_vols, ef_rets = generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
    
    return {
    'optimized_weights': optimal_weights_dict,   #FIXED
    'current_weights': {
        valid_tickers[i]: round(current_weights_arr[i], 4)
        for i in range(len(valid_tickers))
    },
    'trade_suggestions': trade_suggestions,
    'ranknet_scores': {k: round(v, 2) for k, v in ranknet_scores.items()},
    'ranking_context': ranking_context,
    'ranking_benchmark': ranking_benchmark,
    'ranking_validation': ranking_validation,
    'capm_metrics': capm_metrics,
    'stock_comparison': stock_comparison,
    'portfolio_assumptions': {
        'ranking_method': 'heuristic_momentum',
        'ranking_model_used': 'tensorflow_placeholder' if TF_AVAILABLE else 'heuristic_only',
        'ranking_formula': '0.20*ret_1m + 0.25*ret_3m + 0.20*price_vs_ma50 - 0.15*volatility - 0.20*max_drawdown',
        'optimizer_objective': 'maximize_sharpe_ratio',
        'risk_free_rate': risk_free_rate,
        'weight_bounds': {'min': 0.01, 'max': 1.0},
    },

    # UI FRIENDLY METRICS
    'metrics': {
        'optimal': {
            'return': p_ret,
            'volatility': p_vol,
            'sharpe': p_sharpe
        },
        'current': {
            'return': c_ret,
            'volatility': c_vol,
            'sharpe': c_sharpe
        },
        'invested': total_invested,
        'current_value': current_value,
        'turnover_ratio': total_turnover / 2
    },

    'efficient_frontier': {
        'vols': ef_vols,
        'rets': ef_rets,
        'optimal_point': {
            'volatility': p_vol,
            'return': p_ret,
        },
        'current_point': {
            'volatility': c_vol,
            'return': c_ret,
        },
    }
}
