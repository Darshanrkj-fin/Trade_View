from data import fetch_stock_data
from model import predict_lstm

def manage_portfolio(portfolio):
    # Mock portfolio (replace with user input or file)
    mock_portfolio = {"RELIANCE.NS": 100, "TCS.NS": 50}  # Shares owned
    total_value = 0
    predicted_gain = 0

    for ticker, shares in mock_portfolio.items():
        data = fetch_stock_data(ticker, period="1y")
        if not data.empty:
            last_price = data['Close'].iloc[-1]
            result = predict_lstm(data['Close'].values, forecast_days=30)
            predictions = result['predictions'] if isinstance(result, dict) else result
            
            total_value += last_price * shares
            predicted_gain += (predictions[-1] - last_price) / last_price * 100 * shares

    avg_gain = predicted_gain / sum(mock_portfolio.values()) if mock_portfolio else 0
    return {"total_value": total_value, "predicted_gain": avg_gain}