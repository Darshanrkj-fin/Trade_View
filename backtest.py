import numpy as np
from model import predict_lstm

def backtest_strategy(historical, test_days=30):
    if historical.empty or 'Close' not in historical or len(historical) < test_days * 2:
        return {"profit": 0, "accuracy": 0, "buy_hold": 0}
        
    # Split data: train on everything EXCEPT the last test_days
    train_data = historical['Close'].values[:-test_days]
    actual_test_data = historical['Close'].values[-test_days:]
    
    # Generate predictions strictly based on past data
    result = predict_lstm(train_data, forecast_days=test_days)
    
    if isinstance(result, dict):
        predictions = result['predictions']
    else:
        predictions = result
        
    if len(predictions) != test_days:
        return {"profit": 0, "accuracy": 0, "buy_hold": 0}

    # 1. ACCURACY CALCULATION (MAPE - Mean Absolute Percentage Error)
    # How close were the predictions to the actuals?
    mape = np.mean(np.abs((actual_test_data - predictions) / actual_test_data))
    accuracy = max(0, (1 - mape) * 100)
    
    # 2. ADVANCED STRATEGY: Buy/sell on every prediction
    # Simulate: If tomorrow's predicted price > today's, buy (or hold); if <, sell (or stay out)
    capital = 1.0  # Start with 1 unit of capital
    position = 0   # 0 = out, 1 = in
    entry_price = 0
    for i in range(test_days - 1):
        today_price = actual_test_data[i]
        next_pred = predictions[i + 1]
        if next_pred > today_price:
            if position == 0:
                # Buy
                position = 1
                entry_price = today_price
        else:
            if position == 1:
                # Sell
                capital *= actual_test_data[i] / entry_price
                position = 0
    # If still holding at the end, sell at last price
    if position == 1:
        capital *= actual_test_data[-1] / entry_price
    strategy_return = (capital - 1) * 100

    # Buy & Hold as before
    start_price = actual_test_data[0]
    final_price = actual_test_data[-1]
    buy_and_hold_return = ((final_price - start_price) / start_price) * 100

    return {
        "profit": strategy_return,
        "accuracy": accuracy,
        "buy_hold": buy_and_hold_return
    }