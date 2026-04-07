from __future__ import annotations


def validate_ticker(raw_ticker: object) -> str:
    ticker = str(raw_ticker or "").strip().upper()
    if not ticker:
        raise ValueError("Ticker is required.")
    return ticker


def validate_forecast_days(raw_forecast_days: object, allowed_values: tuple[int, ...] = (7, 30, 90)) -> int:
    try:
        forecast_days = int(raw_forecast_days)
    except (TypeError, ValueError):
        raise ValueError("Forecast days must be a valid integer.") from None

    if forecast_days not in allowed_values:
        raise ValueError(f"Forecast days must be one of {', '.join(map(str, allowed_values))}.")
    return forecast_days


def validate_holdings(raw_holdings: object) -> list[dict[str, object]]:
    holdings = raw_holdings if isinstance(raw_holdings, list) else []
    if len(holdings) < 2:
        raise ValueError("Please provide at least 2 holdings for optimization.")

    cleaned_holdings = []
    for item in holdings:
        if not isinstance(item, dict):
            raise ValueError("Each holding must be an object with ticker, qty, and buy_price.")
        ticker = validate_ticker(item.get("ticker", ""))
        try:
            qty = int(item.get("qty", 0))
            buy_price = float(item.get("buy_price", 0))
        except (TypeError, ValueError):
            raise ValueError("Each holding must include numeric qty and buy_price.") from None
        if qty <= 0 or buy_price <= 0:
            raise ValueError("Each holding must include a valid ticker, qty, and buy_price.")
        cleaned_holdings.append({"ticker": ticker, "qty": qty, "buy_price": buy_price})
    return cleaned_holdings
