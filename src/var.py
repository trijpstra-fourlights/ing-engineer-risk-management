import numpy as np


def calculate_returns(prices, horizon=1, sort="ascending"):
    """Calculate (simple) returns from price action"""
    step = 1 if sort == "ascending" else -1
    if horizon == 1:
        returns = prices / prices.shift(step) - 1
    else:
        returns = np.exp(np.log(prices / prices.shift(step)) * np.sqrt(horizon)) - 1
    return returns.dropna()


def var(portfolio, prices, sort="ascending"):
    """Calculate VaR 1-day for portfolio"""
    returns = calculate_returns(prices, 1, sort)
    pnl = portfolio.dot(returns.T)
    daily_pnl = pnl.sum(axis=0)
    daily_pnl_asc = daily_pnl.sort_values(ascending=True)
    return 0.4 * daily_pnl_asc[1] + 0.6 * daily_pnl_asc[2]
