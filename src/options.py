import pandas as pd
import numpy as np
from scipy.stats import norm


def as_df(option):
    """Creates a dataframe holding the option, and calculates time to maturity"""
    df = pd.DataFrame([option])
    df["T"] = (df["t1"] - df["t0"]) / np.timedelta64(365, "D")
    return df


def black_scholes_option_price_spot(S, K, T, r, sigma, option_type="call"):
    """Calculates option price using Black-Scholes formula with spot price assuming zero dividend"""
    sigma_scaled = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_scaled
    d2 = d1 - sigma_scaled

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)  # put


def black_scholes_option_price_forward(F, K, T, r, sigma, option_type="call"):
    """Calculates option price using Black-Scholes formula with forward price assuming zero dividend"""
    sigma_scaled = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / sigma_scaled
    d2 = d1 - sigma_scaled

    discount = np.exp(-r * T)

    if option_type == "call":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))  # put


def forward_price(S, T, r):
    """Calculates the forward price assuming zero dividend"""
    return S * np.exp(r * T)


def moneyness(strike_price, current_price, option_type="call", threshold=0.005):
    """Calculates if an option is ITM, ATM, or OTM"""
    fuzzy_price = threshold * strike_price

    if option_type == "put":
        if current_price < strike_price - fuzzy_price:
            return "ITM"
        if current_price > strike_price + fuzzy_price:
            return "OTM"
        return "ATM"

    if option_type == "call":
        if current_price > strike_price + fuzzy_price:
            return "ITM"
        if current_price < strike_price - fuzzy_price:
            return "OTM"
        return "ATM"

    return ""
