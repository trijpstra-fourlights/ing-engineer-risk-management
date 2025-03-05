import pytest
import pandas as pd
from pandas.testing import assert_series_equal

from . import options


# https://analystprep.com/study-notes/actuarial-exams/soa/ifm-investment-and-financial-markets/black-scholes-option-pricing-model/
@pytest.mark.parametrize(
    "S, K, T, r, sigma, option_type, expected", [(100, 90, 0.5, 0.10, 0.25, "call", 16.11)]
)
def test_black_scholes_option_price_spot(S, K, T, r, sigma, option_type, expected):
    assert (
        abs(options.black_scholes_option_price_spot(S, K, T, r, sigma, option_type)) - expected
        < 0.05
    )


@pytest.mark.parametrize(
    "F, K, T, r, sigma, option_type, expected", [(100, 90, 0.5, 0.10, 0.25, "call", 16.11)]
)
def test_black_scholes_option_price_forward(F, K, T, r, sigma, option_type, expected):
    assert (
        abs(options.black_scholes_option_price_forward(F, K, T, r, sigma, option_type)) - expected
        < 0.05
    )


@pytest.mark.parametrize(
    "current_price, strike_price, option_type, expected",
    [
        (1, 10, "call", "ITM"),
        (10, 1, "call", "OTM"),
        (10, 10, "call", "ATM"),
        (1, 10, "put", "OTM"),
        (10, 1, "put", "ITM"),
        (10, 10, "put", "ATM"),
    ],
)
def test_moneyness(current_price, strike_price, option_type, expected):
    assert options.moneyness(current_price, strike_price, option_type) == expected


def test_end_to_end():
    option = {
        "t0": pd.Timestamp("2024-11-23"),
        "t1": pd.Timestamp("2025-05-10"),
        "S": 19,
        "K": 17,
        "r": 0.005,
        "sigma": 0.3,
    }
    options_df = options.as_df(option)

    expected = {
        "F": 19.04367,
        "C_F": 2.70,
        "C_S": 2.70,
        "P_F": 0.66,
        "P_PCP": 0.66,
    }
    expected_df = pd.DataFrame([expected])

    assert_series_equal(
        options.forward_price(options_df["S"], options_df["T"], options_df["r"]),
        expected_df["F"],
        check_names=False,
        check_exact=False,
    )

    assert_series_equal(
        options.black_scholes_option_price_spot(
            options_df["S"],
            options_df["K"],
            options_df["T"],
            options_df["r"],
            options_df["sigma"],
            "call",
        ),
        expected_df["C_S"],
        rtol=0.01,
        check_names=False,
        check_exact=False,
    )

    assert_series_equal(
        options.black_scholes_option_price_forward(
            options.forward_price(options_df["S"], options_df["T"], options_df["r"]),
            options_df["K"],
            options_df["T"],
            options_df["r"],
            options_df["sigma"],
            "call",
        ),
        expected_df["C_F"],
        rtol=0.01,
        check_names=False,
        check_exact=False,
    )

    assert_series_equal(
        options.black_scholes_option_price_forward(
            options.forward_price(options_df["S"], options_df["T"], options_df["r"]),
            options_df["K"],
            options_df["T"],
            options_df["r"],
            options_df["sigma"],
            "put",
        ),
        expected_df["P_F"],
        rtol=0.01,
        check_names=False,
        check_exact=False,
    )
