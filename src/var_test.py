import os
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from . import var


@pytest.mark.parametrize(
    "prices, horizon, ascending, expected",
    [
        ([1, 2, 3, 4, 5], 1, "ascending", [1.0, 0.5, 0.333333, 0.25]),
        ([5, 4, 3, 2, 1], 1, "descending",[0.25, 0.333333, 0.5, 1.0])
    ],
)
def test_calculate_returns(prices, horizon, ascending, expected):
    assert_series_equal(
        pd.Series(var.calculate_returns(pd.DataFrame(prices), horizon, ascending).squeeze()),
        pd.Series(expected),
        check_names=False,
        check_exact=False,
        check_index=False,
    )

@pytest.mark.parametrize(
    "portfolio, prices, expected",
    [
        ({"FX-1": [100.00]}, {"FX-1": [1, 1.1, 0.9, 0.8]}, -6.909090),
        ({"FX-1": [100.00]}, {"FX-1": [1, 0.9, 0.8, 0.8]}, -10.66666),
    ],
)
def test_var(portfolio, prices, expected):
    assert var.var(pd.DataFrame(portfolio), pd.DataFrame(prices)) == pytest.approx(expected)


def test_end_to_end():
    portfolio = pd.DataFrame([{"ccy-1": 153084.81, "ccy-2": 95891.51}])
    fx_prices = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "var_fx_prices.csv"),
        parse_dates=["date"],
        index_col="date",
        dtype=float,
    )
    fx_prices.sort_index(inplace=True, ascending=True)
    assert var.var(portfolio, fx_prices) == pytest.approx(-13572.73)
