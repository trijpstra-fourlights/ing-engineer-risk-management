## Answer 1 - Option

* Implied Volatility (IV) provides a prediction of the future, while Historical Volatility (HV) provides an observation of the past.
* IV is derived from option pricing models while HV is calculated directly from historical price data.
* IV provides a subjective measure on future volatility, while HV provides an objective measure of baseline volatility.

## Answer 2 - VaR

Value-at-Risk (VaR) is a measurement of the maximum potential loss in value of a portfolio over a defined period (e.g. 1 day) and for a given confidence interval (e.g. 95%).

There are various methods of calculation that mainly differ in the way they approximate portfolio volatility.

The simplest method (parametric) assumes that portfolio volatility is constant and thus return variability can be approximated using a normal distribution. It requires very little data (only the current portfolio value and a measure of volatility, e.g. annual volatility of the portfolio) and is easily computed but has practical limitations due to its assumptions.

A more complex method is the Peaks Over Threshold (POT) method. This method requires more data, specifically containing sufficient extreme events, as it uses historical excesses in portfolio returns to extrapolate risk beyond historical observations.
In other words, it estimates the chance of future excesses in returns.
This estimation is dependent on the method's parameters, so stress testing is needed make sure the parameters are aptly chosen.

Because VaR is used in risk management, and specifically to reduce portfolio risk if it's deemed to large, a method that specifically models extreme events (i.e. "tail focus") is preferable.

## Answer 3 - Option

Refer to `options.py` for the implementation and `options_test.py` for both unit- and end-to-end tests.

## Answer 4 - VaR

Refer to `var.py` for the implementation  and `var_test.py` for both unit- and end-to-end tests.

