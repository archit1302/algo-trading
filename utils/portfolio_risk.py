"""
Portfolio Risk Analytics — compute VaR, Sharpe, drawdown, and correlation.

Lightweight risk toolkit for algo-trading portfolios. Works with
pandas DataFrames of daily returns (columns = tickers, rows = dates).

Usage:
    import pandas as pd
        from utils.portfolio_risk import PortfolioRisk

            returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
                risk = PortfolioRisk(returns)
                    print(risk.summary())
                    """

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class PortfolioRisk:
      """Compute common risk metrics from a DataFrame of daily returns."""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
              """
                      Args:
                                  returns: DataFrame where each column is a ticker and each row
                                                       is a daily simple return (e.g. 0.02 for +2%).
                                                                   risk_free_rate: Annualized risk-free rate (default 5% for India).
                                                                           """
              self.returns = returns.dropna()
              self.rf = risk_free_rate
              self._trading_days = 252

    # ---- Core metrics ----

    def annualized_return(self) -> pd.Series:
              """Geometric annualized return per asset."""
              cum = (1 + self.returns).prod()
              n_years = len(self.returns) / self._trading_days
              return cum ** (1 / n_years) - 1

    def annualized_volatility(self) -> pd.Series:
              """Annualized standard deviation of returns."""
              return self.returns.std() * np.sqrt(self._trading_days)

    def sharpe_ratio(self) -> pd.Series:
              """Annualized Sharpe ratio per asset."""
              excess = self.annualized_return() - self.rf
              return excess / self.annualized_volatility()

    def sortino_ratio(self) -> pd.Series:
              """Sortino ratio — penalizes only downside volatility."""
              downside = self.returns[self.returns < 0].fillna(0)
              down_vol = downside.std() * np.sqrt(self._trading_days)
              excess = self.annualized_return() - self.rf
              return excess / down_vol

    # ---- Value at Risk ----

    def var_historical(self, confidence: float = 0.95) -> pd.Series:
              """Historical VaR at given confidence level (daily)."""
              return self.returns.quantile(1 - confidence)

    def var_parametric(self, confidence: float = 0.95) -> pd.Series:
              """Parametric (Gaussian) VaR at given confidence level (daily)."""
              from scipy.stats import norm
              z = norm.ppf(1 - confidence)
              return self.returns.mean() + z * self.returns.std()

    def cvar(self, confidence: float = 0.95) -> pd.Series:
              """Conditional VaR (Expected Shortfall) — mean of losses beyond VaR."""
              var = self.var_historical(confidence)
              mask = self.returns.le(var, axis=1)
              return self.returns.where(mask).mean()

    # ---- Drawdown ----

    def max_drawdown(self) -> pd.Series:
              """Maximum drawdown per asset."""
              cum = (1 + self.returns).cumprod()
              peak = cum.cummax()
              dd = (cum - peak) / peak
              return dd.min()

    def drawdown_series(self) -> pd.DataFrame:
              """Full drawdown time-series."""
              cum = (1 + self.returns).cumprod()
              peak = cum.cummax()
              return (cum - peak) / peak

    # ---- Correlation ----

    def correlation_matrix(self) -> pd.DataFrame:
              """Pairwise Pearson correlation of daily returns."""
              return self.returns.corr()

    def rolling_correlation(
              self, ticker_a: str, ticker_b: str, window: int = 60
    ) -> pd.Series:
              """Rolling correlation between two tickers."""
              return self.returns[ticker_a].rolling(window).corr(self.returns[ticker_b])

    # ---- Portfolio-level (equal-weight) ----

    def portfolio_return(self, weights: Optional[np.ndarray] = None) -> float:
              """Weighted annualized return. Defaults to equal-weight."""
              n = len(self.returns.columns)
              w = weights if weights is not None else np.ones(n) / n
              port_daily = self.returns.values @ w
              cum = np.prod(1 + port_daily)
              n_years = len(self.returns) / self._trading_days
              return float(cum ** (1 / n_years) - 1)

    def portfolio_volatility(self, weights: Optional[np.ndarray] = None) -> float:
              """Weighted annualized volatility."""
              n = len(self.returns.columns)
              w = weights if weights is not None else np.ones(n) / n
              cov = self.returns.cov().values * self._trading_days
              return float(np.sqrt(w @ cov @ w))

    # ---- Summary ----

    def summary(self) -> pd.DataFrame:
              """One-line-per-asset risk summary."""
              return pd.DataFrame({
                  "ann_return": self.annualized_return(),
                  "ann_vol": self.annualized_volatility(),
                  "sharpe": self.sharpe_ratio(),
                  "sortino": self.sortino_ratio(),
                  "max_dd": self.max_drawdown(),
                  "var_95": self.var_historical(0.95),
                  "cvar_95": self.cvar(0.95),
              }).round(4)


if __name__ == "__main__":
      # Quick demo with random data
      np.random.seed(42)
      tickers = ["SBIN", "RELIANCE", "TCS", "INFY"]
      dates = pd.bdate_range("2024-01-01", periods=252)
      data = np.random.normal(0.0005, 0.02, (252, 4))
      df = pd.DataFrame(data, index=dates, columns=tickers)

    risk = PortfolioRisk(df)
    print(risk.summary())
    print(f"\nPortfolio Sharpe (equal-weight): "
                    f"{risk.portfolio_return() / risk.portfolio_volatility():.3f}")
