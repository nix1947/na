import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NepsePredictor:
    """
    Predicts future NEPSE stock prices using time-series + macroeconomic models.
    DISCLAIMER: For educational/informational use only. Not financial advice.
    """

    def __init__(self, conn: sqlite3.Connection, macro_df: Optional[pd.DataFrame] = None):
        self.conn = conn
        self.scanner = NepseScanner(conn)  # external loader assumed
        self.macro_df = macro_df  # must include "date" column

    # ------------------------
    # Synthetic Macro Generator
    # ------------------------
    @staticmethod
    def generate_synthetic_macro_data(start_date="2023-01-01", end_date="2025-01-01") -> pd.DataFrame:
        """
        Generate synthetic macro data with mixed frequencies and forward-fill to daily.
        Quarterly: gdp_growth, fy_budget_ceiling, govt_debt_increase, govt_spending
        Monthly: inflation, lending_rate, custom_rate, exchange_rate
        Policy rate: random step function (few changes per year)
        """
        np.random.seed(42)
        idx_daily = pd.date_range(start=start_date, end=end_date, freq="D")

        # Quarterly data
        idx_q = pd.date_range(start=start_date, end=end_date, freq="Q")
        quarterly = pd.DataFrame({
            "date": idx_q,
            "gdp_growth": np.random.normal(4.5, 0.4, len(idx_q)).round(2),
            "fy_budget_ceiling": np.random.normal(1800, 80, len(idx_q)).round(0),  # in billions
            "govt_debt_increase": np.random.normal(35, 2.5, len(idx_q)).round(2),   # % of GDP
            "govt_spending": np.random.normal(12, 1.5, len(idx_q)).round(2)        # % growth
        }).set_index("date")

        # Monthly data
        idx_m = pd.date_range(start=start_date, end=end_date, freq="M")
        monthly = pd.DataFrame({
            "date": idx_m,
            "inflation": np.random.normal(6, 0.6, len(idx_m)).round(2),
            "lending_rate": np.random.normal(11, 0.8, len(idx_m)).round(2),
            "custom_rate": np.random.normal(15, 1, len(idx_m)).round(2),
            "exchange_rate": np.random.normal(132, 1, len(idx_m)).round(2)
        }).set_index("date")

        # Policy rate (step changes few times per year)
        idx_p = pd.date_range(start=start_date, end=end_date, freq="180D")
        policy = pd.DataFrame({
            "date": idx_p,
            "policy_rate": np.random.choice([6.5, 7.0, 7.5, 8.0], len(idx_p))
        }).set_index("date")

        # Merge all and forward-fill to daily
        macro = pd.concat([quarterly, monthly, policy]).sort_index()
        macro_daily = macro.reindex(idx_daily).ffill().reset_index().rename(columns={"index": "date"})
        return macro_daily

    # ------------------------
    # Stock Loader
    # ------------------------
    def _load_symbol_history(self, symbol: str, days: int) -> pd.Series:
        """Loads close price series for symbol."""
        df = self.scanner._load_history(days=days)
        stock_data = df[df['symbol'] == symbol].set_index("date")["close"]
        return stock_data

    # ------------------------
    # Technical Models
    # ------------------------
    def predict_next_day_linear(self, symbol: str, windowsize: int = 30) -> Dict:
        data = self._load_symbol_history(symbol, days=windowsize)
        if len(data) < windowsize:
            return {'error': 'Not enough history.'}

        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(data)]])[0]
        return {"symbol": symbol, "method": "Linear Trend", "prediction": float(pred), "last_close": float(y[-1])}

    def predict_arima(self, symbol: str, days_history: int = 180, horizon: int = 7) -> Dict:
        data = self._load_symbol_history(symbol, days=days_history)
        if len(data) < 50:
            return {"error": "Not enough history."}
        try:
            model = ARIMA(data, order=(5, 1, 0)).fit()
            forecast = model.get_forecast(steps=horizon)
            return {
                "symbol": symbol, "method": "ARIMA", "horizon": horizon,
                "predictions": forecast.predicted_mean.tolist(),
                "conf_int": forecast.conf_int().values.tolist(),
                "last_close": float(data.iloc[-1])
            }
        except Exception as e:
            return {"error": f"ARIMA failed: {e}"}

    def predict_holt_winters(self, symbol: str, days_history: int = 365, horizon: int = 30, seasonal_periods: int = 30) -> Dict:
        data = self._load_symbol_history(symbol, days=days_history)
        if len(data) < seasonal_periods * 2:
            return {"error": "Not enough history."}
        try:
            model = ExponentialSmoothing(data, trend="add", seasonal="add",
                                         seasonal_periods=seasonal_periods).fit()
            forecast = model.forecast(horizon)
            return {
                "symbol": symbol, "method": "Holt-Winters", "horizon": horizon,
                "predictions": forecast.tolist(), "last_close": float(data.iloc[-1])
            }
        except Exception as e:
            return {"error": f"Holt-Winters failed: {e}"}

    # ------------------------
    # Macro Models
    # ------------------------
    def predict_with_macro_factors(self, symbol: str, days: int = 180) -> Dict:
        if self.macro_df is None:
            return {"error": "Macro dataset not provided."}

        prices = self._load_symbol_history(symbol, days=days)
        if prices.empty:
            return {"error": "No stock data."}

        df = self.macro_df.set_index("date").join(prices.rename("close")).dropna()
        if len(df) < 30:
            return {"error": "Not enough aligned data."}

        X = df.drop(columns=["close"]).values
        y = df["close"].values

        model = LinearRegression().fit(X, y)
        latest_macro = X[-1].reshape(1, -1)
        pred = model.predict(latest_macro)[0]

        return {
            "symbol": symbol, "method": "Macro Regression",
            "prediction": float(pred), "last_close": float(y[-1]),
            "coefficients": dict(zip(df.drop(columns=["close"]).columns, model.coef_.round(4)))
        }


# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    conn = sqlite3.connect("nepse.db")

    # 1. Generate macro dataset
    macro_data = NepsePredictor.generate_synthetic_macro_data("2023-01-01", "2025-01-01")

    # 2. Init predictor
    predictor = NepsePredictor(conn, macro_df=macro_data)

    # 3. Try different models
    print(predictor.predict_next_day_linear("NABIL"))
    print(predictor.predict_arima("NABIL", horizon=7))
    print(predictor.predict_holt_winters("NABIL", horizon=30))
    print(predictor.predict_with_macro_factors("NABIL", days=180))
