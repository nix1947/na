"""
nepse_analyzer.py

A comprehensive toolkit for scanning and predicting NEPSE stock market data.
This module provides two main classes:
1.  NepseScanner: For identifying current technical patterns, statistical properties,
    and behavioral clusters among stocks.
2.  NepsePredictor: For forecasting future stock prices using time-series models.

Requirements:
    pip install pandas numpy scipy scikit-learn statsmodels

Usage:
    import sqlite3
    from nepse_analyzer import NepseScanner, NepsePredictor, create_sample_database

    # Create a database connection and populate with sample data
    conn = sqlite3.connect(':memory:')
    create_sample_database(conn)
    
    # --- Scanning ---
    scanner = NepseScanner(conn)
    stock_clusters = scanner.get_stock_clusters()
    print("Stock Clusters:")
    print(stock_clusters)
    
    # --- Prediction ---
    predictor = NepsePredictor(conn)
    nabil_prediction = predictor.predict_next_day_arima('NABIL')
    print("\\nNABIL ARIMA Prediction:", nabil_prediction)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import sqlite3
import warnings
from typing import Iterable, List, Optional, Tuple, Dict

# Statistical and ML Libraries
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA




import sqlite3
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class NepseScanner:
    """
    Scans NEPSE history for technical patterns, statistical properties, and clusters.
    Improved version with docstrings and usage examples.
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize the scanner with a SQLite connection.
        """
        self.conn = conn

    # -------------------------------
    # Private Helper Methods
    # -------------------------------


    def _load_history(self, days: Optional[int] = 400, table: str = 'price_history') -> pd.DataFrame:
        """
        Load historical data from NEPSE MySQL database.

        Parameters:
        - days (int): Number of days to fetch. Default = 400.
        - table (str): Database table name. Default = 'nepse_history'.

        Returns:
        - DataFrame with standardized column names.
        """
        where_sql = f"WHERE BusinessDate >= DATE_SUB(CURDATE(), INTERVAL {int(days)} DAY)" if days else ""
        sql = f"""
            SELECT BusinessDate, Symbol, SecurityName, OpenPrice, HighPrice, LowPrice, ClosePrice,
                TotalTradedQuantity, PreviousDayClosePrice, FiftyTwoWeekHigh, FiftyTwoWeekLow,
                AverageTradedPrice, MarketCapitalization
            FROM `{table}`
            {where_sql}
            ORDER BY BusinessDate ASC, Symbol ASC
        """
        try:
        
            df = pd.read_sql(sql, self.conn)

        except Exception as e:
            print(f"Error loading data from MySQL database: {e}")
            return pd.DataFrame()

        rename_map = {
            'BusinessDate': 'date',
            'Symbol': 'symbol',
            'SecurityName': 'security_name',
            'OpenPrice': 'open',
            'HighPrice': 'high',
            'LowPrice': 'low',
            'ClosePrice': 'close',
            'TotalTradedQuantity': 'volume',
            'PreviousDayClosePrice': 'prev_close',
            'FiftyTwoWeekHigh': 'high_52w',
            'FiftyTwoWeekLow': 'low_52w',
            'AverageTradedPrice': 'avg_price',
            'MarketCapitalization': 'mcap'
        }
        df = df.rename(columns=rename_map)

        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'prev_close',
                        'high_52w', 'low_52w', 'avg_price', 'mcap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna(subset=['date', 'symbol', 'close']).reset_index(drop=True)

    
    
    
    @staticmethod
    def _find_swings(g: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
        """
        Identify swing highs and lows.

        Parameters:
        - left/right: Window size for comparison.

        Returns:
        - DataFrame with swing high/low levels.
        """
        n = len(g)
        sh = np.zeros(n, dtype=bool)
        sl = np.zeros(n, dtype=bool)
        highs, lows = g['high'].values, g['low'].values
        for i in range(left, n - right):
            if highs[i] == max(highs[i-left:i+right+1]):
                sh[i] = True
            if lows[i] == min(lows[i-left:i+right+1]):
                sl[i] = True
        out = g[['date']].copy()
        out['level_high'] = np.where(sh, g['high'], np.nan)
        out['level_low'] = np.where(sl, g['low'], np.nan)
        return out

    @staticmethod
    def _cluster_levels(levels: np.ndarray, band_width: float) -> List[Tuple[float, int]]:
        """
        Cluster levels for support/resistance.
        """
        levels = levels[~np.isnan(levels)]
        if len(levels) == 0 or band_width <= 0:
            return []
        bins = np.floor(levels / band_width).astype(int)
        clusters: Dict[int, List[float]] = {}
        for b, lvl in zip(bins, levels):
            clusters.setdefault(b, []).append(lvl)
        summary = sorted(
            [(float(np.mean(arr)), len(arr)) for arr in clusters.values()],
            key=lambda x: x[1], reverse=True
        )
        return summary

    # -------------------------------
    # Public Scanner Methods
    # -------------------------------

    def get_max_variation_stocks(self, windowsize: int = 20, top_n: int = 10) -> pd.DataFrame:
        """Find top stocks with largest ATR % variation."""
        df = self._load_history(days=windowsize * 3)
        if df.empty:
            print("DF epty")
            return pd.DataFrame()
        g = df.groupby('symbol', group_keys=False)
        df['atr'] = g.apply(lambda x: (x['high'] - x['low']).rolling(windowsize).mean())
        df['atr_pct'] = 100 * df['atr'] / df['close']
        last = df.groupby('symbol').tail(1)
        return last.sort_values('atr_pct', ascending=False).head(top_n)[['symbol', 'date', 'close', 'atr_pct']]

    def get_long_consolidated_stocks(self, windowsize: int = 60, band_pct: float = 5.0) -> pd.DataFrame:
        """Find stocks consolidating in tight ranges."""
        df = self._load_history(days=windowsize * 2)
        if df.empty:
            return pd.DataFrame()
        out = []
        for sym, g in df.groupby('symbol'):
            tail = g.tail(windowsize)
            if len(tail) < windowsize * 0.8:
                continue
            median = tail['close'].median()
            realized_band = 100 * (tail['high'].max() - tail['low'].min()) / median
            if realized_band <= band_pct:
                out.append({'symbol': sym, 'median_close': median, 'realized_band_pct': realized_band})
        return pd.DataFrame(out).sort_values('realized_band_pct')

    def get_support_resistance_levels(self, windowsize: int = 90, tolerance_pct: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate major support and resistance levels."""
        df = self._load_history(days=windowsize * 2)
        supports, resistances = [], []
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        for sym, g in df.groupby('symbol'):
            g_window = g.tail(windowsize)
            if len(g_window) < 20:
                continue
            swings = self._find_swings(g_window)
            band_width = g_window['close'].median() * (tolerance_pct / 100.0)
            s_levels = self._cluster_levels(swings['level_low'].dropna().values, band_width)
            if s_levels:
                supports.append({'symbol': sym, 'level': s_levels[0][0], 'touches': s_levels[0][1]})
            r_levels = self._cluster_levels(swings['level_high'].dropna().values, band_width)
            if r_levels:
                resistances.append({'symbol': sym, 'level': r_levels[0][0], 'touches': r_levels[0][1]})
        return pd.DataFrame(supports), pd.DataFrame(resistances)

    def get_statistical_summary(self, windowsize: int = 90) -> pd.DataFrame:
        """Return Sharpe ratio and ADF test results."""
        df = self._load_history(days=windowsize + 10)
        if df.empty:
            return pd.DataFrame()
        results = []
        for sym, g in df.groupby('symbol'):
            if len(g) < windowsize:
                continue
            returns = g['close'].pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            adf_result = adfuller(g['close'].dropna(), autolag='AIC')
            adf_p_value = adf_result[1]
            behavior = 'Mean-Reverting (Stationary)' if adf_p_value < 0.05 else 'Trending (Non-Stationary)'
            results.append({
                'symbol': sym,
                'sharpe_ratio_annualized': sharpe_ratio,
                'adf_p_value': adf_p_value,
                'inferred_behavior': behavior,
                'last_close': g['close'].iloc[-1]
            })
        return pd.DataFrame(results).sort_values('sharpe_ratio_annualized', ascending=False)

    def get_stock_clusters(self, windowsize: int = 30, n_clusters: int = 4) -> pd.DataFrame:
        """Cluster stocks by momentum and volatility."""
        df = self._load_history(days=windowsize * 3)
        if df.empty:
            return pd.DataFrame()
        g = df.groupby('symbol', group_keys=False)
        df['slope'] = g['close'].apply(lambda x: x.rolling(windowsize).apply(lambda y: stats.linregress(np.arange(len(y)), y).slope if len(y) > 1 else 0, raw=False))
        df['volatility'] = g['close'].pct_change().rolling(windowsize).std()
        df['volume_z'] = g['volume'].transform(lambda z: (z - z.rolling(windowsize).mean()) / z.rolling(windowsize).std())
        latest = df.groupby('symbol').tail(1).copy().dropna(subset=['slope', 'volatility', 'volume_z'])
        features = latest[['slope', 'volatility', 'volume_z']]
        if len(features) < n_clusters:
            return pd.DataFrame()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(scaled_features)
        latest['cluster'] = kmeans.labels_
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_map = {}
        for i, center in enumerate(cluster_centers):
            slope, vol, vol_z = center
            desc = []
            if slope > 0.5:
                desc.append("High Momentum")
            elif slope < -0.5:
                desc.append("Bearish")
            else:
                desc.append("Sideways")
            if vol > 0.02:
                desc.append("High Volatility")
            elif vol < 0.01:
                desc.append("Low Volatility")
            else:
                desc.append("Mid Volatility")
            cluster_map[i] = ", ".join(desc)
        latest['cluster_desc'] = latest['cluster'].map(cluster_map)
        return latest[['symbol', 'last_close', 'cluster', 'cluster_desc', 'slope', 'volatility', 'volume_z']].rename(columns={'close': 'last_close'}).sort_values('cluster')

    def get_current_positive_momentum_stocks(self, windowsize: int = 20, top_n: int = 10) -> pd.DataFrame:
        """Find stocks with strong positive momentum."""
        df = self._load_history(days=windowsize * 2)
        if df.empty:
            return pd.DataFrame()
        out = []
        for sym, g in df.groupby('symbol'):
            tail = g.tail(windowsize)
            if len(tail) < windowsize * 0.8:
                continue
            y = tail['close'].values
            x = np.arange(len(y))
            slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
            if slope > 0 and r_val**2 > 0.6:
                out.append({
                    'symbol': sym,
                    'last_close': y[-1],
                    'momentum_slope': slope,
                    'r_squared': r_val**2
                })
        return pd.DataFrame(out).sort_values('momentum_slope', ascending=False).head(top_n)





# ==============================================================================
# Class 2: NepsePredictor - For Forecasting Future Prices
# ==============================================================================



# -------------------------------
# Usage Examples (20)
# -------------------------------

# Assume we already have a connection:
# conn = sqlite3.connect("nepse.db")
# scanner = NepseScanner(conn)

"""
Examples:
1. scanner.get_max_variation_stocks()
2. scanner.get_max_variation_stocks(windowsize=50)
3. scanner.get_max_variation_stocks(top_n=5)
4. scanner.get_long_consolidated_stocks()
5. scanner.get_long_consolidated_stocks(windowsize=90, band_pct=3)
6. scanner.get_support_resistance_levels()
7. supports, resistances = scanner.get_support_resistance_levels(120, 1.5)
8. scanner.get_statistical_summary()
9. scanner.get_statistical_summary(windowsize=120)
10. scanner.get_stock_clusters()
11. scanner.get_stock_clusters(n_clusters=5)
12. scanner.get_current_positive_momentum_stocks()
13. scanner.get_current_positive_momentum_stocks(windowsize=40, top_n=5)
14. df = scanner._load_history(200)
15. swings = scanner._find_swings(df[df.symbol=='NABIL'])
16. levels = scanner._cluster_levels(np.array([500,505,507,600]), 10)
17. scanner.get_max_variation_stocks(windowsize=10, top_n=3)
18. scanner.get_long_consolidated_stocks(windowsize=30)
19. scanner.get_stock_clusters(windowsize=60, n_clusters=3)
20. scanner.get_current_positive_momentum_stocks(windowsize=25)
"""
