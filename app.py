from task import scrape_and_insert
from datetime import datetime
from db import get_mysql_connection, get_sqlalchemy_engine
from models.nepse_signals import NepseScanner

# Usage example
engine = get_sqlalchemy_engine()


if __name__ == '__main__':
    # scrape_and_insert()
    conn = get_mysql_connection()
    scanner = NepseScanner(engine)
    scanner.get_max_variation_stocks()
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
conn.close()


