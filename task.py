import logging
from datetime import datetime
from scraper import initiate_scraper
from db import get_mysql_connection, insert_data_for_date, query_by_symbol

# Configure logging
logging.basicConfig(
    filename="schedule.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def scrape_and_insert(today=datetime.today()):
    today_weekday = datetime.today().weekday()  # Monday=0, Sunday=6
    try:
        if today_weekday in [6, 0, 1, 2, 3]:  # Sun–Thu
            logging.info("Starting scraper and database insert. has been started")

            # Run scraper
            initiate_scraper()

            # Connect to MySQL
            mysql_connection = get_mysql_connection()

            # Insert CSV data
            insert_data_for_date(today, mysql_connection)

            # Query sample symbol
            query_by_symbol(mysql_connection, "SNLI")

            mysql_connection.close()
            logging.info(f"Task completed successfully. Database connection closed. {today}")
        else:
            logging.info(f"Skipped today (Friday or Saturday).- {today}")
    except Exception as e:
        logging.error(f"Error while running task: {e}", exc_info=True)