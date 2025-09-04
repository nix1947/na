import os
import pandas as pd
from datetime import datetime
import sqlite3
import csv
import mysql.connector


# --- CONFIGURATION ---
downloads_dir = './downloads'
table_name = 'price_history'
log_file = 'log.txt'



import mysql.connector

def get_mysql_connection():
    """Sets up the MySQL database and creates the table if it doesn't exist."""
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",  # update if needed
        database="nepse"  # make sure this DB exists
    )
    cursor = conn.cursor()
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        Id INT,
        BusinessDate VARCHAR(20),
        SecurityId INT,
        Symbol VARCHAR(20),
        SecurityName VARCHAR(255),
        OpenPrice DOUBLE,
        HighPrice DOUBLE,
        LowPrice DOUBLE,
        ClosePrice DOUBLE,
        TotalTradedQuantity DOUBLE,
        TotalTradedValue DOUBLE,
        PreviousDayClosePrice DOUBLE,
        FiftyTwoWeekHigh DOUBLE,
        FiftyTwoWeekLow DOUBLE,
        LastUpdatedTime VARCHAR(50),
        LastUpdatedPrice DOUBLE,
        TotalTrades INT,
        AverageTradedPrice DOUBLE,
        MarketCapitalization DOUBLE,
        PRIMARY KEY (BusinessDate, SecurityId)
    )
    ''')
    conn.commit()
    return conn


from sqlalchemy import create_engine
import pandas as pd

def get_sqlalchemy_engine():
    """Create SQLAlchemy engine using the same MySQL credentials."""
    user = "root"
    password = "root"
    host = "localhost"
    port = 3306
    database = "nepse"

    # Using mysql-connector driver
    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}")
    return engine





def insert_data_for_date(date, conn):
    """
    Inserts data from the CSV file for a given date into a MySQL database.
    Logs any problematic rows and continues processing other rows.
    """
    cursor = conn.cursor()
    file_name = f"Today's Price - {date.strftime('%Y-%m-%d')}.csv"
    file_path = os.path.join(downloads_dir, file_name)

    if not os.path.exists(file_path):
        print(f"File not found, skipping: {file_name}")
        with open(log_file, 'a') as log:
            log.write(f"File not found: {file_name}\n")
        return

    insert_query = f'''
    INSERT IGNORE INTO {table_name} (
        Id, BusinessDate, SecurityId, Symbol, SecurityName, OpenPrice, HighPrice, LowPrice, ClosePrice,
        TotalTradedQuantity, TotalTradedValue, PreviousDayClosePrice, FiftyTwoWeekHigh, FiftyTwoWeekLow,
        LastUpdatedTime, LastUpdatedPrice, TotalTrades, AverageTradedPrice, MarketCapitalization
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row

            for row_num, row in enumerate(reader, start=2):  # start=2 → account for header
                try:
                    if len(row) != 19:
                        raise ValueError(f"Expected 19 columns, but found {len(row)}")

                    cursor.execute(insert_query, (
                        int(row[0]),
                        str(row[1]),
                        int(row[2]),
                        str(row[3]),
                        str(row[4]),
                        float(row[5]),
                        float(row[6]),
                        float(row[7]),
                        float(row[8]),
                        float(row[9]),
                        float(row[10]),
                        float(row[11]),
                        float(row[12]),
                        float(row[13]),
                        str(row[14]),
                        float(row[15]),
                        int(row[16]),
                        float(row[17]),
                        float(row[18])
                    ))
                except Exception as e:
                    with open(log_file, 'a') as log:
                        log.write(
                            f"Error processing row {row_num} from '{file_name}': {e} | Row Data: {row}\n"
                        )
                    continue  # Skip to the next line of the CSV

        conn.commit()
        print(f"Successfully processed file: {file_name}")

    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Error opening or reading file '{file_name}': {e}\n")
        print(f"Skipping file due to a file-level error: {file_name}")



def query_all_data(conn):
    """Queries and returns all data from the price_history table."""
    if conn is None:
        return pd.DataFrame() # Return an empty DataFrame on error
    
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        return df
    except pd.io.sql.DatabaseError as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

def query_by_date(conn, business_date):
    """
    Queries data for a specific BusinessDate.
    
    Args:
        conn (sqlite3.Connection): The database connection object.
        business_date (str): The date to query in 'YYYY-MM-DD' format.
    """
    if conn is None:
        return pd.DataFrame()
        
    try:
        query = f"SELECT * FROM {table_name} WHERE BusinessDate = ?"
        df = pd.read_sql_query(query, conn, params=(business_date,))
        return df
    except pd.io.sql.DatabaseError as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

def query_by_symbol(conn, symbol):
    """
    Queries data for a specific stock symbol using MySQL.

    Args:
        conn: The MySQL connection object.
        symbol (str): The stock symbol to query.
    """
    if conn is None:
        return pd.DataFrame()

    try:
        query = f"SELECT * FROM {table_name} WHERE Symbol = %s"
        df = pd.read_sql_query(query, conn, params=[symbol])
        
        print(f"Data of Symbol {symbol}")
        print(df)
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()
        
def query_by_date_and_symbol(conn, business_date, symbol):
    """
    Queries data for a specific date and stock symbol.
    
    Args:
        conn (sqlite3.Connection): The database connection object.
        business_date (str): The date to query in 'YYYY-MM-DD' format.
        symbol (str): The stock symbol to query.
    """
    if conn is None:
        return pd.DataFrame()

    try:
        query = f"SELECT * FROM {table_name} WHERE BusinessDate = ? AND Symbol = ?"
        df = pd.read_sql_query(query, conn, params=(business_date, symbol))
        return df
    except pd.io.sql.DatabaseError as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()



