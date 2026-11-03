# Data Sources and Access

## Data Source

During my internship, I contributed to parts of the development and maintenance of the team's internal database. The data used in this project came from the company's internal MySQL database and were retrieved with SQL queries.

For privacy and confidentiality reasons, this repo does not include the original data, database credentials, server details, or other internal database materials.

## Data Retrieval Workflow

The research code connected to the database with `PyMySQL` and used `pandas.read_sql()` to run SQL queries and load the results into DataFrames. The workflow shown in the example below was:

1. Query the database for the distinct tickers associated with a selected portfolio or universe.
2. Use that ticker list to query daily closing prices over a chosen date range.
3. Sort the results by date and ticker for downstream analysis and backtesting.

The example is sanitized and illustrative. Replace `source_table`, `selection_field`, and the placeholder values with authorized names and values before running it. Connection details and data are intentionally omitted.

```python
import pandas as pd
import pymysql

conn = pymysql.connect(
    host="<DB_HOST>",
    port=3306,
    user="<DB_USER>",
    password="<DB_PASSWORD>",
    database="<DB_NAME>",
    charset="utf8"
)

try:
    query_tickers = """
    SELECT DISTINCT ticker
    FROM source_table
    WHERE selection_field = %s
    """
    selection_value = "<SELECTION_VALUE>"
    df_tickers = pd.read_sql(
        query_tickers,
        conn,
        params=[selection_value]
    )
    tickers = df_tickers["ticker"].tolist()

    if tickers:
        ticker_marks = ", ".join(["%s"] * len(tickers))
        query_prices = f"""
        SELECT dt, ticker, close
        FROM source_table
        WHERE ticker IN ({ticker_marks})
          AND dt BETWEEN %s AND %s
        ORDER BY dt, ticker
        """
        params = tickers + ["<START_DATE>", "<END_DATE>"]
        df_prices = pd.read_sql(query_prices, conn, params=params)
finally:
    conn.close()
```

The ticker-selection query and the price query are shown as separate steps to illustrate how the database was accessed. Actual table and column names, filters, credentials, and query results are not published here.
