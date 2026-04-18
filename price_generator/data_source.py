import yfinance as yf

class YahooDataSource:
    def get_last_session_minutes(self, symbol):
        """
        Returns last trading session minute data
        Format (one dict per bar):
            timestamp, price (Close), Open, High, Low, Close, Volume
        """

        ticker = yf.Ticker(symbol)

        # Pull last 5 days minute data (safe window)
        df = ticker.history(interval="1m", period="5d")

        if df.empty:
            return []

        df = df.reset_index()

        # --- Find Last Trading Day ---
        df["date"] = df["Datetime"].dt.date
        last_day = df["date"].iloc[-1]

        session_df = df[df["date"] == last_day]

        # --- Convert To Tick Format (full OHLCV + price alias) ---
        ticks = []
        for _, row in session_df.iterrows():
            vol = row.get("Volume", 0)
            ticks.append({
                "timestamp": row["Datetime"].to_pydatetime(),
                "price": float(row["Close"]),
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": float(vol) if vol is not None and vol == vol else 0.0,
            })

        return ticks
