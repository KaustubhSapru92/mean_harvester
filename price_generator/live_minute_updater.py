import yfinance as yf
class LiveMinuteUpdater:
    def __init__(self, symbol, queue_manager, market_status):
        self.symbol = symbol
        self.queue_manager = queue_manager
        self.market_status = market_status
        self.ticker = yf.Ticker(symbol)

        self.last_timestamp = None

    def update_once(self):
        """
        Call this every minute from scheduler
        """

        if not self.market_status.is_market_open():
            return

        df = self.ticker.history(interval="1m", period="1d")

        if df.empty:
            return

        latest_row = df.tail(1).iloc[0]
        latest_ts = df.tail(1).index[0].to_pydatetime()

        # Prevent duplicate insert
        if self.last_timestamp == latest_ts:
            return

        vol = latest_row.get("Volume", 0)
        tick = {
            "timestamp": latest_ts,
            "price": float(latest_row["Close"]),
            "Open": float(latest_row["Open"]),
            "High": float(latest_row["High"]),
            "Low": float(latest_row["Low"]),
            "Close": float(latest_row["Close"]),
            "Volume": float(vol) if vol is not None and vol == vol else 0.0,
        }

        self.queue_manager.push(tick)
        self.last_timestamp = latest_ts
