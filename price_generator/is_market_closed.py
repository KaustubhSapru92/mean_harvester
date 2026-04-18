from datetime import datetime, time
import pytz


class MarketStatus:
    def __init__(self, timezone="Asia/Kolkata", open_time=time(9, 15),
                 close_time=time(15, 30),
                 trading_days=None,):  # Mon=0 ... Sun=6
        if trading_days is None:
            trading_days = {0, 1, 2, 3, 4}
        self.tz = pytz.timezone(timezone)
        self.open_time = open_time
        self.close_time = close_time
        self.trading_days = trading_days

    def is_market_open(self, current_dt=None):
        if current_dt is None:
            current_dt = datetime.now(self.tz)

        # Check weekday
        if current_dt.weekday() not in self.trading_days:
            return False

        # Check time window
        current_time = current_dt.time()
        return self.open_time <= current_time <= self.close_time