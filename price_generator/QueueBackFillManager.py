from collections import deque
from is_market_closed import MarketStatus
from data_source import YahooDataSource
from datetime import datetime

class QueueBackfillManager:
    def __init__(self, data_source: YahooDataSource = YahooDataSource,
                 market_status: MarketStatus = MarketStatus,
                 max_size=500):
        self.queue = deque(maxlen=max_size)
        self.data_source = data_source
        self.market_status = market_status

    # =====================
    # Queue Operations
    # =====================
    def push(self, tick):
        self.queue.append(tick)

    def clear(self):
        self.queue.clear()

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

    def get_latest(self):
        if self.is_empty():
            return None
        return self.queue[-1]

    def get_all(self):
        return list(self.queue)

    # Backfill Logic
    def run_backfill_if_needed(self, symbol, force=False):
        """
        Backfills ONLY if:
        - Market is closed
        - Queue is empty
        """
        if self.market_status.is_market_open(current_dt=datetime.now()) and not self.is_empty():
            return

        self.clear()
        self._backfill_last_session(symbol)

    def _backfill_last_session(self, symbol):
        data = self.data_source.get_last_session_minutes(symbol)

        for tick in data:
            self.push(tick)
