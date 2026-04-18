import time
from datetime import datetime

class StockEngine:
    def __init__(self, symbol, market_status, queue_manager, live_updater, data_queue):
        self.symbol = symbol
        self.market_status = market_status
        self.queue_manager = queue_manager
        self.live_updater = live_updater

        self.previous_market_state = False
        self.backfill_done = False
        self.running = False

        self.data_queue = data_queue

    def stop(self):
        self.running = False

    def _run_cycle(self):
        print("Cycle executed at:", datetime.now())
        is_open = self.market_status.is_market_open()

        # --- Market just opened ---
        if is_open and self.previous_market_state is False:
            self.queue_manager.clear()
            self.backfill_done = False  # reset so next close triggers a fresh backfill

        # --- Market closed ---
        if not is_open:
            if not self.backfill_done:
                self.queue_manager.run_backfill_if_needed(self.symbol)
                self.backfill_done = True

        # --- Transmit every minute regardless of market state ---
        if not self.queue_manager.is_empty():
            if is_open:
                self.data_queue.put({"type": "tick", "data": self.queue_manager.get_latest()})
            else:
                self.data_queue.put({"type": "full_day", "data": self.queue_manager.get_all()})

        # --- Live update if market open ---
        if is_open:
            self.live_updater.update_once()

        self.previous_market_state = is_open

    # --- NEW METHOD ---
    def _sleep_until_next_minute(self):
        now = datetime.now()
        seconds_to_wait = 60 - now.second - now.microsecond / 1_000_000 + 1
        time.sleep(seconds_to_wait)

    def start(self, cycles=7):
        self.running = True
        count = 0

        while self.running and count < cycles:
            self._sleep_until_next_minute()
            self._run_cycle()
            count += 1
