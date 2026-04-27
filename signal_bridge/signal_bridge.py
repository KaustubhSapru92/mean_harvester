import multiprocessing
from math_generator.workflow.fetcher import YahooFetcher
from math_generator.workflow.cleaner import OHLVCleaner
from math_generator.workflow.bars_adapter import bars_to_ohlcv_dataframe
from math_generator.workflow.vwap import VWAPCalculator
from math_generator.diagnostics.rolling_vwap_stability import RollingVWAPStability
from datetime import datetime, timedelta


class SignalBridge:
    def __init__(self, signal_queue, data_queue, symbol,
                 vwap_windows=[20, 50], roll_len=100):
        self.signal_queue = signal_queue
        self.data_queue = data_queue
        self.symbol = symbol
        self.vwap_windows = vwap_windows
        self.roll_len = roll_len
        self.live_df = None

    def _bootstrap(self):
        fetcher = YahooFetcher(symbol=self.symbol, interval="1m")
        end = datetime.now()
        start = end - timedelta(days=5)
        raw_df = fetcher.fetch(start=start, end=end)
        clean_df = OHLVCleaner.clean(raw_df)
        self.live_df = clean_df

    def _compute_and_push(self):
        if self.live_df is None or self.live_df.empty:
            return

        vwap_df = VWAPCalculator.compute_rolling_vwap(
            self.live_df, windows=self.vwap_windows
        )

        stability_map = {}
        for w in self.vwap_windows:
            stability = RollingVWAPStability.compute(
                vwap_df, window=w, roll_len=self.roll_len
            )
            if not stability.empty:
                stability_map[w] = stability

        self.data_queue.put({
            "type": "diagnostics",
            "data": {
                "vwap_df": vwap_df,
                "stability_map": stability_map
            }
        })

    def _process_tick(self, tick):
        import pandas as pd
        new_row = bars_to_ohlcv_dataframe([tick])
        self.live_df = pd.concat([self.live_df, new_row]).sort_index()

    def start(self):
        self._bootstrap()
        self._compute_and_push()

        while True:
            try:
                message = self.signal_queue.get()
                msg_type = message.get("type")
                data = message.get("data")

                if msg_type == "tick":
                    self._process_tick(data)
                    self._compute_and_push()
                elif msg_type == "full_day":
                    self.live_df = bars_to_ohlcv_dataframe(data)
                    self._compute_and_push()

            except Exception as e:
                print(f"SignalBridge error: {e}")