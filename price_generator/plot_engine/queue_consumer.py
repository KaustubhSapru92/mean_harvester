import threading
import queue
from .logger import get_logger

logger = get_logger("QueueConsumer")


class QueueConsumer:
    def __init__(self, data_queue: queue.Queue):
        self.data_queue = data_queue
        self._lock = threading.Lock()
        self._ticks = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._max_len = 500
        self._latest_dev_z = {"DEV_20": [], "DEV_50": []}
        self._market_open = False
        self._vwap_df = None
        self._stability_map = {}

    def start(self):
        self._thread.start()
        logger.info("QueueConsumer polling thread started")

    def get_ticks(self):
        with self._lock:
            return list(self._ticks)

    def get_dev_z(self):
        with self._lock:
            return {k: list(v) for k, v in self._latest_dev_z.items()}

    def is_market_open(self):
        with self._lock:
            return self._market_open

    def get_vwap_df(self):
        with self._lock:
            return self._vwap_df

    def get_stability_map(self):
        with self._lock:
            return dict(self._stability_map)

    def _poll(self):
        while True:
            try:
                message = self.data_queue.get_nowait()
                msg_type = message.get("type")
                data = message.get("data")

                with self._lock:
                    if msg_type == "full_day":
                        self._ticks = data
                        logger.debug(f"Full day data received — {len(data)} ticks")
                    elif msg_type == "tick":
                        self._ticks.append(data)
                        logger.debug(f"Live tick received: {data}")
                    elif msg_type == "signal":
                        for key, value in data.items():
                            self._latest_dev_z[key].append(value)
                            if len(self._latest_dev_z[key]) > self._max_len:
                                self._latest_dev_z[key].pop(0)
                    elif msg_type == "status":
                        self._market_open = data
                    elif msg_type == "diagnostics":
                        self._vwap_df = data["vwap_df"]
                        self._stability_map = data["stability_map"]
                        logger.debug("Diagnostics received")

            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)

            threading.Event().wait(1)