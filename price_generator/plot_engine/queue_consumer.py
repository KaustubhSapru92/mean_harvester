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

    def start(self):
        self._thread.start()
        logger.info("QueueConsumer polling thread started")

    def get_ticks(self):
        with self._lock:
            return list(self._ticks)

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

            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)

            threading.Event().wait(1)