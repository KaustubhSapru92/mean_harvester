import queue
from dash import Dash
from .queue_consumer import QueueConsumer
from .layout import build_layout
from .callbacks import register_callbacks
from .logger import get_logger


class PlotEngine:
    def __init__(self, data_queue: queue.Queue, symbol: str = ""):
        self.logger = get_logger(self.__class__.__name__)
        self._consumer = QueueConsumer(data_queue)
        self._app = Dash(__name__)
        self._app.layout = build_layout(symbol)
        register_callbacks(self._app, self._consumer)

    def start(self):
        self.logger.info("PlotEngine starting on http://localhost:8050")
        self._consumer.start()
        self._app.run(debug=False, host="0.0.0.0", port=8050)
