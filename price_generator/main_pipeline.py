import multiprocessing
from is_market_closed import MarketStatus
from data_source import YahooDataSource
from QueueBackFillManager import QueueBackfillManager
from live_minute_updater import LiveMinuteUpdater
from StockEngine import StockEngine
from plot_engine.plot_engine import PlotEngine
from logger import get_logger
from signal_bridge.signal_bridge import SignalBridge
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SYMBOL = "RELIANCE.NS"

def _run_plot_engine(data_queue, symbol):
    engine = PlotEngine(data_queue=data_queue, symbol=symbol)
    engine.start()

def _run_signal_bridge(signal_queue, data_queue, symbol):
    bridge = SignalBridge(signal_queue=signal_queue,
                          data_queue=data_queue,
                          symbol=symbol)
    bridge.start()

class StockPipeline:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = get_logger(self.__class__.__name__)
        self.data_queue = multiprocessing.Queue()
        self.signal_queue = multiprocessing.Queue()

        self.market_status = MarketStatus()
        self.data_source = YahooDataSource()
        self.queue_manager = QueueBackfillManager(data_source=self.data_source,
                                                  market_status=self.market_status)
        self.live_updater = LiveMinuteUpdater(symbol=self.symbol,
                                              queue_manager=self.queue_manager,
                                              market_status=self.market_status)

        self.engine = StockEngine(symbol=self.symbol,
                                  market_status=self.market_status,
                                  queue_manager=self.queue_manager,
                                  live_updater=self.live_updater,
                                  data_queue=self.data_queue, signal_queue=self.signal_queue)
        self.plot_engine = PlotEngine(data_queue=self.data_queue,
                                      symbol=self.symbol)


    def start(self):
        self.logger.info(f"Pipeline starting for symbol={self.symbol}")
        try:
            plot_process = multiprocessing.Process(target=_run_plot_engine,
                                                   args=(self.data_queue, self.symbol),
                                                   daemon=True)
            plot_process.start()
            self.logger.info("***PlotEngine process started***")
            signal_process = multiprocessing.Process(target=_run_signal_bridge,
                                                     args=(self.signal_queue,
                                                           self.data_queue,
                                                           self.symbol),
                                                     daemon=True)
            signal_process.start()
            self.logger.info("SignalBridge process started")

            self.engine.start()

        except KeyboardInterrupt:
            self.logger.info("Pipeline stopped by user")
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            self.logger.info("Pipeline shut down")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    pipeline = StockPipeline(symbol=SYMBOL)
    pipeline.start()
