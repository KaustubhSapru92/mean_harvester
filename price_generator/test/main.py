from is_market_closed import MarketStatus
from data_source import YahooDataSource
from QueueBackFillManager import QueueBackfillManager
from live_minute_updater import LiveMinuteUpdater
from StockEngine import StockEngine
from logger import get_logger

SYMBOL = "RELIANCE.NS"

class StockPipeline:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = get_logger(self.__class__.__name__)

        self.market_status = MarketStatus()
        self.data_source = YahooDataSource()
        self.queue_manager = QueueBackfillManager(data_source=self.data_source, market_status=self.market_status)
        self.live_updater = LiveMinuteUpdater(symbol=self.symbol, queue_manager=self.queue_manager,
                                              market_status=self.market_status)
        self.engine = StockEngine(symbol=self.symbol,
                                  market_status=self.market_status,
                                  queue_manager=self.queue_manager,
                                  live_updater=self.live_updater)

    def start(self):
        self.logger.info(f"Pipeline starting for symbol={self.symbol}")
        try:
            self.engine.start()
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            self.logger.info("Pipeline shut down")


if __name__ == "__main__":
    pipeline = StockPipeline(symbol=SYMBOL)
    pipeline.start()