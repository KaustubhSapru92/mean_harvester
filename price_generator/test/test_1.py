from is_market_closed import MarketStatus
from data_source import YahooDataSource
from QueueBackFillManager import QueueBackfillManager
from live_minute_updater import LiveMinuteUpdater
from StockEngine import StockEngine

symbol = "RELIANCE.NS"

market_status = MarketStatus()
data_source = YahooDataSource()

queue_manager = QueueBackfillManager(data_source=data_source, market_status=market_status)

live_updater = LiveMinuteUpdater(symbol=symbol, queue_manager=queue_manager, market_status=market_status)

print(market_status.is_market_open())
print(data_source.get_last_session_minutes(symbol=symbol))


queue_manager.run_backfill_if_needed(symbol)

print("Queue size after backfill:", queue_manager.size())

live_updater.update_once()

print("Latest tick:", queue_manager.get_latest())

engine = StockEngine(
    symbol=symbol,
    market_status=market_status,
    queue_manager=queue_manager,
    live_updater=live_updater
)

engine._run_cycle()

market_status.is_market_open = lambda: True

engine.start()

print("Queue size after market open transition:", queue_manager.size())
