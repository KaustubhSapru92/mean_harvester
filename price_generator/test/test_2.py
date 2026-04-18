from is_market_closed import MarketStatus
from data_source import YahooDataSource
from QueueBackFillManager import QueueBackfillManager
from live_minute_updater import LiveMinuteUpdater
from StockEngine import StockEngine
from logger import get_logger

logger = get_logger("main")

SYMBOL = "RELIANCE.NS"


def build_pipeline(symbol: str):
    """Constructs and wires all pipeline components."""
    logger.info(f"Building pipeline for symbol={symbol}")

    market_status = MarketStatus()
    logger.info("MarketStatus created")

    data_source = YahooDataSource()
    logger.info("YahooDataSource created")

    queue_manager = QueueBackfillManager(
        data_source=data_source,
        market_status=market_status
    )
    logger.info("QueueBackfillManager created")

    live_updater = LiveMinuteUpdater(
        symbol=symbol,
        queue_manager=queue_manager,
        market_status=market_status
    )
    logger.info("LiveMinuteUpdater created")

    engine = StockEngine(
        symbol=symbol,
        market_status=market_status,
        queue_manager=queue_manager,
        live_updater=live_updater
    )
    logger.info("StockEngine created")

    return engine, queue_manager, market_status


def run_diagnostics(queue_manager, market_status, symbol: str):
    """Runs startup checks and loggers current state."""
    logger.info("=== Running diagnostics ===")

    is_open = market_status.is_market_open()
    logger.info(f"Market open: {is_open}")

    session_data = queue_manager.data_source.get_last_session_minutes(symbol=symbol)
    logger.info(f"Last session tick count: {len(session_data) if session_data else 0}")

    queue_manager.run_backfill_if_needed(symbol)
    logger.info(f"Queue size after backfill: {queue_manager.size()}")

    latest = queue_manager.get_latest()
    logger.info(f"Latest tick after backfill: {latest}")

    logger.info("=== Diagnostics complete ===")


def main():
    logger.info("========== Pipeline starting ==========")

    engine, queue_manager, market_status = build_pipeline(SYMBOL)

    run_diagnostics(queue_manager, market_status, SYMBOL)

    logger.info("Running single test cycle")
    engine._run_cycle()

    logger.info("Starting live engine loop")
    engine.start()

    logger.info("========== Pipeline shut down ==========")


if __name__ == "__main__":
    main()