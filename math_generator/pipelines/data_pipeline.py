from datetime import datetime, timedelta

from workflow.fetcher import YahooFetcher
from workflow.cleaner import OHLVCleaner
from workflow.resampler import OHLCVResampler


def run_data_pipeline(config):

    symbol = config.get("market_data", "symbol")
    interval = config.get("market_data", "base_interval")
    lookback_days = config.get("market_data", "lookback_days")

    #resample_cfg = config.get("resample")
    resample_enabled = config.get("resample", "enabled") or False
    resample_tfs     = config.get("resample", "timeframes") or []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    fetcher = YahooFetcher(symbol, interval)
    raw_df = fetcher.fetch(start_date, end_date)

    clean_df = OHLVCleaner.clean(raw_df)

    resampled_map = {}

    if resample_enabled:
        for tf in resample_tfs:
            resampled_map[tf] = OHLCVResampler.resample(clean_df, tf)

    return clean_df, resampled_map
