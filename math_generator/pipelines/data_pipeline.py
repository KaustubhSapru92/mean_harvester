from datetime import datetime, timedelta

from math_generator.workflow.fetcher import YahooFetcher, LocalFileFetcher
from math_generator.workflow.cleaner import OHLVCleaner
from math_generator.workflow.resampler import OHLCVResampler


def run_data_pipeline(config):
    """
    Fetches, cleans, and optionally resamples OHLCV data.

    Data source is controlled by config:
        data_source.type == "yahoo"  → YahooFetcher (live, symbol-based)
        data_source.type == "local"  → LocalFileFetcher (CSV on disk)

    When type is "local" and data_source.resample_to is set, the raw
    1m bars from the CSV are resampled to that interval BEFORE being
    passed to OHLVCleaner and the rest of the pipeline.  This keeps the
    contract identical to the Yahoo path where base_interval drives the
    bar frequency.

    Returns
    -------
    clean_df     : cleaned OHLCV DataFrame at base_interval frequency
    resampled_map: { timeframe_str: DataFrame } for any additional
                   timeframes configured under resample.timeframes
    """

    # ---------------------------------------------------------------
    # 1. Resolve data source type
    # ---------------------------------------------------------------
    source_type = (config.get("data_source", "type") or "yahoo").lower()

    # ---------------------------------------------------------------
    # 2. Fetch raw data
    # ---------------------------------------------------------------
    if source_type == "local":
        raw_df = _fetch_local(config)
    elif source_type == "yahoo":
        raw_df = _fetch_yahoo(config)
    else:
        raise ValueError(
            f"run_data_pipeline: unknown data_source.type '{source_type}'. "
            f"Must be 'yahoo' or 'local'."
        )

    # ---------------------------------------------------------------
    # 3. If local and resample_to is set, bring raw bars up to
    #    base_interval before cleaning.  Yahoo already returns data
    #    at the requested interval, so this step is local-only.
    # ---------------------------------------------------------------
    if source_type == "local":
        resample_to = config.get("data_source", "resample_to")
        if resample_to:
            raw_df = OHLCVResampler.resample(raw_df, resample_to)

    # ---------------------------------------------------------------
    # 4. Clean
    # ---------------------------------------------------------------
    clean_df = OHLVCleaner.clean(raw_df)

    # ---------------------------------------------------------------
    # 5. Optional additional resamples (same for both paths)
    # ---------------------------------------------------------------
    resample_enabled = config.get("resample", "enabled") or False
    resample_tfs     = config.get("resample", "timeframes") or []

    resampled_map = {}
    if resample_enabled:
        for tf in resample_tfs:
            resampled_map[tf] = OHLCVResampler.resample(clean_df, tf)

    return clean_df, resampled_map


# ---------------------------------------------------------------------------
# Private helpers — keep fetch logic out of the main function
# ---------------------------------------------------------------------------

def _fetch_yahoo(config) -> object:
    """Fetch from Yahoo Finance using symbol + lookback_days from config."""
    symbol       = config.get("market_data", "symbol")
    interval     = config.get("market_data", "base_interval")
    lookback_days = config.get("market_data", "lookback_days")

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    fetcher = YahooFetcher(symbol, interval)
    return fetcher.fetch(start_date, end_date)


def _fetch_local(config) -> object:
    """
    Load a local CSV file via LocalFileFetcher.

    Optionally filters to a date window if local_start / local_end are
    set under data_source in config.  If not set, the full CSV is loaded
    (all available history).
    """
    file_path = config.get("data_source", "file_path")
    if not file_path:
        raise ValueError(
            "data_source.type is 'local' but data_source.file_path is not set in config."
        )

    # Optional date window — useful for smoke tests or partial loads
    start_str = config.get("data_source", "local_start")   # e.g. "2020-01-01"
    end_str   = config.get("data_source", "local_end")     # e.g. "2024-01-01"

    start = datetime.strptime(start_str, "%Y-%m-%d") if start_str else None
    end   = datetime.strptime(end_str,   "%Y-%m-%d") if end_str   else None

    fetcher = LocalFileFetcher(file_path)
    return fetcher.fetch(start=start, end=end)