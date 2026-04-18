import yfinance as yf
from is_market_closed import MarketStatus

class FetchLatestPrice:
    def __init__(self, symbol: str, start_date: str, end_date: str, market_status: MarketStatus):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self._market_status = market_status.is_market_open()

    def fetch_price(self):
        latest_price = yf.download(tickers=self.symbol, period="1d", interval="1m")
        return latest_price

    def update_price(self):
        if self._market_status:
            print('yes')
        else:
            print("Not open")

lp = FetchLatestPrice(symbol="ESCORTS.NS", start_date="2026-02-12", end_date="2021-02-13", market_status=MarketStatus())
latest_price = lp.fetch_price()
print(latest_price['High']['ESCORTS.NS'].tolist())
print(lp.update_price())