"""Records a security's historical data using the Alpha Vantage API."""
import numpy as np

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators


class Generator:
    def __init__(self, symbol):
        self.symbol = symbol

        # My Alpha Vantage API Key
        self.api_key = 'EXLOXHXU9RCAEE9W'

        # Create a TimeSeries object to track the security's price
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')

        # Create a TechIndicators object to track the security's indicators
        self.ti = TechIndicators(key=self.api_key, output_format='pandas')

    def get_symbol(self):
        return self.symbol

    def set_symbol(self, symbol):
        self.symbol = symbol

    def get_data(self, print_prices):
        """Creates a DataFrame that contains a stock's price and indicator history."""
        # Pull the security's price history from Alpha Advantage's servers
        prices, meta = self.ts.get_daily(symbol=self.symbol, outputsize='full')

        # Pull the security's indicators' history from Alpha Advantage's servers
        adx, meta = self.ti.get_adx(symbol=self.symbol, interval='daily', time_period=14)
        macd, meta = self.ti.get_macd(symbol=self.symbol, interval='daily', series_type='close')
        obv, meta = self.ti.get_obv(symbol=self.symbol, interval='daily')
        rsi, meta = self.ti.get_rsi(symbol=self.symbol, interval='daily', series_type='close')

        # Add additional indicator features to ADX DF
        adx['MACD'] = macd['MACD_Signal'][::-1]
        adx['OBV'] = obv
        adx['RSI'] = rsi

        # The 'prices' column is inverted so that the oldest entries come first
        # Only the closing price will be considered for classification
        prices = prices['4. close'][:len(adx)][::-1]

        # Makes the output a lot cleaner when collecting data from many securities
        if print_prices:
            print(prices)

        # Add a column for the security's target class (10% price rally)
        adx['class'] = [Generator.get_class(prices[i], prices[i:]) for i in range(len(prices))]

        # Return the DataFrame
        return adx.dropna()

    @staticmethod
    def get_class(orig_price, next_prices):
        """Determines if the next absolute 10% rally will be positive or negative."""
        for cur_price in next_prices:
            if abs((cur_price - orig_price) / orig_price) > .1:
                # Return an int to enable classification
                return int(cur_price > orig_price)

        return np.NaN


def main():
    generator = Generator('IBM')

    # Print data to show it working
    print(generator.get_data(True))


if __name__ == '__main__':
    main()
