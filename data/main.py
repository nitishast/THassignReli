# main.py
import pandas as pd
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
import datetime
import pandas as pd
import pandas_market_calendars as mcal

class DataProcessor:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()

    def get_trading_dates(self):
        # Get today's date
        today = datetime.date.today()

        # Get the NYSE calendar
        nyse = mcal.get_calendar('NYSE')

        # Get the last trading day
        schedule = nyse.schedule(start_date=today - datetime.timedelta(days=10), end_date=today)
        last_trading_day = schedule.iloc[-1].name.date()

        # Define start and end dates
        start_date = last_trading_day - datetime.timedelta(days=735)
        end_date = last_trading_day - datetime.timedelta(days=1)
        
        return start_date, end_date

    def process_data(self):
        start_date, end_date = self.get_trading_dates()
        tickers = self.data_loader.get_sp500_tickers()
        time_series_data, stock_info_data = self.data_loader.load_data(tickers, start_date, end_date)
        time_series_data, stock_info_data = self.feature_engineer.engineer_features(time_series_data, stock_info_data)
        # save the timeseries data to csv with the date given as second argument in load_data function
        print('========SAMPLED DATA=========================')
        time_series_data.to_csv(f'data/prepared_data_{end_date}.csv', index=False)
        print('=================================')
        print(time_series_data.sample(5))

if __name__ == '__main__':
    data_processor = DataProcessor()
    data_processor.process_data()