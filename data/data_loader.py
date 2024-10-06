# data_loader.py
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def get_sp500_tickers(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        rows = table.find_all('tr')[1:]  # Skip the header row
        sp500_companies = []
        for row in rows:
            columns = row.find_all('td')
            ticker = columns[0].text.strip()
            name = columns[1].text.strip()
            sp500_companies.append((ticker, name))
        return [company[0] for company in sp500_companies]

    def load_stock_data(self, ticker, start_date, end_date):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_info = yf.Ticker(ticker).info
        sector = stock_info.get('sector', 'Unknown')
        industry = stock_info.get('industry', 'Unknown')
        stock_data['Ticker'] = ticker
        stock_data = stock_data.reset_index()
        return stock_data, sector, industry

    def load_data(self, tickers, start_date, end_date):
        time_series_data = pd.DataFrame()
        stock_info_data = pd.DataFrame()
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            stock_data, sector, industry = self.load_stock_data(ticker, start_date, end_date)
            time_series_data = pd.concat([time_series_data, stock_data], ignore_index=True)
            stock_info_data = pd.concat([stock_info_data, pd.DataFrame({
                'Ticker': [ticker],
                'Sector': [sector],
                'Industry': [industry]
            })], ignore_index=True)
        return time_series_data, stock_info_data