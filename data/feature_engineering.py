# feature_engineering.py
import pandas as pd
import numpy as np
import talib
# from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self):
        pass

    def calculate_rolling_features(self, df):
        df['30D_MA'] = df.groupby('Ticker')['Close'].rolling(window=30).mean().reset_index(0, drop=True)
        df['60D_MA'] = df.groupby('Ticker')['Close'].rolling(window=60).mean().reset_index(0, drop=True)
        df['30D_STD'] = df.groupby('Ticker')['Close'].rolling(window=30).std().reset_index(0, drop=True)
        return df

    def calculate_volatility(self, df):
        df['Log_Return'] = np.log(df.groupby('Ticker')['Close'].pct_change() + 1)
        df['Volatility'] = df.groupby('Ticker')['Log_Return'].rolling(window=20).std().reset_index(0, drop=True)
        return df

    def calculate_talib_features(self, df):
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        return df

    def calculate_lagged_features(self, df):
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df.groupby('Ticker')['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df.groupby('Ticker')['Volume'].shift(i)
        return df

    def calculate_rsi_features(self, df):
        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_28'] = talib.RSI(df['Close'], timeperiod=28)
        return df

    def calculate_bollinger_bands(self, df):
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df

    def calculate_sma_ema_features(self, df):
        df['SMA_50'] = df.groupby('Ticker')['Close'].rolling(window=50).mean().reset_index(0, drop=True)
        df['EMA_20'] = df.groupby('Ticker')['Close'].ewm(span=20, adjust=False).mean().reset_index(0, drop=True)
        return df

    def calculate_obv_sma_ratio(self, df):
        df['OBV_SMA_ratio'] = df['OBV'] / df['SMA_50']
        return df

    def calculate_high_low_range(self, df):
        df['High_Low_Range'] = df['High'] - df['Low']
        return df

    def engineer_features(self, time_series_data, stock_info_data):
        df = time_series_data.copy()
        df = self.calculate_rolling_features(df)
        df = self.calculate_volatility(df)
        df = self.calculate_talib_features(df)
        df = self.calculate_lagged_features(df)
        df = self.calculate_rsi_features(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_sma_ema_features(df)
        df = self.calculate_obv_sma_ratio(df)
        df = self.calculate_high_low_range(df)
        # Fill Returns with 0
        try:
            df['Returns'] = df['Returns'].fillna(0)
            # Handle other columns
            for col in ['SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'Volume_MA_5']:
                df[col] = df.groupby('Ticker')[col].transform(lambda x: x.ffill().bfill())
        except KeyError:
            pass
        time_series_data = df.copy()
        return time_series_data, stock_info_data