
import pandas as pd 
import numpy as np
import os
import datetime
import pandas_market_calendars as mcal
from scipy import stats

def get_trading_dates():
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

start_date, end_date = get_trading_dates()
# end_date_str = end_date.strftime('%Y-%m-%d')

# Define the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
# Ensure end_date is formatted as a string
file_path = os.path.join(parent_dir, f'data/prepared_data_{end_date}.csv')  # Ensure end_date is a string
print(file_path)



# Function to calculate additional technical indicators
def calculate_technical_indicators(df):
    # Simple Moving Average (SMA) and Exponential Moving Average (EMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)

    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(14)

    # 52-week High and Low
    df['52W_High'] = df['Close'].rolling(window=252).max()
    df['52W_Low'] = df['Close'].rolling(window=252).min()

    return df

# Main function to label stock
def label_stock(df):
    if len(df) < 252:  # Minimum 1 year of data
        df['Label_Daily'] = ['Insufficient Data'] * len(df)
        df['Label_2_Year'] = 'Insufficient Data'
        return df

    df = df.sort_values('Date')
    
    try:
        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Calculate returns
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1

        # Linear regression for trend (calculating slope)
        x = np.arange(len(df))
        slope, _, _, _, _ = stats.linregress(x, df['Close'])

        # Volatility
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Define 52-week metrics
        pct_from_52w_high = (df['Close'].iloc[-1] - df['52W_High'].iloc[-1]) / df['52W_High'].iloc[-1]
        pct_from_52w_low = (df['Close'].iloc[-1] - df['52W_Low'].iloc[-1]) / df['52W_Low'].iloc[-1]

        # RSI-based conditions
        current_rsi = df['RSI_14'].iloc[-1]

        # Momentum
        momentum = df['Momentum'].iloc[-1]

        # 2-Year Label (incorporating slope)
        if total_return > 0.2 and current_rsi > 60 and pct_from_52w_high < 0.1 and slope > 0:
            label_2_year = 'Strong Up'
        elif total_return > 0.05 and current_rsi > 50 and momentum > 0 and slope > 0:
            label_2_year = 'Moderate Up'
        elif total_return < -0.2 and current_rsi < 30 and pct_from_52w_low > -0.1 and slope < 0:
            label_2_year = 'Strong Down'
        elif total_return < -0.05 and current_rsi < 50 and momentum < 0 and slope < 0:
            label_2_year = 'Moderate Down'
        elif volatility > 0.4:
            label_2_year = 'Volatile'
        else:
            label_2_year = 'Flat'

        # Daily labels (simplified)
        daily_returns = df['Close'].pct_change()
        daily_labels = pd.cut(daily_returns, 
                              bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf], 
                              labels=['Strong Down', 'Weak Down', 'Flat', 'Weak Up', 'Strong Up'])

        df['Label_Daily'] = daily_labels
        df['Label_2_Year'] = label_2_year
        return df

    except Exception as e:
        print(f"Error processing group: {e}")
        df['Label_Daily'] = ['Error'] * len(df)
        df['Label_2_Year'] = 'Error'
        return df

if file_path:
    # df = pd.read_csv(f'/Users/nitastha/Desktop/NitishFiles/Projects/Newrelic/prepared_data_{end_date}.csv')
    df = pd.read_csv(file_path)
    stocks = df['Ticker'].unique()

    results = []
    for stock in stocks:
        stock_df = df[df['Ticker'] == stock]
        labeled_stock_df = label_stock(stock_df)
        results.append(labeled_stock_df)

    final_df = pd.concat(results, axis=0, ignore_index=True)
    # final_df.to_csv(f'public_{end_date}.csv', index=False)
    final_df.to_csv(f'public/public_data({end_date}).csv', index=False)
    final_df.to_csv(f'data/training_data.csv', index=False)
    # Display distribution of labels
    print('=================================')
    print("df shape : ", df.shape)
    print("final_df shape: ", final_df.shape)
    print('=================================')
    print("\nDaily Label Distribution:")
    print(final_df['Label_Daily'].value_counts(dropna=False))
    print("\n2-Year Label Distribution:")
    print(final_df['Label_2_Year'].value_counts(dropna=False))


else:
    print("No file found.")

