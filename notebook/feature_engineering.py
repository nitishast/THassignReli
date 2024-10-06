# %%
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# %%
df = pd.read_csv('../time_series_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.head()

# %%
def engineer_features(group):
    group = group.sort_values('Date')
    group['Returns'] = group['Close'].pct_change()
    group['SMA_50'] = group['Close'].rolling(window=50).mean()
    group['EMA_20'] = group['Close'].ewm(span=20, adjust=False).mean()
    group['RSI'] = RSIIndicator(group['Close']).rsi()
    bb = BollingerBands(group['Close'])
    group['BB_upper'] = bb.bollinger_hband()
    group['BB_lower'] = bb.bollinger_lband()
    group['Volume_MA_5'] = group['Volume'].rolling(window=5).mean()
    return group

# %%
# def engineer_features(df):
#     # Calculate returns
#     df['Returns'] = df['Close'].pct_change()
    
#     # Moving averages
#     df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
#     df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    
#     # RSI
#     df['RSI'] = RSIIndicator(df['Close']).rsi()
    
#     # Bollinger Bands
#     bb = BollingerBands(df['Close'])
#     df['BB_upper'] = bb.bollinger_hband()
#     df['BB_lower'] = bb.bollinger_lband()
    
#     # Volume features
#     df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    
#     return df

df = df.groupby('Ticker').apply(engineer_features).reset_index(drop=True)


# %%
# Display the last 5 rows for 3 different stocks
for ticker in df['Ticker'].unique()[:3]:
    print(f"\nLast 5 rows for {ticker}:")
    print(df[df['Ticker'] == ticker].tail())

# %%
df

# %%
df.isnull().sum()

# %%
# Fill Returns with 0
df['Returns'] = df['Returns'].fillna(0)

# For SMA, RSI, BB, and Volume_MA, we'll leave as NaN
# Alternatively, you could forward fill:
# Handle other columns
for col in ['SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'Volume_MA_5']:
    df[col] = df.groupby('Ticker')[col].transform(lambda x: x.ffill().bfill())

# For any remaining NaNs, fill with the column mean
# df = df.fillna(df.mean())

# Check null values again
print(df.isnull().sum())

# %%
df.head(50)

# %%
def label_stock(group):
    if len(group) < 22:  # Not enough data for recent return calculation
        return pd.Series(['Insufficient Data'] * len(group))
    
    group = group.sort_values('Date')
    
    try:
        x = np.arange(len(group))
        slope, _, _, _, _ = stats.linregress(x, group['Close'])
        
        total_return = (group['Close'].iloc[-1] / group['Close'].iloc[0]) - 1
        recent_return = (group['Close'].iloc[-1] / group['Close'].iloc[-22]) - 1
        
        if total_return > 0.2 and recent_return > 0.05:
            label = 'Strong Up'
        elif total_return > 0.1 and recent_return > 0:
            label = 'Moderate Up'
        elif total_return > 0.05 or (slope > 0 and recent_return > 0):
            label = 'Weak Up'
        elif total_return < -0.2 and recent_return < -0.05:
            label = 'Strong Down'
        elif total_return < -0.1 and recent_return < 0:
            label = 'Moderate Down'
        elif total_return < -0.05 or (slope < 0 and recent_return < 0):
            label = 'Weak Down'
        else:
            label = 'Flat'
        
        return pd.Series([label] * len(group))
    except Exception as e:
        print(f"Error processing group: {e}")
        return pd.Series(['Error'] * len(group))

# Apply labeling to each stock
df_grouped = df.groupby('Ticker')
df['Label'] = df_grouped.apply(label_stock).reset_index(drop=True)['Label']

# Display distribution of labels
print(df['Label'].value_counts(normalize=True))

# Check for any stocks with 'Insufficient Data' or 'Error' labels
problematic_stocks = df[df['Label'].isin(['Insufficient Data', 'Error'])]['Ticker'].unique()
if len(problematic_stocks) > 0:
    print(f"Stocks with issues: {problematic_stocks}")

# %%
def label_stock(group):
    if len(group) < 22:  # Not enough data for recent return calculation
        return pd.DataFrame({'Label': ['Insufficient Data'] * len(group)})
    
    group = group.sort_values('Date')
    
    try:
        x = np.arange(len(group))
        slope, _, _, _, _ = stats.linregress(x, group['Close'])
        
        total_return = (group['Close'].iloc[-1] / group['Close'].iloc[0]) - 1
        recent_return = (group['Close'].iloc[-1] / group['Close'].iloc[-22]) - 1
        
        if total_return > 0.2 and recent_return > 0.05:
            label = 'Strong Up'
        elif total_return > 0.1 and recent_return > 0:
            label = 'Moderate Up'
        elif total_return > 0.05 or (slope > 0 and recent_return > 0):
            label = 'Weak Up'
        elif total_return < -0.2 and recent_return < -0.05:
            label = 'Strong Down'
        elif total_return < -0.1 and recent_return < 0:
            label = 'Moderate Down'
        elif total_return < -0.05 or (slope < 0 and recent_return < 0):
            label = 'Weak Down'
        else:
            label = 'Flat'
        
        return pd.DataFrame({'Label': [label] * len(group)})
    except Exception as e:
        print(f"Error processing group: {e}")
        return pd.DataFrame({'Label': ['Error'] * len(group)})

# Apply labeling to each stock
df_grouped = df.groupby('Ticker')
df['Label'] = df_grouped.apply(label_stock).reset_index(drop=True)['Label']

# Display distribution of labels
print(df['Label'].value_counts(normalize=True))

# Check for any stocks with 'Insufficient Data' or 'Error' labels
problematic_stocks = df[df['Label'].isin(['Insufficient Data', 'Error'])]['Ticker'].unique()
if len(problematic_stocks) > 0:
    print(f"Stocks with issues: {problematic_stocks}")

# %%
# Distribution of labels by ticker
label_by_ticker = df.groupby('Ticker')['Label'].last().value_counts()
print("Number of stocks in each category:")
print(label_by_ticker)

# Example stocks for each category
for label in df['Label'].unique():
    example_tickers = df[df['Label'] == label]['Ticker'].unique()[:5]
    print(f"\n{label} examples: {', '.join(example_tickers)}")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df['Label'].value_counts().plot(kind='bar')
plt.title('Distribution of Stock Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
print(f"Total data points: {len(df)}")
print(f"Unique stocks: {df['Ticker'].nunique()}")
print(f"Average data points per stock: {len(df) / df['Ticker'].nunique():.2f}")

# %%
final_labels = df.groupby('Ticker')['Label'].last()
print(final_labels.value_counts())

# %%
df['Date'] = pd.to_datetime(df['Date'])
label_trends = df.groupby('Date')['Label'].value_counts(normalize=True).unstack()
label_trends.plot(figsize=(12, 6), stacked=True)
plt.title('Proportion of Stock Labels Over Time')
plt.ylabel('Proportion')
plt.show()

# %%
df.columns

# %%
import talib 

# %%
import pandas as pd
import numpy as np
import talib 
from sklearn.preprocessing import LabelEncoder

def prepare_data(df):
    # Ensure data is sorted by Date and Ticker
    df = df.sort_values(['Ticker', 'Date'])

    # Create additional features
    df['Close_Pct_Change'] = df.groupby('Ticker')['Close'].pct_change()
    df['Volume_Pct_Change'] = df.groupby('Ticker')['Volume'].pct_change()
    
    # Relative Strength Index (RSI) 14-day (if not already present)
    if 'RSI' not in df.columns:
        df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: talib.RSI(x, timeperiod=14))
    
    # Moving Average Convergence Divergence (MACD)
    df['MACD'], df['MACD_Signal'], _ = df.groupby('Ticker')['Close'].transform(lambda x: talib.MACD(x))
    
    # Bollinger Bands Percentage
    df['BB_Pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Lagged features (previous 5 days)
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df.groupby('Ticker')['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df.groupby('Ticker')['Volume'].shift(i)
    
    # Rolling mean and std of Returns
    df['Returns_Rolling_Mean'] = df.groupby('Ticker')['Returns'].rolling(window=30).mean().reset_index(0, drop=True)
    df['Returns_Rolling_Std'] = df.groupby('Ticker')['Returns'].rolling(window=30).std().reset_index(0, drop=True)
    
    # Encode the Label
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features for model
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_50', 'EMA_20', 'RSI', 
                'BB_upper', 'BB_lower', 'Volume_MA_5', 'Close_Pct_Change', 'Volume_Pct_Change', 
                'MACD', 'MACD_Signal', 'BB_Pct', 'Returns_Rolling_Mean', 'Returns_Rolling_Std'] + \
               [f'Close_Lag_{i}' for i in range(1, 6)] + [f'Volume_Lag_{i}' for i in range(1, 6)]
    
    X = df[features]
    y = df['Label_Encoded']
    
    return X, y, le

# Prepare the data
X, y, label_encoder = prepare_data(df)

print("Features created:")
print(X.columns)
print("\nShape of feature matrix:", X.shape)
print("Number of classes:", len(label_encoder.classes_))

# %%
import pandas as pd
import numpy as np
import talib 
from sklearn.preprocessing import LabelEncoder

def prepare_data(df):
    # Ensure data is sorted by Date and Ticker
    df = df.sort_values(['Ticker', 'Date'])

    # Create additional features
    df['Close_Pct_Change'] = df.groupby('Ticker')['Close'].pct_change()
    df['Volume_Pct_Change'] = df.groupby('Ticker')['Volume'].pct_change()
    
    # Relative Strength Index (RSI) 14-day (if not already present)
    if 'RSI' not in df.columns:
        df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: talib.RSI(x, timeperiod=14))
    
    # Moving Average Convergence Divergence (MACD)
    macd_results = df.groupby('Ticker')['Close'].apply(lambda x: talib.MACD(x)[0])
    macd_signal_results = df.groupby('Ticker')['Close'].apply(lambda x: talib.MACD(x)[1])
    
    df['MACD'] = macd_results.reset_index(level=0, drop=True)
    df['MACD_Signal'] = macd_signal_results.reset_index(level=0, drop=True)
    
    # Bollinger Bands Percentage
    df['BB_Pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Lagged features (previous 5 days)
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df.groupby('Ticker')['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df.groupby('Ticker')['Volume'].shift(i)
    
    # Rolling mean and std of Returns
    df['Returns_Rolling_Mean'] = df.groupby('Ticker')['Returns'].rolling(window=30).mean().reset_index(0, drop=True)
    df['Returns_Rolling_Std'] = df.groupby('Ticker')['Returns'].rolling(window=30).std().reset_index(0, drop=True)

    # Calculate 14-day Relative Strength Index (RSI) for each Ticker group
    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: talib.RSI(x, timeperiod=14))

    # Calculate 28-day Relative Strength Index (RSI) for each Ticker group
    df['RSI_28'] = df.groupby('Ticker')['Close'].transform(lambda x: talib.RSI(x, timeperiod=28))

    # Calculate Average True Range (ATR) for each Ticker group and reset the index
    df['ATR'] = df.groupby('Ticker').apply(lambda x: talib.ATR(x['High'], x['Low'], x['Close'])).reset_index(level=0, drop=True)

    # Calculate On-Balance Volume (OBV) for each Ticker group and reset the index
    df['OBV'] = df.groupby('Ticker').apply(lambda x: talib.OBV(x['Close'], x['Volume'])).reset_index(level=0, drop=True)

    # Calculate 10-day momentum (percentage change) for each Ticker group
    df['Momentum'] = df.groupby('Ticker')['Close'].pct_change(periods=10)

    # Extract the day of the week from the Date column
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek

    # Extract the month from the Date column
    df['Month'] = pd.to_datetime(df['Date']).dt.month

    df['52W_High'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=252).max())
    df['52W_Low'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=252).min())
    df['Pct_From_52W_High'] = (df['Close'] - df['52W_High']) / df['52W_High']
    df['Pct_From_52W_Low'] = (df['Close'] - df['52W_Low']) / df['52W_Low']
    
    # Encode the Label
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # # Select features for model
    # features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_50', 'EMA_20', 'RSI', 
    #             'BB_upper', 'BB_lower', 'Volume_MA_5', 'Close_Pct_Change', 'Volume_Pct_Change', 
    #             'MACD', 'MACD_Signal', 'BB_Pct', 'Returns_Rolling_Mean', 'Returns_Rolling_Std'] + \
    #            [f'Close_Lag_{i}' for i in range(1, 6)] + [f'Volume_Lag_{i}' for i in range(1, 6)]
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_50', 'EMA_20', 
            'RSI', 'RSI_14', 'RSI_28', 'BB_upper', 'BB_lower', 'Volume_MA_5', 
            'Close_Pct_Change', 'Volume_Pct_Change', 'MACD', 'MACD_Signal', 'BB_Pct', 
            'Returns_Rolling_Mean', 'Returns_Rolling_Std', 'ATR', 'OBV', 'Momentum', 
            'DayOfWeek', 'Month', 'Pct_From_52W_High', 'Pct_From_52W_Low'] + \
           [f'Close_Lag_{i}' for i in range(1, 6)] + [f'Volume_Lag_{i}' for i in range(1, 6)]
    
    X = df[features]
    y = df['Label_Encoded']
    
    return X, y, le

# Prepare the data
X, y, label_encoder = prepare_data(df)

print("Features created:")
print(X.columns)
print("\nShape of feature matrix:", X.shape)
print("Number of classes:", len(label_encoder.classes_))

# %%
X.head()

# %%
import pandas as pd
import numpy as np
import talib 
from sklearn.preprocessing import LabelEncoder

def prepare_data(df):
    # Ensure data is sorted by Date and Ticker
    df = df.sort_values(['Ticker', 'Date'])

    # Create additional features
    df['Close_Pct_Change'] = df.groupby('Ticker')['Close'].pct_change()
    df['Volume_Pct_Change'] = df.groupby('Ticker')['Volume'].pct_change()
    
    # Relative Strength Index (RSI) 14-day (if not already present)
    if 'RSI' not in df.columns:
        df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: talib.RSI(x, timeperiod=14))
    
    # Moving Average Convergence Divergence (MACD)
    macd_results = df.groupby('Ticker')['Close'].apply(lambda x: talib.MACD(x)[0])
    macd_signal_results = df.groupby('Ticker')['Close'].apply(lambda x: talib.MACD(x)[1])
    
    df['MACD'] = macd_results.reset_index(level=0, drop=True)
    df['MACD_Signal'] = macd_signal_results.reset_index(level=0, drop=True)
    
    # Bollinger Bands Percentage
    df['BB_Pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Lagged features (previous 5 days)
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df.groupby('Ticker')['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df.groupby('Ticker')['Volume'].shift(i)
    
    # Rolling mean and std of Returns
    df['Returns_Rolling_Mean'] = df.groupby('Ticker')['Returns'].rolling(window=30).mean().reset_index(0, drop=True)
    df['Returns_Rolling_Std'] = df.groupby('Ticker')['Returns'].rolling(window=30).std().reset_index(0, drop=True)

    # Calculate 14-day Relative Strength Index (RSI) for each Ticker group
    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: talib.RSI(x, timeperiod=14))

    # Calculate 28-day Relative Strength Index (RSI) for each Ticker group
    df['RSI_28'] = df.groupby('Ticker')['Close'].transform(lambda x: talib.RSI(x, timeperiod=28))

    # Calculate Average True Range (ATR) for each Ticker group and reset the index
    df['ATR'] = df.groupby('Ticker').apply(lambda x: talib.ATR(x['High'], x['Low'], x['Close'])).reset_index(level=0, drop=True)

    # Calculate On-Balance Volume (OBV) for each Ticker group and reset the index
    df['OBV'] = df.groupby('Ticker').apply(lambda x: talib.OBV(x['Close'], x['Volume'])).reset_index(level=0, drop=True)

    # Calculate 10-day momentum (percentage change) for each Ticker group
    df['Momentum'] = df.groupby('Ticker')['Close'].pct_change(periods=10)

    # Extract the day of the week from the Date column
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek

    # Extract the month from the Date column
    df['Month'] = pd.to_datetime(df['Date']).dt.month

    df['52W_High'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=252).max())
    df['52W_Low'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=252).min())
    df['Pct_From_52W_High'] = (df['Close'] - df['52W_High']) / df['52W_High']
    df['Pct_From_52W_Low'] = (df['Close'] - df['52W_Low']) / df['52W_Low']
    
    # Encode the Label
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # # Select features for model
    # features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_50', 'EMA_20', 'RSI', 
    #             'BB_upper', 'BB_lower', 'Volume_MA_5', 'Close_Pct_Change', 'Volume_Pct_Change', 
    #             'MACD', 'MACD_Signal', 'BB_Pct', 'Returns_Rolling_Mean', 'Returns_Rolling_Std'] + \
    #            [f'Close_Lag_{i}' for i in range(1, 6)] + [f'Volume_Lag_{i}' for i in range(1, 6)]
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_50', 'EMA_20', 
            'RSI', 'RSI_14', 'RSI_28', 'BB_upper', 'BB_lower', 'Volume_MA_5', 
            'Close_Pct_Change', 'Volume_Pct_Change', 'MACD', 'MACD_Signal', 'BB_Pct', 
            'Returns_Rolling_Mean', 'Returns_Rolling_Std', 'ATR', 'OBV', 'Momentum', 
            'DayOfWeek', 'Month', 'Pct_From_52W_High', 'Pct_From_52W_Low'] + \
           [f'Close_Lag_{i}' for i in range(1, 6)] + [f'Volume_Lag_{i}' for i in range(1, 6)]
    
    X = df[features]
    y = df['Label_Encoded']
    
    return X, y, le , df

# Prepare the data
X, y, label_encoder , df_final = prepare_data(df)

print("Features created:")
print(X.columns)
print("\nShape of feature matrix:", X.shape)
print("Number of classes:", len(label_encoder.classes_))

# add code to save X,y and le 
# X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# %%
X.head()

# %%
X.to_csv('prepared_data.csv', index=False)

# %%
df_final.head(3)

# %%
df_final.to_csv('final_df.csv', index=False)

# %%
X, y, label_encoder , df_final = prepare_data(df)


