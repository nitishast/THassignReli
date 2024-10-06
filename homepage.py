import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import datetime
import pandas_market_calendars as mcal

st.set_page_config(layout="wide")

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

@st.cache_data
def load_data():
    # predictions = pd.read_csv('labeled_data_updated_2_year_fixed.csv')
    predictions = pd.read_csv(f'public/public_data({end_date}).csv')
    predictions['Date'] = pd.to_datetime(predictions['Date'], format='mixed')
    return predictions

def calculate_long_term_stats(group):
    total_return = (group['Close'].iloc[-1] / group['Close'].iloc[0]) - 1
    volatility = group['Close'].pct_change().std() * np.sqrt(252)
    positive_days = (group['Close'].pct_change() > 0).sum() / len(group)
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Volatility': f"{volatility:.2%}",
        'Positive Days': f"{positive_days:.2%}"
    }

predictions = load_data()

# Sidebar
st.sidebar.title('Stock Analysis Dashboard')
selected_ticker = st.sidebar.selectbox('Select a stock', sorted(predictions['Ticker'].unique()))

# Date range selector in sidebar with error handling
min_date = predictions['Date'].min()
max_date = predictions['Date'].max()
default_start = max_date - timedelta(days=365)
default_end = max_date

try:
    start_date, end_date = st.sidebar.date_input(
        'Select date range',
        [default_start, default_end],
        min_value=min_date,
        max_value=max_date
    )
    if start_date > end_date:
        st.sidebar.error('End date must be after start date.')
        start_date, end_date = default_start, default_end
except ValueError:
    st.sidebar.error('Invalid date range. Using default values.')
    start_date, end_date = default_start, default_end

# Filter data
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
stock_data = predictions[(predictions['Ticker'] == selected_ticker) & 
                         (predictions['Date'] >= start_date) & 
                         (predictions['Date'] <= end_date)]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.title(f'{selected_ticker} Analysis')

    # Price trend
    st.subheader('Price Trend')
    fig = px.line(stock_data, x='Date', y='Close', title=f'{selected_ticker} Price Trend')
    st.plotly_chart(fig, use_container_width=True)

    # Label trend
    st.subheader('Label Trend')
    fig = px.line(stock_data, x='Date', y='Label_Daily', title=f'{selected_ticker} Label Trend',
                  category_orders={'Label_Daily': ['Strong Up', 'Weak Up', 'Flat', 'Weak Down', 'Strong Down']})
    st.plotly_chart(fig, use_container_width=True) 

with col2:
    # Current trend (most recent label)
    latest_label = stock_data['Label_Daily'].iloc[-1]
    st.metric("Current Trend", latest_label, f"as of {stock_data['Date'].iloc[-1].date()}")

    # Long-term trend (2-year label)
    latest_2year_label = stock_data['Label_2_Year'].iloc[-1]
    st.metric("Long-term Trend (2 Years)", latest_2year_label, f"as of {stock_data['Date'].iloc[-1].date()}")

    # Label distribution
    st.subheader('Daily Label Distribution')
    label_counts = stock_data['Label_Daily'].value_counts()
    if not label_counts.empty:
        fig = px.pie(values=label_counts.values, names=label_counts.index, title='Daily Label Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No label data available for the selected date range.")

    # # 2-Year Label distribution
    # st.subheader('2-Year Label Distribution')
    # label_2year_counts = stock_data['Label_2_Year'].value_counts()
    # if not label_2year_counts.empty:
    #     fig = px.pie(values=label_2year_counts.values, names=label_2year_counts.index, title='2-Year Label Distribution')
    #     st.plotly_chart(fig, use_container_width=True)
    # else:
    #     st.write("No 2-year label data available for the selected date range.")

    # Long-term statistics
    st.subheader('Long-term Statistics')
    long_term_stats = calculate_long_term_stats(stock_data)
    for stat, value in long_term_stats.items():
        st.metric(stat, value)

    # Summary statistics
    st.subheader('Summary Statistics')
    summary = stock_data[['Close', 'Volume', 'RSI_14', 'MACD']].describe().round(2)
    st.dataframe(summary, use_container_width=True)

# Additional features
st.sidebar.subheader('Additional Features')
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(stock_data)

# Display unique labels for the selected stock
st.sidebar.subheader('Unique Labels')
unique_daily_labels = stock_data['Label_Daily'].unique()
unique_2year_labels = stock_data['Label_2_Year'].unique()
st.sidebar.write(f"Daily Labels for {selected_ticker}: {', '.join(map(str, unique_daily_labels))}")
st.sidebar.write(f"2-Year Labels for {selected_ticker}: {', '.join(map(str, unique_2year_labels))}")