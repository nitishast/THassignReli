import datetime
import pandas as pd
import pandas_market_calendars as mcal
import os
import sys
# Add the parent directory to the Python path


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

