import pandas as pd
import numpy as np

def categorize_fluctuation(rate, min_rate, q1_rate, q3_rate, max_rate):
    """
    Categorize the fluctuation based on the rate of change.
    """
    if pd.isna(rate):
        return "UNKNOWN"
    elif rate < min_rate or rate > max_rate:
        return "CRITICAL"
    elif rate < q1_rate:
        return "GRADUAL"
    elif q1_rate <= rate < q3_rate:
        return "REGULAR"
    else:
        return "STEEP"


def calculate_fluctuation_rate(df_glucose, event_time, pre_interval, post_interval):
    """
    Calculate blood glucose fluctuations within the specified pre and post time intervals.
    """
    start_time = event_time - pd.Timedelta(minutes=pre_interval)
    end_time = event_time + pd.Timedelta(minutes=post_interval)

    df_window = df_glucose[(df_glucose['timestamp'] >= start_time) & (df_glucose['timestamp'] <= end_time)].copy()

    if df_window.empty or len(df_window) < 2:
        return None

    df_window['time_diff'] = df_window['timestamp'].diff().dt.total_seconds() / 60
    df_window['glucose_diff'] = df_window['value'].diff()
    df_window['rate_of_change'] = df_window['glucose_diff'] / df_window['time_diff']

    min_rate = df_window['rate_of_change'].min()
    q1_rate = df_window['rate_of_change'].quantile(0.25)
    q3_rate = df_window['rate_of_change'].quantile(0.75)
    max_rate = df_window['rate_of_change'].max()

    df_window['fluctuation_category'] = df_window['rate_of_change'].apply(
        lambda rate: categorize_fluctuation(rate, min_rate, q1_rate, q3_rate, max_rate)
    )

    glucose_start = df_window['value'].iloc[0]  
    glucose_end = df_window['value'].iloc[-1]  
    fluctuation_direction = "UP" if glucose_end > glucose_start else "DOWN" if glucose_end < glucose_start else "STABLE"

    return {
        'rate_of_change': df_window['rate_of_change'].mean(),
        'fluctuation_category': df_window['fluctuation_category'].mode()[0],
        'fluctuation_direction': fluctuation_direction
    }


def calculate_fluctuations(df_events, df_glucose, event_type, pre_interval, post_interval):
    """
    Calculate blood glucose fluctuations for a specific event type.
    """
    df_event = df_events[df_events['type'] == event_type].copy()
    if df_event.empty:
        return []

    fluctuation_results = []
    
    for _, row in df_event.iterrows():
        event_time = row['timestamp']
        event_value = row['value']

        # Calculate fluctuation before and after the event
        pre_fluctuation = calculate_fluctuation_rate(df_glucose, event_time, pre_interval, 0)
        post_fluctuation = calculate_fluctuation_rate(df_glucose, event_time, 0, post_interval)

        # If no fluctuation data is available, use 'No data' as placeholders
        pre_category = pre_fluctuation['fluctuation_category'] if pre_fluctuation else 'No data'
        post_category = post_fluctuation['fluctuation_category'] if post_fluctuation else 'No data'
        pre_direction = pre_fluctuation['fluctuation_direction'] if pre_fluctuation else 'No data'
        post_direction = post_fluctuation['fluctuation_direction'] if post_fluctuation else 'No data'

        fluctuation_results.append([event_type, event_time, event_value, pre_category, post_category, pre_direction, post_direction])

    return fluctuation_results



