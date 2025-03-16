import pandas as pd
import numpy as np
import os
from src.data_extraction import load_patient_data
from src.data_extraction import extract_all_events
from src.data_extraction import load_glucose_data
from src.data_extraction import find_nearest_bolus

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


# ---------------- Data Loading ---------------- #
def process_patient_data(data_dir, patient_file, output_dir):
    """
    Process patient data including event extraction, glucose fluctuation analysis, and save results.
    
    Args:
        data_dir (str): Path to the data directory.
        patient_file (str): JSON filename containing patient data.
        output_dir (str): Directory to save the processed results.
    
    Returns:
        None
    """
    file_path = os.path.join(data_dir, patient_file)

    # 1️⃣ Load patient data
    print(f"Loading patient data from {file_path}...")
    data, patient_id = load_patient_data(file_path)

    # 2️⃣ Extract event data
    print("Extracting event data...")
    df_events = extract_all_events(data)
    print(f"Total events extracted: {len(df_events)}")

    # 3️⃣ Load glucose data
    print("Loading glucose data...")
    df_glucose = load_glucose_data(file_path)
    print(f"Total glucose readings: {len(df_glucose)}")

    # 4️⃣ Find meal-with-bolus events
    print(f"Finding meal_with_nearest_bolus events for patient {patient_id}...")
    df_meal_with_bolus = find_nearest_bolus(df_events, patient_id,output_dir)

    # 5️⃣ Apply glucose fluctuation analysis
    print("Applying glucose fluctuation analysis...")
    fluctuation_df_meal_with_bolus = apply_glucose_fluctuation(df_meal_with_bolus, df_glucose)

    # 6️⃣ Save processed results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{patient_id}_fluctuations.csv")
    fluctuation_df_meal_with_bolus.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# ---------------- Data Processing ---------------- #

def apply_glucose_fluctuation(df_bolus_meal, df_glucose):
    """
    Calculate blood glucose fluctuations for meal events and include bolus event details.

    Args:
        df_bolus_meal (pd.DataFrame): DataFrame containing meal events and their nearest bolus events.
        df_glucose (pd.DataFrame): DataFrame containing glucose readings with timestamps.

    Returns:
        pd.DataFrame: A DataFrame with glucose fluctuation information for each meal event, including bolus event details.
    """
    fluctuation_results = []

    for _, row in df_bolus_meal.iterrows():
        event_type = "meal"
        event_time = row["meal_time"]
        event_value = row["meal_value"]

        # Calculate glucose fluctuation 30 minutes before and 60 minutes after the meal
        pre_fluctuation = calculate_fluctuation_rate(df_glucose, event_time, pre_interval=30, post_interval=0)
        post_fluctuation = calculate_fluctuation_rate(df_glucose, event_time, pre_interval=0, post_interval=60)

        # Handle missing fluctuation data to avoid KeyError
        pre_category = pre_fluctuation.get("fluctuation_category", "No data") if pre_fluctuation else "No data"
        post_category = post_fluctuation.get("fluctuation_category", "No data") if post_fluctuation else "No data"
        pre_direction = pre_fluctuation.get("fluctuation_direction", "No data") if pre_fluctuation else "No data"
        post_direction = post_fluctuation.get("fluctuation_direction", "No data") if post_fluctuation else "No data"

        # Store the results, including bolus event details
        fluctuation_results.append({
            "patient_id": row["patient_id"],
            "event_type": event_type,
            "event_time": event_time,
            "event_value": event_value,
            "nearest_bolus_time": row["nearest_bolus_time"],
            "nearest_bolus_value": row["nearest_bolus_value"],
            "time_gap_minutes": row["time_gap_minutes"],
            "direction": row["direction"],
            "pre_fluctuation_category": pre_category,
            "post_fluctuation_category": post_category,
            "pre_fluctuation_direction": pre_direction,
            "post_fluctuation_direction": post_direction
        })

    return pd.DataFrame(fluctuation_results)




