import pandas as pd
import json
import os

# ---------------- Data Extraction Functions ---------------- #

def load_glucose_data(file_path):
    """
    Load glucose data from the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    glucose_data = data['patient'].get('glucose_level', {}).get('event', [])

    if not glucose_data:
        print("⚠️ Warning: No glucose data found!")

    df_glucose = pd.DataFrame(glucose_data)

    if 'ts' in df_glucose.columns and 'value' in df_glucose.columns:
        df_glucose['timestamp'] = pd.to_datetime(df_glucose['ts'], format='%d-%m-%Y %H:%M:%S')
        df_glucose['value'] = pd.to_numeric(df_glucose['value'], errors='coerce')
    elif 'ts' in df_glucose.columns and 'glucose_value' in df_glucose.columns:
        df_glucose['timestamp'] = pd.to_datetime(df_glucose['ts'], format='%d-%m-%Y %H:%M:%S')
        df_glucose.rename(columns={'glucose_value': 'value'}, inplace=True)  # 确保列名统一
        df_glucose['value'] = pd.to_numeric(df_glucose['value'], errors='coerce')
    else:
        print(f"⚠️ Error: 'ts' or glucose value column missing in glucose data! Columns found: {df_glucose.columns}")
        return pd.DataFrame(columns=['timestamp', 'value'])

    df_glucose = df_glucose[['timestamp', 'value']].sort_values(by='timestamp')
    return df_glucose


def extract_event_data(event_list, event_name, value_field, use_ts_begin=False):
    """
    Extract event data (Bolus, Meal, Basal) from the event list.
    """
    if not event_list:
        print(f"No {event_name} events found.")
        return pd.DataFrame(columns=['timestamp', 'type', 'value'])

    timestamps = [event['ts_begin'] if use_ts_begin else event['ts'] for event in event_list]
    values = [event[value_field] for event in event_list]
    return pd.DataFrame({'timestamp': pd.to_datetime(timestamps, format='%d-%m-%Y %H:%M:%S'), 'type': event_name, 'value': values})


def load_patient_data(file_path):
    """
    Load patient data from the JSON file and extract the patient ID from the filename.
    """
    patient_id = os.path.basename(file_path).split('-')[0]  # Extract patient ID from filename (e.g., "563-ws-training.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Patient ID: {patient_id}")  

    return data, patient_id


def extract_all_events(data):
    """
    Extract event data (Bolus, Meal, Basal) for all event types from patient data.
    """
    # Extract bolus data (value = dose)
    df_bolus = extract_event_data(data['patient'].get('bolus', {}).get('event', []), 'bolus', 'dose', use_ts_begin=True)
    print(f"bolus count:", len(df_bolus))

    # Extract meal data (value = carbs)
    df_meal = extract_event_data(data['patient'].get('meal', {}).get('event', []), 'meal', 'carbs')
    print(f"meal count:", len(df_meal))

    # Extract basal data (value = value)
    df_basal = extract_event_data(data['patient'].get('basal', {}).get('event', []), 'basal', 'value')
    print(f"basal count:", len(df_basal))

    # Merge all events into one DataFrame
    df_events = pd.concat([df_bolus, df_meal, df_basal]).sort_values(by='timestamp')
    print(f"event count:", len(df_events))
    return df_events

import pandas as pd
import os

def find_bolus_meal_within_15min(df_events, save_to_csv=False, output_dir="data/processed", patient_id=None):
    """
    Find meal events that occur within 15 minutes of bolus events.
    
    Args:
        df_events (pd.DataFrame): DataFrame containing bolus, meal, and other events.
        save_to_csv (bool): Whether to save the resulting dataframes to CSV.
        output_dir (str): Directory path to save the CSV files.
        patient_id (str): The patient ID to include in the CSV filenames.
        
    Returns:
        pd.DataFrame, pd.DataFrame: DataFrames with meal and bolus events within 15 minutes, and those excluded.
    """
    # Filter only bolus and meal events
    df_bolus_meal = df_events[df_events['type'].isin(['bolus', 'meal'])].copy()

    # Sort by timestamp
    df_bolus_meal = df_bolus_meal.sort_values(by='timestamp')

    # Initialize list to store bolus-meal pairs within 15 minutes
    bolus_meal_within_15min = []
    excluded_bolus_meal = []

    # Iterate over the events to find meal events within 15 minutes of bolus events
    for i, row in df_bolus_meal.iterrows():
        if row['type'] == 'bolus':
            bolus_time = row['timestamp']
            bolus_value = row['value']
            
            # Find meal events within 15 minutes after the bolus event
            meal_events = df_bolus_meal[(df_bolus_meal['type'] == 'meal') & 
                (df_bolus_meal['timestamp'] >= bolus_time) & 
                (df_bolus_meal['timestamp'] <= bolus_time + pd.Timedelta(minutes=15))]
            
            if not meal_events.empty:
                # If there are meal events within 15 minutes, store them in the list
                for _, meal_row in meal_events.iterrows():
                    time_gap = (meal_row['timestamp'] - bolus_time).total_seconds() / 60  # Time gap in minutes
                    bolus_meal_within_15min.append({
                        'patient_id': patient_id,
                        'event_type': 'meal',
                        'event_time': meal_row['timestamp'],
                        'event_value': meal_row['value'],
                        'time_gap': time_gap
                    })
                    bolus_meal_within_15min.append({
                        'patient_id': patient_id,
                        'event_type': 'bolus',
                        'event_time': bolus_time,
                        'event_value': bolus_value,
                        'time_gap': time_gap
                    })
            else:
                # If no meal event is within 15 minutes, store the bolus event in excluded list
                excluded_bolus_meal.append({
                    'patient_id': patient_id,
                    'event_type': 'bolus',
                    'event_time': bolus_time,
                    'event_value': bolus_value,
                    'time_gap': None
                })

    # Convert lists to DataFrames
    df_bolus_meal_within_15min = pd.DataFrame(bolus_meal_within_15min)
    df_excluded_bolus_meal = pd.DataFrame(excluded_bolus_meal)
    
    # Save both DataFrames to CSV files if requested
    if save_to_csv:
        os.makedirs(output_dir, exist_ok=True)
        df_bolus_meal_within_15min.to_csv(os.path.join(output_dir, f"{patient_id}_bolus_meal_within_15min.csv"), index=False)
        df_excluded_bolus_meal.to_csv(os.path.join(output_dir, f"{patient_id}_excluded_bolus_meal.csv"), index=False)
        print(f"CSV files for patient {patient_id} have been saved to:", output_dir)

    return df_bolus_meal_within_15min, df_excluded_bolus_meal


# import pandas as pd
# import os

def find_nearest_bolus(df_events, patient_id, output_dir, save_to_csv= True):
    """
    For each meal event, find the closest bolus event (before or after) and indicate direction.
    
    Args:
        df_events (pd.DataFrame): DataFrame containing bolus, meal, and other events.
        save_to_csv (bool): Whether to save the resulting DataFrame to CSV.
        output_dir (str): Directory path to save the CSV file.
        patient_id (str): The patient ID to include in the CSV filename.
        
    Returns:
        pd.DataFrame: DataFrame with meal events, their nearest bolus event, and direction.
    """
    print(f"Inside find_nearest_bolus: patient_id={patient_id}") 
    # Filter only bolus and meal events
    df_bolus_meal = df_events[df_events['type'].isin(['bolus', 'meal'])].copy()
    
    # Sort by timestamp
    df_bolus_meal = df_bolus_meal.sort_values(by='timestamp')
    
    # Separate bolus and meal events
    df_bolus = df_bolus_meal[df_bolus_meal['type'] == 'bolus']
    df_meal = df_bolus_meal[df_bolus_meal['type'] == 'meal']
    
    # Initialize list to store results
    meal_with_nearest_bolus = []
    
    for _, meal_row in df_meal.iterrows():
        meal_time = meal_row['timestamp']
        meal_value = meal_row['value']
        
        # Find the nearest bolus event (before or after)
        # df_bolus['time_diff'] = (df_bolus['timestamp'] - meal_time).abs()
        df_bolus = df_events[df_events['type'] == 'bolus'].copy()  # 用 .copy() 避免修改视图
        df_bolus.loc[:, 'time_diff'] = (df_bolus['timestamp'] - meal_time).abs()

        nearest_bolus = df_bolus.loc[df_bolus['time_diff'].idxmin()]
        
        time_gap = nearest_bolus['time_diff'].total_seconds() / 60  # Convert to minutes
        direction = "before" if nearest_bolus['timestamp'] < meal_time else "after"
        
        meal_with_nearest_bolus.append({
            'patient_id': patient_id,
            'meal_time': meal_time,
            'meal_value': meal_value,
            'nearest_bolus_time': nearest_bolus['timestamp'],
            'nearest_bolus_value': nearest_bolus['value'],
            'time_gap_minutes': time_gap,
            'direction': direction
        })
    
    # Convert list to DataFrame
    df_result = pd.DataFrame(meal_with_nearest_bolus)
    
    # Save to CSV if requested
    if save_to_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{patient_id}_meal_nearest_bolus.csv")
        df_result.to_csv(output_path, index=False)
        print(f"CSV file saved for patient {patient_id}: {output_path}")
    
    return df_result



# Example Usage
if __name__ == "__main__":
    json_path = "/data/raw/OhioT1DM/2018/train/json/563-ws-training.json"

    # Load glucose data
    df_glucose = load_glucose_data(json_path)
    print(df_glucose.head())

    # Load event data
    data, patient_id = load_patient_data(json_path)
    df_events = extract_all_events(data)

    # Find bolus-meal pairs
    df_bolus_meal_within_15min, df_excluded_bolus_meal = find_bolus_meal_within_15min(df_events, save_to_csv=True, output_dir="data/processed", patient_id=patient_id)
