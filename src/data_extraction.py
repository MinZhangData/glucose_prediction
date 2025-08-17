
# ---------------- Data Extraction Functions ---------------- #
import json
import pandas as pd
import os

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
        df_glucose.rename(columns={'glucose_value': 'value'}, inplace=True)  
        df_glucose['value'] = pd.to_numeric(df_glucose['value'], errors='coerce')
    else:
        print(f"⚠️ Error: 'ts' or glucose value column missing in glucose data! Columns found: {df_glucose.columns}")
        return pd.DataFrame(columns=['timestamp', 'value'])

    df_glucose = df_glucose[['timestamp', 'value']].sort_values(by='timestamp')

    return df_glucose


def extract_event_data(event_list, event_name, value_field, use_ts_begin=False):
    """
    Extract event data (Bolus, Meal, Basal,exercise) from the event list.
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


def extract_event_data(event_list, event_type, value_field, *extra_fields, use_ts_begin=False):
    """
    Converts a list of events to a DataFrame, extracting relevant fields.
    """
    records = []
    for event in event_list:
        if not isinstance(event, dict):
            continue

        record = {
            'event_type': event_type,
            'timestamp': event.get('ts_begin') if use_ts_begin else event.get('ts'),
            value_field: event.get(value_field)
        }

        for field in extra_fields:
            record[field] = event.get(field)

        # For bolus events, ensure all relevant fields are captured
        if event_type == 'bolus':
            record['ts_begin'] = event.get('ts_begin')
            record['ts_end'] = event.get('ts_end')
            record['type'] = event.get('type')
            record['dose'] = event.get('dose')
            record['timestamp'] = event.get('ts_begin')  # explicitly set for sorting
            record['timestamp2'] = event.get('ts_end')

        records.append(record)

    return pd.DataFrame(records)


def extract_all_events(data,patient_id,output_dir='data/processed/2018/processed'):
    """
    Extract event data (Bolus, Meal, Basal, Exercise) from patient data safely.
    """
    if not isinstance(data, dict):
        print("❌ Error: `data` is not a dictionary")
        return pd.DataFrame()

    patient_data = data.get('patient')
    if not isinstance(patient_data, dict):
        print("⚠️ Warning: `data['patient']` is missing or not a dictionary.")
        return pd.DataFrame()

    def safe_extract(event_type, value_field, *extra_fields, use_ts_begin=False):
        event_source = patient_data.get(event_type, {})
        if not isinstance(event_source, dict):
            print(f"⚠️ Warning: `data['patient']['{event_type}']` is not a dictionary")
            return pd.DataFrame()

        event_data = event_source.get('event', [])
        if not isinstance(event_data, list):
            print(f"⚠️ Warning: `data['patient']['{event_type}']['event']` is not a list")
            return pd.DataFrame()

        df = extract_event_data(event_data, event_type, value_field, *extra_fields, use_ts_begin=use_ts_begin)
        print(f"{event_type} count:", len(df))
        return df

    # Extracting each event type
    df_bolus = safe_extract('bolus', 'dose', use_ts_begin=True)
    df_meal = safe_extract('meal', 'carbs','type')
    df_exercise = safe_extract('exercise', 'intensity', 'duration')

    print(f"df_bolus count: {len(df_bolus)}")
    print(f"examples of df_bolus: {df_bolus.head()}")
    print(f"df_bolus columns: {df_bolus.columns}")
    
    print(f"df_meal count: {len(df_meal)}")
    print(f"examples of df_meal: {df_meal}") 
    df_basal = safe_extract('basal', 'value')
    
    print(f"df_exercise count: {len(df_exercise)}")
    print(f"examples of df_exercise: {df_exercise.head()}")

    df_result = find_bolus_and_exercise_near_meals(df_meal, df_bolus, df_exercise, patient_id, output_dir)

    return df_result



def find_bolus_and_exercise_near_meals(df_meal, df_bolus, df_exercise, patient_id, output_dir=None, save_to_csv=True):
    df_meal['timestamp'] = pd.to_datetime(df_meal['timestamp'], dayfirst=True, errors='coerce')
    df_bolus['ts_begin'] = pd.to_datetime(df_bolus['ts_begin'], dayfirst=True, errors='coerce')
    df_bolus['ts_end'] = pd.to_datetime(df_bolus['ts_end'],dayfirst=True,  errors='coerce')
    if not df_exercise.empty:
        df_exercise['timestamp'] = pd.to_datetime(df_exercise['timestamp'], dayfirst=True, errors='coerce')

    results = []

    for _, meal_row in df_meal.iterrows():
        meal_time = meal_row['timestamp']
        meal_carbs = meal_row.get('carbs')
        meal_type = meal_row.get('type')

        matched_bolus = df_bolus[
            ((df_bolus['ts_begin'] - meal_time).abs() <= pd.Timedelta(minutes=45)) |
            ((df_bolus['ts_end'] - meal_time).abs() <= pd.Timedelta(minutes=45))
        ].copy()
        matched_bolus['gap_minutes'] = (matched_bolus['ts_begin'] - meal_time).dt.total_seconds() / 60
        matched_bolus.sort_values(by='gap_minutes', inplace=True)

        bolus_data = {}
        for i in range(2):
            if i < len(matched_bolus):
                b = matched_bolus.iloc[i]
                bolus_data.update({
                    f'bolus_time_{i+1}': b['ts_begin'],
                    f'bolus_dose_{i+1}': b.get('dose'),
                    f'bolus_type_{i+1}': b.get('type'),
                    f'bolus_ts_begin_{i+1}': b.get('ts_begin'),
                    f'bolus_ts_end_{i+1}': b.get('ts_end'),
                    f'bolus_time_gap_minutes_{i+1}': round((b['ts_begin'] - meal_time).total_seconds() / 60, 2),
                    f'bolus_direction_{i+1}': 'before' if b['ts_begin'] < meal_time else 'after'
                })
            else:
                bolus_data.update({
                    f'bolus_time_{i+1}': None,
                    f'bolus_dose_{i+1}': None,
                    f'bolus_type_{i+1}': None,
                    f'bolus_ts_begin_{i+1}': None,
                    f'bolus_ts_end_{i+1}': None,
                    f'bolus_time_gap_minutes_{i+1}': None,
                    f'bolus_direction_{i+1}': None
                })


        exercise_data = {
            'exercise_time': None,
            'exercise_intensity': None,
            'exercise_duration': None,
            'exercise_time_gap_minutes': None,
            'exercise_direction': None
        }

        # Look for nearest exercise within 60 minutes
        if not df_exercise.empty:
            df_exercise['gap_minutes'] = (df_exercise['timestamp'] - meal_time).abs().dt.total_seconds() / 60
            near_exercise = df_exercise[df_exercise['gap_minutes'] <= 60]

            if not near_exercise.empty:
                exercise_row = near_exercise.sort_values(by='gap_minutes').iloc[0]
                exercise_data.update({
                    'exercise_time': exercise_row['timestamp'],
                    'exercise_intensity': exercise_row.get('intensity'),
                    'exercise_duration': exercise_row.get('duration'),
                    'exercise_time_gap_minutes': round(exercise_row['gap_minutes'], 2),
                    'exercise_direction': 'before' if exercise_row['timestamp'] < meal_time else 'after'
                })


        results.append({
            'patient_id': patient_id,
            'meal_time': meal_time,
            'meal_carbs': meal_carbs,
            'meal_type': meal_type,
            **bolus_data,
            **exercise_data
        })

    df_result = pd.DataFrame(results)

    if save_to_csv and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{patient_id}_meal_bolus_exercise.csv")
        df_result.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")

    return df_result


def find_nearest_bolus_and_exercise(df_events, patient_id, output_dir, save_to_csv=True):
    """
    For each meal event, find the closest bolus and exercise event (before or after) and indicate direction.
    """

    print(f"Inside find_nearest_bolus_and_exercise: patient_id={patient_id}") 

    # Ensure timestamp is datetime
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], errors='coerce')

    # Filter only relevant event types
    df_relevant = df_events[df_events['type'].isin(['bolus', 'meal', 'exercise'])].copy()
    df_relevant = df_relevant.sort_values(by='timestamp')

    # Separate event types
    df_bolus = df_relevant[df_relevant['type'] == 'bolus'].copy()
    df_meal = df_relevant[df_relevant['type'] == 'meal'].copy()
    df_exercise = df_relevant[df_relevant['type'] == 'exercise'].copy()
    print(f"df_exercise count in find_nearest_bolus_and_exercise: {len(df_exercise)}")

    results = []

    for _, meal_row in df_meal.iterrows():
        meal_time = meal_row['timestamp']
        meal_value = meal_row.get('value')

        # Find nearest bolus
        df_bolus['time_diff'] = (df_bolus['timestamp'] - meal_time).abs()
        nearest_bolus = df_bolus.loc[df_bolus['time_diff'].idxmin()] if not df_bolus.empty else None

        # Find nearest exercise
        df_exercise['time_diff'] = (df_exercise['timestamp'] - meal_time).abs()
        nearest_exercise = df_exercise.loc[df_exercise['time_diff'].idxmin()] if not df_exercise.empty else None

        bolus_gap = round(nearest_bolus['time_diff'].total_seconds() / 60, 2) if nearest_bolus is not None else None
        exercise_gap = round(nearest_exercise['time_diff'].total_seconds() / 60, 2) if nearest_exercise is not None else None

        result_entry = {
            'patient_id': patient_id,
            'meal_time': meal_time,
            'meal_value': meal_value,

            'nearest_bolus_time': nearest_bolus['timestamp'] if nearest_bolus is not None else None,
            'nearest_bolus_value': nearest_bolus.get('value') if nearest_bolus is not None else None,
            'bolus_time_gap_minutes': 999 if bolus_gap is not None and bolus_gap > 45 else bolus_gap,
            'bolus_direction': "before" if nearest_bolus is not None and nearest_bolus['timestamp'] < meal_time else "after" if nearest_bolus is not None else None,

            'nearest_exercise_time': nearest_exercise['timestamp'] if nearest_exercise is not None else None,
            'nearest_exercise_intensity': nearest_exercise.get('value') if nearest_exercise is not None else None,
            'nearest_exercise_duration': nearest_exercise.get('value2') if nearest_exercise is not None else None,
            'exercise_time_gap_minutes': 999 if exercise_gap is not None and exercise_gap > 60 else exercise_gap,
            'exercise_direction': "before" if nearest_exercise is not None and nearest_exercise['timestamp'] < meal_time else "after" if nearest_exercise is not None else None,
        }

        results.append(result_entry)

    # Convert to DataFrame
    df_result = pd.DataFrame(results)

    # Save to CSV if requested
    if save_to_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{patient_id}_meal_nearest_bolus_exercise.csv")
        df_result.to_csv(output_path, index=False)
        print(f"✅ CSV file saved for patient {patient_id}: {output_path}")

    return df_result

