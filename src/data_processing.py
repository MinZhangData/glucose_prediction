import pandas as pd
import numpy as np
import os
from src.data_extraction import load_patient_data,extract_all_events,load_glucose_data
import matplotlib.pyplot as plt



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
        Calculate the rate of glucose fluctuation within a specified time window and determine the direction of change.

        :param df_glucose: DataFrame containing glucose data; must include ['timestamp', 'value']
        :param event_time: Timestamp indicating the time of the event
        :param pre_interval: Number of minutes before the event to consider for fluctuation calculation
        :param post_interval: Number of minutes after the event to consider for fluctuation calculation
        :return: Glucose change information, including rate_of_change, fluctuation_category, and fluctuation_direction
    """


    # Ensure 'timestamp' is of datetime type
    df_glucose = df_glucose.copy()
    df_glucose['timestamp'] = pd.to_datetime(df_glucose['timestamp'])

    # get glucose value at event_time
    glucose_event = df_glucose.loc[df_glucose['timestamp'] == event_time, 'value']
    if glucose_event.empty:
        glucose_event = df_glucose.iloc[(df_glucose['timestamp'] - event_time).abs().idxmin()]['value']  
    else:
        glucose_event = glucose_event.iloc[0]

    # Calculate using the pre-event window (pre_interval > 0, post_interval == 0)
    if pre_interval > 0 and post_interval == 0:
        target_time = event_time - pd.Timedelta(minutes=pre_interval)
        closest_index = (df_glucose['timestamp'] - target_time).abs().idxmin()  
        glucose_compare = df_glucose.loc[closest_index, 'value']  
        fluctuation_direction = "UP" if glucose_event > glucose_compare else "DOWN" if glucose_event < glucose_compare else "STABLE"

    # Calculate using the post-event window (pre_interval == 0, post_interval > 0)
    elif pre_interval == 0 and post_interval > 0:
        target_time = event_time + pd.Timedelta(minutes=post_interval)
        closest_index = (df_glucose['timestamp'] - target_time).abs().idxmin()  
        glucose_compare = df_glucose.loc[closest_index, 'value']  
        fluctuation_direction = "UP" if glucose_event > glucose_compare else "DOWN" if glucose_event < glucose_compare else "STABLE"

    else:
        return None  

   # Retrieve data for event_time and its surrounding time window

    df_window = df_glucose[(df_glucose['timestamp'] >= event_time - pd.Timedelta(minutes=pre_interval)) &
                           (df_glucose['timestamp'] <= event_time + pd.Timedelta(minutes=post_interval))].copy()

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
    df_events = extract_all_events(data,patient_id,output_dir)
    print(f"Total events extracted: {len(df_events)}")

    # 3️⃣ Load glucose data
    print("Loading glucose data...")
    df_glucose = load_glucose_data(file_path)
    print(f"Total glucose readings: {len(df_glucose)}")

    # 4️⃣ Apply glucose fluctuation analysis
    print("Applying glucose fluctuation analysis...")
    fluctuation_df_meal_with_bolus = apply_glucose_fluctuation(df_events, df_glucose)

    # 5️⃣ Save processed results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{patient_id}_fluctuations_beforecleanning.csv")

    fluctuation_df_meal_with_bolus.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    cols_to_check = ['glucose_30min_before_event', 'glucose_at_event_time', 'glucose_60min_after_event']

    rows_before = len(fluctuation_df_meal_with_bolus)
    rows_to_delete = fluctuation_df_meal_with_bolus[
        fluctuation_df_meal_with_bolus[cols_to_check].isin([999]).any(axis=1)
    ].shape[0]

    fluctuation_df_cleaned = fluctuation_df_meal_with_bolus[
        ~fluctuation_df_meal_with_bolus[cols_to_check].isin([999]).any(axis=1)
    ].copy()

    rows_after = len(fluctuation_df_cleaned)

    print(f"Deleted {rows_to_delete} rows where any of {cols_to_check} == 999.")
    print(f"Rows before: {rows_before}, after cleaning: {rows_after}")

    cleaned_output_file = os.path.join(output_dir, f"{patient_id}_fluctuations.csv")
    fluctuation_df_cleaned.to_csv(cleaned_output_file, index=False)
    print(f"Cleaned results saved to {cleaned_output_file}")


# ---------------- Data Processing ---------------- #

def apply_glucose_fluctuation(df_bolus_meal, df_glucose):
    """
    Calculate blood glucose fluctuations for meal events and include bolus and exercise details.

    Args:
        df_bolus_meal (pd.DataFrame): DataFrame containing meal events and their nearest bolus events.
        df_glucose (pd.DataFrame): DataFrame containing glucose readings with timestamps.

    Returns:
        pd.DataFrame: A DataFrame with glucose fluctuation information for each meal event, including bolus and exercise details.
    """
    fluctuation_results = []

    df_glucose.to_csv("/Users/min/Documents/git/glucose_prediction/scripts/570_glucose_data.csv", index=False)



    for _, row in df_bolus_meal.iterrows():
        # event_type = "meal"
        meal_time = row["meal_time"]
        meal_carbs = row["meal_carbs"]
        meal_type = row["meal_type"]

        # Calculate glucose fluctuation 30 minutes before and 60 minutes after the meal
        pre_fluctuation = calculate_fluctuation_rate(df_glucose, meal_time, pre_interval=30, post_interval=0)
        post_fluctuation = calculate_fluctuation_rate(df_glucose, meal_time, pre_interval=0, post_interval=60)

        # Extract glucose values at specific time points
        glucose_30min_before_event = get_glucose_value(df_glucose, meal_time, offset=-30)
        # print(f"Glucose 30min before event: {glucose_30min_before_event} at {event_time - pd.Timedelta(minutes=30)}")
        glucose_at_event_time = get_glucose_value(df_glucose, meal_time, offset=0)
        glucose_60min_after_event = get_glucose_value(df_glucose, meal_time, offset=60)

        # Handle missing fluctuation data to avoid KeyError
        pre_category = pre_fluctuation.get("fluctuation_category", "No data") if pre_fluctuation else "No data"
        post_category = post_fluctuation.get("fluctuation_category", "No data") if post_fluctuation else "No data"
        pre_direction = pre_fluctuation.get("fluctuation_direction", "No data") if pre_fluctuation else "No data"
        post_direction = post_fluctuation.get("fluctuation_direction", "No data") if post_fluctuation else "No data"

        # Prepare result dictionary
        result = {
            "patient_id": row["patient_id"],
            "meal_time": meal_time,
            "meal_carbs": meal_carbs,
            "meal_type": meal_type,

            "bolus_time_1": row["bolus_time_1"],
            "bolus_dose_1": row["bolus_dose_1"],
            "bolus_time_gap_minutes_1": row["bolus_time_gap_minutes_1"],
            "bolus_direction_1": row["bolus_direction_1"],
            "bolus_type_1": row["bolus_type_1"],

            "bolus_time_2": row["bolus_time_2"],
            "bolus_dose_2": row["bolus_dose_2"],
            "bolus_time_gap_minutes_2": row["bolus_time_gap_minutes_2"],
            "bolus_direction_2": row["bolus_direction_2"],
            "bolus_type_2": row["bolus_type_2"],

            "nearest_exercise_time": row["exercise_time"],
            "nearest_exercise_intensity": row["exercise_intensity"],
            "nearest_exercise_duration": row["exercise_duration"],
            "exercise_time_gap_minutes": row["exercise_time_gap_minutes"],
            "exercise_direction": row["exercise_direction"],

            "pre_fluctuation_category": pre_category,
            "post_fluctuation_category": post_category,
            "pre_fluctuation_direction": pre_direction,
            "post_fluctuation_direction": post_direction,

            "glucose_30min_before_event": glucose_30min_before_event,
            "glucose_at_event_time": glucose_at_event_time,
            "glucose_60min_after_event": glucose_60min_after_event
        }

        if pd.notna(row["bolus_type_2"]):
            glucose_60min_after_bolus = get_glucose_value(df_glucose, row["bolus_ts_end_2"], offset=60)
            result["glucose_60min_after_double_bolus"] = glucose_60min_after_bolus
        if pd.notna(row["exercise_time_gap_minutes"]) and row["exercise_time_gap_minutes"] < 999:

            post_exercise_fluctuation = calculate_fluctuation_rate(df_glucose, row["exercise_time"], pre_interval=0, post_interval=60)
            result["post_exercise_fluctuation_category"] = post_exercise_fluctuation.get("fluctuation_category", "No data") if post_exercise_fluctuation else "No data"
            result["post_exercise_fluctuation_direction"] = post_exercise_fluctuation.get("fluctuation_direction", "No data") if post_exercise_fluctuation else "No data"

            glucose_60min_after_exercise = get_glucose_value(df_glucose, row["exercise_time"], offset=60)
            result["glucose_60min_after_exercise"] = glucose_60min_after_exercise

        fluctuation_results.append(result)

    return pd.DataFrame(fluctuation_results)

# commented on 2025-06-07

def get_glucose_value(df_glucose, reference_time, offset):
    """
    Get the glucose value closest to a specified time offset from the reference time.
    """
    df_glucose = df_glucose.copy()
    df_glucose["timestamp"] = pd.to_datetime(df_glucose["timestamp"])
    
    target_time = reference_time + pd.Timedelta(minutes=offset)
    
    df_glucose["time_diff"] = (df_glucose["timestamp"] - target_time).abs()
    
    if df_glucose.empty:
        print(f"❌ DataFrame is empty for target time: {target_time}")
        return "No data"
    
    closest_row = df_glucose.loc[df_glucose["time_diff"].idxmin()]

    return closest_row["value"]

def get_glucose_value(df_glucose, reference_time, offset):
    """
    Get the glucose value closest to a specified time offset from the reference time.
    If the closest glucose reading is more than 5 minutes away, return 999.
    """
    df_glucose = df_glucose.copy()
    df_glucose["timestamp"] = pd.to_datetime(df_glucose["timestamp"])

    target_time = reference_time + pd.Timedelta(minutes=offset)

    if df_glucose.empty:
        print(f"❌ DataFrame is empty for target time: {target_time}")
        return 999

    df_glucose["time_diff"] = (df_glucose["timestamp"] - target_time).abs()
    closest_idx = df_glucose["time_diff"].idxmin()
    closest_row = df_glucose.loc[closest_idx]

    if closest_row["time_diff"] > pd.Timedelta(minutes=5):
        return 999
    else:
        return closest_row["value"]

def update_fluctuation_data(train_data_path: str, train_data_path_new: str, patient_id: str):
    """ 
    Load the fluctuation data file, update specific columns if exercise_time_gap_minutes != 999,
    and save the updated DataFrame to the specified output file.
    """

    if not os.path.exists(train_data_path):
        print(f"❌ Error: File not found - {train_data_path}")
        return

    print(f"📂 Loading data from {train_data_path}...")
    df = pd.read_csv(train_data_path)

    required_columns = [
        "exercise_time_gap_minutes",
        "post_fluctuation_category",
        "post_fluctuation_direction",
        "glucose_60min_after_event",
        "post_exercise_fluctuation_category",
        "post_exercise_fluctuation_direction",
        "glucose_60min_after_exercise"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Error: Missing required columns: {missing_columns}")
        return

    mask = df["exercise_time_gap_minutes"] != 999
    updated_rows = mask.sum()

    df.loc[mask, "post_fluctuation_category"] = df.loc[mask, "post_exercise_fluctuation_category"]
    df.loc[mask, "post_fluctuation_direction"] = df.loc[mask, "post_exercise_fluctuation_direction"]
    df.loc[mask, "glucose_60min_after_event"] = df.loc[mask, "glucose_60min_after_exercise"]

    df.to_csv(train_data_path_new, index=False)
    print(f"✅ Updated file saved to {train_data_path_new}")
    print(f"🔄 Updated {updated_rows} rows where exercise_time_gap_minutes != 999")


def enrich_with_out_of_range_samples(train_data_path, test_data_path, desired_range, output_path):
    print("🔍 Enriching test dataset with out-of-range training samples...")

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    out_of_range_train = train_df[
        (train_df["glucose_60min_after_event"] < desired_range[0]) |
        (train_df["glucose_60min_after_event"] > desired_range[1])
    ]

    print(f"📊 Found {len(out_of_range_train)} out-of-range training samples to enrich.")

    enriched_df = pd.concat([test_df, out_of_range_train], ignore_index=True)

    enriched_df.to_csv(output_path, index=False)
    print(f"✅ Enriched dataset saved to {output_path}. Total records: {len(enriched_df)}")

    return enriched_df


def analyze_glucose_distribution(filepath, column_name='glucose_60min_after_event'):
    """Read CSV, check for column, describe and plot data distribution."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

    if column_name not in df.columns:
        print(f"Column '{column_name}' does not exist in the file.")
        return None

    data = df[column_name].dropna()

    # Print basic statistics
    print(f"\nBasic statistics for column '{column_name}':")
    print(data.describe())

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {column_name}")
    plt.xlabel('Glucose Level')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, vert=False)
    plt.title(f"Boxplot of {column_name}")
    plt.xlabel('Glucose Level')
    plt.show()

    return data

def clarke_error_evaluation(csv_path, patient_id):
    """
    Generate Clarke Error Grid from a CSV file containing reference and prediction values.
    Grid lines follow standard Clarke Error Grid zone definitions.
    """
    # Load data
    reference_col = "glucose_aftermeal"
    prediction_col = "glucose_predicted"
    df = pd.read_csv(csv_path)
    if reference_col not in df.columns or prediction_col not in df.columns:
        raise ValueError(f"Columns '{reference_col}' and/or '{prediction_col}' not found in CSV file.")
    

    title = f"Clarke Error Grid - Patient_{patient_id}" 

    ref_values = df[reference_col].dropna().values
    pred_values = df[prediction_col].dropna().values

    min_len = min(len(ref_values), len(pred_values))
    ref_values = ref_values[:min_len]
    pred_values = pred_values[:min_len]

    # Plot base grid
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.xlabel("Reference Glucose (mg/dL)")
    plt.ylabel("Predicted Glucose (mg/dL)")
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)

    # Ideal 45° line
    plt.plot([0, 400], [0, 400], 'k:', label='Ideal')

    # Zone A (±20%)
    x_vals = np.linspace(0, 400, 1000)
    plt.plot(x_vals, x_vals * 1.2, 'k')  # upper
    plt.plot(x_vals, x_vals * 0.8, 'k')  # lower

    # Additional zone lines (matching visual grid you gave)
    plt.plot([0, 70], [180, 180], 'k')
    # plt.plot([70, 70], [180, 400], 'k')
    plt.plot([70, 70], [80, 400], 'k')

    plt.plot([70, 290], [180, 400], 'k')
    plt.plot([180, 180], [0, 70], 'k')
    plt.plot([180, 400], [70, 70], 'k')

    plt.plot([240, 240], [70, 180], 'k')
    plt.plot([240, 400], [180, 180], 'k')
    plt.plot([130, 180], [0, 70], 'k')
    plt.plot([70, 70], [0, 56], 'k')
    plt.plot([70, 400], [56, 320], 'k')
    plt.plot([0, 70], [70, 70], 'k')
    plt.plot([70, 0], [180, 180], 'k')

    # Zone labels
    plt.text(20, 20, "A", fontsize=12)
    plt.text(320, 320, "A", fontsize=12)
    plt.text(160, 250, "B", fontsize=12)
    plt.text(150, 80, "B", fontsize=12)
    plt.text(100, 360, "C", fontsize=12)
    plt.text(160, 20, "C", fontsize=12)
    plt.text(20, 150, "D", fontsize=12)
    plt.text(300, 100, "D", fontsize=12)
    plt.text(20, 360, "E", fontsize=12)
    plt.text(360, 20, "E", fontsize=12)

    # Classify zones
    zone = [0] * 5  # A, B, C, D, E
    for ref, pred in zip(ref_values, pred_values):
        if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
            zone[0] += 1  # Zone A
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zone[4] += 1  # Zone E
        elif (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
            zone[2] += 1  # Zone C
        elif (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or (
                175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    # Plot data points
    plt.scatter(ref_values, pred_values, color='black', s=8, zorder=5)

    plt.tight_layout()

    plt.savefig(f"results/model_evaluation/clarke_plot_{patient_id}.png")  

    zone_labels = ['A', 'B', 'C', 'D', 'E']
    zone_df = pd.DataFrame({
    'Zone': zone_labels,
    'Count': zone
})
    zone_df.to_csv(f'results/model_evaluation/{patient_id}_zone_counts.csv', index=False)
    print("✅ Zone counts saved to zone_counts.csv")


    return plt, zone