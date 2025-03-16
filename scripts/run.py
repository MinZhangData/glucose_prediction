# import sys
# sys.path.append('/Users/min/Documents/machine-learning-project')  # Ensure the correct path to the src directory
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import pandas as pd

from src.data_extraction import extract_event_data, load_patient_data, load_glucose_data,extract_all_events, find_bolus_meal_within_15min, find_nearest_bolus
from src.data_processing import calculate_fluctuations, calculate_fluctuation_rate
from src.model_training import train_model, predict_and_evaluate


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






def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))  
    root_dir = os.path.join(base_dir, "..")  


    # patient_id = "588"
    # training_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2018/train/json/")
    # testing_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2018/test/json/")
    # training_output_dir = os.path.join(root_dir, "data/processed/2018/training")
    # testing_output_dir = os.path.join(root_dir, "data/processed/2018/testing")



    patient_id = "596"
    training_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2020/train/json/")
    testing_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2020/test/json/")
    training_output_dir = os.path.join(root_dir, "data/processed/2020/training")
    testing_output_dir = os.path.join(root_dir, "data/processed/2020/testing")

    training_patient_file = f"{patient_id}-ws-training.json"
    testing_patient_file = f"{patient_id}-ws-testing.json"

    print("\nProcessing TRAINING data...")
    process_patient_data(training_data_dir, training_patient_file, training_output_dir)

    print("\nProcessing TESTING data...")
    process_patient_data(testing_data_dir, testing_patient_file, testing_output_dir)

    train_data_path = f"{training_output_dir}/{patient_id}_fluctuations.csv"
    test_data_path = f"{testing_output_dir}/{patient_id}_fluctuations.csv"
    model_output_category = f"models/{patient_id}_fluctuation_category_model.pkl"
    model_output_direction = f"models/{patient_id}_direction_model.pkl"

    print("Starting training process...")
    train_model(train_data_path, model_output_category, "post_fluctuation_category")
    train_model(train_data_path, model_output_direction, "post_fluctuation_direction")

    print("Running predictions and evaluations...")
    predict_and_evaluate(test_data_path, model_output_category, "post_fluctuation_category", patient_id)
    predict_and_evaluate(test_data_path, model_output_direction, "post_fluctuation_direction", patient_id)
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()


