import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import process_patient_data, enrich_with_out_of_range_samples,analyze_glucose_distribution,clarke_error_evaluation
from src.model_training import explain_with_counterfactuals,train_model,predict_and_evaluate_glucose_60min,evaluate_counterfactuals_regression



def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))  
    root_dir = os.path.join(base_dir, "..")  


    patient_id = "570"
    training_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2018/train/json/")
    testing_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2018/test/json/")
    training_output_dir = os.path.join(root_dir, "data/processed/2018/training")
    testing_output_dir = os.path.join(root_dir, "data/processed/2018/testing")



    # patient_id = "596"
    # training_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2020/train/json/")
    # testing_data_dir = os.path.join(root_dir, "data/raw/OhioT1DM/2020/test/json/")
    # training_output_dir = os.path.join(root_dir, "data/processed/2020/training")
    # testing_output_dir = os.path.join(root_dir, "data/processed/2020/testing")

    training_patient_file = f"{patient_id}-ws-training.json"
    testing_patient_file = f"{patient_id}-ws-testing.json"

  
    print("\nProcessing TRAINING data...")
    process_patient_data(training_data_dir, training_patient_file, training_output_dir)

    print("\nProcessing TESTING data...")
    process_patient_data(testing_data_dir, testing_patient_file, testing_output_dir)

    print("Starting training process...")

    train_data_path = f"{training_output_dir}/{patient_id}_fluctuations.csv"
    test_data_path = f"{testing_output_dir}/{patient_id}_fluctuations.csv"
    model_output_regression = f"models/{patient_id}_regression_model.pkl"

    result_file = f"results/prediction/prediction_results_{patient_id}.csv"
    eval_file = f"results/model_evaluation/evaluation_metrics_{patient_id}.csv"

    train_model(train_data_path, model_output_regression, "glucose_60min_after_event",patient_id,result_file)

    predict_and_evaluate_glucose_60min(test_data_path, model_output_regression, patient_id,result_file,eval_file)


    plt_obj, zone_counts = clarke_error_evaluation(result_file, patient_id)

    print(f"✅ Zone counts: A={zone_counts[0]}, B={zone_counts[1]}, C={zone_counts[2]}, D={zone_counts[3]}, E={zone_counts[4]}")
    # plt_obj.layout()
    # plt_obj.savefig("your_path.png")
    plt_obj.show() 

    user_input = input("Please enter the upper bound for glucose level (default is 200): ")

    # If user doesn't enter anything, use default value
    upper_bound = int(user_input) if user_input.strip() else 200

    # Example usage in a desired range
    desired_range = (70, upper_bound)
    print(f"Using desired glucose range: {desired_range}")

    enriched_output_path = f"models/{patient_id}_{desired_range}_enriched_test_data.csv"

    enrich_with_out_of_range_samples(train_data_path, test_data_path, desired_range,enriched_output_path)

    
    explain_with_counterfactuals_path = f"results/cf/{patient_id}{desired_range}_explain_with_counterfactuals.csv"

    query_input, explanation, regression_model, desired_range = explain_with_counterfactuals(
    patient_id=patient_id,
    model_path=model_output_regression,
    data_path=enriched_output_path,
    explain_with_counterfactuals_path=explain_with_counterfactuals_path,
    desired_range=desired_range,
    round_decimals=1
)

    # evaluate_counterfactual_quality(explain_with_counterfactuals_path,explain_with_counterfactuals_path_xls)

    _ = evaluate_counterfactuals_regression(query_input, explanation, regression_model, desired_range,patient_id)

    print("All tasks completed successfully!")



if __name__ == "__main__":
    main()


