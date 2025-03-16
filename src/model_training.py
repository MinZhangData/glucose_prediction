import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(data_file: str, output_model_file: str, target: str):
    print(f"Loading training data from {data_file}...")
    df = pd.read_csv(data_file)

    features = ["event_value", "nearest_bolus_value", "time_gap_minutes", 
                "direction", "pre_fluctuation_category", "pre_fluctuation_direction"]
    
    categorical_cols = ["direction", "pre_fluctuation_category", "pre_fluctuation_direction"]
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df["is_time_gap_large"] = (df["time_gap_minutes"] > 30).astype(int)
    features.append("is_time_gap_large")

    X_train = df[features]
    y_train = df[target]

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    trained_models = {}
    for model_name, model in models.items():
        print(f"Training {model_name} for {target}...")
        trained_models[model_name] = model.fit(X_train, y_train)

    model_data = {"models": trained_models, "label_encoders": label_encoders, "features": features}
    joblib.dump(model_data, output_model_file)
    print(f"✅ Models for {target} saved to {output_model_file}")

def predict_and_evaluate(test_data_file: str, model_file: str, target: str, patient_id: str):
    print(f"Loading test data from {test_data_file}...")
    df_test = pd.read_csv(test_data_file)

    model_data = joblib.load(model_file)
    models = model_data["models"]
    label_encoders = model_data["label_encoders"]
    features = model_data["features"]

    categorical_cols = ["direction", "pre_fluctuation_category", "pre_fluctuation_direction"]
    for col in categorical_cols:
        df_test[col] = label_encoders[col].transform(df_test[col])

    df_test["is_time_gap_large"] = (df_test["time_gap_minutes"] > 30).astype(int)
    X_test = df_test[features]

    results = {}
    metrics = ["Precision", "Recall", "F1-Score", "Accuracy"]

    for model_name, model in models.items():
        print(f"Predicting with {model_name} for {target}...")
        df_test[f"predicted_{target}_{model_name}"] = model.predict(X_test)

        scores = [
            precision_score(df_test[target], df_test[f"predicted_{target}_{model_name}"], average='weighted', zero_division=1),
            recall_score(df_test[target], df_test[f"predicted_{target}_{model_name}"], average='weighted', zero_division=1),
            f1_score(df_test[target], df_test[f"predicted_{target}_{model_name}"], average='weighted', zero_division=1),
            accuracy_score(df_test[target], df_test[f"predicted_{target}_{model_name}"])
        ]

        results[model_name] = scores
        print(f"{model_name} - {target}: {scores}")

    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.2

    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        scores = [results[m][i] for m in model_names]
        bars = ax.bar(x + i * width, scores, width, label=metric, color=palette[i])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_title(f"{target} Prediction Performance - Patient {patient_id}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.show()

# example usage

if __name__ == "__main__":
    train_data_path = "data/processed/training/559_fluctuations.csv"
    test_data_path = "data/processed/testing/559_fluctuations.csv"
    patient_id = "559"
    
    model_output_category = "models/fluctuation_category_model.pkl"
    model_output_direction = "models/direction_model.pkl"

    print("Starting training process...")
    train_model(train_data_path, model_output_category, "post_fluctuation_category")
    train_model(train_data_path, model_output_direction, "post_fluctuation_direction")

    print("Running predictions and evaluations...")
    predict_and_evaluate(test_data_path, model_output_category, "post_fluctuation_category", patient_id)
    predict_and_evaluate(test_data_path, model_output_direction, "post_fluctuation_direction", patient_id)
    print("All tasks completed successfully!")