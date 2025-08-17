# Standard libraries
import os
import joblib

# Data manipulation & visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: preprocessing & data splitting
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Scikit-learn: models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

# Scikit-learn: metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Statistical test
from scipy.stats import ttest_rel

# Additional models
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor

from dice_ml import Dice
import dice_ml
from itertools import combinations


# Patient feature configuration
PATIENT_FEATURE_CONFIG = {
    "570": "all",
    "544": "no_dual",
    "552": "no_dual",
    "584": "no_dual",
    "596": "no_dual",
    "559": "no_dual",
    "563": "no_dual",
    "588": "no_dual",
    "591": "no_dual",
    "540": "no_exercise",
    "575": "no_exercise",
}

def load_data(data_file):
    print(f"Loading training data from {data_file} ...")
    df = pd.read_csv(data_file)
    print(f"Columns loaded: {list(df.columns)}")
    return df

def preprocess_data(df):
    # Fill missing values
    fill_999_cols = ["bolus_time_gap_minutes_1", "exercise_time_gap_minutes", "bolus_time_gap_minutes_2"]
    for col in fill_999_cols:
        if col in df.columns:
            df[col] = df[col].fillna(999)
        else:
            print(f"⚠️ Warning: Column '{col}' missing in data, creating with default 999.")
            df[col] = 999
    
    if "bolus_dose_2" in df.columns:
        df["bolus_dose_2"] = df["bolus_dose_2"].fillna(0)
    else:
        print("⚠️ Warning: 'bolus_dose_2' missing, creating with default 0.")
        df["bolus_dose_2"] = 0

    # Binary indicator features
    df["has_bolus"] = (df["bolus_time_gap_minutes_1"] != 999).astype(int)
    df["has_exercise"] = (df["exercise_time_gap_minutes"] != 999).astype(int)
    df["has_dual_bolus"] = (df["bolus_time_gap_minutes_2"] != 999).astype(int)

    # Replace placeholders for categorical variables
    for col, replacement in [("bolus_direction_1", "No_Bolus"), ("exercise_direction", "No_Exercise")]:
        if col in df.columns:
            df[col] = df[col].replace(999, replacement)
        else:
            print(f"⚠️ Warning: Column '{col}' missing, creating with default '{replacement}'.")
            df[col] = replacement

    return df

def build_features(df):
    # Compute derived features
    # df["applied_bolus_value"] = df.get("bolus_dose_1", 0) * df.get("has_bolus", 0)
    df["applied_bolus_value"] = df["bolus_dose_1"].fillna(0) * df["has_bolus"].fillna(0)

    df["applied_dual_bolus_value"] = df.get("bolus_dose_2", 0) * df.get("has_dual_bolus", 0)
    df["applied_exercise_intensity"] = df.get("nearest_exercise_intensity", 0) * df.get("has_exercise", 0)
    df["applied_exercise_duration"] = df.get("nearest_exercise_duration", 0) * df.get("has_exercise", 0)

    # Fill possible NaN
    df["applied_exercise_intensity"] = df["applied_exercise_intensity"].fillna(0)
    df["applied_exercise_duration"] = df["applied_exercise_duration"].fillna(0)

    # Replace glucose value after double bolus if available
    if "has_dual_bolus" in df.columns and "glucose_60min_after_double_bolus" in df.columns:
        mask = (df["has_dual_bolus"] == 1) & df["glucose_60min_after_double_bolus"].notna()
        df.loc[mask, "glucose_60min_after_event"] = df.loc[mask, "glucose_60min_after_double_bolus"]

    return df

def get_feature_sets(patient_id):
    basic_features = ["meal_carbs", "bolus_direction_1", "applied_bolus_value", "pre_fluctuation_category", "has_bolus"]
    exercise_features = ["applied_exercise_intensity", "applied_exercise_duration", "exercise_direction", "has_exercise"]
    dual_bolus_features = ["applied_dual_bolus_value", "has_dual_bolus"]
    classification_features = basic_features + exercise_features + dual_bolus_features

    mode = PATIENT_FEATURE_CONFIG.get(patient_id, "basic")

    if mode == "all":
        regression_features = classification_features + ["glucose_30min_before_event", "glucose_at_event_time"]
    elif mode == "no_dual":
        regression_features = basic_features + exercise_features + ["glucose_30min_before_event", "glucose_at_event_time"]
    elif mode == "no_exercise":
        regression_features = basic_features + dual_bolus_features + ["glucose_30min_before_event", "glucose_at_event_time"]
    else:
        regression_features = basic_features + ["glucose_30min_before_event", "glucose_at_event_time"]

    return classification_features, regression_features

def encode_categorical(df, columns):
    label_encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            print(f"⚠️ Warning: Categorical column '{col}' missing, skipped encoding.")
    return df, label_encoders

def validate_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return False
    return True

def train_classification(df, features, target):
    from sklearn.exceptions import NotFittedError

    X = df[features]
    y = df[target]

    if y.isna().sum() > 0 or len(y.unique()) <= 1:
        print("❌ Classification target invalid: missing values or single class")
        return None

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        print(f"Training classifier: {name} ...")
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"✅ {name} training accuracy: {acc:.4f}")
        trained_models[name] = model
    return trained_models

def train_regression(df, features, target):
    if df[target].isna().sum() > 0:
        print("❌ Regression target contains missing values. Abort training.")
        return None

    X = df[features]
    y = df[target]

    print("Training SVR regression model...")
    model = SVR(kernel="rbf", C=1000.0, epsilon=0.01)
    print("Checking missing values in training features:")
    print(X.isna().sum())

    model.fit(X, y)

    # Predict and generate fluctuation direction
    df["predicted_glucose"] = model.predict(X)
    df["predicted_post_fluctuation_direction"] = np.where(
        df["predicted_glucose"] > df["glucose_at_event_time"], "UP",
        np.where(df["predicted_glucose"] < df["glucose_at_event_time"], "DOWN", "STABLE")
    )
    print("✅ Regression training complete.")
    return {"SVR": model}, df

def save_model_data(model_data, output_model_file):
    print(f"Saving model data to {output_model_file} ...")
    joblib.dump(model_data, output_model_file)
    print("✅ Model saved successfully.")

def train_model(data_file: str, output_model_file: str, target: str, patient_id: str, result_file: str):
    df = load_data(data_file)

    if target not in df.columns:
        print(f"❌ Target column '{target}' not found. Abort.")
        return

    print(f"Target '{target}' unique values: {df[target].unique()}")
    print(f"Missing values in target: {df[target].isna().sum()}")

    # Load existing model data
    if os.path.exists(output_model_file):
        model_data = joblib.load(output_model_file)
        if not isinstance(model_data, dict):
            print("⚠️ Model file corrupted, reinitializing.")
            model_data = {}
        model_data.setdefault("models", {})
        model_data.setdefault("label_encoders", {})
        model_data.setdefault("features", {})
    else:
        model_data = {"models": {}, "label_encoders": {}, "features": {}}

    df = preprocess_data(df)
    df = build_features(df)

    classification_features, regression_features = get_feature_sets(patient_id)

    # Validate feature columns
    required_features = classification_features if target == "post_fluctuation_category" else regression_features
    if not validate_columns(df, required_features):
        return

    # Encode categorical features
    categorical_cols = ["bolus_direction_1", "exercise_direction", "pre_fluctuation_category"]
    df, label_encoders = encode_categorical(df, categorical_cols)

    trained_models = {}

    if target == "post_fluctuation_category":
        models = train_classification(df, classification_features, target)
        if models:
            trained_models[target] = models

    elif target == "glucose_60min_after_event":
        models, df = train_regression(df, regression_features, target)
        if models:
            trained_models[target] = models

    if not trained_models:
        print("❌ No models trained successfully.")
        return

    # Save models and encoders
    model_data["models"].update(trained_models)
    model_data["label_encoders"].update(label_encoders)
    model_data["features"] = {
        "classification": classification_features,
        "regression": regression_features
    }

    save_model_data(model_data, output_model_file)

    try:
        save_cols = classification_features + list(set(regression_features) - set(classification_features))
        if "post_fluctuation_direction" in df.columns:
            save_cols.append("post_fluctuation_direction")
        df[save_cols].to_csv(result_file, index=False)
        print(f"📁 Feature data saved to: {result_file}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save feature data - {e}")



def explain_with_counterfactuals(
    patient_id: str,
    model_path: str,
    data_path: str,
    explain_with_counterfactuals_path: str,
    desired_range: list,
    round_decimals: int = 1
):
    print("\n🔍 Generating counterfactual explanation using DiCE...")

    # -- Step 1: Load model and features --
    try:
        model_data = joblib.load(model_path)
        trained_models = model_data.get("models", {})
        if "glucose_60min_after_event" not in trained_models:
            raise KeyError("Target model 'glucose_60min_after_event' not found in model data.")
        regression_model = trained_models["glucose_60min_after_event"]
        if isinstance(regression_model, dict):
            regression_model = next(iter(regression_model.values()))

        features_dict = model_data.get("features", {})
        continuous_features = features_dict.get("regression", [])
        all_features = features_dict.get("all", continuous_features)
    except Exception as e:
        print(f"❌ Failed to load model or features: {e}")
        return

    # -- Step 2: Load and preprocess data --
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Feature engineering: binary flags for bolus and exercise presence
    df["has_bolus"] = (df.get("bolus_time_gap_minutes_1", pd.Series()) != 999).astype(int)
    df["has_exercise"] = (df.get("exercise_time_gap_minutes", pd.Series()) != 999).astype(int)
    df["has_dual_bolus"] = (df.get("bolus_time_gap_minutes_2", pd.Series()) != 999).astype(int)

    df["applied_bolus_value"] = df.get("bolus_dose_1", 0) * df["has_bolus"]
    df["applied_dual_bolus_value"] = df.get("bolus_dose_2", 0) * df["has_dual_bolus"]
    df["applied_dual_bolus_value"] = df["applied_dual_bolus_value"].fillna(0)

    df["nearest_exercise_intensity"] = df.get("nearest_exercise_intensity", pd.Series()).fillna(0)
    df["nearest_exercise_duration"] = df.get("nearest_exercise_duration", pd.Series()).fillna(0)

    df["applied_exercise_intensity"] = df["nearest_exercise_intensity"] * df["has_exercise"]
    df["applied_exercise_duration"] = df["nearest_exercise_duration"] * df["has_exercise"]

    # -- Step 3: Safe categorical encoding --
    categorical_candidates = [f for f in all_features if f in df.columns and df[f].dtype == "object"]
    manual_categoricals = ['bolus_direction_1', 'exercise_direction', 'pre_fluctuation_category']
    categorical_features = list(set(categorical_candidates + manual_categoricals))

    label_encoders = {}
    for col in categorical_features:
        if col in df.columns:
            try:
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna("UNKNOWN")
                le.fit(df[col])
                df[col] = df[col].map(lambda x: x if x in le.classes_ else "UNKNOWN")
                df[col] = le.transform(df[col])
                label_encoders[col] = le
            except Exception as e:
                print(f"⚠️ Failed to encode column '{col}': {e}")
                df[col] = -1  # fallback

    # -- Step 4: Select features based on patient --
    basic = ["meal_carbs", "bolus_direction_1", "applied_bolus_value", "pre_fluctuation_category", "has_bolus"]
    exercise = ["applied_exercise_intensity", "applied_exercise_duration", "exercise_direction", "has_exercise"]
    dual = ["applied_dual_bolus_value", "has_dual_bolus"]
    common = ["glucose_30min_before_event", "glucose_at_event_time"]

    if patient_id in ["570"]:
        regression_features = basic + exercise + dual + common
    elif patient_id in ["552", "584", "596", "559", "563", "588", "591", "544"]:
        regression_features = basic + exercise + common
    elif patient_id in ["540", "575"]:
        regression_features = basic + dual + common
    else:
        regression_features = basic + common

    all_features_final = [col for col in regression_features if col in df.columns]
    continuous_features_final = list(set(continuous_features) & set(all_features_final))

    # -- Step 5: Filter and drop NA --
    required_cols = [col for col in all_features_final + ['glucose_60min_after_event'] if col in df.columns]
    df = df[required_cols]

    print("❗ Missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])

    df = df.dropna()
    if df.empty:
        print("❌ No data left after dropping missing values.")
        return

    # Filter samples exceeding upper bound
    upper_bound = desired_range[1]
    sample_input = df[df["glucose_60min_after_event"] > upper_bound].copy()
    if sample_input.empty:
        print("❌ No valid out-of-range samples to explain.")
        return

    print(f"📦 Selected {len(sample_input)} samples for counterfactual explanation.")

    # -- Step 6: Setup DiCE explainer --
    fixed = ['glucose_30min_before_event', 'glucose_at_event_time']
    cont_feats = [f for f in continuous_features_final if f not in fixed]
    cat_feats = [f for f in categorical_features if f not in fixed]

    data_dice = dice_ml.Data(
        dataframe=sample_input,
        continuous_features=cont_feats,
        categorical_features=cat_feats,
        outcome_name="glucose_60min_after_event"
    )
    model_dice = dice_ml.Model(model=regression_model, backend="sklearn", model_type="regressor")
    exp = Dice(data_dice, model_dice, method="random")

    # -- Step 7: Generate counterfactuals --
    query_input = sample_input.drop(columns=["glucose_60min_after_event"])
    original_carbs = float(sample_input["meal_carbs"].values[0])

    features_to_vary = ["meal_carbs", "applied_bolus_value", "bolus_direction_1"]
    optional_feats = ["applied_dual_bolus_value", "exercise_direction", "applied_exercise_intensity", "applied_exercise_duration"]
    features_to_vary.extend([f for f in optional_feats if f in sample_input.columns])

    permitted_range = {}
    if "meal_carbs" in sample_input.columns:
        permitted_range["meal_carbs"] = [0, original_carbs]
    if "applied_exercise_duration" in sample_input.columns:
        permitted_range["applied_exercise_duration"] = [0, 240]
    if "applied_exercise_intensity" in sample_input.columns:
        permitted_range["applied_exercise_intensity"] = [0, 10]

    try:
        explanation = exp.generate_counterfactuals(
            query_input,
            total_CFs=5,
            desired_range=desired_range,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,
            verbose=True,
            proximity_weight=0.2,
            stopping_threshold=0.05,
            desired_class="desired"
        )
    except Exception as e:
        print(f"❌ Failed to generate counterfactuals: {e}")
        return

    # -- Step 8: Extract results --
    results = []
    total_queries = len(query_input)
    queries_with_cf = 0

    for idx, cf in enumerate(explanation.cf_examples_list):
        original = cf.test_instance_df.iloc[0].to_dict()
        original_outcome = original.pop("glucose_60min_after_event", None)
        cf_df = cf.final_cfs_df

        if cf_df is None or cf_df.empty:
            results.append({
                "query_index": idx,
                "changed_glucose_60min_after_event_from": round(original_outcome, round_decimals) if original_outcome is not None else None,
                "changed_glucose_60min_after_event_to": None,
                "changed_features": "",
                "has_counterfactual": 0
            })
            continue

        queries_with_cf += 1
        for _, row in cf_df.iterrows():
            changed = {
                "query_index": idx,
                "changed_glucose_60min_after_event_from": round(original_outcome, round_decimals),
                "changed_glucose_60min_after_event_to": round(row["glucose_60min_after_event"], round_decimals),
                "changed_features": [],
                "has_counterfactual": 1
            }

            for feat, orig_val in original.items():
                cf_val = row.get(feat, None)
                if cf_val is None:
                    continue
                if isinstance(orig_val, (int, float)) and isinstance(cf_val, (int, float)):
                    if round(orig_val, round_decimals) != round(cf_val, round_decimals):
                        changed[f"{feat}_from"] = round(orig_val, round_decimals)
                        changed[f"{feat}_to"] = round(cf_val, round_decimals)
                        changed["changed_features"].append(feat)
                elif orig_val != cf_val:
                    changed[f"{feat}_from"] = orig_val
                    changed[f"{feat}_to"] = cf_val
                    changed["changed_features"].append(feat)

            changed["changed_features"] = ", ".join(changed["changed_features"])
            results.append(changed)

    # -- Step 9: Save results --
    try:
        os.makedirs(os.path.dirname(explain_with_counterfactuals_path), exist_ok=True)
        cf_results_df = pd.DataFrame(results)
        cf_results_df.to_csv(explain_with_counterfactuals_path, index=False)

        summary = pd.DataFrame({
            "query_index": ["SUMMARY"],
            "queries_with_cf": [queries_with_cf],
            "total_queries": [total_queries],
            "percentage_with_cf": [queries_with_cf / total_queries * 100 if total_queries else 0]
        })
        summary.to_csv(explain_with_counterfactuals_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"❌ Failed to save explanation results: {e}")
        return

    print(f"📊 Summary: {queries_with_cf}/{total_queries} queries have counterfactuals.")
    print(f"✅ Explanation saved to {explain_with_counterfactuals_path}")

    return query_input, explanation, regression_model, desired_range



def evaluate_counterfactuals_regression(query_input, explanation, regression_model, desired_range=None, patient_id=None, output_dir="results/cf_evaluation"):
    """
    Evaluate counterfactual explanations for regression tasks and optionally save results to CSV.

    Args:
        query_input (pd.DataFrame): Original input features.
        explanation (dice_ml.explainer_interfaces.ExplainerBase): DiCE explanation object.
        regression_model (sklearn.base.RegressorMixin): Trained regression model.
        desired_range (tuple): Desired prediction range (lower, upper). Default is (70, 120).
        patient_id (str): Patient identifier used for naming output file.
        output_dir (str): Directory to save results CSV.
    
    Returns:
        results_dict: dict with Validity, Proximity, Sparsity, Diversity scores.
    """
    if desired_range is None:
        desired_range = (70, 120)
        print("ℹ️ Using default desired_range = (70, 120)")
    else:
        print(f"✅ Using user-provided desired_range = {desired_range}")

    print("\n🔍 Evaluating Counterfactual Explanations for Regression...")

    proximities = []
    sparsities = []
    validities = []
    cf_vectors = []

    feature_names = query_input.columns.tolist()
    query_array = query_input.values

    for idx, cf_example in enumerate(explanation.cf_examples_list):
        original_instance_df = cf_example.test_instance_df.drop(columns=["glucose_60min_after_event"], errors="ignore")
        original_instance = original_instance_df.values[0]

        if cf_example.final_cfs_df is not None and not cf_example.final_cfs_df.empty:
            cf_instances_df = cf_example.final_cfs_df.drop(columns=["glucose_60min_after_event"], errors="ignore")
            cf_instances = cf_instances_df.values

            for cf in cf_instances:
                proximity = np.linalg.norm(cf - original_instance, ord=2)
                proximities.append(proximity)

                sparsity = np.sum(np.abs(cf - original_instance) > 1e-3)
                sparsities.append(sparsity)

                cf_df = pd.DataFrame([cf], columns=feature_names)
                cf_pred = regression_model.predict(cf_df)[0]
                is_valid = desired_range[0] <= cf_pred <= desired_range[1]
                validities.append(int(is_valid))

                cf_vectors.append(cf)

    if len(cf_vectors) > 1:
        diversity_scores = [np.linalg.norm(a - b, ord=2) for a, b in combinations(cf_vectors, 2)]
        diversity = np.mean(diversity_scores)
    else:
        diversity = 0.0

    results = {
        "Validity (%)": np.mean(validities) * 100 if validities else 0.0,
        "Average Proximity (L2)": np.mean(proximities) if proximities else 0.0,
        "Average Sparsity (# features changed)": np.mean(sparsities) if sparsities else 0.0,
        "Diversity (avg L2 between CFs)": diversity
    }

    print("📊 Evaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # 🔽 Save to CSV
    if patient_id:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{patient_id}_evaluation.csv")
        df_result = pd.DataFrame([results])
        df_result.to_csv(csv_path, index=False)
        print(f"✅ Results saved to: {csv_path}")

    return results





def safe_label_encode(df, col, le, unknown_label="UNKNOWN"):
    """
    Safely encode categorical column using a LabelEncoder.
    If the value is not recognized or missing, encode as 'UNKNOWN' if available,
    else encode as -1.
    """
    def encode_val(x):
        if pd.isna(x):
            # Missing value
            if unknown_label in le.classes_:
                return le.transform([unknown_label])[0]
            else:
                return -1
        elif x in le.classes_:
            return le.transform([x])[0]
        elif unknown_label in le.classes_:
            return le.transform([unknown_label])[0]
        else:
            return -1

    return df[col].map(encode_val)


def predict_and_evaluate(test_data_file: str, model_file: str, target: str, patient_id: str):
    try:
        print(f"📂 Loading test data from {test_data_file}...")
        df_test = pd.read_csv(test_data_file)
    except Exception as e:
        print(f"❌ Failed to load test data file: {e}")
        return

    if target not in df_test.columns:
        print(f"❌ Error: Target column '{target}' not found in test data!")
        print(f"Available columns: {df_test.columns.tolist()}")
        return

    try:
        print(f"📂 Loading model from {model_file}...")
        model_data = joblib.load(model_file)
    except Exception as e:
        print(f"❌ Failed to load model file: {e}")
        return

    models = model_data.get("models", {})
    label_encoders = model_data.get("label_encoders", {})
    all_features = model_data.get("features", {})

    if target not in models:
        print(f"❌ Error: No model found for target '{target}' in the model file!")
        return

    classification_features = [
        "meal_carbs", "applied_bolus_value", "bolus_direction_1",
        "applied_exercise_intensity", "applied_exercise_duration",
        "exercise_direction", "pre_fluctuation_category", "has_bolus", "has_exercise"
    ]

    if target in ["post_fluctuation_category", "post_fluctuation_direction"]:
        features = classification_features
    else:
        features = all_features.get("regression", [])

    if not features:
        print(f"⚠️ Warning: No features found for target '{target}', using empty feature list.")
        features = []

    print(f"📌 Using features for prediction: {features}")

    # Encode categorical variables safely
    categorical_cols = ["bolus_direction_1", "exercise_direction", "pre_fluctuation_category"]
    for col in categorical_cols:
        if col in df_test.columns and col in label_encoders:
            try:
                df_test[col] = safe_label_encode(df_test, col, label_encoders[col])
            except Exception as e:
                print(f"⚠️ Warning: Encoding failed for column '{col}': {e}")
                # fallback: fill with -1
                df_test[col] = -1

    # Fill missing features with default values
    for col in features:
        if col not in df_test.columns:
            print(f"⚠️ Missing feature '{col}', filling with 0.")
            df_test[col] = 0

    # Convert all features to numeric, coerce errors, fill missing with mean or default 0
    for col in features:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        if df_test[col].isnull().all():
            print(f"⚠️ All values missing in feature '{col}', filling with 0.")
            df_test[col].fillna(0, inplace=True)
        else:
            mean_val = df_test[col].mean()
            df_test[col].fillna(mean_val, inplace=True)

    X_test = df_test[features]

    target_models = models[target]

    results = {}
    metrics = ["Precision", "Recall", "F1-Score", "Accuracy"]

    result_file = f"results/prediction/prediction_results_{patient_id}.csv"
    if os.path.exists(result_file):
        try:
            df_results = pd.read_csv(result_file)
            # To avoid misalignment, reindex by df_test index if possible
            if len(df_results) != len(df_test):
                print("⚠️ Warning: result file row count differs from test data. Overwriting results.")
                df_results = df_test[["meal_carbs"]].copy()
        except Exception as e:
            print(f"⚠️ Warning: Failed to read existing result file: {e}. Starting new result DataFrame.")
            df_results = df_test[["meal_carbs"]].copy()
    else:
        df_results = df_test[["meal_carbs"]].copy()

    true_value_col = "category_true_value" if target == "post_fluctuation_category" else "direction_true_value"
    df_results[true_value_col] = df_test[target]

    for model_name, model in target_models.items():
        print(f"🚀 Predicting with {model_name} for {target}...")
        try:
            preds = model.predict(X_test)
        except Exception as e:
            print(f"❌ Prediction failed for model '{model_name}': {e}")
            continue

        df_test[f"predicted_{target}_{model_name}"] = preds
        df_results[f"predicted_{target}_{model_name}"] = preds

        if df_test[target].nunique() > 1:
            try:
                scores = [
                    precision_score(df_test[target], preds, average='weighted', zero_division=1),
                    recall_score(df_test[target], preds, average='weighted', zero_division=1),
                    f1_score(df_test[target], preds, average='weighted', zero_division=1),
                    accuracy_score(df_test[target], preds)
                ]
            except Exception as e:
                print(f"⚠️ Warning: Metric calculation failed for model '{model_name}': {e}")
                scores = [0.0] * 4
        else:
            scores = [1.0] * 4
            print(f"⚠️ Only one class present in true labels for {target}, assigning perfect scores.")

        results[model_name] = scores
        print(f"✅ {model_name} - {target} Scores: {dict(zip(metrics, scores))}")

    try:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        df_results.to_csv(result_file, index=False)
        print(f"📂 Final prediction results saved to {result_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

    # Visualization
    if results:
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
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_title(f"{target} Prediction Performance - Patient {patient_id}")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.set_xlabel("Models")
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No results to visualize.")


def predict_and_evaluate_direction(test_data_file: str, patient_id: str):
    print(f"📂 Loading test data from {test_data_file}...")

    # Read test data
    df_test = pd.read_csv(test_data_file)

    # Ensure the data contains the required columns
    required_columns = ["post_fluctuation_direction", "glucose_at_event_time", "glucose_60min_after_event"]
    missing_columns = [col for col in required_columns if col not in df_test.columns]

    if missing_columns:
        print(f"❌ Error: Missing columns {missing_columns} in test data!")
        return

    print("🔢 Computing predicted_post_fluctuation_direction...")

    # Calculate predicted_post_fluctuation_direction
    df_test["predicted_post_fluctuation_direction"] = df_test.apply(
        lambda row: "Up" if row["glucose_60min_after_event"] > row["glucose_at_event_time"]
        else ("Down" if row["glucose_60min_after_event"] < row["glucose_at_event_time"] else "Stable"), axis=1
    )

    # Calculate classification metrics
    true_values = df_test["post_fluctuation_direction"]
    pred_values = df_test["predicted_post_fluctuation_direction"]

    precision = precision_score(true_values, pred_values, average='weighted', zero_division=1)
    recall = recall_score(true_values, pred_values, average='weighted', zero_division=1)
    f1 = f1_score(true_values, pred_values, average='weighted', zero_division=1)
    accuracy = accuracy_score(true_values, pred_values)

    results = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy
    }

    print(f"✅ Evaluation Results:\n"
          f"   Precision: {precision:.4f}\n"
          f"   Recall: {recall:.4f}\n"
          f"   F1-Score: {f1:.4f}\n"
          f"   Accuracy: {accuracy:.4f}")

    # Save prediction results
    result_file = f"data/processed/2018/training/prediction_results_{patient_id}.csv"
    df_results = df_test[["event_value", "post_fluctuation_direction", "predicted_post_fluctuation_direction"]]

    if os.path.exists(result_file):
        df_results.to_csv(result_file, mode='a', header=False, index=False)
        print(f"📂 Appended prediction results to {result_file}")
    else:
        df_results.to_csv(result_file, index=False)
        print(f"📂 Saved prediction results to {result_file}")

    # Visualize Precision, Recall, F1-Score, Accuracy
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    
    metrics = list(results.keys())
    scores = list(results.values())

    sns.barplot(x=metrics, y=scores, palette="Set2")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Performance Metrics - Patient {patient_id}")

    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=12)

    plt.show()



def predict_and_evaluate_glucose_60min(test_data_file: str, model_file: str, patient_id: str,result_file, eval_file):
    """
    Predict glucose_60min_after_event using a trained model and evaluate performance.
    Save prediction results and evaluation metrics to files.
    """

    target = "glucose_60min_after_event"

    # Load test data
    print(f"Loading test data from {test_data_file}...")
    df_test = pd.read_csv(test_data_file)

    if target not in df_test.columns:
        print(f"Error: Target column '{target}' not found in test data.")
        return

    # Load model dictionary and extract the SVR regression model
    print(f"Loading model from {model_file}...")
    model_data = joblib.load(model_file)
    model_dict = model_data.get("models", {})
    model = model_dict.get(target, None)

    # If model is stored as dict (multiple models), try to get the SVR model explicitly
    if isinstance(model, dict):
        model = model.get("SVR", None)

    if model is None:
        print(f"Error: No trained SVR model found for target '{target}' in {model_file}.")
        return

    print(f"Model loaded: {type(model).__name__}")

    # Get features for regression
    features = model_data.get("features", {}).get("regression", [])
    if not features:
        print("Error: Regression features not found in model data.")
        return

    print(f"Using features: {features}")

    # Check missing features in test data and fill with default 0
    missing_feats = [f for f in features if f not in df_test.columns]
    if missing_feats:
        print(f"Warning: Missing features {missing_feats} in test data, filling with zeros.")
        for feat in missing_feats:
            df_test[feat] = 0

    # Prepare test feature matrix, convert to numeric
    X_test = df_test[features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predict
    print(f"Predicting {target}...")
    df_test["glucose_predicted"] = model.predict(X_test)

    # Convert target column to numeric for evaluation
    df_test[target] = pd.to_numeric(df_test[target], errors='coerce')
    actual = df_test[target].dropna()
    predicted = df_test.loc[actual.index, "glucose_predicted"]

    if len(actual) == 0:
        print("Warning: No valid actual values available for evaluation.")
        mae = mse = r2 = p_value = None
    else:
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        _, p_value = ttest_rel(actual, predicted)

        print("Evaluation metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  p-value (paired t-test): {p_value:.4e}")

    # Rename target for saving
    df_test.rename(columns={target: "glucose_aftermeal"}, inplace=True)

    # Save predictions
    pred_df = df_test[["glucose_aftermeal", "glucose_predicted"]].copy()

    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file)

        # Align lengths by padding or truncating
        len_diff = len(pred_df) - len(existing_df)
        if len_diff > 0:
            extra_rows = pd.DataFrame(np.nan, index=range(len_diff), columns=existing_df.columns)
            existing_df = pd.concat([existing_df, extra_rows], ignore_index=True)
        elif len_diff < 0:
            existing_df = existing_df.iloc[:len(pred_df)]

        # Avoid column name clashes
        base_actual_col = "glucose_aftermeal"
        base_pred_col = "glucose_predicted"
        actual_col = base_actual_col
        pred_col = base_pred_col
        count = 1
        while actual_col in existing_df.columns or pred_col in existing_df.columns:
            actual_col = f"{base_actual_col}_new{count}"
            pred_col = f"{base_pred_col}_new{count}"
            count += 1

        existing_df[actual_col] = pred_df["glucose_aftermeal"].values
        existing_df[pred_col] = pred_df["glucose_predicted"].values

        existing_df.to_csv(result_file, index=False)
    else:
        pred_df.to_csv(result_file, index=False)

    print(f"Prediction results saved to {result_file}")

    # Save evaluation metrics
    eval_df = pd.DataFrame({
        "MAE": [mae],
        "MSE": [mse],
        "R2": [r2],
        "p_value": [p_value]
    })
    eval_df.to_csv(eval_file, index=False)
    print(f"Evaluation metrics saved to {eval_file}")

from collections import Counter



def generate_symmetric_ranges(start=(70, 120), min_low=50, max_high=280, steps=15):
    ranges = []
    low, high = start
    for i in range(steps + 1):
        new_low = max(min_low, low - i * 5)
        new_high = min(max_high, high + i * 5)
        if new_low < new_high:
            ranges.append((new_low, new_high))
    return ranges


def find_best_desired_range_by_distribution(model_path, data_path, round_decimals=1, steps=15):
    print("🔍 Automatically searching best desired_range from (70, 120), expanding outward...")

    # Load model and config
    model_data = joblib.load(model_path)
    regression_model = model_data["models"].get("glucose_60min_after_event")
    features_dict = model_data["features"]
    continuous_features = features_dict.get("regression", [])
    all_features = features_dict.get("all", continuous_features)

    # Load and preprocess data
    df = pd.read_csv(data_path)
    df["has_bolus"] = (df["bolus_time_gap_minutes_1"] != 999).astype(int)
    df["has_exercise"] = (df["exercise_time_gap_minutes"] != 999).astype(int)
    df["applied_bolus_value"] = df["bolus_dose_1"] * df["has_bolus"]
    df["applied_exercise_intensity"] = df["nearest_exercise_intensity"] * df["has_exercise"]
    df["applied_exercise_duration"] = df["nearest_exercise_duration"] * df["has_exercise"]

    # Encode categoricals
    label_encoders = {}
    categorical_features = [f for f in all_features if f not in continuous_features and df[f].dtype == 'object']
    for col in categorical_features + ['bolus_direction_1', 'exercise_direction', 'pre_fluctuation_category']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    df = df.dropna(subset=all_features + ['glucose_60min_after_event'])

    # Filter samples outside initial desired range
    sample_input = df[(df["glucose_60min_after_event"] < 70) | (df["glucose_60min_after_event"] > 120)]
    if sample_input.empty:
        raise ValueError("❌ No valid out-of-range samples to explain.")

    sample_input = sample_input[all_features + ['glucose_60min_after_event']].copy()

    # Prepare DiCE
    fixed_features = ['glucose_30min_before_event', 'glucose_at_event_time']
    cont_used = [f for f in continuous_features if f not in fixed_features]
    cat_used = [f for f in categorical_features if f not in fixed_features]

    data_dice = dice_ml.Data(
        dataframe=sample_input,
        continuous_features=cont_used,
        categorical_features=cat_used,
        outcome_name="glucose_60min_after_event"
    )
    model_dice = dice_ml.Model(model=regression_model, backend="sklearn", model_type="regressor")
    exp = Dice(data_dice, model_dice, method="random")

    query_input = sample_input.drop(columns=["glucose_60min_after_event"])

    # Generate candidate ranges
    candidate_ranges = generate_symmetric_ranges(start=(70, 120), min_low=50, max_high=280, steps=steps)
    print(f"📊 Candidate desired_ranges (symmetric expansion): {candidate_ranges}")

    best_range = None
    max_success = 0

    # Test each range
    for r in candidate_ranges:
        explanation = exp.generate_counterfactuals(
            query_input,
            total_CFs=1,
            desired_range=r,
            features_to_vary=["event_value", "applied_bolus_value", "bolus_direction_1", "exercise_direction",
                              "applied_exercise_intensity", "applied_exercise_duration"],
            permitted_range={
                "event_value": [0, float(sample_input["event_value"].max())],
                "applied_exercise_duration": [0, 180],
                "applied_exercise_intensity": [1, 10]
            },
            verbose=False
        )

        queries_with_cf = sum([
            1 for ex in explanation.cf_examples_list if ex.final_cfs_df is not None and not ex.final_cfs_df.empty
        ])

        print(f"🔎 Range {r}: {queries_with_cf}/{len(query_input)} counterfactuals")

        if queries_with_cf > max_success:
            max_success = queries_with_cf
            best_range = r

    if best_range:
        print(f"\n✅ Best desired_range: {best_range} with {max_success} successful counterfactuals.")
    else:
        print("❌ No desired_range found that works for any samples.")

    return best_range
