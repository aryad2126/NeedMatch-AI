"""
NeedMatch AI - Machine Learning Model Training Pipeline
Trains a Scikit-learn RandomForestRegressor to predict 30-day future demand
using chronological train/test splitting, baseline benchmarking, and model persistence.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_generator import generate_historical_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "historical_demand.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "rf_demand_model.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")

def train_and_evaluate_model(csv_path: str = DATA_PATH):
    """
    Executes the end-to-end ML training and evaluation pipeline:
    1. Loads historical data (or generates if missing).
    2. Chronological train/test split to prevent temporal leakage.
    3. Fits Scikit-learn Pipeline with OneHotEncoder + RandomForestRegressor.
    4. Evaluates on test set alongside a baseline predictor.
    5. Persists artifacts and metrics.
    """
    if not os.path.exists(csv_path):
        print(f"Data file not found at {csv_path}. Generating dataset now...")
        generate_historical_dataset(csv_path)

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # Feature definitions
    cat_features = ["category", "org_id"]
    num_features = [
        "month",
        "historical_demand_qty",
        "fulfilled_qty",
        "inventory_level",
        "outstanding_requests",
        "season_factor"
    ]
    target_col = "target_next_30d_demand"

    # Chronological Split (Last 20% distinct monthly cycles as held-out test set)
    unique_dates = sorted(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    X_train = train_df[cat_features + num_features]
    y_train = train_df[target_col]

    X_test = test_df[cat_features + num_features]
    y_test = test_df[target_col]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    # Regressor
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf)
    ])

    print("Fitting Random Forest pipeline on chronological training set...")
    pipeline.fit(X_train, y_train)

    # Predictions on test set
    y_pred_rf = pipeline.predict(X_test)

    # Baseline Model: Mean target demand grouped by (category, org_id) from training set
    # (Fallback to category mean or overall train mean if unseen)
    baseline_lookup = train_df.groupby(["category", "org_id"])[target_col].mean().to_dict()
    category_mean_lookup = train_df.groupby("category")[target_col].mean().to_dict()
    overall_mean = train_df[target_col].mean()

    def predict_baseline(row):
        key = (row["category"], row["org_id"])
        if key in baseline_lookup:
            return baseline_lookup[key]
        return category_mean_lookup.get(row["category"], overall_mean)

    y_pred_baseline = X_test.apply(predict_baseline, axis=1).values

    # Evaluation Metrics
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    rf_r2 = float(r2_score(y_test, y_pred_rf))

    baseline_mae = float(mean_absolute_error(y_test, y_pred_baseline))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    baseline_r2 = float(r2_score(y_test, y_pred_baseline))

    mae_improvement_pct = float(((baseline_mae - rf_mae) / baseline_mae) * 100)

    # Feature Importances Extraction
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_rf = pipeline.named_steps["regressor"]
    cat_encoder = fitted_preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    feature_names = encoded_cat_names + num_features
    importances = fitted_rf.feature_importances_

    feature_imp_list = sorted(
        [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    # Summary metrics payload
    metrics = {
        "model_type": "RandomForestRegressor",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_samples": int(len(df)),
        "train_date_range": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
        "test_date_range": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
        "evaluation": {
            "rf_mae": round(rf_mae, 2),
            "rf_rmse": round(rf_rmse, 2),
            "rf_r2": round(rf_r2, 4),
            "baseline_mae": round(baseline_mae, 2),
            "baseline_rmse": round(baseline_rmse, 2),
            "baseline_r2": round(baseline_r2, 4),
            "mae_improvement_pct": round(mae_improvement_pct, 2)
        },
        "feature_importances": feature_imp_list[:12],
        "is_synthetic": True,
        "synthetic_disclaimer": "The model was trained on a calibrated synthetic historical dataset with realistic seasonal distributions."
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model successfully saved to {MODEL_FILE}")
    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Random Forest MAE: {rf_mae:.2f} units | Baseline MAE: {baseline_mae:.2f} units")
    print(f"Accuracy Improvement over Baseline: {mae_improvement_pct:.1f}%")

    # Generate and refresh predictions in DB
    try:
        from predict import generate_and_save_predictions
        generate_and_save_predictions(pipeline, metrics["evaluation"]["rf_mae"])
    except Exception as e:
        print(f"Note: Prediction refresh will occur once predict.py is available: {e}")

    return pipeline, metrics

if __name__ == "__main__":
    train_and_evaluate_model()
