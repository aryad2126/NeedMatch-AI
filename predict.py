"""
NeedMatch AI - Predictive Demand Inference Engine (Audited & Enhanced)
Loads the persisted Scikit-learn RandomForestRegressor pipeline and computes
30-day future demand predictions based on live conditions and verified historical data.
"""

import os
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

import database as db
from data_generator import CATEGORIES, ORGANIZATIONS, get_seasonal_multiplier, BASE_DEMAND

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "historical_demand.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "rf_demand_model.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")

# Cache recent historical baseline stats for high-performance inference
_HISTORICAL_CACHE = None

def get_recent_history_lookup() -> Dict[tuple, Dict[str, float]]:
    """
    Computes recent historical demand and fulfillment averages for each (org_id, category)
    from the historical dataset to ensure feature consistency during inference.
    """
    global _HISTORICAL_CACHE
    if _HISTORICAL_CACHE is not None:
        return _HISTORICAL_CACHE

    lookup = {}
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Take average of last 3 recorded months per (org_id, category)
            df_sorted = df.sort_values(by="date")
            grouped = df_sorted.groupby(["org_id", "category"]).tail(3)
            agg = grouped.groupby(["org_id", "category"])[["historical_demand_qty", "fulfilled_qty"]].mean()
            for (oid, cat), row in agg.iterrows():
                lookup[(oid, cat)] = {
                    "hist_demand": float(row["historical_demand_qty"]),
                    "fulfilled": float(row["fulfilled_qty"])
                }
        except Exception:
            pass

    _HISTORICAL_CACHE = lookup
    return lookup

def load_trained_model():
    """Loads the persisted Scikit-learn Pipeline or trains if not found."""
    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found at {MODEL_FILE}. Training model now...")
        from train_model import train_and_evaluate_model
        pipeline, _ = train_and_evaluate_model()
        return pipeline
    return joblib.load(MODEL_FILE)

def load_metrics() -> Dict[str, Any]:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {}

def predict_single(
    pipeline,
    org_id: str,
    category: str,
    month: Optional[int] = None,
    horizon_days: int = 30
) -> Dict[str, Any]:
    """
    Generates a demand prediction for an organization and category using the trained model.
    Feature schema strictly matches the training pipeline:
    - category (str)
    - org_id (str)
    - month (int)
    - historical_demand_qty (float)
    - fulfilled_qty (float)
    - inventory_level (int)
    - outstanding_requests (int)
    - season_factor (float)
    """
    if month is None:
        month = datetime.now().month

    # Retrieve live current inventory from SQLite
    conn = db.get_connection()
    inv_row = conn.execute(
        "SELECT quantity FROM inventory WHERE org_id = ? AND category = ?",
        (org_id, category)
    ).fetchone()
    current_inv = inv_row["quantity"] if inv_row else 0

    # Retrieve live open requests backlog from SQLite
    req_row = conn.execute("""
        SELECT COALESCE(SUM(requested_qty - fulfilled_qty), 0) as backlog
        FROM donation_requests
        WHERE org_id = ? AND category = ? AND status IN ('OPEN', 'PARTIALLY_FULFILLED')
    """, (org_id, category)).fetchone()
    backlog = req_row["backlog"] if req_row else 0

    conn.close()

    # Domain & historical features
    season_factor = get_seasonal_multiplier(category, month)
    hist_lookup = get_recent_history_lookup()
    cached = hist_lookup.get((org_id, category))

    if cached:
        hist_demand = cached["hist_demand"]
        fulfilled = cached["fulfilled"]
    else:
        base = BASE_DEMAND.get(category, 100)
        hist_demand = max(15.0, base * season_factor)
        fulfilled = max(10.0, hist_demand * 0.75)

    input_df = pd.DataFrame([{
        "category": category,
        "org_id": org_id,
        "month": int(month),
        "historical_demand_qty": float(hist_demand),
        "fulfilled_qty": float(fulfilled),
        "inventory_level": int(current_inv),
        "outstanding_requests": int(backlog),
        "season_factor": float(season_factor)
    }])

    raw_pred_30d = pipeline.predict(input_df)[0]
    pred_30d = max(5.0, round(float(raw_pred_30d), 1))

    # Horizon handling
    if horizon_days == 30:
        scaled_pred = pred_30d
        horizon_note = "Native 30-Day ML Target Prediction"
    else:
        # Transparent proportional scaling with explicit disclosure
        ratio = max(0.1, horizon_days / 30.0)
        scaled_pred = max(2.0, round(pred_30d * ratio, 1))
        horizon_note = f"Proportionally scaled ({horizon_days}d) from native 30-day model"

    return {
        "org_id": org_id,
        "category": category,
        "predicted_demand_30d": pred_30d,
        "predicted_demand_selected_horizon": scaled_pred,
        "horizon_days": horizon_days,
        "horizon_note": horizon_note,
        "current_inventory": current_inv,
        "outstanding_requests": backlog,
        "input_features": {
            "month": month,
            "season_factor": season_factor,
            "historical_demand_qty": round(hist_demand, 1),
            "fulfilled_qty": round(fulfilled, 1),
            "inventory_level": current_inv,
            "outstanding_requests": backlog
        }
    }

def generate_and_save_predictions(pipeline=None, error_margin: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Infers 30-day demand for all 36 NGO-category combinations and updates SQLite.
    """
    if pipeline is None:
        pipeline = load_trained_model()

    if error_margin is None:
        metrics = load_metrics()
        error_margin = metrics.get("evaluation", {}).get("rf_mae", 15.0)

    current_month = datetime.now().month
    records = []

    for org in ORGANIZATIONS:
        for cat in CATEGORIES:
            res = predict_single(pipeline, org["id"], cat, current_month, horizon_days=30)
            records.append({
                "org_id": org["id"],
                "category": cat,
                "predicted_demand": res["predicted_demand_30d"],
                "mae_error_margin": error_margin
            })

    db.save_predictions(records)
    print(f"Generated and persisted {len(records)} demand predictions to SQLite.")
    return records

def get_demand_forecast_table() -> pd.DataFrame:
    """
    Retrieves full forecast table enriched with current inventory,
    open requests, and net projected gap.
    """
    preds = db.get_predictions_from_db()
    if not preds:
        generate_and_save_predictions()
        preds = db.get_predictions_from_db()

    inv_list = db.get_inventory()
    inv_map = {(r["org_id"], r["category"]): r["quantity"] for r in inv_list}

    req_list = db.get_donation_requests(status=None)
    req_map = {}
    for r in req_list:
        if r["status"] in ["OPEN", "PARTIALLY_FULFILLED"]:
            key = (r["org_id"], r["category"])
            backlog = r["requested_qty"] - r["fulfilled_qty"]
            req_map[key] = req_map.get(key, 0) + backlog

    enriched = []
    for p in preds:
        key = (p["org_id"], p["category"])
        inv = inv_map.get(key, 0)
        backlog = req_map.get(key, 0)
        pred = p["predicted_demand"]
        # Estimated Shortage represents predicted demand minus current inventory: max(0, predicted_demand - current_inventory)
        shortfall = max(0.0, round(pred - inv, 1))

        enriched.append({
            "org_id": p["org_id"],
            "org_name": p["org_name"],
            "city": p["city"],
            "category": p["category"],
            "current_inventory": inv,
            "open_requests": backlog,
            "predicted_30d_demand": round(pred, 1),
            "mae_margin": p.get("mae_error_margin", 0),
            "projected_shortfall": round(shortfall, 1),
            "prediction_date": p["prediction_date"]
        })

    return pd.DataFrame(enriched)

if __name__ == "__main__":
    p = load_trained_model()
    res = predict_single(p, "ORG001", "Notebooks", horizon_days=30)
    print("Inference Test:")
    print(f"Category: {res['category']} | Predicted 30d: {res['predicted_demand_30d']} units")
    print(f"Input features used: {res['input_features']}")
