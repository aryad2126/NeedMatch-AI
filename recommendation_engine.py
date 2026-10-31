"""
NeedMatch AI - Recommendation & Allocation Engine (Audited & Enhanced)
Combines ML demand predictions with live SQLite inventory, active requests,
urgency levels, and budget constraints to compute transparent unmet-need scores
and recommend optimal donation distributions.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import database as db
from predict import get_demand_forecast_table, load_trained_model, predict_single

URGENCY_WEIGHTS = {
    "High": 1.5,
    "Medium": 1.2,
    "Low": 1.0
}

def calculate_unmet_need(
    requested_qty: int,
    predicted_30d_demand: float,
    current_inventory: int,
    pending_donations: int = 0
) -> float:
    """
    Computes genuine unmet need avoiding double-counting:
    Unmet Need = max(0, (Requested_Qty + 0.6 * Predicted_Demand) - (Current_Inventory + Pending_Donations))

    Assumptions & Rationale:
    1. Requested_Qty represents immediate, outstanding backlogged requests.
    2. 0.6 * Predicted_Demand represents the expected 30-day requirement excluding backlogged items.
       Using a 0.6 factor prevents double-counting the immediate need twice.
    3. We subtract both Current_Inventory and Pending_Donations (already pledged by other donors)
       to avoid recommending items where supplies are already en route.
    """
    total_requirement = requested_qty + (0.6 * predicted_30d_demand)
    available_supply = current_inventory + pending_donations
    return max(0.0, round(total_requirement - available_supply, 1))

def compute_priority_score(
    unmet_need: float,
    urgency: str,
    current_inventory: int,
    predicted_demand: float
) -> float:
    """
    Computes an interpretable priority score from 0 to 100 based on:
    1. Scale of unmet need relative to predicted demand
    2. Urgency multiplier (High = 1.5, Medium = 1.2, Low = 1.0)
    3. Severe inventory deficit factor
    """
    if unmet_need <= 0:
        return 0.0

    urgency_mult = URGENCY_WEIGHTS.get(urgency, 1.0)
    deficit_ratio = unmet_need / max(1.0, current_inventory + 5.0)
    deficit_factor = min(2.0, max(0.5, deficit_ratio))

    raw_score = (unmet_need * 0.45 + predicted_demand * 0.25) * urgency_mult * deficit_factor
    normalized_score = min(99.0, max(15.0, raw_score / 3.2))
    return round(normalized_score, 1)

def recommend_ngos_for_item(
    category: str,
    available_qty: int = 1,
    condition: str = "Good",
    org_id_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Recommends and ranks NGOs for a donor donating existing items in a specific category.
    Returns sorted list with full context, scores, and dynamic explanations.
    """
    conn = db.get_connection()
    orgs = db.get_organizations()

    inv_rows = conn.execute(
        "SELECT org_id, quantity FROM inventory WHERE category = ?",
        (category,)
    ).fetchall()
    inv_map = {r["org_id"]: r["quantity"] for r in inv_rows}

    # Fetch pending donations
    pending_rows = conn.execute("""
        SELECT org_id, COALESCE(SUM(quantity), 0) as pending_qty
        FROM donations
        WHERE category = ? AND status IN ('Pending Receipt', 'PENDING')
        GROUP BY org_id
    """, (category,)).fetchall()
    pending_map = {r["org_id"]: r["pending_qty"] for r in pending_rows}

    # Fetch open requests
    req_rows = conn.execute("""
        SELECT org_id, requested_qty, fulfilled_qty, urgency, notes
        FROM donation_requests
        WHERE category = ? AND status IN ('OPEN', 'PARTIALLY_FULFILLED')
        ORDER BY CASE urgency WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
    """, (category,)).fetchall()
    req_map = {}
    for r in req_rows:
        oid = r["org_id"]
        backlog = r["requested_qty"] - r["fulfilled_qty"]
        if oid not in req_map:
            req_map[oid] = {"backlog": backlog, "urgency": r["urgency"], "notes": r["notes"]}
        else:
            req_map[oid]["backlog"] += backlog

    # Fetch ML predictions
    pred_rows = conn.execute(
        "SELECT org_id, predicted_demand, mae_error_margin FROM demand_predictions WHERE category = ?",
        (category,)
    ).fetchall()
    pred_map = {r["org_id"]: (r["predicted_demand"], r["mae_error_margin"]) for r in pred_rows}

    conn.close()

    candidates = []
    for org in orgs:
        oid = org["id"]
        if org_id_filter and org_id_filter != "ALL" and oid != org_id_filter:
            continue

        inv = inv_map.get(oid, 0)
        pending = pending_map.get(oid, 0)
        req_info = req_map.get(oid, {"backlog": 0, "urgency": "Low", "notes": ""})
        backlog = req_info["backlog"]
        urgency = req_info["urgency"]

        pred_info = pred_map.get(oid, (60.0, 15.0))
        predicted_demand = pred_info[0]
        mae_margin = pred_info[1]

        unmet_need = calculate_unmet_need(backlog, predicted_demand, inv, pending)
        score = compute_priority_score(unmet_need, urgency, inv, predicted_demand)

        # Suggested quantity
        if unmet_need > 0:
            suggested_qty = min(available_qty, int(round(unmet_need)))
            if available_qty <= unmet_need:
                suggested_qty = available_qty
        else:
            suggested_qty = 0

        # Rationale formulation
        if unmet_need > 0:
            pct_fulfillment = round((min(available_qty, unmet_need) / unmet_need) * 100) if unmet_need > 0 else 100
            reason = (
                f"High priority ({urgency} urgency): Has {backlog} units of open requests "
                f"and ML predicted 30-day demand of {predicted_demand:.0f} units against only "
                f"{inv} in inventory ({pending} pending). Your {min(available_qty, unmet_need)} units "
                f"fulfill ~{pct_fulfillment}% of their immediate deficit."
            )
        else:
            reason = (
                f"Low immediate need: Current inventory ({inv}) and pending shipments ({pending}) "
                f"sufficiently cover current requests ({backlog}). Donating here may cause inventory surplus."
            )

        candidates.append({
            "org_id": oid,
            "org_name": org["name"],
            "city": org["city"],
            "focus": org["focus"],
            "category": category,
            "current_inventory": inv,
            "pending_donations": pending,
            "requested_backlog": backlog,
            "urgency": urgency,
            "predicted_demand": round(predicted_demand, 1),
            "mae_margin": round(mae_margin, 1),
            "unmet_need": round(unmet_need, 1),
            "priority_score": score,
            "suggested_qty": suggested_qty,
            "reason": reason,
            "is_recommended": score >= 50.0 and unmet_need > 0
        })

    candidates.sort(key=lambda x: (x["priority_score"], x["unmet_need"]), reverse=True)
    return candidates

def recommend_purchase_plan(
    budget_inr: float,
    selected_categories: List[str],
    preferred_org_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Budget-constrained recommendation algorithm.
    Optimizes allocation of donor's INR budget across highest unmet needs.
    Distinguishes:
    - ML demand prediction
    - Recommendation ranking logic
    - Feasibility calculation (Unit price x Quantity <= Budget)
    """
    if budget_inr <= 0:
        return {"items": [], "total_cost": 0.0, "remaining_budget": budget_inr, "message": "Budget must be greater than INR 0."}

    prices = db.get_item_prices()
    if not selected_categories:
        selected_categories = list(prices.keys())

    opportunities = []
    for cat in selected_categories:
        if cat not in prices:
            continue
        unit_price = prices[cat]["unit_price_inr"]
        org_recs = recommend_ngos_for_item(cat, available_qty=500)

        for rec in org_recs:
            if preferred_org_id and preferred_org_id != "ALL" and rec["org_id"] != preferred_org_id:
                continue
            if rec["unmet_need"] > 0:
                cost_effectiveness = (rec["priority_score"] * 10.0) / unit_price
                opportunities.append({
                    "category": cat,
                    "unit_price": unit_price,
                    "unit_name": prices[cat]["unit_name"],
                    "org_id": rec["org_id"],
                    "org_name": rec["org_name"],
                    "city": rec["city"],
                    "urgency": rec["urgency"],
                    "unmet_need": rec["unmet_need"],
                    "priority_score": rec["priority_score"],
                    "cost_effectiveness": cost_effectiveness,
                    "predicted_demand": rec["predicted_demand"],
                    "current_inventory": rec["current_inventory"],
                    "reason": rec["reason"]
                })

    if not opportunities:
        return {
            "items": [],
            "total_cost": 0.0,
            "remaining_budget": budget_inr,
            "message": "No critical unmet needs currently found for the selected categories."
        }

    # Sort opportunities by cost effectiveness and priority score
    opportunities.sort(key=lambda x: (x["cost_effectiveness"], x["priority_score"]), reverse=True)

    remaining_budget = float(budget_inr)
    allocated_plan = []

    for opp in opportunities:
        if remaining_budget < opp["unit_price"]:
            continue

        max_affordable = int(remaining_budget // opp["unit_price"])
        if max_affordable <= 0:
            continue

        target_qty = min(int(opp["unmet_need"]), max_affordable)
        if len(selected_categories) > 1 and target_qty > 25 and max_affordable > 30:
            target_qty = min(target_qty, max(10, int(opp["unmet_need"] * 0.6)))

        if target_qty <= 0:
            target_qty = 1

        cost = target_qty * opp["unit_price"]
        if cost > remaining_budget:
            target_qty = int(remaining_budget // opp["unit_price"])
            cost = target_qty * opp["unit_price"]

        if target_qty > 0:
            remaining_budget -= cost
            allocated_plan.append({
                "category": opp["category"],
                "unit_name": opp["unit_name"],
                "unit_price": opp["unit_price"],
                "recommended_qty": target_qty,
                "total_cost": cost,
                "org_id": opp["org_id"],
                "org_name": opp["org_name"],
                "city": opp["city"],
                "urgency": opp["urgency"],
                "unmet_need": opp["unmet_need"],
                "priority_score": opp["priority_score"],
                "predicted_demand": opp["predicted_demand"],
                "current_inventory": opp["current_inventory"],
                "reason": (
                    f"Allocated {target_qty} {opp['unit_name']}(s) to {opp['org_name']} "
                    f"based on {opp['urgency']} urgency, INR {opp['unit_price']:.0f}/unit price, "
                    f"and an unmet need of {opp['unmet_need']:.0f} units."
                )
            })

    total_cost = sum(item["total_cost"] for item in allocated_plan)

    return {
        "items": allocated_plan,
        "total_cost": round(total_cost, 2),
        "remaining_budget": round(remaining_budget, 2),
        "initial_budget": budget_inr,
        "selected_categories": selected_categories
    }

if __name__ == "__main__":
    recs = recommend_ngos_for_item("Notebooks", 50)
    print(f"Top Recommendation for Notebooks: {recs[0]['org_name']} - Score: {recs[0]['priority_score']}")
