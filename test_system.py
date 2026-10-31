"""
NeedMatch AI - Automated Business Logic & System Test Suite
Verifies all 14 core requirements across ML pipeline, database integrity,
inference, recommendations, and receipt workflows.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np

import database as db
import predict
from recommendation_engine import (
    recommend_ngos_for_item,
    recommend_purchase_plan,
    calculate_unmet_need,
    compute_priority_score
)
from train_model import train_and_evaluate_model
from data_generator import CATEGORIES, ORGANIZATIONS

def run_all_tests():
    print("==================================================")
    print("      NEEDMATCH AI — COMPREHENSIVE TEST SUITE     ")
    print("==================================================")
    passed_tests = 0
    total_tests = 16

    # ---------------- TEST 1: Dataset Generation & Coverage ----------------
    print("\n[TEST 1] Verifying Historical Demand Dataset...")
    assert os.path.exists("data/historical_demand.csv"), "historical_demand.csv missing!"
    df = pd.read_csv("data/historical_demand.csv")
    assert len(df) >= 1000, f"Expected at least 1000 rows, found {len(df)}"
    cats_in_df = set(df["category"].unique())
    assert cats_in_df == set(CATEGORIES), f"Category mismatch: {cats_in_df} vs {set(CATEGORIES)}"
    orgs_in_df = set(df["org_id"].unique())
    expected_orgs = set([o["id"] for o in ORGANIZATIONS])
    assert orgs_in_df == expected_orgs, f"Org mismatch: {orgs_in_df} vs {expected_orgs}"
    print(f"-> PASS: Dataset contains {len(df)} records covering all 6 categories and 6 NGOs.")
    passed_tests += 1

    # ---------------- TEST 2: Model Training & Saving ----------------
    print("\n[TEST 2] Verifying Model Training Pipeline...")
    pipeline, metrics = train_and_evaluate_model()
    assert os.path.exists("models/rf_demand_model.joblib"), "Model artifact not saved!"
    assert os.path.exists("models/metrics.json"), "Metrics file not saved!"
    print("-> PASS: Pipeline successfully trained and persisted with Joblib.")
    passed_tests += 1

    # ---------------- TEST 3: Model Loading ----------------
    print("\n[TEST 3] Verifying Model Artifact Loading...")
    loaded_pipe = predict.load_trained_model()
    assert loaded_pipe is not None, "Failed to load trained pipeline!"
    print("-> PASS: Model artifact successfully loaded into memory.")
    passed_tests += 1

    # ---------------- TEST 4: Actual Inference Execution ----------------
    print("\n[TEST 4] Verifying Machine Learning Inference...")
    pred_res = predict.predict_single(loaded_pipe, "ORG001", "Notebooks", horizon_days=30)
    pred_val = pred_res["predicted_demand_30d"]
    assert isinstance(pred_val, float) and pred_val > 0, f"Invalid prediction output: {pred_val}"
    print(f"-> PASS: Model generated genuine demand prediction: {pred_val} units (ORG001 - Notebooks).")
    passed_tests += 1

    # ---------------- TEST 5: Baseline vs Model MAE ----------------
    print("\n[TEST 5] Verifying Baseline Comparison & Error Reduction...")
    rf_mae = metrics["evaluation"]["rf_mae"]
    baseline_mae = metrics["evaluation"]["baseline_mae"]
    assert rf_mae < baseline_mae, f"RF MAE ({rf_mae}) should be strictly lower than baseline ({baseline_mae})"
    improvement = metrics["evaluation"]["mae_improvement_pct"]
    assert improvement > 20.0, f"Expected significant improvement, got {improvement}%"
    print(f"-> PASS: Random Forest MAE ({rf_mae:.2f}) beats Baseline MAE ({baseline_mae:.2f}) by +{improvement:.1f}%.")
    passed_tests += 1

    # ---------------- TEST 6: AI Recommendation Logic ----------------
    print("\n[TEST 6] Verifying Existing-Item Recommendation Engine...")
    recs = recommend_ngos_for_item("Schoolbags", available_qty=20, condition="Good")
    assert len(recs) == len(ORGANIZATIONS), "Expected recommendations for all NGOs"
    top_rec = recs[0]
    assert "priority_score" in top_rec and top_rec["priority_score"] > 0
    assert "reason" in top_rec and len(top_rec["reason"]) > 10
    print(f"-> PASS: Generated recommendations for Schoolbags. Top NGO: {top_rec['org_name']} (Score: {top_rec['priority_score']}).")
    passed_tests += 1

    # ---------------- TEST 7: Budget Constraints in Purchase Plan ----------------
    print("\n[TEST 7] Verifying Budget Purchase Planner Feasibility...")
    test_budget = 6000.0
    plan = recommend_purchase_plan(budget_inr=test_budget, selected_categories=["Notebooks", "Schoolbags", "Books"])
    assert plan["total_cost"] <= test_budget, f"Plan total cost ({plan['total_cost']}) exceeded budget ({test_budget})!"
    assert plan["remaining_budget"] >= 0, "Remaining budget cannot be negative!"
    assert len(plan["items"]) > 0, "Expected allocated items in purchase plan"
    print(f"-> PASS: Plan allocated INR {plan['total_cost']:.2f} within INR {test_budget:.2f} budget.")
    passed_tests += 1

    # ---------------- TEST 8: Existing-Item Donation Submission ----------------
    print("\n[TEST 8] Verifying Donation Submission & 'Pending Receipt' State...")
    inv_before = db.get_inventory("ORG001")
    inv_books_before = next((i["quantity"] for i in inv_before if i["category"] == "Books"), 0)

    don_id = db.create_donation(
        donation_type="EXISTING_ITEMS",
        donor_name="Test Donor",
        donor_email="test.donor@example.com",
        donor_phone="9999999999",
        org_id="ORG001",
        category="Books",
        quantity=15,
        condition="Good",
        notes="Automated test donation"
    )
    assert don_id > 0, "Donation submission failed"

    # Verify inventory was NOT altered upon submission
    inv_after = db.get_inventory("ORG001")
    inv_books_after = next((i["quantity"] for i in inv_after if i["category"] == "Books"), 0)
    assert inv_books_before == inv_books_after, "Inventory should NOT increase upon donation submission!"

    # Verify status in database
    conn = db.get_connection()
    don_row = conn.execute("SELECT * FROM donations WHERE id = ?", (don_id,)).fetchone()
    conn.close()
    assert don_row["status"] == "Pending Receipt", f"Expected 'Pending Receipt', found {don_row['status']}"
    print(f"-> PASS: Donation #{don_id} created in 'Pending Receipt' state without incrementing physical inventory.")
    passed_tests += 1

    # ---------------- TEST 9: NGO Request Creation & Closure ----------------
    print("\n[TEST 9] Verifying NGO Request Creation & Lifecycle...")
    req_id = db.create_donation_request("ORG002", "Blankets", requested_qty=50, urgency="High", notes="Test request")
    assert req_id > 0, "Request creation failed"
    db.close_donation_request(req_id)
    conn = db.get_connection()
    req_row = conn.execute("SELECT status FROM donation_requests WHERE id = ?", (req_id,)).fetchone()
    conn.close()
    assert req_row["status"] == "CLOSED", f"Expected CLOSED, got {req_row['status']}"
    print(f"-> PASS: Request #{req_id} created and closed cleanly in SQLite.")
    passed_tests += 1

    # ---------------- TEST 10: Receipt Confirmation & Atomic Stock Increment ----------------
    print("\n[TEST 10] Verifying Receipt Confirmation & Stock Increment...")
    conf_res = db.confirm_donation_receipt(don_id, confirming_org_id="ORG001")
    assert conf_res["success"], f"Confirmation failed: {conf_res['message']}"

    inv_confirmed = db.get_inventory("ORG001")
    inv_books_confirmed = next((i["quantity"] for i in inv_confirmed if i["category"] == "Books"), 0)
    assert inv_books_confirmed == inv_books_before + 15, f"Inventory expected {inv_books_before + 15}, got {inv_books_confirmed}"
    print(f"-> PASS: Donation #{don_id} confirmed as Received. Warehouse stock atomically incremented by +15 units.")
    passed_tests += 1

    # ---------------- TEST 11: Duplicate Confirmation Prevention ----------------
    print("\n[TEST 11] Verifying Duplicate Receipt Prevention...")
    dup_res = db.confirm_donation_receipt(don_id, confirming_org_id="ORG001")
    assert not dup_res["success"], "Duplicate confirmation should have been rejected!"
    inv_post_dup = db.get_inventory("ORG001")
    inv_books_post_dup = next((i["quantity"] for i in inv_post_dup if i["category"] == "Books"), 0)
    assert inv_books_post_dup == inv_books_confirmed, "Duplicate confirmation illegally incremented inventory!"
    print("-> PASS: Duplicate confirmation was blocked. Inventory was NOT double-counted.")
    passed_tests += 1

    # ---------------- TEST 12: Cross-NGO Confirmation Prevention ----------------
    print("\n[TEST 12] Verifying Cross-NGO Confirmation Security...")
    # Create another donation addressed to ORG003
    don_org3 = db.create_donation(
        donation_type="EXISTING_ITEMS",
        donor_name="Security Test",
        donor_email="sec@example.com",
        donor_phone="1234567890",
        org_id="ORG003",
        category="Hygiene Products",
        quantity=10
    )
    # Attempt confirmation by ORG001 (unauthorized)
    cross_res = db.confirm_donation_receipt(don_org3, confirming_org_id="ORG001")
    assert not cross_res["success"], "Cross-organization receipt confirmation must be blocked!"
    assert "Unauthorized" in cross_res["message"]
    print("-> PASS: Cross-NGO confirmation was successfully blocked.")
    passed_tests += 1

    # ---------------- TEST 13: Purchase Plans Table & Separation ----------------
    print("\n[TEST 13] Verifying Dedicated Purchase Plans Storage...")
    sample_plan_items = [{"category": "Notebooks", "quantity": 20, "unit_price": 45.0, "cost": 900.0}]
    plan_id = db.save_purchase_plan(
        donor_name="Plan Donor",
        donor_email="plan@example.com",
        total_budget_inr=2000.0,
        total_allocated_inr=900.0,
        remaining_budget_inr=1100.0,
        items=sample_plan_items
    )
    assert plan_id > 0, "Failed to save purchase plan!"
    all_plans = db.get_purchase_plans()
    assert any(p["id"] == plan_id for p in all_plans), "Saved plan not found in database!"
    print(f"-> PASS: Purchase Plan #{plan_id} stored in dedicated purchase_plans table.")
    passed_tests += 1

    # ---------------- TEST 14: Persistence Across Connection Re-open ----------------
    print("\n[TEST 14] Verifying SQLite Persistence...")
    # Open a completely fresh connection to test database
    fresh_conn = sqlite3.connect("data/needmatch.db")
    fresh_cur = fresh_conn.cursor()
    fresh_cur.execute("SELECT status FROM donations WHERE id = ?", (don_id,))
    persisted_status = fresh_cur.fetchone()[0]
    fresh_conn.close()
    assert persisted_status == "Received", f"Expected Received, got {persisted_status}"
    print("-> PASS: SQLite persistence verified across independent database connections.")
    passed_tests += 1

    # ---------------- TEST 15: Outstanding Backlog Reconciliation ----------------
    print("\n[TEST 15] Verifying Outstanding Backlog Reconciliation & Request Schema...")
    backlog_reqs = db.get_open_backlog_requests()
    stats = db.get_overall_stats()
    assert len(backlog_reqs) == stats["open_requests_count"], f"Mismatch in count: {len(backlog_reqs)} vs {stats['open_requests_count']}"
    calc_total_rem = sum(r["remaining_qty"] for r in backlog_reqs)
    assert calc_total_rem == stats["total_outstanding_requested_units"], f"Backlog mismatch: {calc_total_rem} != {stats['total_outstanding_requested_units']}"
    for r in backlog_reqs:
        assert r["org_name"], "Missing NGO name"
        assert r["category"], "Missing category"
        assert r["requested_qty"] > 0, "Requested quantity must be positive"
        assert r["fulfilled_qty"] >= 0, "Fulfilled quantity cannot be negative"
        assert r["remaining_qty"] == (r["requested_qty"] - r["fulfilled_qty"]), "Remaining calculation invalid"
        assert r["status"] in ("OPEN", "PARTIALLY_FULFILLED"), f"Invalid status: {r['status']}"
    print(f"-> PASS: Reconciled {len(backlog_reqs)} open requests totaling {calc_total_rem} units with 0 variance.")
    passed_tests += 1

    # ---------------- TEST 16: Complete End-to-End Workflow & IST Timestamps ----------------
    print("\n[TEST 16] Verifying 9-Step Complete Donation, Receipt & IST Timestamp Workflow...")
    wf_org = "ORG001"
    wf_cat = "Books"
    wf_qty = 30
    inv_start = {r["category"]: r["quantity"] for r in db.get_inventory(wf_org)}
    start_qty = inv_start.get(wf_cat, 0)

    # 1. Submit donation for NGO A
    wf_don_id = db.create_donation(
        donation_type="EXISTING_ITEMS",
        donor_name="Workflow Verifier",
        donor_email="workflow.verifier@test.org",
        donor_phone="9876543210",
        org_id=wf_org,
        category=wf_cat,
        quantity=wf_qty,
        condition="New",
        estimated_value_inr=3600.0,
        notes="Automated comprehensive workflow audit test"
    )
    assert wf_don_id > 0, "Donation creation failed!"

    # 2. Verify appears in Donation History
    hist_dons = db.get_donations()
    assert any(d["id"] == wf_don_id for d in hist_dons), "Donation missing from Donation History!"

    # 3. Verify appears in NGO A Incoming Donations
    ngo_inbound = db.get_donations(org_id=wf_org, status="Pending Receipt")
    assert any(d["id"] == wf_don_id for d in ngo_inbound), "Donation missing from NGO incoming donations!"

    # 4. Verify initial status is Pending Receipt
    pending_record = [d for d in ngo_inbound if d["id"] == wf_don_id][0]
    assert pending_record["status"] == "Pending Receipt"
    assert pending_record["confirmed_at"] is None

    # 5. Verify NGO A inventory has not increased yet
    inv_unconfirmed = {r["category"]: r["quantity"] for r in db.get_inventory(wf_org)}
    assert inv_unconfirmed.get(wf_cat, 0) == start_qty, "Inventory must NOT increase while Pending Receipt!"

    # 6. Confirm receipt from NGO Portal
    conf_res = db.confirm_donation_receipt(wf_don_id, confirming_org_id=wf_org)
    assert conf_res["success"] is True, f"Receipt confirmation failed: {conf_res}"

    # 7. Verify donation status changes to Received
    ngo_all_dons = db.get_donations(org_id=wf_org)
    received_record = [d for d in ngo_all_dons if d["id"] == wf_don_id][0]
    assert received_record["status"] == "Received"
    assert received_record["confirmed_at"] is not None

    # 8. Verify inventory increases by exactly the donated quantity
    inv_confirmed = {r["category"]: r["quantity"] for r in db.get_inventory(wf_org)}
    assert inv_confirmed.get(wf_cat, 0) == start_qty + wf_qty, f"Expected {start_qty + wf_qty}, got {inv_confirmed.get(wf_cat, 0)}"

    # 9. Verify timestamps are correct in IST everywhere (DD-MM-YYYY HH:MM:SS)
    sub_ist = db.format_datetime_ist(received_record["created_at"])
    rec_ist = db.format_datetime_ist(received_record["confirmed_at"])
    now_ist_obj = db.get_current_ist_time()
    today_prefix = now_ist_obj.strftime("%d-%m-%Y")
    assert sub_ist.startswith(today_prefix), f"Submitted At {sub_ist} does not match current IST date {today_prefix}"
    assert rec_ist.startswith(today_prefix), f"Receipt Confirmed At {rec_ist} does not match current IST date {today_prefix}"
    assert len(sub_ist) == 19 and sub_ist[2] == "-" and sub_ist[5] == "-" and sub_ist[13] == ":", f"Invalid format {sub_ist}"
    assert len(rec_ist) == 19 and rec_ist[2] == "-" and rec_ist[5] == "-" and rec_ist[13] == ":", f"Invalid format {rec_ist}"

    print(f"-> PASS: 9-step workflow verified (ID #{wf_don_id}, +{wf_qty} {wf_cat}, IST: {sub_ist} -> {rec_ist}).")
    passed_tests += 1

    print("\n==================================================")
    print(f"TEST RESULTS: {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY! (100%)")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
