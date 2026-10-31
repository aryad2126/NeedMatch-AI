# NeedMatch AI — Complete Project Audit Report

**Date of Audit:** October 28, 2026  
**Auditor:** Senior AI/ML Full-Stack Engineer  
**Project:** NeedMatch AI — An AI-Based Predictive Donation Recommendation and Allocation System  

---

## 1. Executive Summary

An in-depth, code-level audit was conducted across all files, database tables, machine learning models, inference scripts, recommendation logic, and frontend Streamlit views in the `NeedMatch AI` workspace.

### Core Verdict
- **Genuine ML Pipeline Exists:** The project does **not** use hardcoded predictions, random numbers, or pure rule-based mocks. It uses a trained Scikit-learn `RandomForestRegressor` with a strict chronological 80/20 train/test split.
- **Model Accuracy Verified:** Test MAE is **61.09 units** vs. Baseline MAE of **103.30 units** (40.86% error reduction) evaluated on identical held-out test data (April–October 2026).
- **Persistent SQLite Storage Exists:** Database tables exist with atomic transactions and inventory increment logic.
- **Areas Requiring Structural & Functional Repair:**
  1. **Navigation Organization:** The previous UI had 7 navigation items where "Demand Forecast" was merged with Analytics, and "AI Recommendations" was embedded inside donation forms rather than being standalone pages. The application must be reorganized into the exact **8 requested pages**.
  2. **Feature Consistency in Inference:** In `predict.py`, recent historical demand was estimated rather than queried directly from the historical dataset for that specific NGO and category.
  3. **Purchase Plans Separation:** Saved purchase plans were previously stored in the `donations` table under `donation_type='PURCHASE_PLEDGE'`. They must be cleanly stored in a dedicated `purchase_plans` table to avoid conflation with physically received donations.
  4. **Cross-NGO Confirmation Guard:** `confirm_donation_receipt()` lacked a check ensuring an NGO cannot confirm donations intended for a different NGO.
  5. **Beginner Transparency:** Every page must clearly answer the 3 questions: *What is this page for?*, *What does the user enter?*, and *What happens when the main button is clicked?*

---

## 2. Comprehensive Requirements Audit Table

| Requirement | Status | Evidence in Codebase | Required Action |
|---|---|---|---|
| **Historical Dataset** | Verified | `data/historical_demand.csv` (1,224 rows, 34 months, all 6 categories, 6 NGOs, realistic seasonality, seed 42). | Document synthetic origin clearly on UI and guide. |
| **ML Model Training** | Verified | `train_model.py` fits Scikit-learn `Pipeline([OneHotEncoder, RandomForestRegressor])`. | Keep working code; verify features during retraining. |
| **Chronological Train/Test Split** | Verified | `train_model.py` lines 56–68 (first 80% dates for train: Jan 2024–Mar 2026; last 20% for test: Apr 2026–Oct 2026). No leakage. | Preserve strictly. |
| **Model Evaluation** | Verified | `train_model.py` lines 113–122 (RF MAE: 61.09, Baseline MAE: 103.30 on identical test set, saved in `models/metrics.json`). | Display prominently on the Model Info page. |
| **Demand Prediction** | Verified (Needs Improvement) | `predict.py` loads `rf_demand_model.joblib` and calls `.predict()`. However, `hist_demand` was estimated instead of pulled from recent dataset history. | Refactor `predict_single()` to pull recent rolling demand from dataset for exact feature consistency. |
| **Recommendation Engine** | Verified | `recommendation_engine.py` implements `calculate_unmet_need()` and `compute_priority_score()`. | Create dedicated **AI Recommendations** page (#3) separate from raw demand forecast and donation forms. |
| **SQLite Persistence** | Verified | `database.py` manages tables in `data/needmatch.db`. | Add dedicated `purchase_plans` table and cross-NGO confirmation validation. |
| **Existing-Item Donation** | Verified | `database.create_donation()` sets status to pending receipt. Inventory is not incremented at submission. | Standardize status label to `Pending Receipt` and integrate on dedicated Page 4. |
| **Budget-Based Purchase Plan** | Partial | Knapsack allocation worked, but pledges were saved into `donations` table. | Store confirmed plans in a dedicated `purchase_plans` table and separate in Donation History. |
| **NGO Requests & Inventory** | Verified | `database.create_donation_request()` and `update_inventory()` persist in SQLite. | Add request update/close buttons and NGO stock management on Page 6. |
| **Donation Receipt Confirmation** | Verified (Needs Guard) | `database.confirm_donation_receipt()` uses `BEGIN IMMEDIATE TRANSACTION;` to update status to confirmed and increment stock once. | Add `confirming_org_id` validation to prevent an NGO from confirming another NGO's shipments. |
| **Dashboard Accuracy** | Verified | `app.py` queries live counts from SQLite and predictions from DB. No fake hardcoded numbers. | Reorganize according to Page 1 specifications. |
| **Application Navigation** | Non-Compliant | Had 7 mixed navigation choices rather than the clean 8-page sequential layout. | Rebuild navigation into the exact 8 specified pages. |
| **Beginner Documentation** | Missing | No `PROJECT_GUIDE.md` existed with student walkthrough and code mapping. | Create comprehensive `PROJECT_GUIDE.md`. |
| **Automated Test Suite** | Missing | Manual terminal tests existed, but no unified automated test script. | Create `test_system.py` verifying all 15 business logic assertions. |

---

## 3. Detailed Audit of Potential ML Pitfalls

1. **Target Leakage:** Checked `train_model.py`. The target `target_next_30d_demand` represents the *subsequent* month's demand. Features use *current month* values (`historical_demand_qty`, `fulfilled_qty`, `inventory_level`, `outstanding_requests`). Target is not present in $X_{train}$ or $X_{test}$. **No leakage detected.**
2. **Train/Test Splitting:** Checked split methodology. A date-based chronological cutoff (`split_date = 2026-04-01`) is used. No future dates exist in the training partition. **Chronological integrity verified.**
3. **Baseline Comparison:** Both Baseline and Random Forest were evaluated on the exact same 252 held-out test rows. The baseline uses the training group-mean demand per `(category, org_id)`. **Valid and fair comparison.**
4. **Feature Consistency:** In `predict.py`, `hist_demand` was previously approximated using `base * season_factor`. To guarantee 100% fidelity to the training distribution, `predict.py` should query the actual average of the last 3 historical months from `data/historical_demand.csv`.

---

## 4. Planned Engineering Repairs

1. **`database.py` Enhancements:**
   - Add `purchase_plans` table: `id`, `donor_name`, `donor_email`, `total_budget_inr`, `total_allocated_inr`, `remaining_budget_inr`, `items_json`, `created_at`.
   - Update `confirm_donation_receipt(donation_id, confirming_org_id)` with cross-organization ownership check.
   - Support standard status labels: `Pending Receipt`, `Received`, `Rejected`.
   - Add functions: `save_purchase_plan()`, `get_purchase_plans()`, `update_donation_request()`, `close_donation_request()`.
2. **`predict.py` Improvements:**
   - Enhance `predict_single()` to look up actual recent historical records for that `(org_id, category)`.
   - Explicitly handle and explain forecast horizon (30 days native target).
3. **`recommendation_engine.py` Refinements:**
   - Explicitly document mathematical assumptions of the unmet need formula.
   - Cleanly format recommendation responses for the new dedicated Page 3.
4. **`app.py` Reorganization into Exactly 8 Pages:**
   - **Page 1:** Home / Dashboard
   - **Page 2:** Demand Forecast
   - **Page 3:** AI Recommendations
   - **Page 4:** Donate Existing Items
   - **Page 5:** Plan a Purchase
   - **Page 6:** NGO Portal
   - **Page 7:** Donation History
   - **Page 8:** Model & Data Information
   - Every page will display clear student-friendly headers answering:
     - *1. What is this page for?*
     - *2. What should you enter or select?*
     - *3. What happens when you click the main button?*
5. **Testing & Documentation:**
   - Create and run `test_system.py`.
   - Create `PROJECT_GUIDE.md` with beginner explanations, architectural diagram, and demo walkthrough.
