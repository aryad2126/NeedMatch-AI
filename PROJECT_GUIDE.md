# NeedMatch AI — Beginner & Student Project Guide

Welcome to **NeedMatch AI**! This guide is written specifically for students and beginners in full-stack AI/ML development. It explains every concept, file, button, and user interaction in plain, easy-to-understand language.

---

## 1. What NeedMatch AI Does

In real-world charity work, donations often fail to help because of **information gaps**:
- One NGO might receive 500 blankets it doesn't need, while another organization in the next district has zero notebooks for school children.
- Donors want to help, but don't know *which* items are urgently needed, *where* to send them, or what to buy within their budget.

**NeedMatch AI solves this by connecting three layers:**
1. **Machine Learning Model (Random Forest):** Analyzes historical demand patterns, seasonal trends (e.g. school reopening in May–July, winter cold waves in Nov–Jan), and predicts what each NGO will need over the **next 30 days**.
2. **AI Recommendation Engine:** Compares the ML predictions against what the NGO already has in its warehouse and open requests, then mathematically calculates the **net unmet need** to rank the best recipient organizations.
3. **Operational Database (SQLite):** Tracks donations from submission to delivery. When a donor gives items, the donation is marked as **Pending Receipt**. Only when the NGO physically receives the package does the inventory increase.

---

## 2. Page-by-Page Overview

The application is structured into **7 clearly organized pages**:

| Page # | Page Name | Primary Purpose | User Inputs | Main Button |
|---|---|---|---|---|
| **1** | **🏠 Home / Dashboard** | High-level system overview & key metrics | View stats & charts | `Donate Items I Own` & `Plan a Purchase` (Navigation) |
| **2** | **🔮 Demand Forecast** | Run real ML inference to see future demand | NGO dropdown, Category dropdown, Horizon (30d) | **`🔮 Generate Demand Forecast`** |
| **3** | **📦 Donate Existing Items** | AI-ranked allocation & donation submission for pre-owned items | Category, Qty, Condition, Contact, Recipient NGO | **`🎯 Generate AI Recommendations`** & **`📤 Submit Donation Request`** |
| **4** | **💳 Plan a Purchase** | Budget planner optimizing new supply purchases | INR Budget, Categories, Contact info | **`💡 Generate Purchase Plan`** & **`🤝 Confirm Purchase Plan`** |
| **5** | **🏢 NGO Portal** | NGO manages inventory, requests, and receipts | NGO Switcher, New Request form, Receipt buttons | **`✅ Confirm Receipt`** |
| **6** | **📜 Donation History** | Auditable ledger of transactions & saved purchase plans | Filters (NGO, Category, Status) | **`📥 Export to CSV`** & **`🎁 Donate Items from This Plan`** |
| **7** | **🧠 Model & Data Info** | Academic transparency & model card | View dataset stats & metrics | **`⚡ Train / Retrain Model Pipeline`** |

---

## 3. Frequently Asked Questions (Beginner-Friendly Answers)

### Q4: Which button do I click to get a prediction?
Go to **Page 2: 🔮 Demand Forecast**, choose an organization and category, and click the **`🔮 Generate Demand Forecast`** button.

### Q5: How do I get AI donation recommendations?
Go to **Page 3: 📦 Donate Existing Items**, enter your item details, and click the **`🎯 Generate AI Recommendations`** button to view ranked recipient shelters with live unmet need scores and justifications.

### Q6: How does the existing-item donation workflow work?
1. The donor visits **Page 3: 📦 Donate Existing Items**.
2. They select a category (e.g., `Notebooks`), enter quantity `30`, and choose condition `Good`.
3. They click **`🎯 Generate AI Recommendations`** to run real-time shortage and ML demand scoring.
4. The page displays ranked recipient NGOs with shortage metrics and justifications.
5. The donor selects their target NGO (or selects manually) and clicks **`📤 Submit Donation Request`**.
6. The system records the donation with status **`Pending Receipt`**.
7. **Important:** The NGO's inventory does **not** increase yet. This prevents fake or phantom inventory from being counted.

### Q7: How does the purchase-plan workflow work?
1. The donor visits **Page 4: 💳 Plan a Purchase**.
2. They enter a budget in Indian Rupees (e.g. `₹8,000`).
3. They click **`💡 Generate Purchase Plan`**.
4. The system calculates how many notebooks, books, schoolbags, etc., to buy based on wholesale prices and greatest shortages.
5. The donor can tweak quantities using the interactive steppers. The system guarantees total cost never exceeds the budget.
6. The donor clicks **`🤝 Confirm Purchase Plan`** to save the plan in SQLite.

### Q8: How does an NGO create a request?
1. Open **Page 6: 🏢 NGO Portal**.
2. Select the organization from the dropdown.
3. Click the **Section B: Manage Requests** tab.
4. Fill in the category, quantity needed, urgency tier (`High`, `Medium`, or `Low`), and click **`Create Request in Database`**.

### Q9: How does an NGO confirm a donation?
1. Open **Page 6: 🏢 NGO Portal**.
2. Select the recipient NGO.
3. Click the **Section C: Incoming Donations** tab.
4. Find the pending shipment and click **`✅ Confirm Receipt for Donation #ID`**.

### Q10: Exactly when does inventory change?
Inventory changes **ONLY** when an authorized NGO user clicks **`✅ Confirm Receipt`** in the NGO Portal, or when an authorized staff member manually updates physical counts post-audit. Submitting a donation does **not** increase inventory.

### Q11: Where is the dataset stored?
At **`data/historical_demand.csv`**. It contains 1,224 rows covering 34 continuous months across all 6 categories and 6 partner NGOs.

### Q12: Which file trains the ML model?
**`train_model.py`**. It loads the CSV, splits data chronologically (first 80% train, last 20% test), trains a Scikit-learn `RandomForestRegressor`, evaluates MAE against a baseline, and saves the model to `models/rf_demand_model.joblib`.

### Q13: Which file performs predictions?
**`predict.py`**. It loads the saved model, extracts current features (month, season factor, recent demand, inventory, requests), and calls `pipeline.predict()`.

### Q14: Which file generates recommendations?
**`recommendation_engine.py`**. It calculates the net unmet need using:
$$\text{Unmet Need} = \max\Big(0, (\text{Backlog} + 0.6 \times \text{Predicted 30d Demand}) - (\text{Inventory} + \text{Pending})\Big)$$
and ranks NGOs by priority score.

### Q15: Which file handles database operations?
**`database.py`**. It manages the SQLite database at `data/needmatch.db` with atomic transactions.

### Q16: How does the dashboard obtain its values?
Every metric and chart on the **Home / Dashboard** queries live records from SQLite (`db.get_overall_stats()`) and live predictions (`predict.get_demand_forecast_table()`). There are **no hardcoded values**.

---

## 4. Step-by-Step Demonstration Walkthrough (For Academic Demo)

Follow these exact steps during your presentation to impress your evaluator:

### Step 1: Show System Health (1 minute)
1. Open your browser at `http://localhost:8501`.
2. Look at the sidebar: point out that **Database: Connected**, **ML Model: Ready (RF)**, and **Model Test MAE: 61.09 units**.
3. Point out the exact color palette (Warm cream background `#F9E9D1`, Deep teal headings `#0F4E6C`, Coral accents `#FF7F6E`, White cards `#FFFFFF`, Sky blue `#C8DFF2`).

### Step 2: Generate a Machine Learning Demand Forecast (2 minutes)
1. In the sidebar, click **`2. 🔮 Demand Forecast`**.
2. Select **Organization:** `Asha Education Foundation (Kolkata)`.
3. Select **Donation Category:** `Notebooks`.
4. Click **`🔮 Generate Demand Forecast`**.
5. **Explain to the evaluator:** *"Our trained Random Forest model predicts an upcoming demand of ~378 units for Notebooks based on recent demand history and seasonal academic reopening patterns. Against an inventory of only 40 units and an open backlog of 135 units, the net projected shortage is acute."*

### Step 3: Run AI Recommendations & Submit Existing Items Donation (3 minutes)
1. In the sidebar, click **`3. 📦 Donate Existing Items`**.
2. Select **Category:** `Notebooks`, **Quantity:** `50`, **Condition:** `Good`.
3. Click **`🎯 Generate AI Recommendations`**.
4. **Explain to the evaluator:** *"The integrated recommendation engine evaluates live shortages, open backlogs, and ML demand predictions, ranking organizations using a transparent 0–100 priority score."*
5. Select the recommended top NGO (e.g., `Vidya Jyoti Rural Mission`).
6. Enter donor name `Pooja Sharma` and click **`📤 Submit Donation Request`**.
7. Show the confirmation box: *"Notice that the donation status is **Pending Receipt**. The inventory has not changed yet, preventing phantom stock."*

### Step 4: Plan a Purchase with Budget (2 minutes)
1. In the sidebar, click **`4. 💳 Plan a Purchase`**.
2. Enter budget `₹8,000`, select categories, and click **`💡 Generate Purchase Plan`**.
3. Point out the impact-maximizing basket, wholesale unit prices, and click **`🤝 Confirm & Save Purchase Plan`**.
4. Show that donors can click **`🎁 Donate Items from This Plan`** to easily fulfill allocations.

### Step 5: Confirm Physical Receipt as the NGO (2 minutes)
1. In the sidebar, click **`5. 🏢 NGO Portal`**.
2. Select the recipient NGO you donated to.
3. Click the **Section C: Incoming Donations** tab.
4. Point out the pending shipment from Pooja Sharma.
5. Click **`✅ Confirm Receipt for Donation #ID`**.
6. Show the green success message: *"Now the donation status is 'Received', and warehouse inventory has been increased atomically."*

### Step 6: Verify in Audit History & Model Information (1 minute)
1. In the sidebar, click **`6. 📜 Donation History`**. Show the confirmed receipt in the ledger and the live fulfillment progress of saved purchase plans.
2. In the sidebar, click **`7. 🧠 Model & Data Information`**. Point out the **Random Forest MAE (61.09)** vs **Baseline MAE (103.30)** showing a **+40.9% accuracy lift**, and highlight the transparent synthetic data disclosure.

---

## 5. How to Run & Troubleshoot

### Launching the Application
```bash
# In your project folder
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Running the Automated Test Suite
```bash
python test_system.py
```
Expected output: `14/14 TESTS PASSED SUCCESSFULLY! (100%)`.

### Retraining the Model from Terminal (Optional)
```bash
python train_model.py
```

### Common Troubleshooting Tips
- **Port already in use?** Run `python -m streamlit run app.py --server.port 8502`.
- **Database corrupted or want a clean start?** Click the **`🔄 Reset to Clean Demo State`** button in the sidebar or run `python database.py`.
