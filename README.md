# NeedMatch AI: An AI-Based Predictive Donation Recommendation and Allocation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-0F4E6C.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF7F6E.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-C8DFF2.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/SQLite-Persistent%20Storage-0F4E6C.svg)](https://www.sqlite.org/)

**NeedMatch AI** is an intelligent, full-stack predictive donation allocation platform designed to eliminate donation waste, balance supply and demand across non-governmental organizations (NGOs), and empower donors to make high-impact contribution decisions.

---

## 1. Project Overview & Problem Statement

### The Problem
Traditional donation ecosystems suffer from severe **information asymmetry**:
- **Unbalanced Distribution:** Some urban NGOs receive massive surpluses of clothing and blankets, while grassroots shelters face acute shortages of student notebooks, schoolbags, or hygiene kits.
- **Donor Uncertainty:** Well-intentioned donors often do not know *which* items are urgently needed, *where* their contribution will make the biggest difference, or how to optimally spend a financial budget.
- **Operational Burden for NGOs:** Shelters often spend valuable time managing unwanted inventory rather than receiving items tailored to upcoming demand cycles.

### The Proposed Solution
**NeedMatch AI** bridges this gap by combining:
1. **Machine Learning Demand Forecasting:** A Scikit-learn `RandomForestRegressor` trained on multi-year temporal trends across 6 core donation categories.
2. **Transparent AI Recommendation Engine:** Formulations that calculate net unmet needs and rank beneficiary organizations based on urgency, inventory deficit, and predicted demand.
3. **Knapsack Budget Allocation:** A donor purchase planner that maximizes social impact per rupee within a user's defined INR budget.
4. **NGO Operations & Inventory Portal:** A persistent SQLite operational ledger where incoming donations remain *pending* until physically received and verified, ensuring data integrity.

---

## 2. Visual Design & Color Palette

Adapted strictly from the warm, minimal, editorial design reference:

| Color Name | Hex Code | Application |
|---|---|---|
| **Deep Teal Blue** | `#0F4E6C` | Primary brand color, headers, sidebar accents, chart bars |
| **Coral** | `#FF7F6E` | Primary action accents, CTA buttons, highlight badges, deficits |
| **Warm Cream / Beige** | `#F9E9D1` | Main page canvas background, subtle section panels |
| **Pure White** | `#FFFFFF` | Cards, forms, modal-like panels, content surfaces |
| **Soft Sky Blue** | `#C8DFF2` | Secondary metrics, pill tags, chart supporting series |

---

## 3. Core Features

### Feature 1: Donate Existing Items (Donor Workflow)
- Select from 6 essential donation categories:
  - **Books**
  - **Notebooks**
  - **Schoolbags**
  - **Clothing**
  - **Blankets**
  - **Hygiene Products**
- Input available quantity and condition (*New*, *Good*, *Usable*).
- Real-time ranking of recipient NGOs with transparent priority scores (0–100) and generated justification reasons.
- Submits donation in **PENDING** state (inventory is only updated upon NGO physical receipt confirmation).

### Feature 2: Plan a Donation Purchase with a Budget (Donor Workflow)
- Specify an available budget in Indian Rupees (₹).
- Select supported categories and optional NGO preference.
- System calculates the optimal purchase basket using wholesale INR reference prices:
  - *Notebooks:* ₹45 / pack of 2
  - *Books:* ₹150 / book
  - *Hygiene Kits:* ₹220 / kit
  - *Thermal Blankets:* ₹380 / blanket
  - *Schoolbags:* ₹420 / bag
  - *Clothing Sets:* ₹480 / set
- Interactive quantity steppers with real-time budget feasibility recalculation.
- 1-click purchase pledge registration.

### Feature 3: Machine Learning Demand Forecasting
- Scikit-learn `RandomForestRegressor` forecasting future 30-day demand.
- Trained across 34 monthly cycles for 6 NGOs and 6 categories (1,224 historical records).
- Strict **chronological train/test split** (80% train, 20% held-out test).
- Benchmark comparison against a training group-mean baseline on identical test data.
- Feature importance analysis and on-demand model retraining from the UI.

### Feature 4: Transparent AI Recommendation & Ranking Engine
- **Unmet Need Formula:**
  $$\text{Unmet Need} = \max\Big(0, (\text{Requested Backlog} + 0.6 \times \text{Predicted 30d Demand}) - (\text{Current Inventory} + \text{Pending Donations})\Big)$$
- **Urgency Multipliers:** High = $1.5\times$, Medium = $1.2\times$, Low = $1.0\times$.
- **Score (0–100):** Considers unmet need magnitude, urgency weight, and inventory deficit ratio.

### Feature 5: NGO Management & Inventory Portal
- Switch between 6 partner organizations (e.g., *Asha Education Foundation*, *Goonj Seva Trust*, *Sneha Health & Shelter*, etc.).
- Audit current inventory vs. ML projected demand.
- Broadcast new item requests with quantities and urgency tiers.
- **Confirm Receipt Workflow:** Atomically confirms pending inbound shipments, increments physical stock, and fulfills open requests.

### Feature 6: Donation Ledger & Audit Trail
- Persistent historical ledger of all donation transactions.
- Filter by status (*PENDING*, *CONFIRMED*, *REJECTED*), Category, and Organization.
- 1-click export to CSV.

---

## 4. Machine Learning Methodology & Dataset Disclosure

### Calibrated Synthetic Dataset Disclosure
Because longitudinal multi-NGO demand records across all six categories are not publicly consolidated into a single unified schema, this project employs a **statistically calibrated synthetic dataset** (`data/historical_demand.csv`).
- **Reproducibility:** Seeded with `np.random.seed(42)`.
- **Domain Seasonal Dynamics:**
  - *Education Supplies (Books, Notebooks, Schoolbags):* Peak in May–July ($2.2\times - 2.8\times$) matching Indian academic school reopening, with a secondary mid-term peak in Oct–Nov.
  - *Winter Relief (Blankets, Warm Clothing):* Massive spikes in Nov–Jan ($2.5\times - 3.8\times$) with drops during peak summer.
  - *Hygiene Products:* Year-round baseline with monsoon spikes in July–Sept ($1.7\times$).

### Model Performance Metrics
| Metric | Baseline Model (Group Mean) | Random Forest Regressor | Improvement |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | 103.30 units | **61.09 units** | **+40.86%** |
| **Root Mean Squared Error (RMSE)** | 156.83 units | **96.17 units** | **+38.68%** |
| **R² Score** | 0.4269 | **0.7845** | **+83.77%** |

*Note: Evaluation was conducted strictly on held-out future monthly records (Apr 2026 – Oct 2026) without data leakage.*

---

## 5. Technology Stack

- **Language:** Python 3.10+
- **Frontend / UI:** Streamlit with custom CSS and Plotly
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (`RandomForestRegressor`, `ColumnTransformer`, `OneHotEncoder`, `Pipeline`)
- **Model Serialization:** Joblib
- **Database:** SQLite3 (standard library, zero external DB configuration required)
- **Charts:** Plotly Express & Plotly Graph Objects

---

## 6. Project Structure

```
NeedMatch-AI/
│
├── app.py                      # Main Streamlit dashboard application
├── database.py                 # SQLite schema, transactional queries & inventory logic
├── data_generator.py           # Synthetic dataset generator with realistic seasonality
├── train_model.py              # ML training script, evaluation & pipeline persistence
├── predict.py                  # Inference module generating 30-day demand predictions
├── recommendation_engine.py    # Unmet need calculation, ranking logic & budget planner
├── requirements.txt            # Python dependencies
├── README.md                   # Complete system documentation
│
├── data/
│   ├── historical_demand.csv   # 1,224-row historical demand & inventory dataset
│   └── needmatch.db            # Persistent SQLite database
│
├── models/
│   ├── rf_demand_model.joblib  # Persisted Scikit-learn Pipeline
│   └── metrics.json            # Model evaluation metrics & feature importances
│
├── utils/
│   └── ui_styles.py            # Reference design palette & CSS styling injection
│
└── .streamlit/
    └── config.toml             # Streamlit theme configuration (#0F4E6C, #FF7F6E, #F9E9D1)
```

---

## 7. Installation & Quick Start

### Step 1: Clone or Open the Workspace
```bash
git clone <repository_url>
cd "AI mini project"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model (Optional)
*The repository already includes pre-trained model artifacts, but you can retrain at any time:*
```bash
python train_model.py
```

### Step 4: Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Example Demonstration Workflows

### Scenario A: Donating Pre-Owned School Supplies
1. Go to **"📦 Donate Existing Items"**.
2. Select **Category:** `Schoolbags`, **Quantity:** `20`, **Condition:** `Good`.
3. Notice how **Asha Education Foundation** is ranked #1 because they have an active open request of 75 bags, only 12 in inventory, and a predicted demand of 116 units.
4. Fill in donor contact details and click **"Submit Donation"**.
5. The donation is registered as `PENDING` with a unique ID.

### Scenario B: Corporate CSR or Budget Purchase Pledge
1. Go to **"💳 Plan a Donation Purchase"**.
2. Enter a budget of `₹10,000`.
3. Select categories: `Notebooks`, `Books`, `Schoolbags`.
4. Review the AI-recommended basket allocating units to highest-priority deficits per rupee spent.
5. Click **"Confirm Purchase Pledge"** to register the donation pledges.

### Scenario C: NGO Receiving and Confirming Inbound Shipments
1. Navigate to **"🏢 NGO Management Portal"**.
2. Switch to **"Asha Education Foundation"**.
3. Open the **"📥 Incoming Donations & Confirm Receipt"** tab.
4. Locate the newly submitted shipment and click **"✅ Confirm Physical Receipt"**.
5. Observe that:
   - Donation status changes to `CONFIRMED`.
   - Inventory for `Schoolbags` increments automatically.
   - Open requests are fulfilled in the database.

---

## 9. Current Limitations & Future Roadmap

1. **Synthetic Data Calibration:** While based on real seasonal trends (monsoons, academic calendars, winter relief), models should be re-calibrated against live NGO ERP data in production.
2. **Multi-Currency & Geocoding:** Future iterations can incorporate GPS distance matrix calculations to optimize logistics costs between donors and local shelters.
3. **Automated Logistic API Integration:** Integration with shipping and courier APIs (e.g. India Post, Delhivery) for automated shipment label generation.

---

## 10. License & Credits

Developed as part of the **NeedMatch AI** social impact predictive technology initiative. Built with modern machine learning, reliable software architecture, and human-centered design.
