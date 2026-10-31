"""
NeedMatch AI - Synthetic Historical Dataset Generator
Generates realistic multi-year historical demand and inventory data
across 6 donation categories and 6 partner NGOs with realistic seasonal patterns.
"""

import os
import numpy as np
import pandas as pd

CATEGORIES = [
    "Books",
    "Notebooks",
    "Schoolbags",
    "Clothing",
    "Blankets",
    "Hygiene Products"
]

ORGANIZATIONS = [
    {
        "id": "ORG001",
        "name": "Asha Education Foundation",
        "city": "Kolkata",
        "focus": "Education & Child Literacy",
        "primary_cats": ["Books", "Notebooks", "Schoolbags"]
    },
    {
        "id": "ORG002",
        "name": "Goonj Seva Trust",
        "city": "Delhi",
        "focus": "Disaster & Winter Relief",
        "primary_cats": ["Clothing", "Blankets", "Hygiene Products"]
    },
    {
        "id": "ORG003",
        "name": "Pratham Youth Initiative",
        "city": "Mumbai",
        "focus": "Slum Youth & Remedial Schools",
        "primary_cats": ["Notebooks", "Books", "Schoolbags", "Clothing"]
    },
    {
        "id": "ORG004",
        "name": "Sneha Health & Shelter",
        "city": "Bangalore",
        "focus": "Women & Child Community Health",
        "primary_cats": ["Hygiene Products", "Clothing", "Blankets"]
    },
    {
        "id": "ORG005",
        "name": "Vidya Jyoti Rural Mission",
        "city": "Jaipur",
        "focus": "Rural Schools & Tribal Welfare",
        "primary_cats": ["Schoolbags", "Books", "Blankets", "Notebooks"]
    },
    {
        "id": "ORG006",
        "name": "Karuna Care Society",
        "city": "Chennai",
        "focus": "Elderly Care & Community Welfare",
        "primary_cats": ["Blankets", "Hygiene Products", "Clothing"]
    }
]

# Base monthly demand scales per category
BASE_DEMAND = {
    "Books": 120,
    "Notebooks": 220,
    "Schoolbags": 75,
    "Clothing": 160,
    "Blankets": 90,
    "Hygiene Products": 180
}

def get_seasonal_multiplier(category: str, month: int) -> float:
    """Calculates realistic seasonal multipliers for Indian NGO demand patterns."""
    if category in ["Books", "Notebooks", "Schoolbags"]:
        # Academic reopening peak: May, June, July; minor exam term peak: October, November
        if month in [5, 6]:
            return 2.8
        elif month == 7:
            return 2.2
        elif month in [10, 11]:
            return 1.4
        elif month in [3, 4]:
            return 0.8
        else:
            return 1.0
    elif category == "Blankets":
        # Winter peak: Nov, Dec, Jan, Feb; extreme drop in peak summer
        if month in [12, 1]:
            return 3.8
        elif month in [11, 2]:
            return 2.5
        elif month in [5, 6, 7]:
            return 0.3
        else:
            return 0.7
    elif category == "Clothing":
        # Winter spike (Nov-Jan) and monsoon flood relief (July-Aug)
        if month in [12, 1]:
            return 2.2
        elif month in [7, 8]:
            return 1.6
        elif month in [10, 11]:
            return 1.5
        else:
            return 1.0
    elif category == "Hygiene Products":
        # Monsoon spike (waterborne diseases / health camps in July-Sept)
        if month in [7, 8, 9]:
            return 1.7
        else:
            return 1.1
    return 1.0

def generate_historical_dataset(
    output_path: str = "data/historical_demand.csv",
    start_date: str = "2024-01-01",
    num_months: int = 34,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a reproducible, unified synthetic historical dataset.
    Features realistic demand, inventory, fulfillment, and temporal relationships.
    Target: target_next_30d_demand represents the subsequent month's actual demand.
    """
    np.random.seed(random_seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dates = pd.date_range(start=start_date, periods=num_months, freq="MS")
    rows = []

    for d in dates:
        year = d.year
        month = d.month
        date_str = d.strftime("%Y-%m-%d")

        for org in ORGANIZATIONS:
            for cat in CATEGORIES:
                base = BASE_DEMAND[cat]
                # Boost if category is in org's primary focus
                org_focus_boost = 1.45 if cat in org["primary_cats"] else 0.75
                season_mult = get_seasonal_multiplier(cat, month)
                noise = np.random.normal(loc=1.0, scale=0.12)
                growth_factor = 1.0 + (year - 2024) * 0.08 + (month / 12) * 0.03

                # Current month's demand
                raw_demand = base * org_focus_boost * season_mult * noise * growth_factor
                historical_demand = max(10, int(round(raw_demand)))

                # Inventory at start of month
                inv_ratio = np.random.uniform(0.15, 0.65)
                inventory_level = max(0, int(round(historical_demand * inv_ratio)))

                # Fulfilled quantity typically covers a fraction of demand
                fulfill_rate = np.random.uniform(0.55, 0.90)
                fulfilled_qty = min(historical_demand, int(round(historical_demand * fulfill_rate)))

                # Outstanding backlog requests
                backlog = max(5, int(round((historical_demand - fulfilled_qty) * np.random.uniform(1.0, 1.4))))

                # Calculate future 30-day demand (next month's demand)
                next_date = d + pd.DateOffset(months=1)
                next_month = next_date.month
                next_year = next_date.year
                next_season_mult = get_seasonal_multiplier(cat, next_month)
                next_noise = np.random.normal(loc=1.0, scale=0.12)
                next_growth = 1.0 + (next_year - 2024) * 0.08 + (next_month / 12) * 0.03
                target_demand_val = base * org_focus_boost * next_season_mult * next_noise * next_growth
                target_next_30d_demand = max(12, int(round(target_demand_val)))

                rows.append({
                    "date": date_str,
                    "year": year,
                    "month": month,
                    "org_id": org["id"],
                    "org_name": org["name"],
                    "category": cat,
                    "historical_demand_qty": historical_demand,
                    "fulfilled_qty": fulfilled_qty,
                    "inventory_level": inventory_level,
                    "outstanding_requests": backlog,
                    "season_factor": round(season_mult, 2),
                    "target_next_30d_demand": target_next_30d_demand,
                    "is_synthetic": 1
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic historical records at {output_path}")
    return df

if __name__ == "__main__":
    generate_historical_dataset()
