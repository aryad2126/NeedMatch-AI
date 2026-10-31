"""
NeedMatch AI - SQLite Database Manager (Audited & Enhanced)
Provides persistent storage, transactional integrity, and auditable records for:
- Organizations & Verified Profiles
- Category Inventories
- Item Requests & Status Tracking
- Submitted Donations (Pending Receipt vs Received)
- Saved Purchase Plans (distinct from completed donations)
- Configured Item Prices in INR
- Persisted ML Demand Forecasts
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "needmatch.db")

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_PATH, force_reseed: bool = False):
    """
    Initializes database schema and populates initial demo data if empty.
    Preserves existing user data unless force_reseed is explicitly True.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    if force_reseed:
        cursor.execute("DROP TABLE IF EXISTS purchase_plans;")
        cursor.execute("DROP TABLE IF EXISTS donations;")
        cursor.execute("DROP TABLE IF EXISTS donation_requests;")
        cursor.execute("DROP TABLE IF EXISTS inventory;")
        cursor.execute("DROP TABLE IF EXISTS demand_predictions;")
        cursor.execute("DROP TABLE IF EXISTS item_prices;")
        cursor.execute("DROP TABLE IF EXISTS organizations;")

    # 1. Organizations Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS organizations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT NOT NULL,
        focus TEXT NOT NULL,
        contact_email TEXT,
        phone TEXT,
        description TEXT
    );
    """)

    # 2. Item Prices Table (in INR)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS item_prices (
        category TEXT PRIMARY KEY,
        unit_price_inr REAL NOT NULL,
        unit_name TEXT NOT NULL,
        item_description TEXT
    );
    """)

    # 3. Inventory Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        org_id TEXT NOT NULL,
        category TEXT NOT NULL,
        quantity INTEGER NOT NULL DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (org_id) REFERENCES organizations(id),
        UNIQUE(org_id, category)
    );
    """)

    # 4. Donation Requests Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS donation_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        org_id TEXT NOT NULL,
        category TEXT NOT NULL,
        requested_qty INTEGER NOT NULL,
        fulfilled_qty INTEGER NOT NULL DEFAULT 0,
        urgency TEXT NOT NULL CHECK(urgency IN ('High', 'Medium', 'Low')),
        status TEXT NOT NULL CHECK(status IN ('OPEN', 'PARTIALLY_FULFILLED', 'FULFILLED', 'CLOSED')),
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (org_id) REFERENCES organizations(id)
    );
    """)

    # 5. Donations Table (Physical existing item donations & pledges)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS donations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        donation_type TEXT NOT NULL CHECK(donation_type IN ('EXISTING_ITEMS', 'PURCHASE_PLEDGE')),
        donor_name TEXT NOT NULL,
        donor_email TEXT NOT NULL,
        donor_phone TEXT,
        org_id TEXT NOT NULL,
        category TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        condition TEXT CHECK(condition IN ('New', 'Good', 'Usable', 'N/A')),
        estimated_value_inr REAL DEFAULT 0,
        status TEXT NOT NULL CHECK(status IN ('Pending Receipt', 'Received', 'Cancelled', 'PENDING', 'CONFIRMED', 'REJECTED')),
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confirmed_at TIMESTAMP,
        FOREIGN KEY (org_id) REFERENCES organizations(id)
    );
    """)

    # 6. Purchase Plans Table (Cleanly separated from physical donations)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS purchase_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        donor_name TEXT NOT NULL,
        donor_email TEXT NOT NULL,
        total_budget_inr REAL NOT NULL,
        total_allocated_inr REAL NOT NULL,
        remaining_budget_inr REAL NOT NULL,
        items_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 7. Demand Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS demand_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        org_id TEXT NOT NULL,
        category TEXT NOT NULL,
        predicted_demand REAL NOT NULL,
        mae_error_margin REAL NOT NULL DEFAULT 0,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (org_id) REFERENCES organizations(id),
        UNIQUE(org_id, category)
    );
    """)

    conn.commit()

    # Seed Demo Data if organizations table is empty or force_reseed
    cursor.execute("SELECT COUNT(*) FROM organizations")
    count = cursor.fetchone()[0]

    if count == 0 or force_reseed:
        if force_reseed:
            cursor.execute("DELETE FROM purchase_plans")
            cursor.execute("DELETE FROM donations")
            cursor.execute("DELETE FROM donation_requests")
            cursor.execute("DELETE FROM inventory")
            cursor.execute("DELETE FROM demand_predictions")
            cursor.execute("DELETE FROM item_prices")
            cursor.execute("DELETE FROM organizations")

        from data_generator import ORGANIZATIONS, CATEGORIES

        org_descriptions = {
            "ORG001": "Focusing on underprivileged children in urban slums, providing bridge education and library resources.",
            "ORG002": "Mobilizing urban discard as a tool for rural development and humanitarian winter relief.",
            "ORG003": "Community remedial learning centers serving high-density municipal school children.",
            "ORG004": "Supporting transitional women's shelters, neonatal hygiene, and adolescent girl care centers.",
            "ORG005": "Empowering tribal students and remote desert village schools with essential gear and books.",
            "ORG006": "Community health centers, disaster-vulnerable coastal settlements, and eldercare outreach."
        }

        for org in ORGANIZATIONS:
            cursor.execute("""
            INSERT INTO organizations (id, name, city, focus, contact_email, phone, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                org["id"],
                org["name"],
                org["city"],
                org["focus"],
                f"contact@{org['id'].lower()}.org",
                "+91 98765 43210",
                org_descriptions.get(org["id"], "Community NGO working for social welfare.")
            ))

        # Standard wholesale prices in INR
        standard_prices = [
            ("Books", 150.0, "Book", "Age-appropriate storybooks, curricula textbooks, reference guides"),
            ("Notebooks", 45.0, "Pack of 2", "Single-line & square ruled 120-page student notebooks"),
            ("Schoolbags", 420.0, "Bag", "Durable water-resistant school backpacks with bottle holder"),
            ("Clothing", 480.0, "Garment Set", "New durable daily wear clothing sets for children and youth"),
            ("Blankets", 380.0, "Fleece Blanket", "Warm dual-layered winter thermal fleece blankets"),
            ("Hygiene Products", 220.0, "Kit", "Essential hygiene kit (soap, toothbrush, paste, sanitizers, sanitary pads)")
        ]

        cursor.executemany("""
        INSERT INTO item_prices (category, unit_price_inr, unit_name, item_description)
        VALUES (?, ?, ?, ?)
        """, standard_prices)

        # Initial baseline inventory
        initial_inv = [
            ("ORG001", "Books", 85),
            ("ORG001", "Notebooks", 40),
            ("ORG001", "Schoolbags", 12),
            ("ORG001", "Clothing", 25),
            ("ORG001", "Blankets", 15),
            ("ORG001", "Hygiene Products", 30),
            ("ORG002", "Books", 20),
            ("ORG002", "Notebooks", 35),
            ("ORG002", "Schoolbags", 15),
            ("ORG002", "Clothing", 210),
            ("ORG002", "Blankets", 28),
            ("ORG002", "Hygiene Products", 45),
            ("ORG003", "Books", 95),
            ("ORG003", "Notebooks", 60),
            ("ORG003", "Schoolbags", 18),
            ("ORG003", "Clothing", 50),
            ("ORG003", "Blankets", 10),
            ("ORG003", "Hygiene Products", 40),
            ("ORG004", "Books", 15),
            ("ORG004", "Notebooks", 25),
            ("ORG004", "Schoolbags", 8),
            ("ORG004", "Clothing", 42),
            ("ORG004", "Blankets", 14),
            ("ORG004", "Hygiene Products", 35),
            ("ORG005", "Books", 45),
            ("ORG005", "Notebooks", 50),
            ("ORG005", "Schoolbags", 14),
            ("ORG005", "Clothing", 60),
            ("ORG005", "Blankets", 30),
            ("ORG005", "Hygiene Products", 20),
            ("ORG006", "Books", 10),
            ("ORG006", "Notebooks", 20),
            ("ORG006", "Schoolbags", 5),
            ("ORG006", "Clothing", 75),
            ("ORG006", "Blankets", 22),
            ("ORG006", "Hygiene Products", 30),
        ]

        cursor.executemany("""
        INSERT INTO inventory (org_id, category, quantity)
        VALUES (?, ?, ?)
        """, initial_inv)

        # Initial Active Demo Requests
        demo_requests = [
            ("ORG001", "Notebooks", 180, 45, "High", "OPEN", "Academic term notebooks urgently needed for 240 primary school students."),
            ("ORG001", "Schoolbags", 90, 15, "High", "OPEN", "Annual kit distribution for slum bridge schooling batch."),
            ("ORG002", "Blankets", 250, 60, "High", "OPEN", "North India winter relief drive for night shelter occupants."),
            ("ORG002", "Hygiene Products", 160, 30, "Medium", "OPEN", "Sanitation and hygiene kits for community camps."),
            ("ORG003", "Notebooks", 150, 40, "High", "OPEN", "Exam preparation revision workbooks and notebooks."),
            ("ORG003", "Books", 100, 35, "Medium", "OPEN", "Community library expansion in Dharavi learning center."),
            ("ORG004", "Hygiene Products", 200, 50, "High", "OPEN", "Maternal hygiene and adolescent welfare kits for women's shelter."),
            ("ORG004", "Blankets", 80, 20, "Medium", "OPEN", "Shelter bedding renewal before monsoon cold."),
            ("ORG005", "Schoolbags", 120, 25, "High", "OPEN", "Tribal students walking 4+ km needing durable backpacks."),
            ("ORG005", "Books", 140, 50, "Medium", "OPEN", "Bilingual STEM and storybooks for village school library."),
            ("ORG006", "Blankets", 140, 40, "High", "OPEN", "Thermal blankets for destitute elderly care home residents."),
            ("ORG006", "Clothing", 110, 35, "Medium", "OPEN", "Clean washable everyday clothing sets for shelter elders.")
        ]

        cursor.executemany("""
        INSERT INTO donation_requests (org_id, category, requested_qty, fulfilled_qty, urgency, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, demo_requests)

        # Demo initial donations: clear readable status 'Received' and 'Pending Receipt'
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seed_donations = [
            ("EXISTING_ITEMS", "Sunil Verma", "sunil.v@example.com", "9811223344", "ORG001", "Notebooks", 30, "New", 1350.0, "Received", "Stationery batch from corporate CSR drive.", now_str, now_str),
            ("EXISTING_ITEMS", "Priya Nair", "priya.n@example.com", "9822334455", "ORG002", "Blankets", 25, "New", 9500.0, "Received", "Heavy wool fleece blankets for winter distribution.", now_str, now_str),
            ("EXISTING_ITEMS", "Meera Sen", "meera.sen@example.com", "9844556677", "ORG001", "Schoolbags", 15, "Good", 6300.0, "Pending Receipt", "Pre-loved sturdy backpacks from school drive.", now_str, None),
            ("EXISTING_ITEMS", "Rohan Gupta", "rohan.g@example.com", "9855667788", "ORG002", "Clothing", 40, "Good", 19200.0, "Pending Receipt", "Freshly washed winter woolens and jackets.", now_str, None),
            ("EXISTING_ITEMS", "Ananya Roy", "ananya.r@example.com", "9866778899", "ORG004", "Hygiene Products", 50, "New", 11000.0, "Pending Receipt", "Online merchant delivery dispatched to NGO center.", now_str, None)
        ]

        cursor.executemany("""
        INSERT INTO donations (
            donation_type, donor_name, donor_email, donor_phone, org_id, category,
            quantity, condition, estimated_value_inr, status, notes, created_at, confirmed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_donations)

        # Seed sample purchase plan
        sample_plan_items = json.dumps([
            {"category": "Notebooks", "org_id": "ORG001", "org_name": "Asha Education Foundation", "unit_price": 45.0, "quantity": 50, "cost": 2250.0},
            {"category": "Schoolbags", "org_id": "ORG005", "org_name": "Vidya Jyoti Rural Mission", "unit_price": 420.0, "quantity": 10, "cost": 4200.0}
        ])
        cursor.execute("""
        INSERT INTO purchase_plans (donor_name, donor_email, total_budget_inr, total_allocated_inr, remaining_budget_inr, items_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """, ("Arun Kumar", "arun.k@example.com", 7000.0, 6450.0, 550.0, sample_plan_items))

        conn.commit()
        print("Database initialized and demo data seeded successfully.")

    conn.close()

# ----------------- Database Query & Mutation API -----------------

def get_organizations() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM organizations ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_organization_by_id(org_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM organizations WHERE id = ?", (org_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_item_prices() -> Dict[str, Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM item_prices").fetchall()
    conn.close()
    return {r["category"]: dict(r) for r in rows}

def get_inventory(org_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    if org_id:
        rows = conn.execute("""
            SELECT i.*, o.name as org_name
            FROM inventory i
            JOIN organizations o ON i.org_id = o.id
            WHERE i.org_id = ?
            ORDER BY i.category
        """, (org_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT i.*, o.name as org_name, o.city
            FROM inventory i
            JOIN organizations o ON i.org_id = o.id
            ORDER BY i.org_id, i.category
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def set_inventory_quantity(org_id: str, category: str, quantity: int) -> bool:
    if quantity < 0:
        return False
    conn = get_connection()
    conn.execute("""
        INSERT INTO inventory (org_id, category, quantity, last_updated)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(org_id, category) DO UPDATE SET
            quantity = excluded.quantity,
            last_updated = CURRENT_TIMESTAMP
    """, (org_id, category, quantity))
    conn.commit()
    conn.close()
    return True

def get_donation_requests(org_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    query = """
        SELECT r.*, o.name as org_name, o.city
        FROM donation_requests r
        JOIN organizations o ON r.org_id = o.id
        WHERE 1=1
    """
    params = []
    if org_id:
        query += " AND r.org_id = ?"
        params.append(org_id)
    if status:
        query += " AND r.status = ?"
        params.append(status)
    query += " ORDER BY CASE r.urgency WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, r.created_at DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_donation_request(org_id: str, category: str, requested_qty: int, urgency: str, notes: str = "") -> int:
    if requested_qty <= 0:
        raise ValueError("Requested quantity must be positive")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO donation_requests (org_id, category, requested_qty, fulfilled_qty, urgency, status, notes)
        VALUES (?, ?, ?, 0, ?, 'OPEN', ?)
    """, (org_id, category, requested_qty, urgency, notes))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id

def update_donation_request(request_id: int, requested_qty: int, urgency: str, notes: str = "") -> bool:
    if requested_qty <= 0:
        raise ValueError("Requested quantity must be positive")
    conn = get_connection()
    conn.execute("""
        UPDATE donation_requests
        SET requested_qty = ?, urgency = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (requested_qty, urgency, notes, request_id))
    conn.commit()
    conn.close()
    return True

def close_donation_request(request_id: int) -> bool:
    conn = get_connection()
    conn.execute("""
        UPDATE donation_requests
        SET status = 'CLOSED', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (request_id,))
    conn.commit()
    conn.close()
    return True

def update_request_status(request_id: int, status: str) -> bool:
    conn = get_connection()
    conn.execute("""
        UPDATE donation_requests
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (status, request_id))
    conn.commit()
    conn.close()
    return True

def get_donations(org_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    query = """
        SELECT d.*, o.name as org_name, o.city
        FROM donations d
        JOIN organizations o ON d.org_id = o.id
        WHERE 1=1
    """
    params = []
    if org_id:
        query += " AND d.org_id = ?"
        params.append(org_id)
    if status:
        # Standardize matching for status
        if status in ["Pending Receipt", "PENDING"]:
            query += " AND d.status IN ('Pending Receipt', 'PENDING')"
        elif status in ["Received", "CONFIRMED"]:
            query += " AND d.status IN ('Received', 'CONFIRMED')"
        elif status in ["Cancelled", "REJECTED"]:
            query += " AND d.status IN ('Cancelled', 'REJECTED')"
        else:
            query += " AND d.status = ?"
            params.append(status)
    query += " ORDER BY d.created_at DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_donation(
    donation_type: str,
    donor_name: str,
    donor_email: str,
    donor_phone: str,
    org_id: str,
    category: str,
    quantity: int,
    condition: str = "Good",
    estimated_value_inr: float = 0.0,
    notes: str = ""
) -> int:
    """Submits a donation record in 'Pending Receipt' state. Does NOT alter inventory."""
    if quantity <= 0:
        raise ValueError("Donation quantity must be greater than 0")
    if not donor_name or not donor_email:
        raise ValueError("Donor name and email are required")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO donations (
            donation_type, donor_name, donor_email, donor_phone, org_id, category,
            quantity, condition, estimated_value_inr, status, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pending Receipt', ?, CURRENT_TIMESTAMP)
    """, (
        donation_type, donor_name, donor_email, donor_phone,
        org_id, category, quantity, condition, estimated_value_inr, notes
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id

def confirm_donation_receipt(donation_id: int, confirming_org_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Transactionally confirms a donation receipt:
    1. Validates that the donation exists and is currently pending receipt.
    2. Enforces that the confirming NGO matches the intended recipient organization.
    3. Atomically updates donation status to 'Received' with timestamp.
    4. Atomically increments physical inventory for that NGO and category.
    5. Allocates quantity to open requests for that NGO & category if any exist.
    6. Prevents duplicate confirmation or negative inventory state.
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("BEGIN IMMEDIATE TRANSACTION;")

        cur.execute("SELECT * FROM donations WHERE id = ?", (donation_id,))
        don = cur.fetchone()

        if not don:
            conn.rollback()
            return {"success": False, "message": f"Donation #{donation_id} not found."}

        # Validate authorization: NGO cannot confirm donations intended for another NGO
        if confirming_org_id and don["org_id"] != confirming_org_id:
            conn.rollback()
            return {
                "success": False,
                "message": f"Unauthorized: Donation #{donation_id} is addressed to {don['org_id']}, but {confirming_org_id} attempted confirmation."
            }

        if don["status"] in ["Received", "CONFIRMED"]:
            conn.rollback()
            return {"success": False, "message": f"Donation #{donation_id} has already been confirmed as Received."}

        if don["status"] in ["Cancelled", "REJECTED"]:
            conn.rollback()
            return {"success": False, "message": f"Donation #{donation_id} was previously cancelled."}

        org_id = don["org_id"]
        category = don["category"]
        quantity = don["quantity"]

        # 1. Update Donation Status to 'Received'
        cur.execute("""
            UPDATE donations
            SET status = 'Received', confirmed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (donation_id,))

        # 2. Update Inventory
        cur.execute("""
            INSERT INTO inventory (org_id, category, quantity, last_updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(org_id, category) DO UPDATE SET
                quantity = quantity + excluded.quantity,
                last_updated = CURRENT_TIMESTAMP
        """, (org_id, category, quantity))

        # 3. Check and allocate to Open/Partially Fulfilled Requests
        cur.execute("""
            SELECT * FROM donation_requests
            WHERE org_id = ? AND category = ? AND status IN ('OPEN', 'PARTIALLY_FULFILLED')
            ORDER BY CASE urgency WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, created_at ASC
        """, (org_id, category))
        open_requests = cur.fetchall()

        remaining_alloc = quantity
        allocated_details = []

        for req in open_requests:
            if remaining_alloc <= 0:
                break
            needed = req["requested_qty"] - req["fulfilled_qty"]
            alloc = min(remaining_alloc, needed)
            new_fulfilled = req["fulfilled_qty"] + alloc
            new_status = "FULFILLED" if new_fulfilled >= req["requested_qty"] else "PARTIALLY_FULFILLED"

            cur.execute("""
                UPDATE donation_requests
                SET fulfilled_qty = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_fulfilled, new_status, req["id"]))

            remaining_alloc -= alloc
            allocated_details.append(f"Request #{req['id']} updated with +{alloc} units ({new_status})")

        conn.commit()
        return {
            "success": True,
            "message": f"Donation #{donation_id} confirmed as Received! Added {quantity} units to {category} inventory.",
            "allocated_details": allocated_details
        }
    except Exception as e:
        conn.rollback()
        return {"success": False, "message": f"Transaction failed: {str(e)}"}
    finally:
        conn.close()

# ----------------- Purchase Plans API -----------------

def save_purchase_plan(
    donor_name: str,
    donor_email: str,
    total_budget_inr: float,
    total_allocated_inr: float,
    remaining_budget_inr: float,
    items: List[Dict[str, Any]]
) -> int:
    """Saves an optimized purchase plan distinctly from completed donations."""
    if not donor_name or not donor_email:
        raise ValueError("Donor name and email are required")
    if total_allocated_inr > total_budget_inr:
        raise ValueError("Allocated cost exceeds total available budget")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO purchase_plans (
            donor_name, donor_email, total_budget_inr, total_allocated_inr, remaining_budget_inr, items_json
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        donor_name, donor_email, total_budget_inr, total_allocated_inr,
        remaining_budget_inr, json.dumps(items)
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id

def get_purchase_plans() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM purchase_plans ORDER BY created_at DESC").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["items"] = json.loads(d["items_json"])
        except Exception:
            d["items"] = []
        results.append(d)
    return results

def save_predictions(pred_records: List[Dict[str, Any]]):
    """Saves or updates ML 30-day demand predictions in SQLite."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS demand_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id TEXT NOT NULL,
            category TEXT NOT NULL,
            predicted_demand REAL NOT NULL,
            mae_error_margin REAL NOT NULL DEFAULT 0,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (org_id) REFERENCES organizations(id),
            UNIQUE(org_id, category)
        );
    """)

    for p in pred_records:
        cur.execute("""
            INSERT INTO demand_predictions (org_id, category, predicted_demand, mae_error_margin, prediction_date)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(org_id, category) DO UPDATE SET
                predicted_demand = excluded.predicted_demand,
                mae_error_margin = excluded.mae_error_margin,
                prediction_date = CURRENT_TIMESTAMP
        """, (p["org_id"], p["category"], float(p["predicted_demand"]), float(p.get("mae_error_margin", 0))))

    conn.commit()
    conn.close()

def get_predictions_from_db(org_id: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    query = """
        SELECT p.*, o.name as org_name, o.city
        FROM demand_predictions p
        JOIN organizations o ON p.org_id = o.id
        WHERE 1=1
    """
    params = []
    if org_id:
        query += " AND p.org_id = ?"
        params.append(org_id)
    if category:
        query += " AND p.category = ?"
        params.append(category)

    query += " ORDER BY p.predicted_demand DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_overall_stats() -> Dict[str, Any]:
    """Computes operational and analytics summary statistics from live SQLite records."""
    conn = get_connection()

    total_orgs = conn.execute("SELECT COUNT(*) FROM organizations").fetchone()[0]
    total_donations = conn.execute("SELECT COUNT(*) FROM donations").fetchone()[0]
    pending_donations = conn.execute("SELECT COUNT(*) FROM donations WHERE status IN ('Pending Receipt', 'PENDING')").fetchone()[0]
    confirmed_donations = conn.execute("SELECT COUNT(*) FROM donations WHERE status IN ('Received', 'CONFIRMED')").fetchone()[0]
    total_units_donated = conn.execute("SELECT COALESCE(SUM(quantity), 0) FROM donations WHERE status IN ('Received', 'CONFIRMED')").fetchone()[0]
    pending_units = conn.execute("SELECT COALESCE(SUM(quantity), 0) FROM donations WHERE status IN ('Pending Receipt', 'PENDING')").fetchone()[0]

    open_requests_count = conn.execute("SELECT COUNT(*) FROM donation_requests WHERE status IN ('OPEN', 'PARTIALLY_FULFILLED')").fetchone()[0]
    total_outstanding_requested_units = conn.execute("""
        SELECT COALESCE(SUM(requested_qty - fulfilled_qty), 0)
        FROM donation_requests
        WHERE status IN ('OPEN', 'PARTIALLY_FULFILLED')
    """).fetchone()[0]

    total_inventory_units = conn.execute("SELECT COALESCE(SUM(quantity), 0) FROM inventory").fetchone()[0]
    total_purchase_plans = conn.execute("SELECT COUNT(*) FROM purchase_plans").fetchone()[0]

    conn.close()
    return {
        "total_orgs": total_orgs,
        "total_donations": total_donations,
        "pending_donations": pending_donations,
        "confirmed_donations": confirmed_donations,
        "total_units_donated": total_units_donated,
        "pending_units": pending_units,
        "open_requests_count": open_requests_count,
        "total_outstanding_requested_units": total_outstanding_requested_units,
        "total_inventory_units": total_inventory_units,
        "total_purchase_plans": total_purchase_plans
    }

if __name__ == "__main__":
    init_db(force_reseed=True)
    print("Database re-initialized and audited successfully.")
    print("Stats:", get_overall_stats())
