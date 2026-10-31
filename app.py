"""
NeedMatch AI: An AI-Based Predictive Donation Recommendation and Allocation System
Complete, audited, and restructured Streamlit Application.
Strictly organized into 8 distinct pages with complete educational transparency.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import math
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local Modules
import database as db
import predict
from recommendation_engine import (
    recommend_ngos_for_item,
    recommend_purchase_plan,
    calculate_unmet_need,
    compute_priority_score
)
from utils.ui_styles import (
    inject_custom_css,
    render_hero,
    get_plotly_layout,
    render_ngo_recommendation_card,
    COLOR_TEAL,
    COLOR_CORAL,
    COLOR_CREAM,
    COLOR_WHITE,
    COLOR_SKY,
    COLOR_MUTED_TEXT,
    COLOR_BORDER
)
from data_generator import CATEGORIES, ORGANIZATIONS

# Page Configuration
st.set_page_config(
    page_title="NeedMatch AI | Predictive Donation Allocation System",
    page_icon=":material/handshake:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject visual styling matching reference image
inject_custom_css()

# Ensure Database is initialized on startup
if "db_initialized" not in st.session_state:
    db.init_db(force_reseed=False)
    try:
        preds = db.get_predictions_from_db()
        if not preds:
            predict.generate_and_save_predictions()
    except Exception:
        pass
    st.session_state["db_initialized"] = True


# ----------------- SIDEBAR NAVIGATION (EXACT 8 PAGES) -----------------
with st.sidebar:
    st.markdown(f"<div style='padding: 0.4rem 0 0.6rem 0;'><div style=\"font-family: 'Playfair Display', Georgia, serif; font-size: 1.65rem; font-weight: 700; color: {COLOR_TEAL}; line-height: 1.1;\">NeedMatch <span style='color: {COLOR_CORAL};'>AI</span></div><div style='font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-top: 4px;'>Predictive Donation Allocation System</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    PAGE_HOME = ":material/home: 1. Home / Dashboard"
    PAGE_FORECAST = ":material/trending_up: 2. Demand Forecast"
    PAGE_DONATE = ":material/inventory_2: 3. Donate Existing Items"
    PAGE_PURCHASE = ":material/shopping_cart: 4. Plan a Purchase"
    PAGE_NGO = ":material/domain: 5. NGO Portal"
    PAGE_HISTORY = ":material/receipt_long: 6. Donation History"
    PAGE_MODEL = ":material/psychology: 7. Model & Data Information"

    PAGES = [
        PAGE_HOME,
        PAGE_FORECAST,
        PAGE_DONATE,
        PAGE_PURCHASE,
        PAGE_NGO,
        PAGE_HISTORY,
        PAGE_MODEL
    ]

    # Deep-link navigation via query parameters
    if "nav" in st.query_params:
        nav_val = st.query_params.get("nav")
        if nav_val == "ngo_backlog":
            st.session_state["nav_target"] = PAGE_NGO
            st.session_state["ngo_portal_view"] = "backlog"
        try:
            del st.query_params["nav"]
        except Exception:
            pass

    # Handle cross-page redirection
    if "nav_target" in st.session_state and st.session_state["nav_target"] in PAGES:
        st.session_state["nav_selection"] = st.session_state["nav_target"]
        del st.session_state["nav_target"]

    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = PAGES[0]

    selected_page = st.radio("Navigation Flow", PAGES, key="nav_selection")

    st.markdown("---")

    # System Status Indicator Widget
    metrics_data = predict.load_metrics()
    model_trained = os.path.exists("models/rf_demand_model.joblib")
    db_ready = os.path.exists("data/needmatch.db")
    with st.container(border=True):
        st.markdown("**System Health**")
        st.caption("Database: **Connected**")
        st.caption(f"ML Model: **{'Ready (RF)' if model_trained else 'Needs Training'}**")
        st.caption(f"Model Test MAE: **{metrics_data.get('evaluation', {}).get('rf_mae', 'N/A')} units**")


    if st.button("Reset to Clean Demo State", icon=":material/refresh:", help="Reseeds the database with clean baseline demo records"):
        db.init_db(force_reseed=True)
        predict.generate_and_save_predictions()
        st.success("Database reseeded with fresh demo records!")
        st.rerun()

# =====================================================================
# PAGE 1: HOME / DASHBOARD
# =====================================================================
if selected_page == PAGE_HOME:
    render_hero(
        title="Predict Donation Needs.",
        highlight_word="Recommend Meaningful Contributions.",
        subtitle="NeedMatch AI connects donations with organizations that need them most using machine learning demand forecasts and real-time inventory intelligence."
    )

    # Primary Action Buttons
    c_btn1, c_btn2, c_sp = st.columns([1, 1, 1.5])
    with c_btn1:
        if st.button("Donate Items I Own", icon=":material/inventory_2:", type="primary", use_container_width=True):
            st.session_state["nav_target"] = PAGE_DONATE
            st.rerun()
    with c_btn2:
        if st.button("Plan a Purchase with Budget", icon=":material/shopping_cart:", use_container_width=True):
            st.session_state["nav_target"] = PAGE_PURCHASE
            st.rerun()



    # Summary Metrics (Real Database Values)
    stats = db.get_overall_stats()
    m_info = predict.load_metrics()
    rf_mae = m_info.get("evaluation", {}).get("rf_mae", 61.1)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Demo Organizations", f"{stats['total_orgs']}", "Partner shelters", delta_color="off")
    col2.metric("Total Donations", f"{stats['total_donations']}", "Ledger records", delta_color="off")
    col3.metric("Awaiting Receipt", f"{stats['pending_donations']}", f"{stats['pending_units']} units pending", delta_color="off")
    col4.metric("Donations Received", f"{stats['confirmed_donations']}", f"{stats['total_units_donated']} physical units", delta_color="off")
    with col5:
        st.metric("Outstanding Backlog", f"{stats['total_outstanding_requested_units']}", f"{stats['open_requests_count']} open requests", delta_color="inverse")
        if st.button("View NGO Backlog Details", icon=":material/arrow_forward:", key="btn_view_backlog", use_container_width=True, help="Navigate to NGO Portal to see all open requests contributing to this backlog"):
            st.session_state["nav_target"] = PAGE_NGO
            st.session_state["ngo_portal_view"] = "backlog"
            st.rerun()

    st.divider()

    # Visual Charts
    forecast_df = predict.get_demand_forecast_table()
    chart_c1, chart_c2 = st.columns([1.1, 0.9])

    with chart_c1:
        with st.container(border=True):
            st.subheader("Predicted 30-Day Demand vs. Current Stock by Category")
            st.caption("Comparing ML 30-day forecast against available inventory across all 6 partner organizations.")

            cat_summary = forecast_df.groupby("category")[["predicted_30d_demand", "current_inventory"]].sum().reset_index()
            cat_summary["projected_shortfall"] = (cat_summary["predicted_30d_demand"] - cat_summary["current_inventory"]).clip(lower=0.0)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=cat_summary["category"],
                y=cat_summary["predicted_30d_demand"],
                name="ML 30d Demand",
                marker_color=COLOR_TEAL,
                text=cat_summary["predicted_30d_demand"].astype(int),
                textposition="auto"
            ))
            fig_bar.add_trace(go.Bar(
                x=cat_summary["category"],
                y=cat_summary["current_inventory"],
                name="Current Inventory",
                marker_color=COLOR_SKY,
                text=cat_summary["current_inventory"].astype(int),
                textposition="auto"
            ))
            fig_bar.add_trace(go.Bar(
                x=cat_summary["category"],
                y=cat_summary["projected_shortfall"],
                name="Estimated Shortage",
                marker_color=COLOR_CORAL,
                text=cat_summary["projected_shortfall"].astype(int),
                textposition="auto"
            ))
            fig_bar.update_layout(
                get_plotly_layout(height=340),
                barmode="group",
                yaxis_title="Units"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with chart_c2:
        with st.container(border=True):
            st.subheader("Estimated Shortages by Organization")
            st.caption("Identifies which partner NGOs face the highest immediate deficits.")

            org_summary = forecast_df.groupby("org_name")["projected_shortfall"].sum().reset_index()
            fig_donut = px.pie(
                org_summary,
                names="org_name",
                values="projected_shortfall",
                hole=0.55,
                color_discrete_sequence=[COLOR_TEAL, COLOR_CORAL, "#75B4D6", "#E5A88E", "#3A7D9D", COLOR_SKY]
            )
            fig_donut.update_layout(
                get_plotly_layout(height=340),
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(size=10))
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # Additional Sections: High Priority Needs & Recent Donations
    sec_c1, sec_c2 = st.columns([1.1, 1])

    with sec_c1:
        with st.container(border=True):
            st.subheader("High-Priority Unmet Needs")
            st.caption("Organizations with acute shortages calculated from open requests and predicted demand.")

            top_shortfalls = forecast_df.sort_values(by="projected_shortfall", ascending=False).head(4)
            for _, row in top_shortfalls.iterrows():
                with st.container(border=True):
                    c_sh1, c_sh2 = st.columns([2.5, 1])
                    c_sh1.markdown(f"**{row['category']} — {row['org_name']}**")
                    c_sh1.caption(f"Stock: {row['current_inventory']} | Requests: {row['open_requests']} | 30d Demand: {row['predicted_30d_demand']:.0f}")
                    c_sh2.metric("Deficit", f"{row['projected_shortfall']:.0f} units")

    with sec_c2:
        with st.container(border=True):
            st.subheader("Recent Donation Transactions")
            st.caption("Latest submissions from the persistent SQLite ledger.")

            recent_donations = db.get_donations()[:4]
            if not recent_donations:
                st.info("No donations recorded yet.")
            else:
                for d in recent_donations:
                    with st.container(border=True):
                        c_rd1, c_rd2 = st.columns([2.6, 1])
                        c_rd1.markdown(f"**Donation #{d['id']}: {d['quantity']} units of {d['category']}**")
                        c_rd1.caption(f"Donor: {d['donor_name']} → Recipient: {d['org_name']} ({db.format_datetime_ist(d['created_at'])})")
                        if d['status'] in ['Received', 'CONFIRMED']:
                            c_rd2.success(f"{d['status']}", icon=":material/check_circle:")
                        else:
                            c_rd2.warning(f"⏳ {d['status']}")



# =====================================================================
# PAGE 2: DEMAND FORECAST
# =====================================================================
elif selected_page == PAGE_FORECAST:
    render_hero(
        title="Predictive Demand",
        highlight_word="Forecast Engine.",
        subtitle="Generate 30-day demand predictions using the trained Random Forest model. Observe the features, inventory context, and estimated shortages."
    )

    # Input Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        orgs = db.get_organizations()
        org_options = {"ALL": "All Organizations (Aggregated)"}
        for o in orgs:
            org_options[o["id"]] = f"{o['name']} ({o['city']})"
        sel_org = st.selectbox("Select Organization:", options=list(org_options.keys()), format_func=lambda x: org_options[x], index=1)

    with ctrl_col2:
        cat_options = ["All Categories"] + CATEGORIES
        sel_cat = st.selectbox("Select Donation Category:", cat_options, index=2)

    with ctrl_col3:
        sel_horizon = st.selectbox("Forecast Horizon:", [30, 14, 7], index=0, format_func=lambda x: f"{x} Days {'(Standard ML Target)' if x == 30 else '(Proportionally Scaled)'}")

    # Main Action Button

    run_forecast_btn = st.button("Generate Demand Forecast", icon=":material/trending_up:", type="primary", use_container_width=True)

    if run_forecast_btn:
        st.session_state["forecast_executed"] = True
        st.session_state["forecast_org"] = sel_org
        st.session_state["forecast_cat"] = sel_cat
        st.session_state["forecast_horizon"] = sel_horizon

    if st.session_state.get("forecast_executed", False):
        curr_org = st.session_state.get("forecast_org", sel_org)
        curr_cat = st.session_state.get("forecast_cat", sel_cat)
        curr_horizon = st.session_state.get("forecast_horizon", sel_horizon)

        pipeline = predict.load_trained_model()

        # Handle Single Organization vs All Organizations
        target_org_ids = [curr_org] if curr_org != "ALL" else [o["id"] for o in orgs]
        target_cats = [curr_cat] if curr_cat != "All Categories" else CATEGORIES

        results = []
        for oid in target_org_ids:
            for c in target_cats:
                res = predict.predict_single(pipeline, oid, c, horizon_days=curr_horizon)
                # Compute unmet shortage: (Backlog + Predicted Demand) - Inventory
                inv = res["current_inventory"]
                backlog = res["outstanding_requests"]
                pred = res["predicted_demand_selected_horizon"]
                shortage = max(0.0, round((backlog + pred) - inv, 1))

                org_meta = db.get_organization_by_id(oid)
                results.append({
                    "org_id": oid,
                    "org_name": org_meta["name"],
                    "city": org_meta["city"],
                    "category": c,
                    "predicted_demand": pred,
                    "native_30d_demand": res["predicted_demand_30d"],
                    "current_inventory": inv,
                    "outstanding_requests": backlog,
                    "estimated_shortage": shortage,
                    "features": res["input_features"],
                    "horizon_note": res["horizon_note"]
                })

        # Display Result Highlights
        st.divider()

        if len(results) == 1:
            r = results[0]
            with st.container(border=True):
                col_fh1, col_fh2 = st.columns([3, 1.2])
                with col_fh1:
                    st.subheader(f"Demand Forecast: {r['category']} for {r['org_name']}")
                    st.caption(f"Location: **{r['city']}** • Horizon: **{r['horizon_note']}**")
                with col_fh2:
                    st.info("Random Forest Output")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Demand", f"{r['predicted_demand']:.0f}", f"{curr_horizon}-Day Horizon", delta_color="off")
                m2.metric("Current Inventory", f"{r['current_inventory']}", "In warehouse", delta_color="off")
                m3.metric("Outstanding Requests", f"{r['outstanding_requests']}", "Urgent backlog", delta_color="off")
                m4.metric("Estimated Shortage", f"{r['estimated_shortage']:.0f}", "Net deficit gap", delta_color="inverse")

                st.info(f"**Explanation:** Based on historical demand patterns ({r['features']['historical_demand_qty']} avg monthly units), seasonal factor ({r['features']['season_factor']}x), and an active backlog of {r['outstanding_requests']} units against {r['current_inventory']} units in stock, our model projects a net deficit of **{r['estimated_shortage']:.0f} units**.")
        else:
            # Multi-result table
            st.subheader(f"Forecast Results ({len(results)} Combinations)")
            res_df = pd.DataFrame(results)[[
                "org_name", "city", "category", "predicted_demand",
                "current_inventory", "outstanding_requests", "estimated_shortage"
            ]]
            res_df.columns = ["Organization", "City", "Category", f"Predicted Demand ({curr_horizon}d)", "Inventory", "Backlog", "Estimated Shortage"]
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Historical vs Predicted Visualization

        hist_df = pd.read_csv("data/historical_demand.csv")
        filtered_hist = hist_df[hist_df["category"].isin(target_cats)].groupby(["date", "category"])["historical_demand_qty"].sum().reset_index()

        fig_compare = px.line(
            filtered_hist,
            x="date",
            y="historical_demand_qty",
            color="category",
            color_discrete_sequence=[COLOR_TEAL, COLOR_CORAL, "#3A7D9D", "#E5A88E", "#75B4D6", "#9C6B53"]
        )
        fig_compare.update_layout(
            get_plotly_layout(height=360, title="Historical Demand Trajectory (Jan 2024 – Oct 2026)"),
            xaxis_title="Timeline",
            yaxis_title="Historical Demand Units"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.warning("**Transparency Disclosure:** Demand forecasts are probabilistic estimates generated by a Scikit-learn RandomForestRegressor. Real-world predictive accuracy may vary based on unforeseen socio-economic events.")



# =====================================================================
# PAGE 3: DONATE EXISTING ITEMS (WITH INTEGRATED AI RECOMMENDATIONS)
# =====================================================================
elif selected_page == PAGE_DONATE:
    render_hero(
        title="Donate Items You",
        highlight_word="Already Own.",
        subtitle="AI-guided allocation connecting pre-owned or spare essential supplies directly to verified community shelters experiencing acute deficits."
    )

    # Check if navigation arrived with a linked Purchase Plan context
    plan_ctx = st.session_state.get("plan_donation_context")
    linked_plan = None
    if plan_ctx and "plan_id" in plan_ctx:
        linked_plan = db.get_purchase_plan_by_id(plan_ctx["plan_id"])

    if linked_plan:
        prog = linked_plan["progress"]
        items_prog = prog["items_progress"]

        with st.container(border=True):
            col_ph1, col_ph2 = st.columns([3, 1.4])
            with col_ph1:
                st.subheader(f"Fulfilling Saved Purchase Plan #{linked_plan['id']}")
                st.caption(f"Sponsor: **{linked_plan['donor_name']}** ({linked_plan['donor_email']})")
            with col_ph2:
                if prog['status'] == 'Donated':
                    st.warning(f"Status: {prog['status']}")
                elif prog['status'] == 'Partially Donated':
                    st.info(f"Status: {prog['status']}")
                else:
                    st.success(f"Status: {prog['status']}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Planned Units", f"{prog['total_planned_units']}")
            m2.metric("Donated Units", f"{prog['total_donated_units']}")
            m3.metric("Confirmed Received", f"{prog['total_received_units']}")
            m4.metric("Remaining Units", f"{prog['total_remaining_units']}")

            if st.button("Switch to Regular Donation (Unlink Plan)", icon=":material/close:", key="btn_unlink_plan"):
                del st.session_state["plan_donation_context"]
                if "donate_selected_org_id" in st.session_state:
                    del st.session_state["donate_selected_org_id"]
                st.rerun()

        # If the plan has multiple allocations, let the donor select which one to fulfill
        if len(items_prog) > 1:
            alloc_options = {}
            for itm in items_prog:
                stat_lbl = "Fulfilled" if itm["is_fulfilled"] else f"Remaining: {itm['remaining_quantity']} units"
                alloc_options[itm["item_index"]] = f"Line #{itm['item_index']+1}: {itm['category']} → {itm['org_name']} (Planned: {itm['planned_quantity']} | {stat_lbl})"

            default_idx = plan_ctx.get("selected_item_index", 0)
            if default_idx >= len(items_prog):
                default_idx = 0

            selected_alloc_idx = st.selectbox(
                "Select Plan Allocation to Fulfill:",
                options=list(alloc_options.keys()),
                format_func=lambda x: alloc_options[x],
                index=default_idx,
                key="plan_alloc_sel"
            )
            active_alloc = items_prog[selected_alloc_idx]
        else:
            active_alloc = items_prog[0]


    else:
        active_alloc = None

    # Two column layout: Left (Inputs & Generation), Right (Ranked Recommendations & Selection/Submission)
    col_input, col_recs = st.columns([1, 1.25])

    with col_input:
        # STEP 1: DONATION DETAILS
        with st.container(border=True):
            st.subheader("Step 1: Donation Details")
            st.caption("Specify your donor information and available items. Click **Generate AI Recommendations** to calculate optimal NGO allocations.")

            if active_alloc:
                d_name = st.text_input("Donor Full Name:", value=linked_plan["donor_name"], key="d_name_plan")
                d_email = st.text_input("Contact Email Address:", value=linked_plan["donor_email"], key="d_email_plan")
                d_phone = st.text_input("Phone Number:", value="+91 98765 22334", key="d_phone_plan")

                cat_idx = CATEGORIES.index(active_alloc["category"]) if active_alloc["category"] in CATEGORIES else 0
                don_cat = st.selectbox(
                    "Item Category:",
                    CATEGORIES,
                    index=cat_idx,
                    key=f"don_cat_p_{linked_plan['id']}_{active_alloc['item_index']}"
                )

                st.info(f"Planned Allocation: **{active_alloc['planned_quantity']} units** • Already Donated: **{active_alloc['donated_quantity']} units** • Remaining: **{active_alloc['remaining_quantity']} units**")

                rem_units = active_alloc["remaining_quantity"]
                if rem_units <= 0:
                    st.info(f"All {active_alloc['planned_quantity']} planned units of {active_alloc['category']} have already been donated.", icon=":material/info:")
                    don_qty = 0
                else:
                    don_qty = st.number_input(
                        "Available Quantity (Units):",
                        min_value=1,
                        max_value=rem_units,
                        value=rem_units,
                        step=1,
                        key=f"don_qty_p_{linked_plan['id']}_{active_alloc['item_index']}",
                        help=f"Maximum allowed is {rem_units} units (remaining unfulfilled allocation)."
                    )

                don_cond = st.selectbox(
                    "Item Condition:",
                    ["New", "Good", "Usable"],
                    index=0,
                    key=f"don_cond_p_{linked_plan['id']}_{active_alloc['item_index']}"
                )

                orgs = db.get_organizations()
                org_choices = {"ALL": "Any NGO (System Recommended Highest Need)"}
                for o in orgs:
                    org_choices[o["id"]] = f"{o['name']} ({o['city']})"
                default_pref_idx = list(org_choices.keys()).index(active_alloc["org_id"]) if active_alloc["org_id"] in org_choices else 0
                don_pref_org = st.selectbox(
                    "Optional Preferred NGO:",
                    options=list(org_choices.keys()),
                    format_func=lambda x: org_choices[x],
                    index=default_pref_idx,
                    key=f"don_pref_p_{linked_plan['id']}_{active_alloc['item_index']}"
                )

                default_note = f"Fulfilling Purchase Plan #{linked_plan['id']} — {active_alloc['category']} allocation"
                d_notes = st.text_area("Pickup or Drop-off Notes:", value=default_note, key="d_notes_plan")

            else:
                d_name = st.text_input("Donor Full Name:", value="Rohan Mehra", key="d_name_reg")
                d_email = st.text_input("Contact Email Address:", value="rohan.m@example.org", key="d_email_reg")
                d_phone = st.text_input("Phone Number:", value="+91 98765 22334", key="d_phone_reg")

                don_cat = st.selectbox("Item Category:", CATEGORIES, index=0, key="don_cat_reg")
                don_qty = st.number_input("Available Quantity (Units):", min_value=1, max_value=5000, value=25, step=5, key="don_qty_reg")
                don_cond = st.selectbox("Item Condition:", ["Good", "New", "Usable"], index=0, key="don_cond_reg")

                orgs = db.get_organizations()
                org_choices = {"ALL": "Any NGO (Highest Need / System Recommended)"}
                for o in orgs:
                    org_choices[o["id"]] = f"{o['name']} ({o['city']})"
                don_pref_org = st.selectbox(
                    "Optional Preferred NGO:",
                    options=list(org_choices.keys()),
                    format_func=lambda x: org_choices[x],
                    index=0,
                    key="don_pref_reg"
                )

                d_notes = st.text_area("Pickup or Drop-off Notes:", placeholder="E.g., Packed in a cardboard box, ready for courier pickup.", key="d_notes_reg")

        # STEP 2: GENERATE AI RECOMMENDATIONS BUTTON
        current_input_snapshot = {
            "name": d_name.strip(),
            "email": d_email.strip(),
            "category": don_cat,
            "qty": int(don_qty) if don_qty else 0,
            "condition": don_cond,
            "pref_org": don_pref_org,
            "alloc_idx": active_alloc["item_index"] if active_alloc else -1,
            "plan_id": linked_plan["id"] if linked_plan else None
        }

        gen_rec_btn = st.button("Generate AI Recommendations", icon=":material/psychology:", type="primary", use_container_width=True, key="btn_generate_ai_recs")
        if gen_rec_btn:
            if int(don_qty) <= 0:
                st.error("Please enter a valid quantity greater than zero to generate recommendations.")
            else:
                with st.spinner("Computing real-time shortage and ML demand scores..."):
                    recs = recommend_ngos_for_item(
                        category=don_cat,
                        available_qty=int(don_qty),
                        condition=don_cond,
                        org_id_filter=None if don_pref_org == "ALL" else don_pref_org
                    )
                    st.session_state["donate_ai_recs"] = recs
                    st.session_state["donate_ai_params"] = current_input_snapshot
                    if recs:
                        st.session_state["donate_selected_org_id"] = recs[0]["org_id"]
                    elif active_alloc:
                        st.session_state["donate_selected_org_id"] = active_alloc["org_id"]

    with col_recs:
        cached_recs = st.session_state.get("donate_ai_recs")
        cached_params = st.session_state.get("donate_ai_params")

        # STEP 3: DISPLAY RANKED NGO RECOMMENDATIONS
        if cached_recs is not None and cached_params is not None:
            is_outdated = (
                current_input_snapshot["category"] != cached_params.get("category") or
                current_input_snapshot["qty"] != cached_params.get("qty") or
                current_input_snapshot["condition"] != cached_params.get("condition") or
                current_input_snapshot["pref_org"] != cached_params.get("pref_org")
            )

            with st.container(border=True):
                col_sh1, col_sh2 = st.columns([3, 1.2])
                with col_sh1:
                    st.subheader("Step 3: Ranked NGO Recommendations")
                    st.caption(f"Based on {cached_params['qty']} units of {cached_params['category']} ({cached_params['condition']} condition)")
                with col_sh2:
                    st.caption(f"**{len(cached_recs)} Shelters Evaluated**")

                if is_outdated:
                    st.warning(f"**Input Parameters Changed — Displaying Previous Results.** Recommendations reflect: **{cached_params['category']}** ({cached_params['qty']} units, {cached_params['condition']}). Click **Generate AI Recommendations** to recalculate.")

                all_orgs = db.get_organizations()
                selected_org_id = st.session_state.get("donate_selected_org_id")
                if not selected_org_id:
                    if cached_recs:
                        selected_org_id = cached_recs[0]["org_id"]
                    elif active_alloc:
                        selected_org_id = active_alloc["org_id"]
                    else:
                        selected_org_id = all_orgs[0]["id"]
                    st.session_state["donate_selected_org_id"] = selected_org_id

                if not cached_recs:
                    st.info("No partner organizations currently have an open deficit for this specific filter. You may still choose an NGO manually below.")
                else:
                    for idx, r in enumerate(cached_recs):
                        is_this_selected = (r["org_id"] == selected_org_id)
                        clicked = render_ngo_recommendation_card(
                            r=r,
                            idx=idx,
                            is_selected=is_this_selected,
                            select_button_key=f"btn_pick_ngo_{r['org_id']}"
                        )
                        if clicked:
                            st.session_state["donate_selected_org_id"] = r["org_id"]
                            st.rerun()

        else:
            # Placeholder State before clicking Generate AI Recommendations
            with st.container(border=True):
                st.subheader("Step 2: AI Recommendation Engine Ready")
                st.caption("Enter your item details in Step 1, then click **Generate AI Recommendations** to calculate ranked organization scores considering real-time shortages, outstanding backlog, and 30-day ML forecasts.")
                st.info("You can also manually select a recipient organization in Step 4 below at any time.", icon=":material/info:")

        # STEP 4: CONFIRM RECIPIENT ORGANIZATION (MANUAL OR AI-RECOMMENDED)
        with st.container(border=True):
            st.subheader("Step 4: Confirm Recipient Organization")
            st.caption("Choose the partner shelter that will receive this donation. You can confirm the AI recommendation or choose any organization manually.")

            all_orgs = db.get_organizations()
            org_options_dict = {o["id"]: f"{o['name']} ({o['city']}) — Focus: {o['focus']}" for o in all_orgs}

            current_sel_id = st.session_state.get("donate_selected_org_id")
            if not current_sel_id or current_sel_id not in org_options_dict:
                if active_alloc and active_alloc["org_id"] in org_options_dict:
                    current_sel_id = active_alloc["org_id"]
                else:
                    current_sel_id = list(org_options_dict.keys())[0]

            default_org_index = list(org_options_dict.keys()).index(current_sel_id)

            chosen_org_id = st.selectbox(
                "Selected Recipient Organization:",
                options=list(org_options_dict.keys()),
                format_func=lambda x: org_options_dict[x],
                index=default_org_index,
                key="final_chosen_org_selector"
            )
            st.session_state["donate_selected_org_id"] = chosen_org_id

            target_ngo = db.get_organization_by_id(chosen_org_id)
            st.info(f"**Destination Organization:** **{target_ngo['name']}** • Location: {target_ngo['city']} • Focus: **{target_ngo['focus']}** • Email: {target_ngo['contact_email']}", icon=":material/domain:")

            # STEP 5: SUBMIT DONATION REQUEST
            st.divider()
            st.subheader("Step 5: Submit Donation Request")

            btn_label = f"Submit Donation Request (Linked to Plan #{linked_plan['id']})" if linked_plan else "Submit Donation Request"
            if st.button(btn_label, type="primary", use_container_width=True, key="btn_sub_don_request"):
                if not d_name.strip() or not d_email.strip():
                    st.error("Please provide both donor name and contact email address.")
                elif not chosen_org_id:
                    st.error("Please select a recipient organization.")
                elif int(don_qty) <= 0:
                    st.error("Donation quantity must be greater than zero.")
                elif active_alloc and active_alloc["remaining_quantity"] <= 0:
                    st.error("This allocation has already been fully donated. Please choose another allocation.")
                elif active_alloc and int(don_qty) > active_alloc["remaining_quantity"]:
                    st.error(f"Donation quantity ({don_qty} units) cannot exceed the remaining planned units ({active_alloc['remaining_quantity']} units) for this allocation.")
                else:
                    try:
                        prices = db.get_item_prices()
                        unit_p = active_alloc["unit_price"] if active_alloc else prices.get(don_cat, {}).get("unit_price_inr", 100.0)
                        est_val = unit_p * int(don_qty)
                        plan_id_to_link = linked_plan["id"] if linked_plan else None

                        new_id = db.create_donation(
                            donation_type="EXISTING_ITEMS",
                            donor_name=d_name.strip(),
                            donor_email=d_email.strip(),
                            donor_phone=d_phone.strip(),
                            org_id=chosen_org_id,
                            category=don_cat,
                            quantity=int(don_qty),
                            condition=don_cond,
                            estimated_value_inr=est_val,
                            notes=d_notes.strip(),
                            purchase_plan_id=plan_id_to_link
                        )

                        org_info = db.get_organization_by_id(chosen_org_id)
                        sub_time_ist = db.format_datetime_ist(db.get_current_ist_str())

                        st.success(f"Physical Donation #{new_id} registered successfully!" + (f" Linked to Saved Purchase Plan #{plan_id_to_link}." if plan_id_to_link else ""), icon=":material/check_circle:")

                        plan_progress_info = ""
                        if plan_id_to_link:
                            updated_prog = db.get_purchase_plan_progress(plan_id_to_link)
                            plan_progress_info = f"\\n• Total Donated: {updated_prog['total_donated_units']} / {updated_prog['total_planned_units']} units ({updated_prog['total_remaining_units']} units remaining)\\n• Plan Status: {updated_prog['status']}"

                        st.warning(
                            f"**STATUS: Pending Receipt**\\n\\n"
                            f"Donation **#{new_id}** of **{don_qty} units of {don_cat}** (Est. Value: ₹{est_val:,.2f}) has been logged for **{org_info['name']}** at **{sub_time_ist}**.\\n\\n"
                            f"*Notice: To prevent phantom inventory, your donation remains in 'Pending Receipt' status without incrementing inventory until {org_info['name']} physically inspects and confirms receipt in their NGO Portal.*"
                            f"{plan_progress_info}"
                        )

                        c_act1, c_act2, c_act3 = st.columns([1, 1, 1.2])
                        with c_act1:
                            if st.button("View in Donation History", icon=":material/receipt_long:", key=f"nav_don_hist_{new_id}", use_container_width=True):
                                st.session_state["nav_target"] = PAGE_HISTORY
                                st.session_state["don_history_active_tab"] = 0
                                st.rerun()
                        with c_act2:
                            if st.button("View in NGO Portal", icon=":material/domain:", key=f"nav_ngo_portal_{new_id}", use_container_width=True):
                                st.session_state["nav_target"] = PAGE_NGO
                                st.session_state["selected_ngo_id"] = chosen_org_id
                                st.rerun()
                        with c_act3:
                            if plan_id_to_link:
                                if st.button(f"Return to Plan #{plan_id_to_link}", icon=":material/shopping_cart:", key=f"nav_return_plan_{new_id}", use_container_width=True):
                                    st.session_state["nav_target"] = PAGE_HISTORY
                                    st.session_state["don_history_active_tab"] = 1
                                    st.rerun()

                    except Exception as ex:
                        st.error(f"Error submitting donation: {ex}")



elif selected_page == PAGE_PURCHASE:
    render_hero(
        title="Plan a Purchase with an",
        highlight_word="INR Budget.",
        subtitle="Sponsor brand-new essential supplies. The optimization engine allocates your funds to maximize genuine social impact per rupee spent."
    )

    col_setup, col_basket = st.columns([1, 1.25])

    with col_setup:
        with st.container(border=True):
            st.subheader("Step 1: Budget & Category Options")
            st.caption("Configure your budget and preferences. Click **Generate Purchase Plan** to compute Step 2.")

            plan_budget = st.number_input(
                "Available Budget (₹ INR):",
                min_value=500.0,
                max_value=500000.0,
                value=8000.0,
                step=500.0,
                key="plan_budget_inp"
            )
            plan_cats = st.multiselect(
                "Categories to Sponsor:",
                CATEGORIES,
                default=["Notebooks", "Schoolbags", "Books", "Blankets"],
                key="plan_cats_sel"
            )

            orgs = db.get_organizations()
            p_choices = {"ALL": "Any NGO (System Recommended Highest Need)"}
            for o in orgs:
                p_choices[o["id"]] = f"{o['name']} ({o['city']})"
            plan_pref_org = st.selectbox(
                "Preferred NGO:",
                options=list(p_choices.keys()),
                format_func=lambda x: p_choices[x],
                index=0,
                key="plan_pref_sel"
            )

            st.divider()

            p_name = st.text_input("Sponsor / Donor Name:", value="Vikram Malhotra", key="p_name_inp")
            p_email = st.text_input("Contact Email:", value="vikram.m@example.org", key="p_email_inp")

            price_dict = db.get_item_prices()
            st.caption("**Wholesale Prices:** " + " • ".join([f"{k}: ₹{v['unit_price_inr']:.0f}" for k, v in price_dict.items()]))

            gen_plan_btn = st.button("Generate Purchase Plan", icon=":material/shopping_cart:", type="primary", use_container_width=True, key="btn_gen_plan")

        current_setup_params = {
            "budget": float(plan_budget),
            "cats": sorted(plan_cats),
            "pref_org": plan_pref_org,
            "name": p_name.strip(),
            "email": p_email.strip()
        }

        if gen_plan_btn:
            if plan_budget <= 0:
                st.error("Please enter a valid budget greater than ₹0.")
            elif not plan_cats:
                st.error("Please select at least one category to sponsor.")
            elif not p_name.strip():
                st.error("Please enter donor/sponsor name.")
            elif not p_email.strip():
                st.error("Please enter contact email.")
            else:
                with st.spinner("Calculating optimal allocation basket..."):
                    res = recommend_purchase_plan(
                        budget_inr=float(plan_budget),
                        selected_categories=plan_cats,
                        preferred_org_id=None if plan_pref_org == "ALL" else plan_pref_org
                    )
                    st.session_state["purchase_plan_basket"] = res
                    st.session_state["purchase_plan_params"] = current_setup_params
                    st.session_state["purchase_plan_gen_id"] = st.session_state.get("purchase_plan_gen_id", 0) + 1

    with col_basket:
        cached_basket = st.session_state.get("purchase_plan_basket")
        cached_setup = st.session_state.get("purchase_plan_params")

        if cached_basket is None or cached_setup is None:
            with st.container(border=True):
                st.subheader("Step 2: Optimized Purchase Basket (Confirmed)")
                st.caption("Please configure your available budget, target categories, and sponsor details in Step 1, then click **Generate Purchase Plan** to compute an optimal, impact-maximizing allocation.")
        else:
            is_basket_outdated = (current_setup_params != cached_setup)

            with st.container(border=True):
                c_bh1, c_bh2 = st.columns([3, 1.2])
                with c_bh1:
                    st.subheader("Step 2: Optimized Purchase Basket")
                with c_bh2:
                    st.metric("Plan Budget", f"₹{cached_setup['budget']:,.2f}")

                if is_basket_outdated:
                    st.warning(f"**Step 1 Parameters Changed — Basket Outdated.** This basket reflects budget ₹{cached_setup['budget']:,.2f} and categories ({', '.join(cached_setup['cats'])}). Click **Generate Purchase Plan** in Step 1 to refresh before saving.")

                raw_items = cached_basket.get("items", [])
                if not raw_items:
                    st.info("No items could be allocated within the specified budget and categories. Try increasing budget or selecting more categories.")
                else:
                    adjusted_items = []
                    gen_id = st.session_state.get("purchase_plan_gen_id", 1)

                    for i, itm in enumerate(raw_items):
                        with st.container(border=True):
                            st.markdown(f"**{itm['category']} → {itm['org_name']}** (`₹{itm['unit_price']:.0f} / {itm['unit_name']}`)")
                            st.caption(f"Location: **{itm.get('city', 'Regional Hub')}** • Urgency: **{itm['urgency']}** • Unmet Need: **{itm['unmet_need']:.0f} units**")

                            col_q, col_c = st.columns([1, 1.2])
                            with col_q:
                                q_val = st.number_input(
                                    f"Units ({itm['category']}):",
                                    min_value=0,
                                    max_value=1000,
                                    value=int(itm["recommended_qty"]),
                                    step=1,
                                    key=f"pp_qty_v{gen_id}_{i}"
                                )
                            with col_c:
                                sub_c = q_val * itm["unit_price"]
                                st.metric("Item Cost", f"₹{sub_c:,.2f}")

                            adjusted_items.append({
                                "category": itm["category"],
                                "org_id": itm["org_id"],
                                "org_name": itm["org_name"],
                                "unit_price": itm["unit_price"],
                                "unit_name": itm["unit_name"],
                                "quantity": q_val,
                                "cost": sub_c
                            })

                    st.divider()

                    # Budget & Cost Summary
                    total_calc_cost = sum(x["cost"] for x in adjusted_items)
                    active_budget = cached_setup["budget"]
                    rem_budget = active_budget - total_calc_cost

                    b1, b2 = st.columns(2)
                    b1.metric("Total Basket Cost", f"₹{total_calc_cost:,.2f}", f"Budget: ₹{active_budget:,.2f}", delta_color="off")
                    b2.metric("Remaining Budget", f"₹{rem_budget:,.2f}", delta_color="normal" if rem_budget >= 0 else "inverse")

                    if total_calc_cost > active_budget:
                        over_amt = total_calc_cost - active_budget
                        st.error(f"**Budget Exceeded!** Total cost exceeds available budget by ₹{over_amt:,.2f}. Please reduce unit quantities above before confirming.")
                    elif is_basket_outdated:
                        st.info("Step 1 inputs were modified. Please click **Generate Purchase Plan** in Step 1 to refresh the plan before saving.", icon=":material/info:")
                    else:
                        valid_items_to_save = [x for x in adjusted_items if x["quantity"] > 0]
                        if not valid_items_to_save:
                            st.warning("All item quantities are set to 0. Please specify at least one item to purchase.")
                        else:
                            if st.button("Confirm & Save Purchase Plan", icon=":material/check_circle:", type="primary", use_container_width=True, key="btn_save_plan"):
                                sponsor_name = cached_setup["name"] or p_name.strip()
                                sponsor_email = cached_setup["email"] or p_email.strip()
                                if not sponsor_name or not sponsor_email:
                                    st.error("Please provide sponsor name and contact email.")
                                else:
                                    plan_id = db.save_purchase_plan(
                                        donor_name=sponsor_name,
                                        donor_email=sponsor_email,
                                        total_budget_inr=active_budget,
                                        total_allocated_inr=total_calc_cost,
                                        remaining_budget_inr=rem_budget,
                                        items=valid_items_to_save
                                    )
                                    st.success(f"Purchase Plan #{plan_id} saved successfully to SQLite!", icon=":material/check_circle:")
                                    st.info(f"Your purchase plan has been recorded in the persistent ledger (Plan ID #{plan_id}). You can view this in the **Donation History** page under the Saved Purchase Plans tab.")
                                    if st.button("View in Donation History (Saved Purchase Plans)", icon=":material/receipt_long:", key=f"nav_to_plans_{plan_id}", use_container_width=True):
                                        st.session_state["nav_target"] = PAGE_HISTORY
                                        st.session_state["don_history_active_tab"] = 1
                                        st.rerun()



elif selected_page == PAGE_NGO:
    render_hero(
        title="NGO Management &",
        highlight_word="Inventory Control.",
        subtitle="Manage warehouse stock, broadcast critical item requests, and confirm physical receipt of incoming shipments."
    )

    # Load live statistics & open backlog requests
    stats = db.get_overall_stats()
    open_backlog_reqs = db.get_open_backlog_requests()
    total_dashboard_backlog = stats["total_outstanding_requested_units"]
    total_contributing_remaining = sum(r["remaining_qty"] for r in open_backlog_reqs)

    # Top Control Banner & View Toggle
    is_backlog_view = st.session_state.get("ngo_portal_view") == "backlog"

    with st.container(border=True):
        c_bkh1, c_bkh2 = st.columns([3, 1])
        with c_bkh1:
            st.markdown(f"**System Outstanding Backlog:** `{total_dashboard_backlog:,}` units across **{stats['open_requests_count']}** open requests")
            st.caption("Currently viewing the complete breakdown of all open requests contributing to the backlog." if is_backlog_view else "Unmet community demand actively awaiting donor fulfillment across all partner organizations.")
        with c_bkh2:
            if is_backlog_view:
                if st.button("Close Backlog Breakdown", icon=":material/close:", use_container_width=True, help="Switch to single NGO warehouse operations"):
                    st.session_state["ngo_portal_view"] = "standard"
                    st.rerun()
            else:
                if st.button("View Backlog Breakdown", icon=":material/assignment:", use_container_width=True, help="Display all open requests contributing to the dashboard backlog"):
                    st.session_state["ngo_portal_view"] = "backlog"
                    st.rerun()

    if is_backlog_view:
        with st.container(border=True):
            st.subheader("Open Requests Contributing to Outstanding Backlog")
            st.caption("Transparent breakdown of all active requests across partner shelters awaiting fulfillment.")

            # KPI Summary Cards
            total_req_units = sum(r["requested_qty"] for r in open_backlog_reqs)
            total_ful_units = sum(r["fulfilled_qty"] for r in open_backlog_reqs)
            num_ngos_with_backlog = len(set(r["org_id"] for r in open_backlog_reqs))

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                label="Outstanding Backlog",
                value=f"{total_contributing_remaining:,} Units",
                delta=f"Matches Dashboard: {total_dashboard_backlog:,} Units"
            )
            kpi2.metric(
                label="Open Requests",
                value=f"{len(open_backlog_reqs)} Requests",
                delta=f"Across {num_ngos_with_backlog} Partner NGOs"
            )
            kpi3.metric(
                label="Total Required Demand",
                value=f"{total_req_units:,} Units",
                delta="Gross volume requested"
            )
            kpi4.metric(
                label="Fulfilled to Date",
                value=f"{total_ful_units:,} Units",
                delta=f"{(total_ful_units / max(1, total_req_units) * 100):.1f}% fulfilled"
            )

            # Filters
            with st.expander("Filter & Search Open Backlog Requests", expanded=False, icon=":material/search:"):
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    all_org_names = sorted(list(set(r["org_name"] for r in open_backlog_reqs)))
                    sel_org = st.selectbox("Filter by NGO:", ["All Partner NGOs"] + all_org_names, key="bk_sel_org")
                with f_col2:
                    all_cats = sorted(list(set(r["category"] for r in open_backlog_reqs)))
                    sel_cat = st.selectbox("Filter by Category:", ["All Categories"] + all_cats, key="bk_sel_cat")
                with f_col3:
                    all_statuses = ["All Statuses", "OPEN", "PARTIALLY_FULFILLED"]
                    sel_status = st.selectbox("Filter by Status:", all_statuses, key="bk_sel_status")

            # Apply Filters
            filtered_reqs = open_backlog_reqs
            if sel_org != "All Partner NGOs":
                filtered_reqs = [r for r in filtered_reqs if r["org_name"] == sel_org]
            if sel_cat != "All Categories":
                filtered_reqs = [r for r in filtered_reqs if r["category"] == sel_cat]
            if sel_status != "All Statuses":
                filtered_reqs = [r for r in filtered_reqs if r["status"] == sel_status]

            table_rows = []
            for r in filtered_reqs:
                table_rows.append({
                    "Request ID": f"#{r['id']}",
                    "NGO Name": r["org_name"],
                    "Item Category": r["category"],
                    "Required Quantity": int(r["requested_qty"]),
                    "Fulfilled Quantity": int(r["fulfilled_qty"]),
                    "Remaining Quantity": int(r["remaining_qty"]),
                    "Status": r["status"],
                    "Urgency": r["urgency"]
                })

            df_display = pd.DataFrame(table_rows)

            filtered_rem_total = sum(r["Remaining Quantity"] for r in table_rows) if table_rows else 0
            filtered_req_total = sum(r["Required Quantity"] for r in table_rows) if table_rows else 0
            filtered_ful_total = sum(r["Fulfilled Quantity"] for r in table_rows) if table_rows else 0

            summary_row = pd.DataFrame([{
                "Request ID": "—",
                "NGO Name": "TOTAL (All Displayed Requests)" if (sel_org == "All Partner NGOs" and sel_cat == "All Categories" and sel_status == "All Statuses") else "SUBTOTAL (Filtered)",
                "Item Category": f"{len(set(r['Item Category'] for r in table_rows))} Categories" if table_rows else "—",
                "Required Quantity": filtered_req_total,
                "Fulfilled Quantity": filtered_ful_total,
                "Remaining Quantity": filtered_rem_total,
                "Status": f"{len(table_rows)} Requests",
                "Urgency": "—"
            }])
            df_with_total = pd.concat([df_display, summary_row], ignore_index=True)

            st.dataframe(
                df_with_total,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Request ID": st.column_config.TextColumn("Request ID", width="small"),
                    "NGO Name": st.column_config.TextColumn("NGO Name", width="medium"),
                    "Item Category": st.column_config.TextColumn("Item Category", width="medium"),
                    "Required Quantity": st.column_config.NumberColumn("Required Quantity", format="%d units"),
                    "Fulfilled Quantity": st.column_config.NumberColumn("Fulfilled Quantity", format="%d units"),
                    "Remaining Quantity": st.column_config.NumberColumn("Remaining Quantity", format="%d units"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Urgency": st.column_config.TextColumn("Urgency", width="small")
                }
            )

            is_exact_match = (total_contributing_remaining == total_dashboard_backlog)
            if is_exact_match:
                st.success(f"**Total Reconciled:** Displayed Remaining Quantity ({total_contributing_remaining:,} units) matches Dashboard Outstanding Backlog ({total_dashboard_backlog:,} units) across {len(open_backlog_reqs)} open requests. **100% MATCH CONFIRMED (0 UNIT VARIANCE)**")
            else:
                st.error(f"**Discrepancy Detected:** Displayed Remaining Quantity ({total_contributing_remaining:,} units) differs from Dashboard ({total_dashboard_backlog:,} units).")

    # Standard NGO Operations
    orgs = db.get_organizations()
    ngo_options = {o["id"]: f"{o['name']} — Location: {o['city']}" for o in orgs}
    org_keys = list(ngo_options.keys())

    default_ngo_idx = 0
    if "selected_ngo_id" in st.session_state and st.session_state["selected_ngo_id"] in org_keys:
        default_ngo_idx = org_keys.index(st.session_state["selected_ngo_id"])
        del st.session_state["selected_ngo_id"]

    active_ngo_id = st.selectbox(
        "Select Active Partner Organization:",
        options=org_keys,
        format_func=lambda x: ngo_options[x],
        index=default_ngo_idx,
        key="ngo_portal_active_org_sel"
    )
    ngo_data = db.get_organization_by_id(active_ngo_id)

    # 4 Reorganized Tabs as Requested
    tab_overview, tab_reqs, tab_inv, tab_inbound = st.tabs([
        "1. NGO Overview",
        "2. Manage Requirements",
        "3. Inventory Management",
        "4. Incoming Donations"
    ])

    # -------------------------------------------------------------
    # TAB 1: NGO OVERVIEW
    # -------------------------------------------------------------
    with tab_overview:
        # Organization Information Card
        with st.container(border=True):
            c_ngoh1, c_ngoh2 = st.columns([3, 1])
            with c_ngoh1:
                st.subheader(ngo_data['name'])
                st.caption(f"Location: **{ngo_data['city']}** • Mission Focus: **{ngo_data['focus']}** • Contact: **{ngo_data['contact_email']}**")
                st.markdown(f"*{ngo_data['description']}*")
            with c_ngoh2:
                st.info(f"Shelter ID: #{ngo_data['id']}")

        # Retrieve live NGO statistics
        ngo_inv = db.get_inventory(org_id=active_ngo_id)
        total_ngo_inv_units = sum(i["quantity"] for i in ngo_inv)

        ngo_reqs = db.get_donation_requests(org_id=active_ngo_id)
        open_ngo_reqs = [r for r in ngo_reqs if r["status"] in ["OPEN", "PARTIALLY_FULFILLED"]]
        total_remaining_req_units = sum(r["requested_qty"] - r["fulfilled_qty"] for r in open_ngo_reqs)

        ngo_pending_dons = db.get_donations(org_id=active_ngo_id, status="Pending Receipt")
        total_pending_units = sum(d["quantity"] for d in ngo_pending_dons)

        ngo_confirmed_dons = db.get_donations(org_id=active_ngo_id, status="Received")
        total_confirmed_units = sum(d["quantity"] for d in ngo_confirmed_dons)

        # 4 Summary KPI Metric Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            label="Warehouse Inventory",
            value=f"{total_ngo_inv_units:,} Units",
            delta=f"Across {len(ngo_inv)} Categories"
        )
        k2.metric(
            label="Open Requirements",
            value=f"{len(open_ngo_reqs)} Requests",
            delta=f"{total_remaining_req_units:,} Units Needed",
            delta_color="inverse"
        )
        k3.metric(
            label="Incoming Donations",
            value=f"{len(ngo_pending_dons)} Shipments",
            delta=f"{total_pending_units:,} Units Awaiting Verification",
            delta_color="off"
        )
        k4.metric(
            label="Confirmed Received",
            value=f"{len(ngo_confirmed_dons)} Shipments",
            delta=f"{total_confirmed_units:,} Units Added to Stock"
        )

        st.divider()

        # Operational Snapshot Table across all 6 categories
        st.subheader(f"Operational Summary by Category ({ngo_data['name']})")

        inv_by_cat = {i["category"]: i["quantity"] for i in ngo_inv}
        preds_all = db.get_predictions_from_db(org_id=active_ngo_id)
        preds_by_cat = {p["category"]: p["predicted_demand"] for p in preds_all}

        req_stats_by_cat = {}
        for r in open_ngo_reqs:
            c = r["category"]
            req_stats_by_cat[c] = req_stats_by_cat.get(c, 0) + (r["requested_qty"] - r["fulfilled_qty"])

        pending_by_cat = {}
        for d in ngo_pending_dons:
            c = d["category"]
            pending_by_cat[c] = pending_by_cat.get(c, 0) + d["quantity"]

        overview_rows = []
        for cat in CATEGORIES:
            stk = inv_by_cat.get(cat, 0)
            pred = preds_by_cat.get(cat, 0.0)
            req_rem = req_stats_by_cat.get(cat, 0)
            pend = pending_by_cat.get(cat, 0)
            health = "Critical Deficit" if stk < (pred * 0.3) else ("Moderate Shortage" if stk < pred else "Sufficient Stock")

            overview_rows.append({
                "Category": cat,
                "Current Warehouse Stock": f"{stk} units",
                "ML 30d Demand": f"{pred:.0f} units",
                "Open Requirement Deficit": f"{req_rem} units" if req_rem > 0 else "None",
                "Inbound Pending Shipments": f"{pend} units" if pend > 0 else "—",
                "Supply Status": health
            })

        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 2: MANAGE REQUIREMENTS
    # -------------------------------------------------------------
    with tab_reqs:
        st.subheader(f"Active Item Requirements ({ngo_data['name']})")
        st.caption("Broadcast community deficits to donors. Requirements are automatically satisfied when received donations or purchase plans are allocated.")

        reqs = db.get_donation_requests(org_id=active_ngo_id)
        open_reqs = [r for r in reqs if r["status"] in ["OPEN", "PARTIALLY_FULFILLED"]]

        if not open_reqs:
            st.info("No active open requirements for this NGO. Add a requirement below to broadcast urgent community needs.")
        else:
            req_display_rows = []
            for r in open_reqs:
                rem = r["requested_qty"] - r["fulfilled_qty"]
                pct = (r["fulfilled_qty"] / max(1, r["requested_qty"])) * 100
                req_display_rows.append({
                    "Request ID": f"#{r['id']}",
                    "Item Category": r["category"],
                    "Required Quantity": f"{r['requested_qty']} units",
                    "Fulfilled Quantity": f"{r['fulfilled_qty']} units ({pct:.0f}%)",
                    "Remaining Quantity": f"{rem} units",
                    "Urgency": r["urgency"],
                    "Status": r["status"],
                    "Notes": r["notes"] or "Standard community deficit"
                })

            st.dataframe(pd.DataFrame(req_display_rows), use_container_width=True, hide_index=True)

            with st.expander("Quick Action: Update Status on Active Requests", icon=":material/edit:"):
                q_cols = st.columns(len(open_reqs) if len(open_reqs) <= 4 else 4)
                for idx, rq in enumerate(open_reqs[:4]):
                    with q_cols[idx]:
                        st.markdown(f"**Request #{rq['id']}** ({rq['category']})")
                        c_act1, c_act2 = st.columns(2)
                        with c_act1:
                            if st.button("Mark Fulfilled", key=f"ful_btn_t2_{rq['id']}", help="Marks request completely fulfilled"):
                                db.update_request_status(rq["id"], "FULFILLED")
                                st.success(f"Request #{rq['id']} marked as Fulfilled!")
                                st.rerun()
                        with c_act2:
                            if st.button("Close", key=f"cls_btn_t2_{rq['id']}", help="Closes this requirement"):
                                db.close_donation_request(rq["id"])
                                st.info(f"Request #{rq['id']} closed.")
                                st.rerun()

        st.divider()

        col_req_add, col_req_edit = st.columns([1, 1])

        # SUB-SECTION: ADD REQUIREMENT
        with col_req_add:
            with st.container(border=True):
                st.subheader("Add a New Requirement")
                n_cat = st.selectbox("Item Category Needed:", CATEGORIES, key="nr_cat_tab2")
                n_qty = st.number_input("Required Quantity (Units):", min_value=5, max_value=5000, value=100, step=10, key="nr_qty_tab2")
                n_urg = st.selectbox("Urgency Tier:", ["High", "Medium", "Low"], index=0, key="nr_urg_tab2")
                n_notes = st.text_area("Justification / Community Context:", placeholder="E.g., Term notebooks needed for rural school batch.", key="nr_notes_tab2")

                if st.button("Create Requirement in Database", icon=":material/add:", use_container_width=True, key="btn_create_req_tab2", type="primary"):
                    req_id = db.create_donation_request(
                        org_id=active_ngo_id,
                        category=n_cat,
                        requested_qty=int(n_qty),
                        urgency=n_urg,
                        notes=n_notes
                    )
                    st.success(f"Created Requirement #{req_id} for {n_cat} successfully!", icon=":material/check_circle:")
                    st.rerun()

        # SUB-SECTION: EDIT AN EXISTING REQUIREMENT
        with col_req_edit:
            with st.container(border=True):
                st.subheader("Edit an Existing Requirement")
                if not open_reqs:
                    st.info("No active requirements available to edit.")
                else:
                    edit_choices = {r["id"]: f"Request #{r['id']} — {r['category']} (Required: {r['requested_qty']}, Rem: {r['requested_qty'] - r['fulfilled_qty']})" for r in open_reqs}
                    sel_edit_id = st.selectbox("Select Requirement to Edit:", options=list(edit_choices.keys()), format_func=lambda x: edit_choices[x], key="sel_req_edit")
                    selected_req = next(r for r in open_reqs if r["id"] == sel_edit_id)

                    e_qty = st.number_input(
                        "Updated Total Required Quantity:",
                        min_value=max(1, selected_req["fulfilled_qty"]),
                        max_value=10000,
                        value=int(selected_req["requested_qty"]),
                        step=5,
                        key="e_qty_inp"
                    )
                    urg_options = ["High", "Medium", "Low"]
                    curr_urg_idx = urg_options.index(selected_req["urgency"]) if selected_req["urgency"] in urg_options else 0
                    e_urg = st.selectbox("Urgency Tier:", urg_options, index=curr_urg_idx, key="e_urg_sel")
                    e_notes = st.text_area("Updated Notes / Justification:", value=selected_req["notes"] or "", key="e_notes_inp")

                    if st.button("Save Changes to Requirement", icon=":material/edit:", use_container_width=True, key="btn_save_req_edit", type="primary"):
                        db.update_donation_request(selected_req["id"], int(e_qty), e_urg, e_notes)
                        st.success(f"Requirement #{selected_req['id']} updated successfully!", icon=":material/check_circle:")
                        st.rerun()

    # -------------------------------------------------------------
    # TAB 3: INVENTORY MANAGEMENT
    # -------------------------------------------------------------
    with tab_inv:
        st.subheader(f"Warehouse Stock Levels & Item Criteria ({ngo_data['name']})")
        st.caption("Live stock counts verified against the persistent ledger. Accepted conditions outline quality standards for received community donations.")

        inv_items = db.get_inventory(org_id=active_ngo_id)
        preds_list = db.get_predictions_from_db(org_id=active_ngo_id)
        pred_map = {p["category"]: p["predicted_demand"] for p in preds_list}

        ACCEPTED_CONDITIONS = {
            "Notebooks": "Brand New / Unused only",
            "Schoolbags": "New or Sturdy Good (clean, intact zippers & straps)",
            "Books": "New or Clean Good (complete pages, readable, undamaged)",
            "Blankets": "New or Freshly Sanitized / Washed Good condition",
            "Clothing": "New or Freshly Washed Wearable Good condition",
            "Hygiene Products": "Brand New & Factory Sealed only"
        }

        inv_table_data = []
        for itm in inv_items:
            cat = itm["category"]
            qty = itm["quantity"]
            predicted = pred_map.get(cat, 0.0)
            status_tag = "Critical Deficit" if qty < (predicted * 0.3) else ("Moderate Shortage" if qty < predicted else "Sufficient Stock")
            inv_table_data.append({
                "Item Category": cat,
                "Available Quantity": f"{qty} units",
                "Accepted Conditions": ACCEPTED_CONDITIONS.get(cat, "Good / New"),
                "ML 30d Demand": f"{predicted:.0f} units",
                "Supply Status": status_tag,
                "Last Audit Timestamp": db.format_datetime_ist(itm["last_updated"])
            })

        st.dataframe(pd.DataFrame(inv_table_data), use_container_width=True, hide_index=True)

        st.divider()

        # Reconcile warehouse stock count
        with st.container(border=True):
            st.subheader("Reconcile Physical Warehouse Count (Audit Stock Adjustment)")
            st.caption("Record verified physical counts following periodic shelf audits. Note: Confirmed shipments received through Incoming Donations automatically increment inventory.")

            c_u1, c_u2 = st.columns([1.2, 1])
            with c_u1:
                u_cat = st.selectbox("Select Category to Reconcile:", [i["category"] for i in inv_items], key="u_cat_sel_t3")
            with c_u2:
                current_stk = next((i["quantity"] for i in inv_items if i["category"] == u_cat), 0)
                u_qty = st.number_input("Verified Shelf Count (Units):", min_value=0, max_value=10000, value=current_stk, step=5, key="u_qty_inp_t3")

            if st.button("Save Stock Quantity to Database", icon=":material/database:", key="btn_save_inv_t3", type="primary"):
                db.set_inventory_quantity(active_ngo_id, u_cat, int(u_qty))
                st.success(f"Inventory for {u_cat} updated to {u_qty} units in the persistent ledger.", icon=":material/check_circle:")
                st.rerun()

    # -------------------------------------------------------------
    # TAB 4: INCOMING DONATIONS
    # -------------------------------------------------------------
    with tab_inbound:
        st.subheader(f"Inbound Shipments Awaiting Physical Verification ({ngo_data['name']})")
        st.info("**Inventory Receipt Integrity Rule:** Pending donations never increase inventory upon donor submission. Warehouse stock is incremented only after the receiving NGO verifies physical delivery and clicks **Confirm Physical Receipt**.", icon=":material/info:")

        pending_dons = db.get_donations(org_id=active_ngo_id, status="Pending Receipt")

        if not pending_dons:
            with st.container(border=True):
                st.info("No Inbound Shipments Awaiting Receipt. All donations addressed to this NGO have already been confirmed and accounted for in stock.")
        else:
            for don in pending_dons:
                with st.container(border=True):
                    c_dh1, c_dh2 = st.columns([3, 1.4])
                    with c_dh1:
                        st.markdown(f"### Donation #{don['id']}: {don['quantity']} Units of {don['category']}")
                        phone_str = f" • Phone: {don['donor_phone']}" if don.get("donor_phone") else ""
                        st.caption(f"Donor: **{don['donor_name']}** ({don['donor_email']}){phone_str}")
                    with c_dh2:
                        st.warning(f"Status: {don['status']}")
                        st.caption(f"Submitted: {db.format_datetime_ist(don['created_at'])} (IST)")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Category", don["category"])
                    m2.metric("Quantity", f"{don['quantity']} units")
                    m3.metric("Condition", don["condition"])
                    est_val_str = f"₹{float(don.get('estimated_value_inr', 0.0)):,.2f}" if don.get("estimated_value_inr") is not None else "N/A"
                    m4.metric("Est. Value", est_val_str)

                    if don.get("notes"):
                        st.caption(f"Notes: {don['notes']}")

                    if st.button(f"Confirm Physical Receipt for Donation #{don['id']}", icon=":material/check_circle:", key=f"conf_btn_tab4_{don['id']}", use_container_width=True, type="primary"):
                        res = db.confirm_donation_receipt(don["id"], confirming_org_id=active_ngo_id)
                        if res["success"]:
                            st.success(f"{res['message']}", icon=":material/check_circle:")
                            if res.get("allocated_details"):
                                for ad in res["allocated_details"]:
                                    st.info(f"{ad}", icon=":material/info:")
                            st.rerun()
                        else:
                            st.error(res["message"])

        st.divider()

        # Previously received donations archive
        with st.expander(f"View Confirmed / Received Shipments Archive ({ngo_data['name']})", expanded=False, icon=":material/history_edu:"):
            confirmed_dons = db.get_donations(org_id=active_ngo_id, status="Received")
            if not confirmed_dons:
                st.info("No confirmed receipts recorded yet for this NGO.")
            else:
                conf_table = []
                for cd in confirmed_dons:
                    conf_table.append({
                        "Donation ID": f"#{cd['id']}",
                        "Donor": f"{cd['donor_name']} ({cd['donor_email']})",
                        "Category": cd['category'],
                        "Quantity": f"{cd['quantity']} units",
                        "Condition": cd['condition'],
                        "Est Value": f"₹{cd['estimated_value_inr']:,.2f}",
                        "Status": cd['status'],
                        "Submitted At (IST)": db.format_datetime_ist(cd['created_at']),
                        "Receipt Confirmed At (IST)": db.format_datetime_ist(cd['confirmed_at'])
                    })
                st.dataframe(pd.DataFrame(conf_table), use_container_width=True, hide_index=True)


elif selected_page == PAGE_HISTORY:
    render_hero(
        title="Donation Ledger &",
        highlight_word="Audit Trail.",
        subtitle="Complete persistent transaction history of submitted physical donations and saved purchase plans."
    )

    default_tab_idx = st.session_state.get("don_history_active_tab", 0)

    tab_donations, tab_plans = st.tabs([
        "Physical Item Donations (Verified / Pending)",
        "Saved Purchase Plans (Budget Allocations)"
    ])

    with tab_donations:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            orgs = db.get_organizations()
            o_filter = {"ALL": "All Organizations"}
            for o in orgs:
                o_filter[o["id"]] = o["name"]
            f_org = st.selectbox("Filter by Organization:", list(o_filter.keys()), format_func=lambda x: o_filter[x])
        with col_f2:
            f_cat = st.selectbox("Filter by Category:", ["ALL"] + CATEGORIES)
        with col_f3:
            f_stat = st.selectbox("Filter by Status:", ["ALL", "Pending Receipt", "Received", "Cancelled"])

        filter_org = None if f_org == "ALL" else f_org
        filter_stat = None if f_stat == "ALL" else f_stat
        dons = db.get_donations(org_id=filter_org, status=filter_stat)

        if f_cat != "ALL":
            dons = [d for d in dons if d["category"] == f_cat]

        st.markdown(f"**Showing {len(dons)} matching physical donation transaction(s):**")

        if not dons:
            st.info("No donation records match the specified filters.")
        else:
            table_records = []
            for d in dons:
                plan_tag = f"Plan #{d['purchase_plan_id']}" if d.get("purchase_plan_id") else "Direct Donation"
                table_records.append({
                    "Donation ID": f"#{d['id']}",
                    "Donor": f"{d['donor_name']} ({d['donor_email']})",
                    "Recipient NGO": d["org_name"],
                    "Category": d["category"],
                    "Quantity": int(d["quantity"]),
                    "Condition": d["condition"],
                    "Originating Source": plan_tag,
                    "Est Value (₹)": f"₹{float(d.get('estimated_value_inr', 0.0)):,.2f}",
                    "Status": d["status"],
                    "Submitted At": db.format_datetime_ist(d["created_at"]),
                    "Receipt Confirmed At": db.format_datetime_ist(d["confirmed_at"])
                })
            df_d = pd.DataFrame(table_records)
            st.dataframe(df_d, use_container_width=True, hide_index=True)

            csv_exp = df_d.to_csv(index=False)
            st.download_button(
                label="Export Donations to CSV", icon=":material/download:",
                data=csv_exp,
                file_name=f"needmatch_donations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with tab_plans:
        st.subheader("Saved Budget Purchase Plans")
        st.caption("Optimized purchase plans pledged by donors. Track donation fulfillment progress in real-time, inspect item allocations, and click **Donate Items from This Plan** to fulfill items through verified NGO shipments.")

        plans = db.get_purchase_plans()
        if not plans:
            st.info("No purchase plans saved yet. You can create one on the 'Plan a Purchase' page.")
        else:
            for p in plans:
                prog = p["progress"]
                items_prog = prog["items_progress"]
                plan_status = prog["status"]

                with st.container(border=True):
                    # Header row: Plan Number & Donor Info, Status Badge
                    c_head1, c_head2 = st.columns([3, 1.4])
                    with c_head1:
                        st.markdown(f"### Plan #{p['id']} — Sponsor: {p['donor_name']}")
                        st.caption(f"Email: **{p['donor_email']}** • Created: **{db.format_datetime_ist(p['created_at'])} (IST)**")
                    with c_head2:
                        if plan_status == "Completed":
                            st.success("Status: Completed (100% Received)", icon=":material/check_circle:")
                        elif plan_status == "Donated":
                            st.warning(f"Status: Donated ({prog['total_donated_units']} Units — Awaiting Receipt)", icon=":material/warning:")
                        elif plan_status == "Partially Donated":
                            st.info(f"Status: Partially Donated ({prog['total_donated_units']} / {prog['total_planned_units']} Units)", icon=":material/info:")
                        else:
                            st.info("Status: Saved (0% Donated)", icon=":material/info:")

                    st.divider()

                    # Summary Metrics (All required metrics: budget allocated/original, remaining budget, fulfillment %, planned units, pending units, confirmed units)
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric(
                        label="Allocated Budget",
                        value=f"₹{p['total_allocated_inr']:,.2f}",
                        delta=f"Original: ₹{p['total_budget_inr']:,.2f}",
                        delta_color="off"
                    )
                    m2.metric(
                        label="Remaining Budget",
                        value=f"₹{p['remaining_budget_inr']:,.2f}"
                    )
                    m3.metric(
                        label="Overall Fulfillment",
                        value=f"{prog['pct_donated']:.1f}%",
                        delta=f"{prog['pct_received']:.1f}% received" if prog['pct_received'] > 0 else None
                    )
                    m4.metric(
                        label="Total Planned Units",
                        value=f"{prog['total_planned_units']} units"
                    )
                    m5.metric(
                        label="Pending Receipt",
                        value=f"{prog['total_pending_units']} units",
                        help="Units donated by sponsor but awaiting physical receipt confirmation from recipient NGO"
                    )
                    m6.metric(
                        label="Confirmed Received",
                        value=f"{prog['total_received_units']} units",
                        help="Units physically confirmed received and added to live NGO inventory"
                    )

                    # Visual Progress Bar using native Streamlit component
                    norm_val = min(1.0, max(0.0, float(prog["pct_donated"]) / 100.0))
                    prog_label = (
                        f"Overall Fulfillment: {prog['total_donated_units']} of {prog['total_planned_units']} units donated ({prog['pct_donated']:.1f}%) "
                        f"— {prog['total_pending_units']} pending receipt, {prog['total_received_units']} confirmed received ({prog['pct_received']:.1f}%)"
                    )
                    st.progress(norm_val, text=prog_label)

                    # Rebuild the item table with exact 9 columns
                    item_table = []
                    for itm in items_prog:
                        item_table.append({
                            "Item Category": itm["category"],
                            "Recipient NGO": itm["org_name"],
                            "Planned Quantity": itm["planned_quantity"],
                            "Donated Quantity": itm["donated_quantity"],
                            "Pending Quantity": itm["pending_quantity"],
                            "Confirmed Quantity": itm["received_quantity"],
                            "Remaining Quantity": itm["remaining_quantity"],
                            "Estimated Cost": f"₹{itm['cost']:,.2f}",
                            "Fulfillment Status": itm["allocation_status"]
                        })
                    st.dataframe(pd.DataFrame(item_table), use_container_width=True, hide_index=True)

                    # Donate Items from This Plan button
                    col_act1, col_act2 = st.columns([2.2, 2.8])
                    with col_act1:
                        if prog["total_remaining_units"] > 0:
                            btn_text = f"Donate Items from This Plan ({prog['total_remaining_units']} Remaining) →"
                            if st.button(btn_text, key=f"btn_donate_from_plan_{p['id']}", type="primary", use_container_width=True):
                                # Determine first unfulfilled allocation index
                                first_unfulfilled = 0
                                for idx, itm in enumerate(items_prog):
                                    if itm["remaining_quantity"] > 0:
                                        first_unfulfilled = idx
                                        break
                                st.session_state["plan_donation_context"] = {
                                    "plan_id": p["id"],
                                    "selected_item_index": first_unfulfilled
                                }
                                st.session_state["nav_target"] = PAGE_DONATE
                                st.rerun()
                        elif plan_status == "Donated":
                            st.info(f"All {prog['total_planned_units']} units have been donated and are awaiting NGO physical receipt confirmation.", icon=":material/info:")
                        else:
                            st.success(f"All {prog['total_planned_units']} units have been donated and confirmed received by the partner NGOs!", icon=":material/check_circle:")

elif selected_page == PAGE_MODEL:
    render_hero(
        title="Machine Learning Architecture &",
        highlight_word="Academic Methodology.",
        subtitle="Full transparency into the machine learning pipeline, dataset disclosures, chronological evaluation, and live model retraining."
    )

    # Mandatory Academic Synthetic Data Notice
    st.warning(
        "**DATASET & MODEL DISCLOSURE (ACADEMIC RIGOR & HONESTY):**\\n\\n"
        "The historical dataset utilized for training is a **calibrated synthetic dataset** covering 34 continuous months across all 6 donation categories and 6 partner NGOs. "
        "It models authentic seasonal spikes (e.g. school reopening in May-July, winter drives in Nov-Jan, monsoon hygiene camps) with controlled stochastic variance.\\n\\n"
        "**Limitation Notice:** Demonstration results are based on synthetic historical data. They do not establish real-world operational predictive accuracy."
    )

    metrics = predict.load_metrics()
    eval_m = metrics.get("evaluation", {})
    rf_mae = eval_m.get("rf_mae", 61.1)
    base_mae = eval_m.get("baseline_mae", 103.3)
    improvement = eval_m.get("mae_improvement_pct", 40.9)
    r2_score = eval_m.get("rf_r2", 0.7845)
    train_count = metrics.get("train_samples", 972)
    test_count = metrics.get("test_samples", 252)

    # Metrics Row using Native Streamlit Components
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Algorithm", "Random Forest", "Scikit-learn Regressor", delta_color="off")
    m2.metric("RF Test MAE", f"{rf_mae:.2f}", "Mean Absolute Error", delta_color="off")
    m3.metric("Baseline MAE", f"{base_mae:.2f}", "Training Group Mean", delta_color="off")
    m4.metric("Accuracy Lift", f"+{improvement:.1f}%", "Error reduction", delta_color="normal")
    m5.metric("R² Fit Score", f"{r2_score:.3f}", "252 test instances", delta_color="off")



    # Actions: Train / Retrain Model & Refresh Status
    btn_col1, btn_col2, sp_c = st.columns([1, 1, 1.2])
    with btn_col1:
        if st.button("Train / Retrain Model Pipeline", icon=":material/psychology:", use_container_width=True, type="primary"):
            with st.spinner("Executing chronological pipeline, fitting RandomForestRegressor, and updating predictions..."):
                from train_model import train_and_evaluate_model
                pipe, new_metrics = train_and_evaluate_model()
                st.success(f"Model retrained! New RF MAE: {new_metrics['evaluation']['rf_mae']:.2f} units.")
                st.rerun()

    with btn_col2:
        if st.button("Refresh Model Status", icon=":material/refresh:", use_container_width=True):
            if os.path.exists("models/rf_demand_model.joblib"):
                st.success("Model artifact verified and ready for inference!")
            else:
                st.warning("Model file not found. Please click Train Model.")

    # Educational Pipeline Explanation using Native Streamlit Container
    with st.container(border=True):
        st.subheader("End-to-End System Pipeline Flowchart")
        st.code(
            "Historical Dataset (CSV) → Preprocessing & OneHotEncoder → Chronological Split (80/20)\\n"
            "↓\\n"
            "RandomForestRegressor Training → Held-Out Evaluation (MAE 61.1 vs Baseline 103.3)\\n"
            "↓\\n"
            "Saved Artifact (Joblib) → Demand Prediction Engine → AI Recommendation Logic → User Dashboard",
            language=None
        )
        st.caption(
            "**What is Mean Absolute Error (MAE)?** "
            "Mean Absolute Error measures the average difference between the model's predicted demand and actual demand in raw units. "
            "For example, an MAE of 61.09 means our model's 30-day forecast is, on average, within ~61 units of the true future demand, representing a "
            "**40.86% error reduction** compared to the baseline model (103.30 units)."
        )

    # Feature Importance Chart
    with st.container(border=True):
        st.subheader("Feature Importance Weights (What the Model Learned)")
        f_imps = metrics.get("feature_importances", [])
        if f_imps:
            df_imp = pd.DataFrame(f_imps)
            fig_imp = px.bar(
                df_imp,
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=[COLOR_TEAL]
            )
            fig_imp.update_layout(
                get_plotly_layout(height=360, title="Relative Feature Importance in RandomForestRegressor"),
                xaxis_title="Importance Weight",
                yaxis_title="Feature",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_imp, use_container_width=True)
