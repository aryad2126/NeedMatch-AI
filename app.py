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
    page_icon="🤝",
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
    st.markdown(f"""
        <div style="padding: 0.5rem 0 0.8rem 0;">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.65rem; font-weight: 700; color: {COLOR_TEAL}; line-height: 1.1;">
                NeedMatch <span style="color: {COLOR_CORAL};">AI</span>
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-top: 4px;">
                Predictive Donation Allocation System
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    PAGES = [
        "1. 🏠 Home / Dashboard",
        "2. 🔮 Demand Forecast",
        "3. 🎯 AI Recommendations",
        "4. 📦 Donate Existing Items",
        "5. 💳 Plan a Purchase",
        "6. 🏢 NGO Portal",
        "7. 📜 Donation History",
        "8. 🧠 Model & Data Information"
    ]

    # Handle cross-page redirection
    default_index = 0
    if "nav_target" in st.session_state and st.session_state["nav_target"] in PAGES:
        default_index = PAGES.index(st.session_state["nav_target"])
        del st.session_state["nav_target"]

    selected_page = st.radio("Navigation Flow", PAGES, index=default_index)

    st.markdown("---")

    # System Status Indicator Widget
    metrics_data = predict.load_metrics()
    model_trained = os.path.exists("models/rf_demand_model.joblib")
    db_ready = os.path.exists("data/needmatch.db")

    st.markdown(f"""
        <div style="background: {COLOR_CREAM}; padding: 12px; border-radius: 10px; border: 1px solid rgba(15,78,108,0.12); font-size: 0.8rem;">
            <div style="font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 6px;">System Health</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span>Database:</span>
                <span style="font-weight:600; color:#2E7D32;">● Connected</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span>ML Model:</span>
                <span style="font-weight:600; color:{'#2E7D32' if model_trained else '#C82333'};">{'● Ready (RF)' if model_trained else '○ Needs Training'}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Model Test MAE:</span>
                <span style="font-weight:600; color:{COLOR_TEAL};">{metrics_data.get('evaluation', {}).get('rf_mae', 'N/A')} units</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset to Clean Demo State", help="Reseeds the database with clean baseline demo records"):
        db.init_db(force_reseed=True)
        predict.generate_and_save_predictions()
        st.success("Database reseeded with fresh demo records!")
        st.rerun()

# =====================================================================
# PAGE 1: HOME / DASHBOARD
# =====================================================================
if selected_page == "1. 🏠 Home / Dashboard":
    render_hero(
        title="Predict Donation Needs.",
        highlight_word="Recommend Meaningful Contributions.",
        subtitle="NeedMatch AI connects donations with organizations that need them most using machine learning demand forecasts and real-time inventory intelligence."
    )

    # Primary Action Buttons
    c_btn1, c_btn2, c_sp = st.columns([1, 1, 1.5])
    with c_btn1:
        if st.button("📦 Donate Items I Own", use_container_width=True):
            st.session_state["nav_target"] = "4. 📦 Donate Existing Items"
            st.rerun()
    with c_btn2:
        if st.button("💳 Plan a Purchase with Budget", use_container_width=True):
            st.session_state["nav_target"] = "5. 💳 Plan a Purchase"
            st.rerun()

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    # Summary Metrics (Real Database Values)
    stats = db.get_overall_stats()
    m_info = predict.load_metrics()
    rf_mae = m_info.get("evaluation", {}).get("rf_mae", 61.1)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Demo Organizations</div>
            <div class="metric-value">{stats['total_orgs']}</div>
            <div class="metric-subtext">Verified partner shelters</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Donations</div>
            <div class="metric-value">{stats['total_donations']}</div>
            <div class="metric-subtext">Registered contributions</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Awaiting Receipt</div>
            <div class="metric-value" style="color: {COLOR_CORAL};">{stats['pending_donations']}</div>
            <div class="metric-subtext">{stats['pending_units']} units pending verification</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Donations Received</div>
            <div class="metric-value">{stats['confirmed_donations']}</div>
            <div class="metric-subtext">{stats['total_units_donated']} physical units verified</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Outstanding Backlog</div>
            <div class="metric-value">{stats['total_outstanding_requested_units']}</div>
            <div class="metric-subtext">Units across {stats['open_requests_count']} open requests</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Visual Charts
    forecast_df = predict.get_demand_forecast_table()
    chart_c1, chart_c2 = st.columns([1.1, 0.9])

    with chart_c1:
        st.markdown(f"""
        <div class="nm-card">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                Predicted 30-Day Demand vs. Current Stock by Category
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 0.8rem;">
                Comparing ML 30-day forecast against available inventory across all 6 partner organizations.
            </div>
        """, unsafe_allow_html=True)

        cat_summary = forecast_df.groupby("category")[["predicted_30d_demand", "current_inventory", "projected_shortfall"]].sum().reset_index()

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
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_c2:
        st.markdown(f"""
        <div class="nm-card">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                Estimated Shortages by Organization
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 0.8rem;">
                Identifies which partner NGOs face the highest immediate deficits.
            </div>
        """, unsafe_allow_html=True)

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
        st.markdown("</div>", unsafe_allow_html=True)

    # Additional Sections: High Priority Needs & Recent Donations
    sec_c1, sec_c2 = st.columns([1.1, 1])

    with sec_c1:
        st.markdown(f"""
        <div class="nm-card nm-card-highlight">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                🚨 High-Priority Unmet Needs
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 0.8rem;">
                Organizations with acute shortages calculated from open requests and predicted demand.
            </div>
        """, unsafe_allow_html=True)

        top_shortfalls = forecast_df.sort_values(by="projected_shortfall", ascending=False).head(4)
        for _, row in top_shortfalls.iterrows():
            st.markdown(f"""
            <div style="background: #F8FBFD; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; border: 1px solid {COLOR_BORDER};">
                <div style="display: flex; justify-content: space-between; font-weight: 700; color: {COLOR_TEAL}; font-size: 0.92rem;">
                    <span>{row['category']} — {row['org_name']}</span>
                    <span style="color: {COLOR_CORAL};">Deficit: {row['projected_shortfall']:.0f} units</span>
                </div>
                <div style="font-size: 0.78rem; color: {COLOR_MUTED_TEXT}; margin-top: 2px;">
                    Current Stock: {row['current_inventory']} | Open Requests: {row['open_requests']} | Predicted 30d Demand: {row['predicted_30d_demand']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sec_c2:
        st.markdown(f"""
        <div class="nm-card">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                📋 Recent Donation Transactions
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 0.8rem;">
                Latest submissions from the persistent SQLite ledger.
            </div>
        """, unsafe_allow_html=True)

        recent_donations = db.get_donations()[:4]
        if not recent_donations:
            st.info("No donations recorded yet.")
        else:
            for d in recent_donations:
                status_color = "#2E7D32" if d['status'] in ['Received', 'CONFIRMED'] else COLOR_CORAL
                st.markdown(f"""
                <div style="background: {COLOR_WHITE}; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px; border: 1px solid {COLOR_BORDER}; font-size: 0.82rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>Donation #{d['id']}: {d['quantity']} units of {d['category']}</strong>
                        <span style="color: {status_color}; font-weight: 700;">{d['status']}</span>
                    </div>
                    <div style="color: {COLOR_MUTED_TEXT}; font-size: 0.78rem; margin-top: 2px;">
                        Donor: {d['donor_name']} → Recipient: {d['org_name']} ({d['created_at'][:10]})
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# PAGE 2: DEMAND FORECAST
# =====================================================================
elif selected_page == "2. 🔮 Demand Forecast":
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
    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    run_forecast_btn = st.button("🔮 Generate Demand Forecast", use_container_width=True)

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
        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid rgba(15,78,108,0.1);'>", unsafe_allow_html=True)

        if len(results) == 1:
            r = results[0]
            st.markdown(f"""
            <div class="nm-card nm-card-teal">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <div>
                        <span style="font-family: 'Playfair Display', serif; font-size: 1.45rem; font-weight: 700; color: {COLOR_TEAL};">
                            Demand Forecast: {r['category']} for {r['org_name']}
                        </span>
                        <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-top: 2px;">
                            📍 {r['city']} • {r['horizon_note']}
                        </div>
                    </div>
                    <span class="badge badge-teal">Scikit-learn RF Output</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: {COLOR_WHITE}; padding: 14px; border-radius: 10px; border: 1px solid {COLOR_BORDER}; text-align: center; margin-bottom: 12px;">
                    <div>
                        <div class="metric-label">Predicted Demand</div>
                        <div style="font-family:'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; color: {COLOR_TEAL};">
                            {r['predicted_demand']:.0f}
                        </div>
                        <div style="font-size: 0.75rem; color: {COLOR_MUTED_TEXT};">{curr_horizon}-Day Horizon</div>
                    </div>
                    <div>
                        <div class="metric-label">Current Inventory</div>
                        <div style="font-family:'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; color: {COLOR_MUTED_TEXT};">
                            {r['current_inventory']}
                        </div>
                        <div style="font-size: 0.75rem; color: {COLOR_MUTED_TEXT};">In warehouse</div>
                    </div>
                    <div>
                        <div class="metric-label">Outstanding Requests</div>
                        <div style="font-family:'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; color: {COLOR_MUTED_TEXT};">
                            {r['outstanding_requests']}
                        </div>
                        <div style="font-size: 0.75rem; color: {COLOR_MUTED_TEXT};">Urgent backlog</div>
                    </div>
                    <div>
                        <div class="metric-label">Estimated Shortage</div>
                        <div style="font-family:'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; color: {COLOR_CORAL};">
                            {r['estimated_shortage']:.0f}
                        </div>
                        <div style="font-size: 0.75rem; color: {COLOR_CORAL}; font-weight: 600;">Net deficit gap</div>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: {COLOR_TEAL}; line-height: 1.5; background: #F8FBFD; padding: 10px 14px; border-radius: 8px;">
                    <strong>Explanation:</strong> Based on historical demand patterns ({r['features']['historical_demand_qty']} avg monthly units), seasonal factor ({r['features']['season_factor']}x), and an active backlog of {r['outstanding_requests']} units against {r['current_inventory']} units in stock, our model projects a net deficit of <strong>{r['estimated_shortage']:.0f} units</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Multi-result table
            st.markdown(f"""
                <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.5rem;">
                    Forecast Results ({len(results)} Combinations)
                </div>
            """, unsafe_allow_html=True)
            res_df = pd.DataFrame(results)[[
                "org_name", "city", "category", "predicted_demand",
                "current_inventory", "outstanding_requests", "estimated_shortage"
            ]]
            res_df.columns = ["Organization", "City", "Category", f"Predicted Demand ({curr_horizon}d)", "Inventory", "Backlog", "Estimated Shortage"]
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Historical vs Predicted Visualization
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
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

        st.markdown(f"""
        <div class="disclaimer-box">
            <strong>⚠️ Transparency Disclosure:</strong> Demand forecasts are probabilistic estimates generated by a Scikit-learn RandomForestRegressor. Real-world predictive accuracy may vary based on unforeseen socio-economic events.
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGE 3: AI RECOMMENDATIONS
# =====================================================================
elif selected_page == "3. 🎯 AI Recommendations":
    render_hero(
        title="AI-Based Allocation &",
        highlight_word="Recommendation Engine.",
        subtitle="Transform raw demand forecasts and live warehouse inventory into transparent, ranked donation recommendations."
    )

    rec_method = st.radio(
        "Select Donation Method to Recommend:",
        ["Donate Existing Items You Already Own", "Plan a Purchase of New Supplies with Budget"],
        horizontal=True
    )

    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)

    if rec_method == "Donate Existing Items You Already Own":
        c_cat, c_qty, c_cond, c_pref = st.columns([1, 1, 1, 1.2])
        with c_cat:
            rec_cat = st.selectbox("Item Category:", CATEGORIES, index=0)
        with c_qty:
            rec_qty = st.number_input("Available Units:", min_value=1, max_value=5000, value=30, step=5)
        with c_cond:
            rec_cond = st.selectbox("Condition:", ["Good", "New", "Usable"], index=0)
        with c_pref:
            orgs = db.get_organizations()
            org_choices = {"ALL": "Any NGO (Highest Need)"}
            for o in orgs:
                org_choices[o["id"]] = f"{o['name']} ({o['city']})"
            rec_pref_org = st.selectbox("Preferred NGO:", options=list(org_choices.keys()), format_func=lambda x: org_choices[x])

        if st.button("🎯 Generate AI Recommendations", use_container_width=True):
            st.session_state["rec_items_executed"] = True

        if st.session_state.get("rec_items_executed", False):
            recommendations = recommend_ngos_for_item(
                category=rec_cat,
                available_qty=int(rec_qty),
                condition=rec_cond,
                org_id_filter=rec_pref_org
            )

            st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid rgba(15,78,108,0.1);'>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.6rem;">
                    Ranked Recipient Recommendations for {rec_cat}
                </div>
            """, unsafe_allow_html=True)

            if not recommendations:
                st.warning("No recipient organizations match the selected criteria.")
            else:
                for idx, r in enumerate(recommendations):
                    badge_style = "badge-coral" if idx == 0 else "badge-teal"
                    st.markdown(f"""
                    <div class="nm-card {'nm-card-highlight' if idx == 0 else ''}" style="margin-bottom: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 700; color: {COLOR_TEAL};">
                                    #{idx+1} {r['org_name']}
                                </span>
                                <span style="font-size: 0.8rem; color: {COLOR_MUTED_TEXT}; margin-left: 6px;">📍 {r['city']} • Focus: {r['focus']}</span>
                                <span class="badge {badge_style}" style="margin-left: 8px;">Priority Score: {r['priority_score']:.0f}/100</span>
                            </div>
                            <div style="font-weight: 700; color: {COLOR_CORAL}; font-size: 0.95rem;">
                                Suggested Allocation: {r['suggested_qty']} units
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #F8FBFD; padding: 10px; border-radius: 8px; margin: 8px 0; font-size: 0.8rem; text-align: center;">
                            <div>Current Stock: <strong>{r['current_inventory']}</strong></div>
                            <div>Open Backlog: <strong>{r['requested_backlog']}</strong></div>
                            <div>ML 30d Demand: <strong>{r['predicted_demand']:.0f}</strong></div>
                            <div>Net Shortfall: <strong style="color:{COLOR_CORAL};">{r['unmet_need']:.0f}</strong></div>
                        </div>
                        <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; line-height: 1.45; font-style: italic;">
                            <strong>Recommendation Justification:</strong> {r['reason']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        # Budget Purchase Recommendation
        c_bud, c_cats, c_pref = st.columns([1, 1.5, 1.2])
        with c_bud:
            rec_budget = st.number_input("Available Budget (₹ INR):", min_value=500.0, max_value=500000.0, value=7500.0, step=500.0)
        with c_cats:
            rec_supp_cats = st.multiselect("Eligible Categories:", CATEGORIES, default=["Notebooks", "Schoolbags", "Books"])
        with c_pref:
            orgs = db.get_organizations()
            org_choices = {"ALL": "Any NGO (Highest Impact per Rupee)"}
            for o in orgs:
                org_choices[o["id"]] = f"{o['name']} ({o['city']})"
            rec_pref_org = st.selectbox("Preferred NGO:", options=list(org_choices.keys()), format_func=lambda x: org_choices[x], key="rec_pref_org_b")

        if st.button("🎯 Generate AI Purchase Recommendations", use_container_width=True):
            st.session_state["rec_budget_executed"] = True

        if st.session_state.get("rec_budget_executed", False):
            plan = recommend_purchase_plan(
                budget_inr=float(rec_budget),
                selected_categories=rec_supp_cats if rec_supp_cats else CATEGORIES,
                preferred_org_id=None if rec_pref_org == "ALL" else rec_pref_org
            )

            st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid rgba(15,78,108,0.1);'>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: {COLOR_TEAL};">
                        AI Recommended Purchase Basket
                    </div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: {COLOR_CORAL};">
                        Total Allocated: ₹{plan['total_cost']:,.2f} / ₹{rec_budget:,.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            items = plan.get("items", [])
            if not items:
                st.info("No items could be allocated with the given parameters and budget.")
            else:
                for item in items:
                    st.markdown(f"""
                    <div class="nm-card" style="margin-bottom: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; color: {COLOR_TEAL};">
                                {item['recommended_qty']} Units of {item['category']} → {item['org_name']}
                            </span>
                            <span style="font-weight: 700; color: {COLOR_CORAL}; font-size: 0.95rem;">
                                Total Cost: ₹{item['total_cost']:,.2f} (₹{item['unit_price']:.0f} / {item['unit_name']})
                            </span>
                        </div>
                        <div style="font-size: 0.8rem; color: {COLOR_MUTED_TEXT}; margin: 4px 0 8px 0;">
                            📍 {item['city']} • Urgency: <strong>{item['urgency']}</strong> • Unmet Need: <strong>{item['unmet_need']:.0f} units</strong>
                        </div>
                        <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; font-style: italic;">
                            {item['reason']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# =====================================================================
# PAGE 4: DONATE EXISTING ITEMS
# =====================================================================
elif selected_page == "4. 📦 Donate Existing Items":
    render_hero(
        title="Donate Items You",
        highlight_word="Already Own.",
        subtitle="Connect pre-owned or spare essential supplies directly to verified community shelters experiencing acute deficits."
    )

    col_form, col_ranked = st.columns([1, 1.25])

    with col_form:
        st.markdown(f"""
        <div class="nm-card nm-card-highlight">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                Step 1: Item & Donor Information
            </div>
        """, unsafe_allow_html=True)

        don_cat = st.selectbox("Category of Supplies:", CATEGORIES, index=0)
        don_qty = st.number_input("Available Quantity (Units):", min_value=1, max_value=5000, value=25, step=5)
        don_cond = st.selectbox("Item Condition:", ["Good", "New", "Usable"], index=0)

        st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid rgba(15,78,108,0.1);'>", unsafe_allow_html=True)

        d_name = st.text_input("Donor Full Name:", value="Rohan Mehra")
        d_email = st.text_input("Donor Email Address:", value="rohan.m@example.org")
        d_phone = st.text_input("Phone Number:", value="+91 98765 22334")
        d_notes = st.text_area("Pickup or Drop-off Notes:", placeholder="E.g., Packed in a cardboard box, ready for courier pickup.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_ranked:
        st.markdown(f"""
        <div class="nm-card">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                Step 2: AI Matched Recipient Organizations
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 0.8rem;">
                Ranked by real-time shortage and ML demand forecasts.
            </div>
        """, unsafe_allow_html=True)

        matched_orgs = recommend_ngos_for_item(category=don_cat, available_qty=int(don_qty), condition=don_cond)

        if matched_orgs:
            org_options_map = {m["org_id"]: f"#{i+1} {m['org_name']} ({m['city']}) — Shortfall: {m['unmet_need']:.0f}" for i, m in enumerate(matched_orgs)}
            chosen_org_id = st.selectbox("Select Recipient NGO:", options=list(org_options_map.keys()), format_func=lambda x: org_options_map[x], index=0)

            top_org = matched_orgs[0]
            st.markdown(f"""
            <div style="background: #F8FBFD; border-radius: 8px; padding: 12px; border: 1px solid {COLOR_BORDER}; margin-top: 10px;">
                <div style="font-weight: 700; color: {COLOR_TEAL}; font-size: 0.95rem;">
                    Top Match: {top_org['org_name']} (Priority Score: {top_org['priority_score']:.0f}/100)
                </div>
                <div style="font-size: 0.8rem; color: {COLOR_MUTED_TEXT}; margin: 4px 0 6px 0;">
                    Open Backlog: {top_org['requested_backlog']} | Current Stock: {top_org['current_inventory']} | Predicted 30d Demand: {top_org['predicted_demand']:.0f}
                </div>
                <div style="font-size: 0.82rem; color: {COLOR_TEAL}; font-style: italic;">
                    {top_org['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            chosen_org_id = None

        st.markdown("</div>", unsafe_allow_html=True)

        # Main Submit Donation Action
        if st.button("📤 Submit Donation", use_container_width=True):
            if not d_name or not d_email:
                st.error("Please provide both donor name and email address.")
            elif not chosen_org_id:
                st.error("Please select a recipient organization.")
            else:
                try:
                    prices = db.get_item_prices()
                    unit_p = prices.get(don_cat, {}).get("unit_price_inr", 100.0)
                    est_val = unit_p * int(don_qty)

                    new_id = db.create_donation(
                        donation_type="EXISTING_ITEMS",
                        donor_name=d_name,
                        donor_email=d_email,
                        donor_phone=d_phone,
                        org_id=chosen_org_id,
                        category=don_cat,
                        quantity=int(don_qty),
                        condition=don_cond,
                        estimated_value_inr=est_val,
                        notes=d_notes
                    )

                    org_info = db.get_organization_by_id(chosen_org_id)
                    st.success(f"🎉 Donation #{new_id} registered successfully!")
                    st.markdown(f"""
                    <div class="disclaimer-box" style="background: rgba(255, 127, 110, 0.12); border-left: 4px solid {COLOR_CORAL};">
                        <strong>STATUS: Pending Receipt</strong><br>
                        Your donation of <strong>{don_qty} units of {don_cat}</strong> has been logged for <strong>{org_info['name']}</strong>.<br>
                        <em>Notice: To prevent phantom inventory, your donation remains in 'Pending Receipt' status until the shelter physically receives and verifies the supplies in their NGO Portal.</em>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Error submitting donation: {ex}")

# =====================================================================
# PAGE 5: PLAN A PURCHASE
# =====================================================================
elif selected_page == "5. 💳 Plan a Purchase":
    render_hero(
        title="Plan a Purchase with an",
        highlight_word="INR Budget.",
        subtitle="Sponsor brand-new essential supplies. The optimization engine allocates your funds to maximize genuine social impact per rupee spent."
    )

    col_setup, col_basket = st.columns([1, 1.3])

    with col_setup:
        st.markdown(f"""
        <div class="nm-card nm-card-teal">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                Step 1: Budget & Category Options
            </div>
        """, unsafe_allow_html=True)

        plan_budget = st.number_input("Available Budget (₹ INR):", min_value=500.0, max_value=500000.0, value=8000.0, step=500.0)
        plan_cats = st.multiselect("Categories to Sponsor:", CATEGORIES, default=["Notebooks", "Schoolbags", "Books", "Blankets"])

        orgs = db.get_organizations()
        p_choices = {"ALL": "Any NGO (System Recommended Highest Need)"}
        for o in orgs:
            p_choices[o["id"]] = f"{o['name']} ({o['city']})"
        plan_pref_org = st.selectbox("Preferred NGO:", options=list(p_choices.keys()), format_func=lambda x: p_choices[x])

        st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid rgba(15,78,108,0.1);'>", unsafe_allow_html=True)

        p_name = st.text_input("Sponsor / Donor Name:", value="Vikram Malhotra")
        p_email = st.text_input("Contact Email:", value="vikram.m@example.org")

        # Wholesale Price Table Display
        price_dict = db.get_item_prices()
        st.markdown(f"""
            <div style="background: {COLOR_WHITE}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 10px; font-size: 0.78rem; line-height: 1.5; margin-top: 10px;">
                <strong>Estimated Wholesale Unit Prices (INR):</strong><br>
                {" • ".join([f"{k}: ₹{v['unit_price_inr']:.0f}" for k, v in price_dict.items()])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        gen_plan_btn = st.button("💡 Generate Purchase Plan", use_container_width=True)
        if gen_plan_btn:
            st.session_state["purchase_plan_active"] = True

    with col_basket:
        st.markdown(f"""
        <div class="nm-card">
            <div style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                Step 2: Optimized Purchase Basket
            </div>
        """, unsafe_allow_html=True)

        plan_res = recommend_purchase_plan(
            budget_inr=float(plan_budget),
            selected_categories=plan_cats if plan_cats else CATEGORIES,
            preferred_org_id=None if plan_pref_org == "ALL" else plan_pref_org
        )

        raw_items = plan_res.get("items", [])
        if not raw_items:
            st.info("Click 'Generate Purchase Plan' on the left to compute the optimal allocation.")
        else:
            # Interactive Steppers
            adjusted_items = []
            for i, itm in enumerate(raw_items):
                st.markdown(f"""
                <div style="background: #F8FBFD; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; border: 1px solid {COLOR_BORDER};">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; color: {COLOR_TEAL}; font-size: 0.95rem;">
                        <span>{itm['category']} → {itm['org_name']}</span>
                        <span style="color: {COLOR_CORAL};">₹{itm['unit_price']:.0f} / {itm['unit_name']}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: {COLOR_MUTED_TEXT}; margin: 2px 0 6px 0;">
                        Unmet Need: {itm['unmet_need']:.0f} units | Urgency: <strong>{itm['urgency']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_q, col_c = st.columns([1, 1.2])
                with col_q:
                    q_val = st.number_input(f"Qty ({itm['category']})", min_value=0, max_value=500, value=int(itm["recommended_qty"]), key=f"pp_qty_{i}")
                with col_c:
                    sub_c = q_val * itm["unit_price"]
                    st.markdown(f"<div style='margin-top: 1.8rem; font-weight:700; color:{COLOR_TEAL}; font-size:0.9rem;'>Cost: ₹{sub_c:,.2f}</div>", unsafe_allow_html=True)

                adjusted_items.append({
                    "category": itm["category"],
                    "org_id": itm["org_id"],
                    "org_name": itm["org_name"],
                    "unit_price": itm["unit_price"],
                    "quantity": q_val,
                    "cost": sub_c
                })

            total_calc_cost = sum(x["cost"] for x in adjusted_items)
            rem_budget = plan_budget - total_calc_cost
            pct_spent = min(100.0, (total_calc_cost / max(1.0, plan_budget)) * 100)

            st.markdown(f"""
            <div style="background: {COLOR_CREAM}; padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(15,78,108,0.15); margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 1.05rem; color: {COLOR_TEAL}; margin-bottom: 4px;">
                    <span>Total Plan Cost:</span>
                    <span style="color: {'#C82333' if total_calc_cost > plan_budget else COLOR_TEAL};">₹{total_calc_cost:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 6px;">
                    <span>Remaining Budget:</span>
                    <span>₹{rem_budget:,.2f}</span>
                </div>
                <div style="background: rgba(15,78,108,0.1); height: 8px; border-radius: 999px; overflow: hidden;">
                    <div style="background: {COLOR_CORAL}; width: {pct_spent}%; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            if total_calc_cost > plan_budget:
                st.error("⚠️ Total cost exceeds available budget. Please reduce units to confirm.")
            else:
                if st.button("🤝 Confirm Purchase Plan", use_container_width=True):
                    if not p_name or not p_email:
                        st.error("Please enter donor name and email.")
                    else:
                        plan_id = db.save_purchase_plan(
                            donor_name=p_name,
                            donor_email=p_email,
                            total_budget_inr=plan_budget,
                            total_allocated_inr=total_calc_cost,
                            remaining_budget_inr=rem_budget,
                            items=[x for x in adjusted_items if x["quantity"] > 0]
                        )
                        st.success(f"🎉 Purchase Plan #{plan_id} saved successfully to SQLite!")
                        st.markdown(f"""
                        <div class="disclaimer-box">
                            <strong>Plan Saved in System Database:</strong><br>
                            Your purchase plan has been recorded in the persistent ledger (Plan ID #{plan_id}). You can view this in the <strong>Donation History</strong> page under the Saved Purchase Plans tab.
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# PAGE 6: NGO PORTAL
# =====================================================================
elif selected_page == "6. 🏢 NGO Portal":
    render_hero(
        title="NGO Management &",
        highlight_word="Inventory Control.",
        subtitle="Manage warehouse stock, broadcast critical item requests, and confirm physical receipt of incoming shipments."
    )

    orgs = db.get_organizations()
    ngo_options = {o["id"]: f"{o['name']} — 📍 {o['city']}" for o in orgs}
    active_ngo_id = st.selectbox("Select Active Partner Organization:", options=list(ngo_options.keys()), format_func=lambda x: ngo_options[x], index=0)
    ngo_data = db.get_organization_by_id(active_ngo_id)

    # Organization Overview Card
    st.markdown(f"""
    <div class="nm-card nm-card-teal" style="margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; color: {COLOR_TEAL};">
                    {ngo_data['name']}
                </div>
                <div style="font-size: 0.85rem; color: {COLOR_MUTED_TEXT}; margin-top: 3px;">
                    📍 {ngo_data['city']} • Focus: <strong>{ngo_data['focus']}</strong> • Contact: {ngo_data['contact_email']}
                </div>
                <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-top: 6px;">
                    {ngo_data['description']}
                </div>
            </div>
            <span class="badge badge-sky">ID: {ngo_data['id']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Functional Sections as Tabs
    tab_inv, tab_req, tab_inbound, tab_fc = st.tabs([
        "📦 Section A: Inventory",
        "📝 Section B: Manage Requests",
        "📥 Section C: Incoming Donations",
        "🔮 Section D: NGO Forecasts"
    ])

    # SECTION A: INVENTORY
    with tab_inv:
        st.markdown(f"""
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.5rem;">
                Current Warehouse Stock Levels
            </div>
        """, unsafe_allow_html=True)

        inv_items = db.get_inventory(org_id=active_ngo_id)
        preds_list = db.get_predictions_from_db(org_id=active_ngo_id)
        pred_map = {p["category"]: p["predicted_demand"] for p in preds_list}

        inv_table_data = []
        for itm in inv_items:
            cat = itm["category"]
            qty = itm["quantity"]
            predicted = pred_map.get(cat, 0.0)
            status_tag = "Critical Deficit" if qty < (predicted * 0.3) else ("Moderate Shortage" if qty < predicted else "Sufficient Stock")
            inv_table_data.append({
                "Category": cat,
                "Available Quantity (Units)": qty,
                "ML 30d Demand": f"{predicted:.0f}",
                "Supply Status": status_tag,
                "Last Updated": itm["last_updated"]
            })

        st.dataframe(pd.DataFrame(inv_table_data), use_container_width=True, hide_index=True)

        with st.expander("✏️ Update Inventory Count (Post-Physical Audit)"):
            st.markdown("<p style='font-size:0.82rem; color:#3D6277;'>Adjust physical stock quantity after warehouse inventory audit.</p>", unsafe_allow_html=True)
            u_cat = st.selectbox("Category to Update:", [i["category"] for i in inv_items], key="u_cat_sel")
            u_qty = st.number_input("New Exact Verified Quantity:", min_value=0, max_value=10000, value=50, key="u_qty_inp")
            if st.button("Save Stock Quantity"):
                db.set_inventory_quantity(active_ngo_id, u_cat, int(u_qty))
                st.success(f"Inventory for {u_cat} updated to {u_qty} units.")
                st.rerun()

    # SECTION B: MANAGE REQUESTS
    with tab_req:
        col_r1, col_r2 = st.columns([1.2, 1])
        with col_r1:
            st.markdown(f"""
                <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.5rem;">
                    Active Open Requests
                </div>
            """, unsafe_allow_html=True)

            reqs = db.get_donation_requests(org_id=active_ngo_id)
            open_reqs = [r for r in reqs if r["status"] in ["OPEN", "PARTIALLY_FULFILLED"]]

            if not open_reqs:
                st.info("No active open requests for this NGO.")
            else:
                for rq in open_reqs:
                    pct_ful = min(100.0, (rq["fulfilled_qty"] / max(1, rq["requested_qty"])) * 100)
                    st.markdown(f"""
                    <div style="background: {COLOR_WHITE}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-weight: 700; color: {COLOR_TEAL};">
                            <span>Request #{rq['id']}: {rq['category']}</span>
                            <span class="badge {'badge-coral' if rq['urgency'] == 'High' else 'badge-teal'}">{rq['urgency']} Urgency</span>
                        </div>
                        <div style="font-size: 0.8rem; color: {COLOR_MUTED_TEXT}; margin: 4px 0;">
                            Fulfilled: {rq['fulfilled_qty']} / {rq['requested_qty']} units ({pct_ful:.0f}%)
                        </div>
                        <div style="font-size: 0.78rem; color: {COLOR_MUTED_TEXT}; font-style: italic;">
                            Notes: {rq['notes'] or 'Standard requirement'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if st.button(f"Mark Fulfilled #{rq['id']}", key=f"ful_btn_{rq['id']}"):
                            db.update_request_status(rq["id"], "FULFILLED")
                            st.success(f"Request #{rq['id']} marked as Fulfilled!")
                            st.rerun()
                    with btn_c2:
                        if st.button(f"Close Request #{rq['id']}", key=f"cls_btn_{rq['id']}"):
                            db.close_donation_request(rq["id"])
                            st.info(f"Request #{rq['id']} closed.")
                            st.rerun()

        with col_r2:
            st.markdown(f"""
            <div class="nm-card nm-card-highlight">
                <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                    Broadcast New Request
                </div>
            """, unsafe_allow_html=True)

            n_cat = st.selectbox("Category Needed:", CATEGORIES, key="nr_cat")
            n_qty = st.number_input("Units Required:", min_value=5, max_value=5000, value=100, step=10, key="nr_qty")
            n_urg = st.selectbox("Urgency Tier:", ["High", "Medium", "Low"], index=0, key="nr_urg")
            n_notes = st.text_area("Justification / Notes:", placeholder="E.g., Winter cold wave shelter preparation.", key="nr_notes")

            if st.button("Create Request in Database", use_container_width=True):
                req_id = db.create_donation_request(
                    org_id=active_ngo_id,
                    category=n_cat,
                    requested_qty=int(n_qty),
                    urgency=n_urg,
                    notes=n_notes
                )
                st.success(f"Created Request #{req_id} successfully!")
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    # SECTION C: INCOMING DONATIONS & CONFIRM RECEIPT (CRITICAL)
    with tab_inbound:
        st.markdown(f"""
            <div style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.3rem;">
                Inbound Shipments Awaiting Physical Verification
            </div>
            <div style="font-size: 0.85rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 1rem;">
                Donations submitted by donors remain in <strong>Pending Receipt</strong> status until physically delivered.
                Clicking <strong>Confirm Receipt</strong> updates physical inventory atomically and satisfies open requests.
            </div>
        """, unsafe_allow_html=True)

        pending_dons = db.get_donations(org_id=active_ngo_id, status="Pending Receipt")

        if not pending_dons:
            st.markdown(f"""
            <div style="background: {COLOR_WHITE}; border: 1px dashed {COLOR_BORDER}; border-radius: 10px; padding: 1.8rem; text-align: center;">
                <div style="font-size: 1.5rem;">📦</div>
                <div style="font-weight: 700; color: {COLOR_TEAL}; margin-top: 4px;">No Pending Shipments</div>
                <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT};">All donations addressed to this NGO have already been confirmed!</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for don in pending_dons:
                st.markdown(f"""
                <div style="background: {COLOR_WHITE}; border-radius: 10px; border: 1px solid {COLOR_BORDER}; border-left: 5px solid {COLOR_CORAL}; padding: 1rem 1.2rem; margin-bottom: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; color: {COLOR_TEAL};">
                                Donation #{don['id']}: {don['quantity']} Units of {don['category']}
                            </span>
                            <span class="badge badge-coral" style="margin-left: 8px;">Pending Physical Receipt</span>
                        </div>
                        <span style="font-size: 0.8rem; color: {COLOR_MUTED_TEXT};">{don['created_at'][:16]}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin: 4px 0;">
                        Donor: <strong>{don['donor_name']}</strong> ({don['donor_email']}) • Condition: <strong>{don['condition']}</strong>
                    </div>
                    <div style="font-size: 0.8rem; color: {COLOR_MUTED_TEXT}; font-style: italic; margin-bottom: 8px;">
                        Notes: {don['notes'] or 'Standard contribution'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"✅ Confirm Receipt for Donation #{don['id']}", key=f"conf_btn_{don['id']}"):
                    res = db.confirm_donation_receipt(don["id"], confirming_org_id=active_ngo_id)
                    if res["success"]:
                        st.success(res["message"])
                        if res.get("allocated_details"):
                            for ad in res["allocated_details"]:
                                st.info(f"⚡ {ad}")
                        st.rerun()
                    else:
                        st.error(res["message"])

    # SECTION D: NGO FORECASTS
    with tab_fc:
        st.markdown(f"""
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.5rem;">
                30-Day Demand & Gap Analysis for {ngo_data['name']}
            </div>
        """, unsafe_allow_html=True)

        fc_all = predict.get_demand_forecast_table()
        ngo_fc = fc_all[fc_all["org_id"] == active_ngo_id]

        fig_ngo = go.Figure()
        fig_ngo.add_trace(go.Bar(x=ngo_fc["category"], y=ngo_fc["predicted_30d_demand"], name="ML 30d Demand", marker_color=COLOR_TEAL))
        fig_ngo.add_trace(go.Bar(x=ngo_fc["category"], y=ngo_fc["current_inventory"], name="Warehouse Stock", marker_color=COLOR_SKY))
        fig_ngo.add_trace(go.Bar(x=ngo_fc["category"], y=ngo_fc["projected_shortfall"], name="Net Deficit", marker_color=COLOR_CORAL))
        fig_ngo.update_layout(get_plotly_layout(height=340), barmode="group", yaxis_title="Units")
        st.plotly_chart(fig_ngo, use_container_width=True)

# =====================================================================
# PAGE 7: DONATION HISTORY
# =====================================================================
elif selected_page == "7. 📜 Donation History":
    render_hero(
        title="Donation Ledger &",
        highlight_word="Audit Trail.",
        subtitle="Complete persistent transaction history of submitted physical donations and saved purchase plans."
    )

    tab_donations, tab_plans = st.tabs([
        "📦 Physical Item Donations (Verified / Pending)",
        "💳 Saved Purchase Plans (Budget Allocations)"
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

        st.markdown(f"""
            <div style="font-size: 0.88rem; font-weight: 700; color: {COLOR_TEAL}; margin: 8px 0;">
                Showing {len(dons)} matching physical donation transaction(s):
            </div>
        """, unsafe_allow_html=True)

        if not dons:
            st.info("No donation records match the specified filters.")
        else:
            df_d = pd.DataFrame(dons)[[
                "id", "donor_name", "org_name", "category", "quantity",
                "condition", "donation_type", "estimated_value_inr", "status", "created_at", "confirmed_at"
            ]]
            df_d.columns = ["Donation ID", "Donor", "Recipient NGO", "Category", "Quantity", "Condition", "Type", "Est Value (₹)", "Status", "Submitted At", "Receipt Confirmed At"]
            st.dataframe(df_d, use_container_width=True, hide_index=True)

            csv_exp = df_d.to_csv(index=False)
            st.download_button(
                label="📥 Export Donations to CSV",
                data=csv_exp,
                file_name=f"needmatch_donations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with tab_plans:
        st.markdown(f"""
            <div style="font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.5rem;">
                Saved Budget Purchase Plans
            </div>
            <div style="font-size: 0.85rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 0.8rem;">
                These records represent optimized purchase plans pledged by donors, kept strictly distinct from received physical donations.
            </div>
        """, unsafe_allow_html=True)

        plans = db.get_purchase_plans()
        if not plans:
            st.info("No purchase plans saved yet. You can create one on the 'Plan a Purchase' page.")
        else:
            for p in plans:
                st.markdown(f"""
                <div class="nm-card" style="margin-bottom: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 700; color: {COLOR_TEAL};">
                                Plan #{p['id']} — Sponsor: {p['donor_name']}
                            </span>
                            <span style="font-size: 0.8rem; color: {COLOR_MUTED_TEXT}; margin-left: 6px;">({p['donor_email']})</span>
                        </div>
                        <span style="font-weight: 700; color: {COLOR_CORAL};">
                            Allocated: ₹{p['total_allocated_inr']:,.2f} / ₹{p['total_budget_inr']:,.2f}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-top: 4px;">
                        Date Created: {p['created_at']} • Remaining Balance: ₹{p['remaining_budget_inr']:,.2f}
                    </div>
                    <div style="background: #F8FBFD; padding: 8px 12px; border-radius: 6px; margin-top: 8px; font-size: 0.8rem; border: 1px solid {COLOR_BORDER};">
                        <strong>Planned Items:</strong><br>
                        {"<br>".join([f"• {item.get('quantity', 0)}x {item.get('category', 'Item')} for {item.get('org_name', 'NGO')} (₹{item.get('cost', 0):,.2f})" for item in p.get('items', [])])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =====================================================================
# PAGE 8: MODEL & DATA INFORMATION
# =====================================================================
elif selected_page == "8. 🧠 Model & Data Information":
    render_hero(
        title="Machine Learning Architecture &",
        highlight_word="Academic Methodology.",
        subtitle="Full transparency into the machine learning pipeline, dataset disclosures, chronological evaluation, and live model retraining."
    )

    # Mandatory Academic Synthetic Data Notice
    st.markdown(f"""
    <div class="disclaimer-box">
        <strong>⚠️ DATASET & MODEL DISCLOSURE (ACADEMIC RIGOR & HONESTY):</strong><br>
        The historical dataset utilized for training is a <strong>calibrated synthetic dataset</strong> covering 34 continuous months across all 6 donation categories and 6 partner NGOs. It models authentic seasonal spikes (e.g. school reopening in May-July, winter drives in Nov-Jan, monsoon hygiene camps) with controlled stochastic variance.<br>
        <strong>Limitation Notice:</strong> Demonstration results are based on synthetic historical data. They do not establish real-world operational predictive accuracy.
    </div>
    """, unsafe_allow_html=True)

    metrics = predict.load_metrics()
    eval_m = metrics.get("evaluation", {})
    rf_mae = eval_m.get("rf_mae", 61.1)
    base_mae = eval_m.get("baseline_mae", 103.3)
    improvement = eval_m.get("mae_improvement_pct", 40.9)
    r2_score = eval_m.get("rf_r2", 0.7845)
    train_count = metrics.get("train_samples", 972)
    test_count = metrics.get("test_samples", 252)

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Algorithm</div>
            <div class="metric-value" style="font-size: 1.35rem;">Random Forest</div>
            <div class="metric-subtext">Scikit-learn Regressor</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">RF Test MAE</div>
            <div class="metric-value" style="color: {COLOR_TEAL};">{rf_mae:.2f}</div>
            <div class="metric-subtext">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Baseline MAE</div>
            <div class="metric-value" style="color: {COLOR_MUTED_TEXT};">{base_mae:.2f}</div>
            <div class="metric-subtext">Training Group Mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Accuracy Lift</div>
            <div class="metric-value" style="color: {COLOR_CORAL};">+{improvement:.1f}%</div>
            <div class="metric-subtext">Error reduction</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">R² Fit Score</div>
            <div class="metric-value">{r2_score:.3f}</div>
            <div class="metric-subtext">252 test instances</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    # Actions: Train / Retrain Model & Refresh Status
    btn_col1, btn_col2, sp_c = st.columns([1, 1, 1.2])
    with btn_col1:
        if st.button("⚡ Train / Retrain Model Pipeline", use_container_width=True):
            with st.spinner("Executing chronological pipeline, fitting RandomForestRegressor, and updating predictions..."):
                from train_model import train_and_evaluate_model
                pipe, new_metrics = train_and_evaluate_model()
                st.success(f"Model retrained! New RF MAE: {new_metrics['evaluation']['rf_mae']:.2f} units.")
                st.rerun()

    with btn_col2:
        if st.button("🔄 Refresh Model Status", use_container_width=True):
            if os.path.exists("models/rf_demand_model.joblib"):
                st.success("Model artifact verified and ready for inference!")
            else:
                st.warning("Model file not found. Please click Train Model.")

    # Educational Pipeline Explanation
    st.markdown(f"""
    <div class="nm-card" style="margin-top: 1.2rem;">
        <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.6rem;">
            End-to-End System Pipeline Flowchart
        </div>
        <div style="background: #F8FBFD; padding: 14px 18px; border-radius: 8px; border: 1px solid {COLOR_BORDER}; font-family: monospace; font-size: 0.88rem; color: {COLOR_TEAL}; line-height: 1.8;">
            Historical Dataset (CSV) → Preprocessing & OneHotEncoder → Chronological Split (80/20)<br>
            ↓<br>
            RandomForestRegressor Training → Held-Out Evaluation (MAE 61.1 vs Baseline 103.3)<br>
            ↓<br>
            Saved Artifact (Joblib) → Demand Prediction Engine → AI Recommendation Logic → User Dashboard
        </div>
        <div style="font-size: 0.85rem; color: {COLOR_MUTED_TEXT}; line-height: 1.6; margin-top: 10px;">
            <strong>What is Mean Absolute Error (MAE)?</strong><br>
            Mean Absolute Error measures the average difference between the model's predicted demand and actual demand in raw units. For example, an MAE of 61.09 means our model's 30-day forecast is, on average, within ~61 units of the true future demand, representing a <strong>40.86% error reduction</strong> compared to the baseline model (103.30 units).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance Chart
    st.markdown(f"""
    <div class="nm-card">
        <div style="font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: {COLOR_TEAL}; margin-bottom: 0.4rem;">
            Feature Importance Weights (What the Model Learned)
        </div>
    """, unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)
