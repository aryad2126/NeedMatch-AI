"""
NeedMatch AI - UI Style & Design System
Applies the visual aesthetics from the reference design:
Deep Teal (#0F4E6C), Coral (#FF7F6E), Cream (#F9E9D1), White (#FFFFFF), and Sky Blue (#C8DFF2).
Strictly professional iconography (Material Symbols) with zero emojis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Exact Color Palette Constants
COLOR_TEAL = "#0F4E6C"
COLOR_CORAL = "#FF7F6E"
COLOR_CREAM = "#F9E9D1"
COLOR_WHITE = "#FFFFFF"
COLOR_SKY = "#C8DFF2"
COLOR_MUTED_TEXT = "#3D6277"
COLOR_BORDER = "rgba(15, 78, 108, 0.15)"

def inject_custom_css():
    """Injects high-end editorial styling matching the reference image."""
    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {COLOR_TEAL};
    }}

    .stApp {{
        background-color: {COLOR_CREAM};
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_WHITE} !important;
        border-right: 1px solid {COLOR_BORDER};
        box-shadow: 2px 0 16px rgba(15, 78, 108, 0.04);
    }}

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLOR_TEAL} !important;
        font-family: 'Playfair Display', Georgia, serif;
    }}

    /* Sidebar Radio Navigation */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label {{
        padding: 0.45rem 0.75rem !important;
        border-radius: 8px !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
        font-weight: 500 !important;
        color: {COLOR_TEAL} !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:hover {{
        background-color: rgba(200, 223, 242, 0.25) !important;
    }}

    /* Editorial Headings */
    h1, h2, h3, .editorial-heading {{
        font-family: 'Playfair Display', Georgia, serif !important;
        color: {COLOR_TEAL} !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    .hero-title {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.85rem;
        line-height: 1.15;
        font-weight: 700;
        color: {COLOR_TEAL};
        margin-bottom: 0.5rem;
    }}

    .hero-title .highlight {{
        color: {COLOR_CORAL};
        font-style: italic;
    }}

    .hero-subtitle {{
        font-size: 1.08rem;
        line-height: 1.6;
        color: {COLOR_MUTED_TEXT};
        max-width: 780px;
        margin-bottom: 1.5rem;
    }}

    /* Native Container Border Styling (White Distinct Editorial Cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {COLOR_WHITE} !important;
        border-radius: 16px !important;
        border: 1px solid {COLOR_BORDER} !important;
        box-shadow: 0 4px 18px rgba(15, 78, 108, 0.05) !important;
        padding: 1.3rem 1.5rem !important;
        margin-bottom: 1.15rem !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 8px 24px rgba(15, 78, 108, 0.08) !important;
    }}

    /* Native Metric Cards (White Cards with Borders & Elevation) */
    [data-testid="stMetric"] {{
        background-color: {COLOR_WHITE} !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.35rem !important;
        border: 1px solid {COLOR_BORDER} !important;
        box-shadow: 0 4px 16px rgba(15, 78, 108, 0.05) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
        min-height: 124px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(15, 78, 108, 0.12) !important;
        border-color: rgba(15, 78, 108, 0.3) !important;
    }}

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p {{
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: {COLOR_MUTED_TEXT} !important;
    }}

    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {{
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: {COLOR_TEAL} !important;
        line-height: 1.15 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: {COLOR_MUTED_TEXT} !important;
    }}

    /* Outstanding Backlog Metric Highlight (Coral Left Border & Gradient) */
    div[data-testid="column"]:has(button[key*="btn_view_backlog"]) [data-testid="stMetric"] {{
        border-left: 5px solid {COLOR_CORAL} !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F2 100%) !important;
        box-shadow: 0 4px 20px rgba(255, 127, 110, 0.14) !important;
    }}

    div[data-testid="column"]:has(button[key*="btn_view_backlog"]) [data-testid="stMetricValue"] > div {{
        color: {COLOR_CORAL} !important;
    }}

    div[data-testid="column"]:has(button[key*="btn_view_backlog"]) [data-testid="stMetricDelta"] {{
        color: #D84D3B !important;
    }}

    /* Color Swatches Badge Bar */
    .swatches-bar {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: {COLOR_WHITE};
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid {COLOR_BORDER};
        font-size: 0.78rem;
        color: {COLOR_MUTED_TEXT};
        font-weight: 600;
    }}

    .swatch-dot {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
        display: inline-block;
        border: 1px solid rgba(0,0,0,0.1);
    }}

    /* Buttons */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {{
        background-color: {COLOR_CORAL} !important;
        color: {COLOR_WHITE} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(255, 127, 110, 0.3) !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background-color: #E86A59 !important;
        box-shadow: 0 6px 18px rgba(255, 127, 110, 0.42) !important;
        transform: translateY(-1px) !important;
    }}

    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"] {{
        background-color: {COLOR_WHITE} !important;
        color: {COLOR_TEAL} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(15, 78, 108, 0.25) !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 2px 8px rgba(15, 78, 108, 0.05) !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {{
        background-color: #F0F7FB !important;
        border-color: {COLOR_TEAL} !important;
        color: {COLOR_TEAL} !important;
        transform: translateY(-1px) !important;
    }}

    /* Secondary Forms */
    div[data-testid="stFormSubmitButton"] > button {{
        background-color: {COLOR_TEAL} !important;
        color: {COLOR_WHITE} !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
    }}

    /* Dataframe Table Cards */
    [data-testid="stDataFrame"] {{
        background-color: {COLOR_WHITE} !important;
        border-radius: 12px !important;
        border: 1px solid {COLOR_BORDER} !important;
        box-shadow: 0 3px 14px rgba(15, 78, 108, 0.04) !important;
        overflow: hidden !important;
    }}

    /* Tabs Styling */
    button[data-baseweb="tab"] {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        color: {COLOR_MUTED_TEXT} !important;
        font-size: 0.95rem !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLOR_TEAL} !important;
        border-bottom-color: {COLOR_CORAL} !important;
        border-bottom-width: 3px !important;
    }}

    /* Form Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
        border-radius: 8px !important;
        border: 1px solid {COLOR_BORDER} !important;
        background-color: {COLOR_WHITE} !important;
        color: {COLOR_TEAL} !important;
    }}

    /* Alerts & Notifications */
    div[data-testid="stAlert"] {{
        border-radius: 10px !important;
        padding: 0.9rem 1.25rem !important;
        font-size: 0.88rem !important;
        box-shadow: 0 2px 10px rgba(15, 78, 108, 0.04) !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) {{
        background-color: rgba(200, 223, 242, 0.45) !important;
        border: 1px solid {COLOR_SKY} !important;
        border-left: 5px solid {COLOR_TEAL} !important;
        color: {COLOR_TEAL} !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) {{
        background-color: rgba(255, 243, 224, 0.8) !important;
        border: 1px solid #FFE0B2 !important;
        border-left: 5px solid {COLOR_CORAL} !important;
        color: #7A3200 !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) {{
        background-color: rgba(237, 247, 237, 0.85) !important;
        border: 1px solid #C8E6C9 !important;
        border-left: 5px solid #2E7D32 !important;
        color: #1B5E20 !important;
    }}

    div[data-testid="stAlert"]:has([data-testid="stNotificationContentError"]) {{
        background-color: rgba(255, 235, 238, 0.85) !important;
        border: 1px solid #FFCDD2 !important;
        border-left: 5px solid #C62828 !important;
        color: #B71C1C !important;
    }}

    /* Progress bar */
    div[data-testid="stProgress"] > div > div {{
        background-color: {COLOR_CORAL} !important;
        border-radius: 999px !important;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_hero(title: str, highlight_word: str, subtitle: str, badge_text: str = ""):
    """Renders the editorial-style hero section matching the reference image layout."""
    html = f'<div style="margin-bottom: 1.6rem; padding-top: 0.5rem;"><div class="hero-title">{title} <span class="highlight">{highlight_word}</span></div><div class="hero-subtitle">{subtitle}</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def render_ngo_recommendation_card(r: dict, idx: int, is_selected: bool, select_button_key: str = None) -> bool:
    """
    Renders an NGO recommendation card using 100% native Streamlit components.
    Guarantees consistent, error-free rendering with zero raw HTML/CSS exposure and zero emojis.
    Returns True if the selection button for this card was clicked.
    """
    button_clicked = False
    with st.container(border=True):
        col_h1, col_h2 = st.columns([3, 1.4])
        with col_h1:
            st.markdown(f"### #{idx+1} {r['org_name']}")
            st.caption(f"**Location:** {r['city']} • **Focus:** {r['focus']} • **Category:** {r['category']}")
        with col_h2:
            if is_selected:
                st.success("Selected Recipient", icon=":material/check_circle:")
            st.metric(
                label="Priority Score",
                value=f"{r['priority_score']:.0f}/100",
                help="Ranked by live deficit, backlog urgency, and ML demand forecast"
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Stock", f"{r['current_inventory']} units", help="Available units in shelter warehouse")
        m2.metric("Open Backlog", f"{r['requested_backlog']} units", help="Outstanding requested requirement")
        m3.metric("Predicted 30d Demand", f"{r['predicted_demand']:.0f} units", help="Forecasted 30-day demand from ML model")
        m4.metric("Net Shortage", f"{r['unmet_need']:.0f} units", help="Unmet requirement gap after accounting for current inventory and pending shipments")

        st.info(f"**Recommendation Justification:** {r['reason']}", icon=":material/info:")
        st.markdown(f"**Suggested Donation Quantity:** `{r['suggested_qty']} units`")

        if not is_selected and select_button_key:
            if st.button(f"Select {r['org_name']}", key=select_button_key, use_container_width=True, icon=":material/arrow_forward:"):
                button_clicked = True

    return button_clicked

def get_plotly_layout(height: int = 380, title: str = "") -> dict:
    """Returns a unified layout dictionary with the exact brand aesthetic."""
    return dict(
        height=height,
        title=dict(
            text=title,
            font=dict(family="Playfair Display, Georgia, serif", size=16, color=COLOR_TEAL),
            x=0.02
        ),
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=COLOR_TEAL, size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(15, 78, 108, 0.08)",
            linecolor="rgba(15, 78, 108, 0.2)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(15, 78, 108, 0.08)",
            linecolor="rgba(15, 78, 108, 0.2)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )
