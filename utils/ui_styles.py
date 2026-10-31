"""
NeedMatch AI - UI Style & Design System
Applies the visual aesthetics from the reference design:
Deep Teal (#0F4E6C), Coral (#FF7F6E), Cream (#F9E9D1), White (#FFFFFF), and Sky Blue (#C8DFF2).
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
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

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
    }}

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLOR_TEAL} !important;
        font-family: 'Playfair Display', serif;
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

    /* Editorial Card */
    .nm-card {{
        background-color: {COLOR_WHITE};
        border-radius: 16px;
        padding: 1.6rem;
        border: 1px solid {COLOR_BORDER};
        box-shadow: 0 4px 18px rgba(15, 78, 108, 0.04);
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .nm-card:hover {{
        box-shadow: 0 8px 24px rgba(15, 78, 108, 0.08);
    }}

    .nm-card-highlight {{
        border-left: 5px solid {COLOR_CORAL};
    }}

    .nm-card-teal {{
        background: linear-gradient(135deg, {COLOR_WHITE} 0%, #F0F7FB 100%);
        border-left: 5px solid {COLOR_TEAL};
    }}

    /* Metric Cards */
    .metric-container {{
        background: {COLOR_WHITE};
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        border: 1px solid {COLOR_BORDER};
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .metric-label {{
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLOR_MUTED_TEXT};
        margin-bottom: 0.3rem;
    }}

    .metric-value {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: {COLOR_TEAL};
        line-height: 1.1;
    }}

    .metric-subtext {{
        font-size: 0.8rem;
        color: {COLOR_MUTED_TEXT};
        margin-top: 0.4rem;
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
    div.stButton > button:first-child {{
        background-color: {COLOR_CORAL} !important;
        color: {COLOR_WHITE} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 4px 12px rgba(255, 127, 110, 0.28) !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button:first-child:hover {{
        background-color: #E86A59 !important;
        box-shadow: 0 6px 16px rgba(255, 127, 110, 0.38) !important;
        transform: translateY(-1px) !important;
    }}

    /* Secondary Buttons */
    div[data-testid="stFormSubmitButton"] > button {{
        background-color: {COLOR_TEAL} !important;
        color: {COLOR_WHITE} !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
    }}

    /* Badges & Tags */
    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    .badge-coral {{
        background-color: rgba(255, 127, 110, 0.15);
        color: #D84D3B;
        border: 1px solid rgba(255, 127, 110, 0.35);
    }}

    .badge-teal {{
        background-color: rgba(15, 78, 108, 0.12);
        color: {COLOR_TEAL};
        border: 1px solid rgba(15, 78, 108, 0.25);
    }}

    .badge-sky {{
        background-color: {COLOR_SKY};
        color: {COLOR_TEAL};
        border: 1px solid rgba(15, 78, 108, 0.2);
    }}

    /* Form Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
        border-radius: 8px !important;
        border: 1px solid {COLOR_BORDER} !important;
        background-color: {COLOR_WHITE} !important;
        color: {COLOR_TEAL} !important;
    }}

    /* Alert / Disclaimer box */
    .disclaimer-box {{
        background-color: rgba(200, 223, 242, 0.45);
        border: 1px solid {COLOR_SKY};
        border-left: 4px solid {COLOR_TEAL};
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: {COLOR_TEAL};
        margin: 1rem 0;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_hero(title: str, highlight_word: str, subtitle: str, badge_text: str = ""):
    """Renders the editorial-style hero section matching the reference image layout."""
    html = f"""
    <div style="margin-bottom: 1.6rem; padding-top: 0.5rem;">
        <div class="hero-title">
            {title} <span class="highlight">{highlight_word}</span>
        </div>
        <div class="hero-subtitle">
            {subtitle}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def get_plotly_layout(height: int = 380, title: str = "") -> dict:
    """Returns a unified layout dictionary with the exact brand aesthetic."""
    return dict(
        height=height,
        title=dict(
            text=title,
            font=dict(family="Playfair Display, serif", size=16, color=COLOR_TEAL),
            x=0.02
        ),
        paper_bgcolor="rgba(255,255,255,0.9)",
        plot_bgcolor="rgba(255,255,255,0.9)",
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
