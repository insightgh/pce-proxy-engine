import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from main import run_live, run_backtest

# ── NLF Color Palette ──
NLF_NAVY      = "#003366"   # rgb(0,51,102)    — primary
NLF_BLUE      = "#0070C0"   # rgb(0,112,192)   — secondary
NLF_LIGHTBLUE = "#BDD7EE"   # rgb(189,215,238) — surface/borders
NLF_TEAL      = "#055B49"   # rgb(5,91,73)     — positive
NLF_GREY      = "#70797D"   # rgb(112,121,125) — neutral text
NLF_SILVER    = "#BFBFBF"   # rgb(191,191,191) — light grey
NLF_SAGE      = "#869D7A"   # rgb(134,157,122) — muted green
NLF_GOLD      = "#D4B483"   # rgb(212,180,131) — accent warm
NLF_OLIVE     = "#8CB369"   # rgb(140,179,105) — success/good
NLF_BROWN     = "#6E675F"   # rgb(110,103,95)  — dark muted
NLF_BURGUNDY  = "#A4243B"   # rgb(164,36,59)   — negative/error

# --- Page Configuration ---
st.set_page_config(
    page_title="PCE Nowcast Engine | NLF",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS with NLF Palette ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f7f8fa;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 1rem;
    }}

    /* ── Header Bar ── */
    .header-bar {{
        background: linear-gradient(135deg, {NLF_NAVY} 0%, #0a4080 60%, {NLF_BLUE} 100%);
        padding: 28px 36px;
        border-radius: 8px;
        margin-bottom: 24px;
    }}
    .header-bar h1 {{
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.3px;
    }}
    .header-bar p {{
        color: {NLF_LIGHTBLUE} !important;
        font-size: 14px !important;
        margin: 4px 0 0 0 !important;
        font-weight: 400;
    }}

    /* ── Metric Cards ── */
    .metric-card {{
        background: #ffffff;
        border: 1px solid {NLF_LIGHTBLUE};
        border-radius: 8px;
        padding: 22px 24px;
        text-align: center;
        transition: box-shadow 0.2s;
    }}
    .metric-card:hover {{
        box-shadow: 0 3px 14px rgba(0, 51, 102, 0.10);
    }}
    .metric-label {{
        color: {NLF_GREY};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }}
    .metric-value {{
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
    }}
    .metric-value.primary   {{ color: {NLF_NAVY}; }}
    .metric-value.positive  {{ color: {NLF_TEAL}; }}
    .metric-value.negative  {{ color: {NLF_BURGUNDY}; }}
    .metric-value.neutral   {{ color: {NLF_GREY}; }}
    .metric-unit {{
        color: {NLF_SILVER};
        font-size: 12px;
        font-weight: 400;
        margin-top: 4px;
    }}

    /* ── Section Headers ── */
    .section-header {{
        font-size: 16px;
        font-weight: 600;
        color: {NLF_NAVY};
        margin: 28px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid {NLF_LIGHTBLUE};
    }}

    /* ── Stat Pills ── */
    .stat-pill {{
        display: inline-block;
        background: {NLF_LIGHTBLUE};
        border-radius: 6px;
        padding: 6px 14px;
        margin-right: 8px;
        font-size: 13px;
        color: {NLF_NAVY};
        font-weight: 500;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background-color: #f9fafb;
        border-right: 1px solid {NLF_LIGHTBLUE};
    }}
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {NLF_NAVY} !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }}

    /* ── Global Overrides ── */
    h1, h2, h3 {{ color: {NLF_NAVY} !important; }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {NLF_NAVY}, {NLF_BLUE});
        border: none;
        padding: 10px 28px;
        font-weight: 600;
        letter-spacing: 0.3px;
        color: #ffffff;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, {NLF_BLUE}, #0088e0);
    }}

    /* ── Footer ── */
    .footer {{
        text-align: center;
        padding: 20px 0 8px 0;
        color: {NLF_GREY};
        font-size: 12px;
        border-top: 1px solid {NLF_LIGHTBLUE};
        margin-top: 32px;
    }}

    /* ── Landing Logo Block ── */
    .landing-block {{
        text-align: center;
        padding: 60px 0 40px 0;
    }}
    .landing-block h3 {{
        color: {NLF_NAVY} !important;
        font-weight: 700;
        margin-bottom: 8px;
    }}
    .landing-block p {{
        color: {NLF_GREY};
        font-size: 15px;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.65;
    }}

    /* ── Methodology Cards ── */
    .method-card {{
        background: #ffffff;
        border: 1px solid {NLF_LIGHTBLUE};
        border-radius: 8px;
        padding: 20px 22px;
        text-align: left;
        min-height: 150px;
        border-top: 3px solid {NLF_NAVY};
    }}
    .method-card .method-title {{
        font-weight: 700;
        color: {NLF_NAVY};
        font-size: 14px;
        margin-bottom: 8px;
    }}
    .method-card .method-desc {{
        color: {NLF_GREY};
        font-size: 13px;
        line-height: 1.55;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown(f"""
<div class="header-bar">
    <h1>PCE Nowcast Engine</h1>
    <p>High-frequency inflation forecasting &mdash; Nittany Lion Fund, LLC</p>
</div>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    # NLF Logo in sidebar
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "nittany_lion_fund-removebg-preview.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
        st.markdown("")

    st.markdown("### Configuration")
    st.markdown("")

    is_core = st.toggle("Core PCE", value=False, help="Excludes Food & Energy components")
    lookback_years = st.slider("Backtest Window", min_value=1, max_value=10, value=3, help="Years of historical data")

    st.markdown("---")

    st.markdown("##### Model Specifications")
    st.markdown(f"""
    <div style="font-size: 13px; color: {NLF_GREY}; line-height: 1.8;">
        <strong style="color: {NLF_NAVY};">Aggregation:</strong> Tornqvist<br>
        <strong style="color: {NLF_NAVY};">Weights:</strong> BEA Dynamic<br>
        <strong style="color: {NLF_NAVY};">Formula Drag:</strong> EWM (span=12)<br>
        <strong style="color: {NLF_NAVY};">Seasonal:</strong> LOESS RSA<br>
        <strong style="color: {NLF_NAVY};">Components:</strong> 23 / 20 (core)<br>
        <strong style="color: {NLF_NAVY};">Cap:</strong> Adaptive (2.5&sigma;)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    mode_label = "Core PCE" if is_core else "Headline PCE"
    st.markdown(f"""
    <div style="text-align: center; padding: 8px 0;">
        <span class="stat-pill"><strong>{mode_label}</strong></span>
        <span class="stat-pill"><strong>{lookback_years}Y</strong> window</span>
    </div>
    """, unsafe_allow_html=True)


# --- Main Content ---
run_btn = st.button("Run Nowcast", type="primary", use_container_width=True)

if run_btn:
    # ── Live Forecast ──
    st.markdown('<div class="section-header">Live Forecast</div>', unsafe_allow_html=True)

    with st.spinner("Fetching BLS, BEA, and FRED data..."):
        try:
            live_data = run_live(is_core=is_core, verbose=False)
        except Exception as e:
            st.error(f"Engine error: {e}")
            st.stop()

    forecast_date = live_data['date_str']
    adj_mom = live_data['adjusted_mom']
    raw_mom = live_data['raw_proxy_mom']
    rsa = live_data['rsa_factor']

    val_class = "positive" if adj_mom > 0 else ("negative" if adj_mom < 0 else "neutral")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecast Period</div>
            <p class="metric-value primary" style="font-size: 22px;">{forecast_date}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Adjusted MoM</div>
            <p class="metric-value {val_class}">{adj_mom:+.3f}%</p>
            <div class="metric-unit">seasonally adjusted</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Raw Proxy</div>
            <p class="metric-value primary">{raw_mom:+.3f}%</p>
            <div class="metric-unit">pre-seasonal</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        rsa_class = "positive" if rsa > 0 else ("negative" if rsa < 0 else "neutral")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSA Factor</div>
            <p class="metric-value {rsa_class}">{rsa:+.4f}</p>
            <div class="metric-unit">percentage points</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Backtest ──
    st.markdown(f'<div class="section-header">Historical Accuracy &mdash; {lookback_years}-Year Backtest</div>', unsafe_allow_html=True)

    with st.spinner("Running backtest..."):
        try:
            backtest_data = run_backtest(years=lookback_years, is_core=is_core, verbose=False)
        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.stop()

    df = backtest_data['dataframe']
    mae = backtest_data['mae']

    clean_df = df[~((df["date"] >= '2020-03-01') & (df["date"] <= '2021-12-01'))]
    rmse = np.sqrt((clean_df["adjusted_error"] ** 2).mean())
    hit_rate_5bps = (clean_df["adjusted_error"].abs() <= 0.05).mean() * 100
    max_miss = clean_df["adjusted_error"].abs().max()

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ex-COVID MAE</div>
            <p class="metric-value primary">{mae:.4f}</p>
            <div class="metric-unit">percentage points</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <p class="metric-value primary">{rmse:.4f}</p>
            <div class="metric-unit">percentage points</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hit Rate (&le;5 bps)</div>
            <p class="metric-value primary">{hit_rate_5bps:.0f}%</p>
            <div class="metric-unit">of months within 5 bps</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Worst Miss</div>
            <p class="metric-value negative">{max_miss:.3f}</p>
            <div class="metric-unit">percentage points</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        fig_track = go.Figure()

        fig_track.add_trace(go.Scatter(
            x=df['date'], y=df['actual_mom'],
            mode='lines',
            name='Actual PCE',
            line=dict(color=NLF_GREY, width=1.5),
        ))
        fig_track.add_trace(go.Scatter(
            x=df['date'], y=df['adjusted_proxy'],
            mode='lines',
            name='Proxy Forecast',
            line=dict(color=NLF_NAVY, width=2.2),
        ))

        fig_track.add_vrect(
            x0='2020-03-01', x1='2021-12-01',
            fillcolor=f'rgba(164, 36, 59, 0.06)',
            line_width=0,
            annotation_text="COVID",
            annotation_position="top left",
            annotation_font=dict(size=10, color=NLF_BURGUNDY),
        )

        fig_track.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=0, r=16, t=36, b=0),
            title=dict(text="Actual vs Proxy MoM", font=dict(size=14, color=NLF_NAVY)),
            hovermode="x unified",
            xaxis=dict(title="", gridcolor="#eef1f5"),
            yaxis=dict(title="MoM %", gridcolor="#eef1f5", zeroline=True, zerolinecolor=NLF_LIGHTBLUE),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11, color=NLF_NAVY),
            ),
            plot_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_track, use_container_width=True)

    with chart_col2:
        fig_hist = go.Figure()

        fig_hist.add_trace(go.Histogram(
            x=clean_df['adjusted_error'],
            nbinsx=30,
            marker_color=NLF_BLUE,
            opacity=0.80,
            name='Adjusted Error',
        ))

        fig_hist.add_vline(x=0, line_dash="dash", line_color=NLF_BURGUNDY, line_width=1.5)

        fig_hist.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=0, r=16, t=36, b=0),
            title=dict(text="Error Distribution (Ex-COVID)", font=dict(size=14, color=NLF_NAVY)),
            xaxis=dict(title="Error (pp)", gridcolor="#eef1f5"),
            yaxis=dict(title="Frequency", gridcolor="#eef1f5"),
            showlegend=False,
            plot_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Error over time ──
    fig_err = go.Figure()

    fig_err.add_trace(go.Bar(
        x=clean_df['date'],
        y=clean_df['adjusted_error'],
        marker_color=clean_df['adjusted_error'].apply(
            lambda e: NLF_TEAL if abs(e) <= 0.05 else (NLF_GOLD if abs(e) <= 0.10 else NLF_BURGUNDY)
        ),
        name='Tracking Error',
        opacity=0.85,
    ))

    fig_err.add_hline(y=0.07, line_dash="dot", line_color=NLF_SILVER, line_width=1,
                      annotation_text="+7 bps", annotation_position="right",
                      annotation_font=dict(size=10, color=NLF_GREY))
    fig_err.add_hline(y=-0.07, line_dash="dot", line_color=NLF_SILVER, line_width=1,
                      annotation_text="-7 bps", annotation_position="right",
                      annotation_font=dict(size=10, color=NLF_GREY))

    fig_err.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=0, r=16, t=36, b=0),
        title=dict(text="Month-by-Month Tracking Error", font=dict(size=14, color=NLF_NAVY)),
        xaxis=dict(title="", gridcolor="#eef1f5"),
        yaxis=dict(title="Error (pp)", gridcolor="#eef1f5", zeroline=True, zerolinecolor=NLF_NAVY),
        showlegend=False,
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # ── Raw Data Table ──
    with st.expander("View Raw Backtest Data"):
        display_df = df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m')

        def color_error(val):
            if isinstance(val, (int, float)):
                color = NLF_TEAL if abs(val) <= 0.05 else NLF_BURGUNDY
                return f'color: {color}'
            return ''

        styled = display_df.style.format({
            'actual_mom': '{:.3f}',
            'proxy_mom_pct': '{:.3f}',
            'rsa_factor': '{:.4f}',
            'adjusted_proxy': '{:.3f}',
            'raw_error': '{:.3f}',
            'adjusted_error': '{:.3f}',
        }).map(color_error, subset=['adjusted_error'])

        st.dataframe(styled, use_container_width=True, height=400)

else:
    # ── Landing Page ──
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "nittany_lion_fund-removebg-preview.png")
    st.markdown('<div class="landing-block">', unsafe_allow_html=True)
    if os.path.exists(logo_path):
        lcol1, lcol2, lcol3 = st.columns([1, 1, 1])
        with lcol2:
            st.image(logo_path, width=260)
    st.markdown(f"""
        <h3>Ready to Forecast</h3>
        <p>
            Configure your parameters in the sidebar, then click <strong>Run Nowcast</strong> to generate
            a live PCE estimate and historical accuracy backtest.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Methodology</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    methods = [
        ("Tornqvist Index", "Geometric aggregation via log-change formula to replicate the BEA's Fisher chained methodology."),
        ("Dynamic Weights", "Monthly expenditure shares from BEA Table U20405 capture real-time shifts in consumer behavior."),
        ("Formula Drag", "Exponentially-weighted trailing spread corrects for structural CPI-to-PCE methodology divergence."),
        ("Residual Seasonal", "LOESS decomposition extracts pure seasonal factors with pandemic anomaly masking."),
    ]
    for col, (title, desc) in zip([m1, m2, m3, m4], methods):
        with col:
            st.markdown(f"""
            <div class="method-card">
                <div class="method-title">{title}</div>
                <div class="method-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Additional upgrades section
    st.markdown('<div class="section-header">Engine Upgrades</div>', unsafe_allow_html=True)
    u1, u2, u3 = st.columns(3)
    upgrades = [
        ("Adaptive Caps", "Rolling 2.5-sigma volatility caps replace static 7 bps limit. Tighter in calm markets, flexible during regime shifts."),
        ("OER Dampening", "Housing OER scaled by 0.85x to match the BEA's smoother imputed rent methodology vs CPI."),
        ("EWM Drag", "Exponential decay weighting on formula drag allows faster adaptation to methodology changes."),
    ]
    for col, (title, desc) in zip([u1, u2, u3], upgrades):
        with col:
            st.markdown(f"""
            <div class="method-card">
                <div class="method-title">{title}</div>
                <div class="method-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# --- Footer ---
st.markdown(f"""
<div class="footer">
    PCE Nowcast Engine &mdash; Nittany Lion Fund, LLC &mdash; Quantitative Research
</div>
""", unsafe_allow_html=True)
