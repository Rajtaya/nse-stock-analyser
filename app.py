"""
app.py  —  NSE Stock Analyser  v12  (final clean build)
========================================================
Run:  streamlit run app.py
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from utils.data_fetcher import fetch_stock_data, fetch_stock_info, TIMEFRAMES, is_commodity
from utils.indicators   import compute_all_indicators
from utils.analysis     import (detect_trend, find_support_resistance,
                                generate_signal, calculate_risk_levels)
from components.charts  import (candlestick_chart, rsi_chart, macd_chart,
                                sr_chart, trend_chart)
from components.metrics import (render_signal_card, render_trend_badge,
                                render_risk_panel, render_signal_reasons,
                                render_latest_indicators)

# ── Live chart helpers ────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_candles(ticker: str):
    """Fetch 1-day 1-min candles from Yahoo Finance for the live chart."""
    import yfinance as yf
    yf_sym = ticker if (ticker.endswith('=F') or ticker.startswith('^')) else ticker + '.NS'
    try:
        df = yf.download(yf_sym, period='1d', interval='1m',
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, __import__('pandas').MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.index = __import__('pandas').to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    except Exception:
        return None

# ── Watchlist ──────────────────────────────────────────────────────────────────
WATCHLIST = {
    "🛢️ NATURAL GAS  —  NYMEX Futures (USD)":   ("NG=F",      "Commodities"),
    "🛢️ CRUDE OIL WTI  —  NYMEX Futures (USD)":  ("CL=F",      "Commodities"),
    "🛢️ BRENT CRUDE  —  ICE Futures (USD)":      ("BZ=F",      "Commodities"),
    "🥇 GOLD  —  COMEX Futures (USD)":            ("GC=F",      "Commodities"),
    "🥈 SILVER  —  COMEX Futures (USD)":          ("SI=F",      "Commodities"),
    "⛽ ONGC  —  Oil & Natural Gas Corp":         ("ONGC",       "Natural Gas & Energy"),
    "⛽ GAIL  —  GAIL India Ltd":                 ("GAIL",       "Natural Gas & Energy"),
    "⛽ IGL   —  Indraprastha Gas":               ("IGL",        "Natural Gas & Energy"),
    "⛽ MGL   —  Mahanagar Gas":                  ("MGL",        "Natural Gas & Energy"),
    "⛽ ATGL  —  Adani Total Gas":                ("ATGL",       "Natural Gas & Energy"),
    "⛽ GUJGAS  —  Gujarat Gas":                  ("GUJGASLTD",  "Natural Gas & Energy"),
    "⛽ GSPL  —  Gujarat State Petronet":         ("GSPL",       "Natural Gas & Energy"),
    "🛡️ BEL   —  Bharat Electronics":            ("BEL",        "Defence & Aerospace"),
    "🛡️ HAL   —  Hindustan Aeronautics":         ("HAL",        "Defence & Aerospace"),
    "💻 TCS   —  Tata Consultancy":              ("TCS",        "Information Technology"),
    "💻 INFY  —  Infosys":                       ("INFY",       "Information Technology"),
    "💻 WIPRO —  Wipro Ltd":                     ("WIPRO",      "Information Technology"),
    "🏦 HDFCBANK  —  HDFC Bank":                 ("HDFCBANK",   "Banking & Finance"),
    "🏭 TATASTEEL  —  Tata Steel":               ("TATASTEEL",  "Metals & Materials"),
    "🏢 RELIANCE  —  Reliance Ind.":             ("RELIANCE",   "Conglomerate"),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {
    background-color: #0d1117 !important;
    color: #c9d1d9;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
}
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; }
.stTabs [data-baseweb="tab-list"] {
    background: transparent; gap: 2px; border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #8b949e !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important; font-weight: 700 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    padding: 10px 18px !important;
}
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }
.stButton > button {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important;
}
.stButton > button:hover { background: #30363d !important; border-color: #58a6ff !important; color: #58a6ff !important; }
.stSelectbox > div > div { background: #161b22 !important; border-color: #30363d !important; }
.stNumberInput > div > div > input { background: #161b22 !important; color: #c9d1d9 !important; }
.stTextInput > div > div > input {
    background: #161b22 !important; color: #c9d1d9 !important;
    border-color: #30363d !important; font-family: 'IBM Plex Mono', monospace !important;
}
.stDataFrame { background: #161b22 !important; }
.stDataFrame th { background: #161b22 !important; color: #8b949e !important; }
.stDataFrame td { background: #0d1117 !important; color: #c9d1d9 !important; }
.streamlit-expanderHeader {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 8px !important; color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important;
}
hr { border-color: #21262d !important; }
.sec-hdr {
    font-size: 0.72rem; font-weight: 700; color: #8b949e; text-transform: uppercase;
    letter-spacing: 2px; border-bottom: 1px solid #21262d; padding-bottom: 6px;
    margin-bottom: 14px; margin-top: 4px; font-family: 'IBM Plex Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 22px;">
        <div style="font-size:2.2rem;">📊</div>
        <div style="font-size:1rem;font-weight:800;color:#58a6ff;letter-spacing:1px;">NSE Analyser</div>
        <div style="font-size:0.7rem;color:#8b949e;letter-spacing:2px;text-transform:uppercase;">Technical Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Sector Filter</div>', unsafe_allow_html=True)
    all_sectors = ["All Sectors"] + sorted(set(v[1] for v in WATCHLIST.values()))
    chosen_sector = st.selectbox("", all_sectors, index=1, label_visibility="collapsed")

    filtered_list = (list(WATCHLIST.keys()) if chosen_sector == "All Sectors"
                     else [k for k, v in WATCHLIST.items() if v[1] == chosen_sector])

    st.markdown('<div class="sec-hdr" style="margin-top:14px">Stock</div>', unsafe_allow_html=True)
    selected_name  = st.selectbox("", filtered_list, index=0, label_visibility="collapsed")
    selected_stock = WATCHLIST[selected_name][0]
    stock_sector   = WATCHLIST[selected_name][1]

    custom = st.text_input("Or type any NSE code:", placeholder="e.g. PETRONET")
    if custom.strip():
        selected_stock = custom.strip().upper()
        stock_sector   = "Custom"

    st.markdown('<div class="sec-hdr" style="margin-top:18px">Timeframe</div>', unsafe_allow_html=True)
    timeframe = st.radio("", list(TIMEFRAMES.keys()), index=2, label_visibility="collapsed")

    st.markdown('<div class="sec-hdr" style="margin-top:18px">Display</div>', unsafe_allow_html=True)
    show_bb = st.checkbox("Bollinger Bands",    value=True)
    show_sr = st.checkbox("Support/Resistance", value=True)

    st.markdown("---")
    col_r, col_c = st.columns(2)
    with col_r:
        if st.button("⟳  Refresh", use_container_width=True):
            st.cache_data.clear()
    with col_c:
        if st.button("🗑 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown(f"""
    <div style="color:#4a5568;font-size:0.7rem;text-align:center;margin-top:12px;">
        {datetime.now().strftime('%d %b %Y  %H:%M')}
    </div>
    """, unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
tf = TIMEFRAMES[timeframe]

@st.cache_data(ttl=60, show_spinner=False)
def load_data(ticker: str, period: str, interval: str):
    raw = fetch_stock_data(ticker, period=period, interval=interval)
    if raw is None or raw.empty:
        return None, {}
    return compute_all_indicators(raw), fetch_stock_info(ticker)

with st.spinner(f"Fetching {selected_stock}…"):
    df, info = load_data(selected_stock, tf["period"], tf["interval"])

if df is None or df.empty:
    st.error(
        f"**Could not fetch data for `{selected_stock}`.**\n\n"
        "Possible reasons: incorrect ticker, no internet, or Yahoo Finance rate-limit. "
        "Try again in a moment."
    )
    st.stop()

# ── Derived values ────────────────────────────────────────────────────────────
latest_price = float(df["Close"].iloc[-1])
prev_price   = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_price
price_change = latest_price - prev_price
pct_change   = (price_change / prev_price * 100) if prev_price else 0.0
chg_color    = "#3fb950" if price_change >= 0 else "#f85149"
arrow        = "▲" if price_change >= 0 else "▼"

trend    = detect_trend(df)
signal   = generate_signal(df)
risk     = calculate_risk_levels(df)
supports, resistances = (find_support_resistance(df) if show_sr else ([], []))

company  = info.get("longName",          selected_stock) if info else selected_stock
mkt_cap  = info.get("marketCap",         None)           if info else None
h52      = info.get("fiftyTwoWeekHigh",  None)           if info else None
l52      = info.get("fiftyTwoWeekLow",   None)           if info else None

is_comm   = is_commodity(selected_stock)
curr_sym  = "$" if is_comm else "₹"
curr_name = "USD" if is_comm else "INR"

mc_str  = f"{curr_sym}{mkt_cap/1e9:,.1f}B" if mkt_cap else "—"
h52_str = f"{curr_sym}{h52:,.2f}"           if h52     else "—"
l52_str = f"{curr_sym}{l52:,.2f}"           if l52     else "—"

if is_comm:
    COMM_NAMES = {
        "NG=F": "Natural Gas Futures", "CL=F": "Crude Oil WTI Futures",
        "BZ=F": "Brent Crude Futures", "GC=F": "Gold Futures", "SI=F": "Silver Futures",
    }
    company = COMM_NAMES.get(selected_stock.upper(), company)

SECTOR_EMOJI = {
    "Commodities": "🛢️", "Natural Gas & Energy": "⛽", "Defence & Aerospace": "🛡️",
    "Information Technology": "💻", "Banking & Finance": "🏦",
    "Metals & Materials": "🏭", "Conglomerate": "🏢", "Custom": "🔍",
}
sector_emoji   = SECTOR_EMOJI.get(stock_sector, "📊")
exchange_label = "NYMEX/COMEX · COMMODITY" if is_comm else "NSE"
ticker_display = selected_stock if is_comm else f"{selected_stock}.NS"

# Live price from Kite fast_info
try:
    import yfinance as yf
    fi = yf.Ticker(ticker_display).fast_info
    live_p = getattr(fi, 'last_price', None)
    if live_p and live_p > 0:
        latest_price = float(live_p)
        price_change = latest_price - prev_price
        pct_change   = (price_change / prev_price * 100) if prev_price else 0.0
        chg_color    = "#3fb950" if price_change >= 0 else "#f85149"
        arrow        = "▲" if price_change >= 0 else "▼"
    live_badge = '<span style="background:#0d2818;color:#3fb950;border:1px solid #3fb950;border-radius:4px;font-size:0.65rem;padding:2px 8px;margin-left:8px;letter-spacing:1px;">● LIVE</span>'
except Exception:
    live_badge = ''

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
     border:1px solid #30363d;border-radius:14px;padding:22px 28px;margin-bottom:20px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="color:#8b949e;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;">
          {exchange_label} · {sector_emoji} {stock_sector}</div>
      <div style="font-size:1.7rem;font-weight:800;color:#58a6ff;letter-spacing:1px;margin:4px 0;">{company}</div>
      <div style="color:#8b949e;font-size:0.8rem;">{ticker_display} · {timeframe} · {len(df)} sessions · {curr_name}</div>
    </div>
    <div style="text-align:right;">
      <div style="color:#8b949e;font-size:0.72rem;letter-spacing:1px;">LAST PRICE {live_badge}</div>
      <div style="font-size:2.1rem;font-weight:800;color:#c9d1d9;line-height:1.1;">{curr_sym} {latest_price:,.2f}</div>
      <div style="font-size:0.95rem;font-weight:700;color:{chg_color};">
          {arrow} {curr_sym}{abs(price_change):.2f} ({abs(pct_change):.2f}%)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
def kpi(label, value, color="#c9d1d9"):
    return (f"<div style='background:#161b22;border:1px solid #21262d;border-radius:8px;"
            f"padding:12px;text-align:center;'>"
            f"<div style='color:#8b949e;font-size:0.7rem;margin-bottom:4px;"
            f"text-transform:uppercase;letter-spacing:1px;'>{label}</div>"
            f"<div style='color:{color};font-weight:700;font-family:monospace;"
            f"font-size:0.95rem;'>{value}</div></div>")

k = st.columns(5)
k[0].markdown(kpi("52W High", h52_str, "#3fb950"), unsafe_allow_html=True)
k[1].markdown(kpi("52W Low",  l52_str, "#f85149"), unsafe_allow_html=True)
k[2].markdown(kpi("Mkt Cap",  mc_str,  "#c9d1d9"), unsafe_allow_html=True)
k[3].markdown(kpi("Currency", curr_name, "#58a6ff"), unsafe_allow_html=True)
k[4].markdown(kpi("Sessions", str(len(df)), "#e3b341"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
render_latest_indicators(df, curr_sym=curr_sym)
st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_price, tab_ind, tab_ai, tab_risk, tab_data = st.tabs([
    "⚡  Live Chart",
    "📈  Price Chart",
    "🔬  Indicators",
    "🤖  AI Analysis",
    "🛡️  Risk Mgmt",
    "📋  Raw Data",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 0 · Live Chart  (Yahoo Finance 1-min, cloud-compatible)
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="sec-hdr">Live 1-Minute Candlestick Chart</div>',
                unsafe_allow_html=True)

    # Status badge
    st.markdown("""
    <div style="background:#1a1500;border:1px solid #f0b429;border-radius:8px;
         padding:10px 16px;margin-bottom:14px;font-size:0.78rem;color:#f0b429;
         display:flex;align-items:center;gap:10px;">
      <span style="font-size:1rem;">●</span>
      <span>Yahoo Finance · 1-min candles · auto-refreshes every 60 seconds</span>
    </div>
    """, unsafe_allow_html=True)

    # Symbol picker
    live_syms = ["BEL","HAL","ONGC","GAIL","IGL","MGL","ATGL","GUJGASLTD","GSPL",
                 "TCS","INFY","WIPRO","HDFCBANK","TATASTEEL","RELIANCE",
                 "CL=F","NG=F","BZ=F","GC=F","SI=F"]
    _cols = st.columns(10)
    for i, sym in enumerate(live_syms):
        if _cols[i % 10].button(sym, key=f"lv_{sym}", use_container_width=True):
            st.session_state["live_sym"] = sym

    live_sym = st.session_state.get("live_sym", selected_stock)
    st.markdown(f"**Showing:** `{live_sym}`")

    live_df = fetch_live_candles(live_sym)

    if live_df is not None and not live_df.empty:
        import plotly.graph_objects as _go
        from plotly.subplots import make_subplots as _msp

        _fig = _msp(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25], vertical_spacing=0.03)

        _green, _red = "#3fb950", "#f85149"
        _colors = [_green if float(c) >= float(o) else _red
                   for c, o in zip(live_df["Close"], live_df["Open"])]

        _fig.add_trace(_go.Candlestick(
            x=live_df.index,
            open=live_df["Open"], high=live_df["High"],
            low=live_df["Low"],  close=live_df["Close"],
            name="1-min",
            increasing=dict(fillcolor=_green, line=dict(color=_green, width=1)),
            decreasing=dict(fillcolor=_red,   line=dict(color=_red,   width=1)),
        ), row=1, col=1)

        if "Volume" in live_df.columns:
            _fig.add_trace(_go.Bar(
                x=live_df.index, y=live_df["Volume"],
                name="Volume", marker=dict(color=_colors, opacity=0.6),
            ), row=2, col=1)

        _fig.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", family="monospace", size=11),
            height=600, margin=dict(l=10, r=60, t=40, b=20),
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            xaxis=dict(gridcolor="#21262d", zeroline=False, color="#8b949e"),
            xaxis2=dict(gridcolor="#21262d", zeroline=False, color="#8b949e"),
            yaxis=dict(gridcolor="#21262d", zeroline=False, side="right", color="#8b949e"),
            yaxis2=dict(gridcolor="#21262d", zeroline=False, side="right", color="#8b949e"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e", size=10)),
            title=dict(text=f"{live_sym} · 1-min · Today", font=dict(color="#c9d1d9", size=13)),
        )

        _last = float(live_df["Close"].iloc[-1])
        _prev = float(live_df["Close"].iloc[-2]) if len(live_df) > 1 else _last
        _chg  = _last - _prev
        _pct  = (_chg / _prev * 100) if _prev else 0
        _sym_c = "$" if is_commodity(live_sym) else "₹"
        _col = "#3fb950" if _chg >= 0 else "#f85149"
        st.markdown(
            f"<div style='font-size:1.4rem;font-weight:800;color:{_col};"
            f"font-family:monospace;margin-bottom:8px;'>"
            f"{_sym_c}{_last:,.2f} &nbsp; "
            f"{'▲' if _chg>=0 else '▼'} {_sym_c}{abs(_chg):.2f} ({abs(_pct):.2f}%)"
            f"</div>", unsafe_allow_html=True)

        st.plotly_chart(_fig, use_container_width=True,
                        config={"displayModeBar": True, "scrollZoom": True})

        # Auto-refresh
        import time as _t
        if st.button("🔄 Refresh Now", key="live_refresh"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Last updated: {_t.strftime('%H:%M:%S')}  ·  {len(live_df)} candles today")
    else:
        st.warning(f"No intraday data available for `{live_sym}` right now. "
                   "Market may be closed or data unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 · Price Chart
# ══════════════════════════════════════════════════════════════════════════════
with tab_price:
    st.plotly_chart(
        candlestick_chart(df, selected_stock),
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    if show_sr and (supports or resistances):
        st.markdown('<div class="sec-hdr">Support & Resistance</div>', unsafe_allow_html=True)
        st.plotly_chart(
            sr_chart(df, supports, resistances, selected_stock, curr_sym=curr_sym),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 Support Levels**")
            for i, lvl in enumerate(supports, 1):
                pct = (latest_price - lvl) / latest_price * 100
                st.markdown(f"S{i}: **{curr_sym}{lvl:,.2f}** &nbsp;&nbsp; `{pct:.1f}% below`")
        with c2:
            st.markdown("**🔴 Resistance Levels**")
            for i, lvl in enumerate(resistances, 1):
                pct = (lvl - latest_price) / latest_price * 100
                st.markdown(f"R{i}: **{curr_sym}{lvl:,.2f}** &nbsp;&nbsp; `{pct:.1f}% above`")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 · Indicators
# ══════════════════════════════════════════════════════════════════════════════
with tab_ind:
    st.markdown('<div class="sec-hdr">RSI — Relative Strength Index (14)</div>', unsafe_allow_html=True)
    if "RSI" in df.columns and df["RSI"].notna().any():
        st.plotly_chart(rsi_chart(df), use_container_width=True)
        rsi_now = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else 50
        if rsi_now > 70:
            st.warning(f"⚠️  RSI **{rsi_now:.1f}** — Overbought zone.")
        elif rsi_now < 30:
            st.success(f"✅  RSI **{rsi_now:.1f}** — Oversold zone.")
        else:
            st.info(f"ℹ️  RSI **{rsi_now:.1f}** — Neutral zone (30–70).")
    else:
        st.info("Not enough data for RSI (need ≥ 14 sessions).")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">MACD — (12, 26, 9)</div>', unsafe_allow_html=True)
    if "MACD" in df.columns and df["MACD"].notna().any():
        st.plotly_chart(macd_chart(df), use_container_width=True)
        hist_now = df["MACD_Hist"].iloc[-1]
        if not pd.isna(hist_now):
            if float(hist_now) > 0:
                st.success("✅  MACD Histogram positive — Bullish momentum.")
            else:
                st.warning("⚠️  MACD Histogram negative — Bearish momentum.")
    else:
        st.info("Not enough data for MACD (need ≥ 26 sessions).")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Price Trend</div>', unsafe_allow_html=True)
    st.plotly_chart(trend_chart(df, selected_stock), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 · AI Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    col_sig, col_info = st.columns([1, 2])
    with col_sig:
        st.markdown('<div class="sec-hdr">Signal</div>', unsafe_allow_html=True)
        render_signal_card(signal)
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Trend</div>', unsafe_allow_html=True)
        render_trend_badge(trend)
    with col_info:
        st.markdown('<div class="sec-hdr">Signal Reasoning</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px 20px;margin-bottom:12px;">
        """, unsafe_allow_html=True)
        render_signal_reasons(signal.get("reasons", []))
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("ℹ️  How are signals generated?"):
            st.markdown("""
| Indicator | Bullish (+2) | Bearish (−2) |
|---|---|---|
| RSI | < 30 oversold | > 70 overbought |
| MACD | Bullish crossover | Bearish crossover |
| Moving Avg | Price > SMA-50 & 200 | Price < SMA-50 & 200 |
| Bollinger | Near lower band | Near upper band |
| Volume | Spike confirms up | Spike confirms down |

**Score ≥ 3 → BUY  ·  Score ≤ −3 → SELL  ·  Otherwise → HOLD**

*Rule-based model for educational purposes. Not financial advice.*
            """)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 · Risk Management
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown('<div class="sec-hdr">Risk / Reward Levels</div>', unsafe_allow_html=True)
        render_risk_panel(risk, curr_sym=curr_sym)
    with r2:
        st.markdown('<div class="sec-hdr">Methodology</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
             padding:16px 20px;font-size:0.85rem;line-height:1.8;color:#8b949e;">
        <strong style="color:#c9d1d9">Stop Loss</strong><br>
        Set at 1.5 × ATR(14) below entry. ATR adapts to current volatility automatically.<br><br>
        <strong style="color:#c9d1d9">Target Prices</strong><br>
        Calculated at 1.5:1, 2:1, and 3:1 reward-to-risk ratios.<br><br>
        <strong style="color:#c9d1d9">Position Sizing Rule</strong><br>
        Risk 1–2% of capital per trade.<br>
        {"Contracts" if is_comm else "Shares"} = (Capital × Risk%) ÷ Stop Distance<br><br>
        <span style="color:#e3b341;font-size:0.78rem;">
        Educational tool only. Not financial advice.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Position Size Calculator</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        capital  = st.number_input(f"Total Capital ({curr_sym})", min_value=100, value=100000, step=5000)
    with pc2:
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.5, 0.5)
    with pc3:
        sl_price = st.number_input(
            f"Stop Loss ({curr_sym})",
            value=float(risk.get("stop_loss", latest_price * 0.95)),
            step=0.01 if is_comm else 0.5,
        )

    entry_p = latest_price
    if entry_p > sl_price > 0:
        risk_amt   = capital * risk_pct / 100
        stop_dist  = entry_p - sl_price
        units      = int(risk_amt / stop_dist) if stop_dist > 0 else 0
        pos_val    = units * entry_p
        unit_label = "Contracts" if is_comm else "Shares"
        cc1, cc2, cc3, cc4 = st.columns(4)
        for col, label, val, color in [
            (cc1, "Risk Amount",    f"{curr_sym}{risk_amt:,.2f}", "#f85149"),
            (cc2, "Stop Distance",  f"{curr_sym}{stop_dist:.3f}", "#e3b341"),
            (cc3, unit_label,       f"{units:,}",                 "#3fb950"),
            (cc4, "Position Value", f"{curr_sym}{pos_val:,.2f}",  "#58a6ff"),
        ]:
            col.markdown(
                f"<div style='background:#161b22;border:1px solid #21262d;"
                f"border-radius:8px;padding:14px;text-align:center;'>"
                f"<div style='color:#8b949e;font-size:0.7rem;margin-bottom:4px;'>{label}</div>"
                f"<div style='color:{color};font-weight:700;font-family:monospace;"
                f"font-size:1.05rem;'>{val}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Stop Loss must be below current price for the calculator to work.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 · Raw Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="sec-hdr">Recent OHLCV (last 30)</div>', unsafe_allow_html=True)
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        recent     = df[ohlcv_cols].tail(30).sort_index(ascending=False).copy()
        recent.index = recent.index.strftime("%d %b %Y")
        fmt = {
            "Open": f"{curr_sym}{{:.2f}}", "High": f"{curr_sym}{{:.2f}}",
            "Low":  f"{curr_sym}{{:.2f}}", "Close": f"{curr_sym}{{:.2f}}",
            "Volume": "{:,.0f}",
        }
        st.dataframe(recent.style.format(fmt), use_container_width=True, height=480)
    with d2:
        st.markdown('<div class="sec-hdr">Indicator Values (last 30)</div>', unsafe_allow_html=True)
        ind_cols_want = ["Close", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_Signal",
                         "BB_Upper", "BB_Lower", "ATR"]
        ind_cols = [c for c in ind_cols_want if c in df.columns]
        ind_df   = df[ind_cols].tail(30).sort_index(ascending=False).copy()
        ind_df.index = ind_df.index.strftime("%d %b %Y")
        st.dataframe(ind_df.style.format({c: "{:.2f}" for c in ind_cols}),
                     use_container_width=True, height=480)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download Full Dataset as CSV",
        data=df.to_csv().encode("utf-8"),
        file_name=f"{selected_stock}_{timeframe.replace(' ','_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.72rem;padding:6px 0 18px;font-family:monospace;">
  NSE Stock Analyser v12  ·  Kite WebSocket + Yahoo Finance  ·  Educational use only
</div>
""", unsafe_allow_html=True)
