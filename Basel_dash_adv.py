# app_advanced.py
# Advanced Basel RWA Dashboard — single file, rich UI + scenario engine
# (Place `RWA_Dataset-Hakan.xlsx` next to this file or upload via the sidebar.)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.parse

# ---------------- Page & style ----------------
st.set_page_config(page_title="Basel RWA Dashboard (Advanced)", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 0.8rem;}
.kpi > div {border:1px solid #E6E6E6;border-radius:10px;padding:0.6rem 0.9rem;background:#fff;}
.kpi .delta {font-size:0.9rem;color:#666}
.stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
.stTabs [data-baseweb="tab"] {padding: 0.4rem 0.8rem;}
.smallnote {color:#666;font-size:0.85rem}
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace("__+", "_", regex=True)
    )
    return df

def col_lookup(df: pd.DataFrame):
    # map lower->original for case-insensitive find
    lut = {c.lower(): c for c in df.columns}
    def pick(*names, default=None):
        for n in names:
            if n.lower() in lut: return lut[n.lower()]
        return default
    return pick

def safe_sum(s): 
    try: return float(np.nansum(s))
    except: return 0.0

def safe_mean(s):
    try: 
        v = float(np.nanmean(s))
        return v
    except: 
        return float("nan")

def human(n):
    try:
        n = float(n)
    except:
        return "—"
    a = abs(n)
    if a >= 1e9:  return f"{n/1e9:.2f}B"
    if a >= 1e6:  return f"{n/1e6:.2f}M"
    if a >= 1e3:  return f"{n/1e3:.2f}K"
    return f"{n:,.0f}"

def pct(x, places=1):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{x*100:.{places}f}%"

# ---------------- Data load ----------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        df = pd.read_excel("RWA_Dataset-Hakan.xlsx", engine="openpyxl")
    return sanitize_columns(df)

with st.sidebar:
    st.subheader("📥 Data source")
    uploaded = st.file_uploader("Upload Excel (optional)", type=["xlsx"], label_visibility="collapsed")

df = load_data(uploaded)

# Detect columns
pick = col_lookup(df)
CUST = pick("Customer_ID","CustomerId","Cust_ID")
PROD = pick("Product_Type")
BIZ  = pick("Business_Type")
FICO = pick("FICO_Score")
FICB = pick("FICO_Band")
SAL  = pick("Salary")
PD   = pick("PD")
EAD  = pick("EAD")
BASE_RW = pick("Base_RW","RW","Risk_Weight")
RWA  = pick("RWA")
REG  = pick("Region")
ZIP  = pick("ZIP","Postal","Postcode")
TOB  = pick("Time_on_Books_mon","TimeOnBooks","Tenor_Months")

# Derived RWA if needed
df = df.copy()
if "RWA_calc" not in df.columns:
    if RWA:
        df["RWA_calc"] = df[RWA]
    elif EAD and BASE_RW:
        df["RWA_calc"] = df[EAD] * df[BASE_RW]
    else:
        df["RWA_calc"] = np.nan

# ---------------- Sidebar: Basel techniques & scenarios ----------------
with st.sidebar:
    st.header("🧭 Basel techniques")
    approach = st.radio(
        "Approach",
        ["Standardized", "FIRB (Foundation IRB)", "AIRB (Advanced IRB)"],
        index=0
    )
    st.caption("This dashboard is for **analytics** and **what‑if** exploration, not regulatory calculations.")

    st.markdown("**Capital ratio**")
    cap_ratio = st.slider("Capital (%)", 4.0, 20.0, 8.0, 0.5) / 100.0

    st.markdown("---")
    st.markdown("**Parameters**")

    # Standardized parameters
    std_scale = 100
    if approach == "Standardized":
        if BASE_RW:
            std_scale = st.slider("Scale Base_RW (%)", 50, 150, 100, 1)
        else:
            st.info("Base_RW column not found — upload a mapping or switch to IRB.")

    # IRB parameters (simple educational treatment – not Basel formula)
    # Default LGD by product type (editable)
    default_lgd = {
        "Mortgage": 0.20, "Corporate Loans": 0.45, "SME Loans": 0.45,
        "Consumer Loan": 0.45, "Auto Loan": 0.35, "Credit Card": 0.90,
        "Other": 0.45
    }
    # Build LGD mapping UI only for IRB approaches
    lgd_map = {}
    if approach in ("FIRB (Foundation IRB)", "AIRB (Advanced IRB)"):
        st.markdown("**LGD by Product (defaults editable)**")
        prod_values = sorted(df[PROD].dropna().unique()) if PROD in df else []
        for p in prod_values[:12]:  # show up to 12 to keep UI compact
            key = p if isinstance(p, str) else str(p)
            default = default_lgd.get(key, default_lgd.get("Other", 0.45))
            lgd_map[key] = st.slider(f"LGD • {key}", 0.05, 1.00, float(default), 0.05)

        if len(prod_values) > 12:
            st.caption("Showing first 12 products. Edit the rest via CSV mapping if needed.")

        irb_pd_floor = st.slider("PD floor (IRB)", 0.0005, 0.05, 0.003, 0.0005)
        irb_maturity = st.slider("Maturity proxy (years)", 1.0, 5.0, 2.5, 0.5)

    st.markdown("---")
    st.markdown("**Stress / Scenario controls**")
    scen = st.radio("Scenario", ["Base", "Scenario A", "Scenario B"], index=0, horizontal=True)
    pd_stress  = st.slider("PD stress ×", 0.5, 2.5, 1.0, 0.05)
    lgd_stress = st.slider("LGD stress ×", 0.5, 1.5, 1.0, 0.05) if approach != "Standardized" else 1.0
    ead_stress = st.slider("EAD CCF uplift ×", 0.8, 1.5, 1.0, 0.05)

    with st.expander("ℹ️ Notes", expanded=False):
        st.write("""
- **Standardized**: uses `Base_RW` (or `RWA` if provided). Optional scaling lets you run what‑ifs.
- **FIRB/AIRB (educational)**: computes **EL = PD × LGD × EAD** and a simple **proxy RW** from PD/LGD/Maturity.  
  **Not** a regulatory reproduction; for analytics only.
- **Stress** multipliers apply to PD, LGD, and EAD before KPIs/plots.
        """)

    st.markdown("---")
    st.subheader("🔎 Filters")
    def opts(c): 
        return sorted(df[c].dropna().astype(str).unique()) if c in df else []
    sel_region = st.multiselect("Region", opts(REG), default=opts(REG))
    sel_prod   = st.multiselect("Product", opts(PROD), default=opts(PROD))
    sel_biz    = st.multiselect("Business Type", opts(BIZ), default=opts(BIZ))
    sel_ficob  = st.multiselect("FICO Band", opts(FICB), default=opts(FICB))

    # numeric ranges
    def bounds(colname, lo=0.0, hi=1.0, cast=float):
        if colname in df and pd.api.types.is_numeric_dtype(df[colname]):
            s = df[colname].dropna()
            if len(s): return cast(np.floor(s.min() if cast is int else s.min())), cast(np.ceil(s.max() if cast is int else s.max()))
        return cast(lo), cast(hi)

    fico_lo, fico_hi = bounds(FICO, 200, 900, int)
    pd_lo, pd_hi     = bounds(PD,   0.0, 0.30, float)
    tob_lo, tob_hi   = bounds(TOB,  0,   120,  int)

    sel_fico = st.slider("FICO score", fico_lo, fico_hi, (fico_lo, fico_hi)) if FICO in df else None
    sel_pd   = st.slider("PD", float(pd_lo), float(pd_hi), (float(pd_lo), float(pd_hi))) if PD in df else None
    sel_tob  = st.slider("Time on Books (months)", tob_lo, tob_hi, (tob_lo, tob_hi)) if TOB in df else None

    cust_q = st.text_input("Search Customer ID", "")

# ---------------- Scenario / computation engine ----------------
def apply_filters(d):
    m = pd.Series(True, index=d.index)
    if REG in d and sel_region: m &= d[REG].astype(str).isin(sel_region)
    if PROD in d and sel_prod:  m &= d[PROD].astype(str).isin(sel_prod)
    if BIZ in d and sel_biz:    m &= d[BIZ].astype(str).isin(sel_biz)
    if FICB in d and sel_ficob: m &= d[FICB].astype(str).isin(sel_ficob)
    if FICO in d and sel_fico:  m &= d[FICO].between(*sel_fico)
    if PD in d and sel_pd:      m &= d[PD].between(*sel_pd)
    if TOB in d and sel_tob:    m &= d[TOB].between(*sel_tob)
    if cust_q and CUST in d:    m &= d[CUST].astype(str).str.contains(cust_q.strip(), case=False, na=False)
    return d[m].copy()

@st.cache_data(show_spinner=False)
def irb_proxy_rw(pd_series, lgd_series, maturity_years=2.5, floor=0.003, advanced=False):
    """
    Educational proxy for IRB risk weight — NOT the Basel formula.
    We use a monotonic transform of PD, LGD, and M to get a 0-100% RW band.
    """
    pd_eff = np.maximum(pd_series.fillna(0).astype(float), floor)
    lgd_eff = lgd_series.fillna(0.45).astype(float)
    # Simple shape: RW ~ PD^(0.5..0.7) * LGD * f(M)
    expo = 0.65 if advanced else 0.55
    m_adj = 0.85 + 0.06 * (maturity_years - 2.5)  # small tilt around 2.5y
    rw = np.clip((pd_eff ** expo) * lgd_eff * 2.2 * m_adj, 0.02, 1.00)
    return rw

def build_lgd_vector(d):
    if PROD in d:
        return d[PROD].astype(str).map(lambda p: lgd_map.get(p, default_lgd.get(p, default_lgd.get("Other", 0.45))))
    return pd.Series(0.45, index=d.index)

def compute_views(d):
    d = d.copy()
    # stressed inputs
    if EAD in d: d["_EAD_"] = d[EAD].astype(float) * ead_stress
    else:        d["_EAD_"] = np.nan
    if PD  in d: d["_PD_"]  = d[PD].astype(float) * pd_stress
    else:        d["_PD_"]  = np.nan

    # route by approach
    if approach == "Standardized":
        # Use RWA or Base_RW; allow scaling
        if RWA in d:
            rwa = d[RWA].astype(float) * (ead_stress if EAD in d else 1.0)  # if EAD stressed, propagate linearly
        elif BASE_RW in d and EAD in d:
            rwa = d[BASE_RW].astype(float) * d["_EAD_"]
        else:
            rwa = d["RWA_calc"].astype(float)
        rwa = rwa * (std_scale / 100.0)
        el  = d["_PD_"] * d["_EAD_"] * 0.45 if PD in d and EAD in d else np.nan  # nominal EL proxy
        d["_RW_used"] = (rwa / d["_EAD_"]).replace([np.inf, -np.inf], np.nan)
        d["_EL_"] = el
        d["_RWA_"] = rwa

    else:
        # IRB-like (educational)
        lgd_vec = build_lgd_vector(d) * lgd_stress
        if PD in d and EAD in d:
            d["_EL_"] = d["_PD_"] * lgd_vec * d["_EAD_"]
        else:
            d["_EL_"] = np.nan

        rw_proxy = irb_proxy_rw(
            d["_PD_"] if PD in d else pd.Series(0.01, index=d.index),
            lgd_vec,
            maturity_years=irb_maturity,
            floor=irb_pd_floor,
            advanced=(approach == "AIRB (Advanced IRB)")
        )
        d["_RW_used"] = rw_proxy
        d["_RWA_"] = rw_proxy * d["_EAD_"]

    d["_Capital_"] = d["_RWA_"] * cap_ratio
    return d

# Apply filters then compute
f = apply_filters(df)
f = compute_views(f)

# Keep a Base scenario to show deltas
if "scenario_store" not in st.session_state:
    st.session_state["scenario_store"] = {}
st.session_state["scenario_store"][scen] = f

def scenario_kpis(d):
    tot_ead = safe_sum(d["_EAD_"])
    tot_rwa = safe_sum(d["_RWA_"])
    tot_cap = safe_sum(d["_Capital_"])
    avg_pd  = safe_mean(d["_PD_"]) if "_PD_" in d else float("nan")
    return tot_ead, tot_rwa, tot_cap, avg_pd, len(d)

# ---------------- KPI header with scenario deltas ----------------
st.markdown("""
<div style="padding:1.75rem 0px 1rem; font-size:1.4rem; font-weight:600;">
📊 Portfolio snapshot
</div>
""", unsafe_allow_html=True)

base = st.session_state["scenario_store"].get("Base", f)
curr = f
def delta_str(curr_v, base_v, is_pct=False):
    if base_v in (None, 0, np.nan) or (isinstance(base_v, float) and np.isnan(base_v)):
        return ""
    if curr_v is None or (isinstance(curr_v, float) and np.isnan(curr_v)): return ""
    dv = curr_v - base_v
    sym = "▲" if dv > 0 else ("▼" if dv < 0 else "—")
    if is_pct:
        return f'<span class="delta">{sym} {pct(dv, 2)} vs Base</span>'
    else:
        return f'<span class="delta">{sym} {human(dv)} vs Base</span>'

E0,R0,C0,P0,N0 = scenario_kpis(base)
E1,R1,C1,P1,N1 = scenario_kpis(curr)

cap_ratio_label = f"{cap_ratio*100:.1f}%"
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi"><div><b>Total EAD</b><br>'
                f'<span style="font-size:1.3rem">{human(E1)}</span><br></div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div><b>Total RWA</b><br>'
                f'<span style="font-size:1.3rem">{human(R1)}</span><br></div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div><b>Capital (@ {cap_ratio_label})</b><br>'
                f'<span style="font-size:1.3rem">{human(C1)}</span><br></div></div>', unsafe_allow_html=True)
with k4:
    avg_pd_display = "—" if (P1!=P1) else f"{P1:.3%}"
    base_pd_display = "—" if (P0!=P0) else f"{P0:.3%}"
    # delta in percentage points
    pd_delta = "" if (P0!=P0 or P1!=P1) else f'<span class="delta">{ "▲" if (P1-P0)>0 else ("▼" if (P1-P0)<0 else "—") } {((P1-P0)*100):.2f} bp vs Base</span>'
    st.markdown(f'<div class="kpi"><div><b>Avg PD</b><br>'
                f'<span style="font-size:1.3rem">{avg_pd_display}</span><br></div></div>', unsafe_allow_html=True)
# with k5:
#     st.markdown(f'<div class="kpi"><div><b>Records</b><br>'
#                 f'<span style="font-size:1.3rem">{N1:,}</span><br>{delta_str(N1,N0)}</div></div>', unsafe_allow_html=True)

# st.caption("Deltas compare the selected scenario to **Base** with the same filters.")

# ---------------- Tabs ----------------
tabs = st.tabs(["Overview", "Segments", "Distributions", "Customer Drill"])

# ---- Overview tab
with tabs[0]:
    c1,c2 = st.columns(2)
    if PROD in f:
        top = (f.groupby(PROD, dropna=False)["_RWA_"].sum().sort_values(ascending=False).head(12).reset_index())
        fig = px.bar(top, x=PROD, y="_RWA_", title="RWA by Product (Top 12)", labels={"_RWA_":"RWA"})
        fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=40))
        c1.plotly_chart(fig, use_container_width=True)
    if REG in f and BIZ in f:
        fig = px.treemap(f, path=[REG, BIZ], values="_RWA_", title="RWA: Region ⮕ Business")
        fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=40))
        c2.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    if FICO in f and PD in f and EAD in f:
        fig = px.scatter(f, x=FICO, y="_PD_", size="_EAD_", color=FICB if FICB in f else PROD,
                         hover_data=[CUST] if CUST in f else None,
                         title="PD vs FICO (bubble ~ stressed EAD)",
                         labels={"_PD_":"PD (stressed)","_EAD_":"EAD (stressed)"})
        fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=40))
        c3.plotly_chart(fig, use_container_width=True)
    if "_EL_" in f:
        # EL by product
        if PROD in f:
            el_top = (f.groupby(PROD, dropna=False)["_EL_"].sum().sort_values(ascending=False).head(12).reset_index())
            fig = px.bar(el_top, x=PROD, y="_EL_", title="Expected Loss (EL) by Product", labels={"_EL_":"EL"})
            fig.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=40))
            c4.plotly_chart(fig, use_container_width=True)

# ---- Segments tab
with tabs[1]:
    st.subheader("Segment views")
    seg_cols = []
    if REG in f:  seg_cols.append(REG)
    if PROD in f: seg_cols.append(PROD)
    if BIZ in f:  seg_cols.append(BIZ)
    if FICB in f: seg_cols.append(FICB)
    if not seg_cols:
        st.info("No segment dimensions found.")
    else:
        dim = st.selectbox("Dimension", seg_cols, index=0)
        metric = st.selectbox("Metric", ["RWA","EAD","EL","Capital"], index=0)
        mapping = {"RWA":"_RWA_", "EAD":"_EAD_", "EL":"_EL_", "Capital":"_Capital_"}
        mcol = mapping[metric]
        agg = (f.groupby(dim, dropna=False)[mcol].sum().sort_values(ascending=False).reset_index())
        fig = px.bar(agg, x=dim, y=mcol, title=f"{metric} by {dim}")
        fig.update_layout(height=400, margin=dict(l=10,r=10,b=10,t=40))
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap PD buckets vs Product (if present)
       
pd_bins = st.slider("PD bucket count", 4, 20, 10, 1)
_raw = f["_PD_"].fillna(0.0)
_bins = pd.qcut(_raw, q=pd_bins, duplicates="drop")  # Interval categories
labels = [f"{c.left:.4f}–{c.right:.4f}" for c in _bins.cat.categories]

f["_PD_Bin"] = pd.Categorical(
    _bins.map(lambda c: f"{c.left:.4f}–{c.right:.4f}"),
    categories=labels,
    ordered=True,
)

hm = f.pivot_table(index="_PD_Bin", columns=PROD, values="_RWA_", aggfunc="sum", fill_value=0)

# Ensure axes are strings for Plotly JSON
hm_plot = hm.copy()
hm_plot.index = hm_plot.index.astype(str)
hm_plot.columns = hm_plot.columns.astype(str)

fig = px.imshow(
    hm_plot,
    aspect="auto",
    color_continuous_scale="Blues",
    title="RWA heatmap: PD bucket × Product"
)
fig.update_layout(height=480, margin=dict(l=10, r=10, b=10, t=40))
st.plotly_chart(fig, use_container_width=True)
# ---- Distributions tab
with tabs[2]:
    c1,c2 = st.columns(2)
    if PD in f:
        fig = px.histogram(f, x="_PD_", nbins=50, marginal="box", title="PD (stressed) distribution")
        fig.update_layout(height=380, margin=dict(l=10,r=10,b=10,t=40))
        c1.plotly_chart(fig, use_container_width=True)
    if "_RW_used" in f:
        fig = px.histogram(f, x="_RW_used", nbins=50, title="Applied RW distribution")
        fig.update_layout(height=380, margin=dict(l=10,r=10,b=10,t=40))
        c2.plotly_chart(fig, use_container_width=True)

    if SAL in f:
        st.markdown("---")
        fig = px.scatter(f, x=SAL, y="_PD_", size="_EAD_", color=PROD if PROD in f else None,
                         title="PD vs Salary (bubble ~ EAD)")
        fig.update_layout(height=420, margin=dict(l=10,r=10,b=10,t=40))
        st.plotly_chart(fig, use_container_width=True)

# ---- Customer Drill tab
with tabs[3]:
    st.subheader("Customer drill-through")
    # quick selector
    if CUST in f:
        top_ids = f.sort_values("_RWA_", ascending=False)[CUST].astype(str).head(200).unique().tolist()
        sel_id = st.selectbox("Customer ID", top_ids)
        d1 = f[f[CUST].astype(str) == sel_id].copy()
        st.write("Snapshot")
        cols_show = [c for c in [CUST, REG, PROD, BIZ, FICB, FICO, PD, EAD, BASE_RW, RWA, "_EAD_","_PD_","_RW_used","_RWA_","_EL_","_Capital_"] if c and c in f.columns]
        st.dataframe(d1[cols_show], use_container_width=True)
        if not d1.empty and FICO in d1 and PD in d1:
            fig = px.scatter(d1, x=FICO, y="_PD_", size="_EAD_", color=PROD if PROD in d1 else None,
                             title=f"Customer {sel_id}: PD vs FICO (bubble~EAD)")
            fig.update_layout(height=320, margin=dict(l=10,r=10,b=10,t=40))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Customer_ID column not found — drill-through disabled.")

# ---------------- Table + Exports ----------------
st.markdown("### 📄 Filtered table")
show_cols = [c for c in [CUST, REG, PROD, BIZ, FICB, FICO, PD, EAD, BASE_RW, RWA] if c and c in f.columns]
calc_cols = ["_EAD_","_PD_","_RW_used","_RWA_","_EL_","_Capital_"]
cols_final = [c for c in show_cols + calc_cols if c in f.columns]
st.dataframe(f[cols_final].sort_values("_RWA_", ascending=False), use_container_width=True)

@st.cache_data
def to_csv(df_): return df_.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download filtered CSV (scenario view)",
    data=to_csv(f[cols_final].rename(columns={
        "_EAD_":"EAD_view","_PD_":"PD_view","_RW_used":"RW_used","_RWA_":"RWA_view","_EL_":"EL_view","_Capital_":"Capital_view"})),
    file_name="filtered_portfolio_scenario.csv",
    mime="text/csv",
)

# ---------------- Shareable link (query params) ----------------
# def encode_params():
#     params = {
#         "approach": approach,
#         "cap": f"{cap_ratio:.4f}",
#         "scen": scen,
#         "pdX": f"{pd_stress:.3f}",
#         "lgdX": f"{lgd_stress:.3f}" if approach!="Standardized" else "1",
#         "eadX": f"{ead_stress:.3f}",
#         "stdScale": f"{std_scale}" if approach=="Standardized" else "",
#         "region": ",".join(sel_region) if sel_region else "",
#         "prod":   ",".join(sel_prod) if sel_prod else "",
#         "biz":    ",".join(sel_biz) if sel_biz else "",
#         "ficob":  ",".join(sel_ficob) if sel_ficob else "",
#         "q": cust_q or ""
#     }
#     return params

# qparams = encode_params()
# st.caption("🔗 Share this state:")
# st.code(st.experimental_get_query_params() or qparams, language="json")
# st.button("Copy current filters to URL", on_click=lambda: st.experimental_set_query_params(**qparams))