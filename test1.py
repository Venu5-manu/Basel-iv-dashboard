import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# =========================
# UI SETUP
# =========================
st.set_page_config(
    page_title="Basel IV Credit Risk & Optimization",
    layout="wide",
 )

# =========================
# SIDE MENU (Short labels)
# =========================
# Map short menu labels -> long descriptions for routing
menu_map = {
    "Retail/LTV/CRM": "Retail mortgage, consumer, auto finance, LTV, collateral, netting, Basel IV band allocation, policies vs regulation",
    "CRR3 Cards": "Check of Credit Card CRR3 treatment (retail QRRE, self-employed, SME, corporate)",
    "SME/Corp Opt": "SME/Corporate lending contracts optimization (structure, mezzanine tranches, collateral, haircuts, guarantees, RWA discounts, netting)",
    "FX Hedge/Coll": "Checking currency hedging and mismatch on collateral",
    "Std RWA Proc": "Review/enhancement of automated Standard RWA calculation process",
    "NPE/Forecast": "Assessment of potential for NPE (Non-Performing Exposure), pre-default forecasting, defaulted exposures, early-recovery modelling",
    "Securitization": "Strategic recommendations for securitization & synergies (private credit on mezzanine tranches)",
    "OBS/CCF": "Analysis of off-balance sheet exposures (CCFs)",
    "Capital Buffer": "Examine impact on capital buffer",
    "Portfolio Opt": "Portfolio-level strategic optimization (retail, SME & industry sectors)"
}

st.sidebar.title("Basel IV Optimization")
selected_short = st.sidebar.radio("Module", list(menu_map.keys()))
selected_opt = menu_map[selected_short]

st.title("Basel IV Credit Risk & Optimization Dashboard")

# =========================
# DATA LOADER & PREP
# =========================
@st.cache_data
def load_and_prepare(uploaded_file=None):
    """Load Excel (uploaded or default), unify columns, derive Basel fields."""
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        # Default sample name; change if yours differs
        df = pd.read_excel("Basel_Combined_Datasets.xlsx", engine="openpyxl")

    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

    # Numeric coercion
    num_cols = [
        "LTV", "PD", "LGD", "EAD", "RiskWeight", "RWA",
        "CapitalRequirement", "InternalCapital", "OutputFloorCapital", "OptimizedCapital",
        "Estimated_RW", "Calculated_RWA", "RWA_Difference", "RWA_Delta_%",
        "M", "Maturity", "CollateralValue", "Collateral_Amount", "Haircut", "Haircut_%", "NetExposure"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clips
    if "PD" in df: df["PD"] = df["PD"].clip(0, 1)
    if "LGD" in df: df["LGD"] = df["LGD"].clip(0, 1)
    if "LTV" in df: df["LTV"] = df["LTV"].clip(lower=0)

    # Estimated RW from PD buckets (simple Basel-like proxy)
    if ("Estimated_RW" not in df) or df["Estimated_RW"].isna().all():
        df["Estimated_RW"] = df["PD"].apply(lambda p: 0.30 if p < 0.20 else (0.50 if p < 0.50 else 0.75))

    # Calculated RWA = EAD × Estimated_RW
    if ("Calculated_RWA" not in df) or df["Calculated_RWA"].isna().all():
        df["Calculated_RWA"] = df["EAD"] * df["Estimated_RW"]

    # Differences & Delta%
    df["RWA_Difference"] = df["Calculated_RWA"] - df.get("RWA", np.nan)
    df["RWA_Delta_%"] = np.where(
        (df.get("RWA", np.nan).notna()) & (df["RWA"] != 0),
        (df["RWA_Difference"] / df["RWA"]) * 100,
        np.nan
    )

    # Risk Level, Bands
    df["RiskLevel"] = df["PD"].apply(lambda p: "Low" if p < 0.20 else ("Medium" if p < 0.50 else "High"))
    df["PD_Band"] = pd.cut(df["PD"], [0, 0.10, 0.20, 0.50, 1.0], labels=["<10%", "10–20%", "20–50%", ">50%"], include_lowest=True)
    if "LTV" in df:
        df["LTV_Band"] = pd.cut(df["LTV"], [0, 0.50, 0.80, 1.00, np.inf], labels=["≤50%", "50–80%", "80–100%", ">100%"])

    return df

# Upload or use default
# uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
# df = load_and_prepare(uploaded)

uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
if uploaded:
    df = pd.read_excel(uploaded, engine="openpyxl")
else:
    # Fallback mock data
    df = pd.DataFrame({
        "LoanType": ["Retail", "SME", "Industry"],
        "EAD": [500, 400, 300],
        "RWA": [200, 150, 100],
        "PD": [0.02, 0.05, 0.10],
        "LGD": [0.45, 0.40, 0.35]
    })


# Common references for visuals
product_col = "LoanType" if "LoanType" in df.columns else None

# =========================
# RETAIL / LTV / CRM MODULE
# =========================
if selected_short == "Retail/LTV/CRM":
    st.header("Retail mortgage, consumer, auto finance, LTV, collateral, netting, Basel IV band allocation")

    # ---- Sidebar Filters (short labels) ----
    st.sidebar.subheader("Filters")
    # Product filter
    if product_col:
        products = sorted(df[product_col].dropna().unique())
        sel_products = st.sidebar.multiselect("Prod", options=products, default=products)
    else:
        sel_products = None

    # PD & LTV ranges
    pd_min, pd_max = float(df["PD"].min()), float(df["PD"].max())
    pd_range = st.sidebar.slider("PD", min_value=0.0, max_value=1.0, value=(round(pd_min, 3), round(pd_max, 3)), step=0.001)
    if "LTV" in df:
        ltv_min, ltv_max = float(df["LTV"].min()), float(df["LTV"].max())
        ltv_range = st.sidebar.slider("LTV", min_value=0.0, max_value=max(1.0, round(ltv_max, 2)),
                                      value=(round(ltv_min, 2), round(ltv_max, 2)), step=0.01)
    else:
        ltv_range = (0.0, 1.0)

    # ---- Apply Filters ----
    filtered = df.copy()
    if product_col and sel_products:
        filtered = filtered[filtered[product_col].isin(sel_products)]
    filtered = filtered[(filtered["PD"] >= pd_range[0]) & (filtered["PD"] <= pd_range[1])]
    if "LTV" in filtered:
        filtered = filtered[(filtered["LTV"] >= ltv_range[0]) & (filtered["LTV"] <= ltv_range[1])]

    # ---- KPIs ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("# Loans", f"{len(filtered):,}")
    k2.metric("Total EAD", f"{filtered['EAD'].sum():,.0f}")
    k3.metric("Total RWA (Reported)", f"{filtered['RWA'].sum():,.0f}" if "RWA" in filtered else "NA")
    k4.metric("Total RWA (Calc)", f"{filtered['Calculated_RWA'].sum():,.0f}")

    st.caption("**Note:** Calculated RWA uses a simple PD→RW bucketing proxy (30%/50%/75%) for validation, not a regulatory engine.")

    # =========================
    # ROW 1: Exposures & Mix
    # =========================
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("EAD by Product")
        if product_col:
            fig = px.bar(
                filtered.groupby(product_col)["EAD"].sum().sort_values(ascending=False).reset_index(),
                x=product_col, y="EAD", text="EAD", color=product_col, title="Total EAD by LoanType"
            )
            fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", opacity=0.85)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "> **Why this matters:** Shows where exposure (and capital) is concentrated. Use filters to isolate segments."
        )

    with c2:
        st.subheader("Portfolio Mix (Count)")
        if product_col:
            pie = px.pie(filtered, names=product_col, title="Share by LoanType")
            st.plotly_chart(pie, use_container_width=True)
        top10 = filtered.nlargest(10, "EAD")[["EAD", "RWA", "PD", "LGD"] + ([product_col] if product_col else [])]
        st.dataframe(top10.style.format({"EAD": "{:,.0f}", "RWA": "{:,.0f}", "PD": "{:.1%}", "LGD": "{:.1%}"}))
        st.markdown(
            "> **Tip:** Focus on high EAD & high PD/LGD combinations—they’re typical capital drivers."
        )

    # =========================
    # ROW 2: LTV Distribution
    # =========================
    if "LTV" in filtered:
        c3, c4 = st.columns(2)

        with c3:
            st.subheader("LTV Distribution")
            st.plotly_chart(px.histogram(filtered, x="LTV", nbins=30, title="LTV (All Loans)", marginal="box"),
                            use_container_width=True)
            st.markdown("> Lower LTV typically → stronger collateral coverage.")

        with c4:
            st.subheader("LTV by Product / PD Band")
            if product_col:
                st.plotly_chart(px.box(filtered, x=product_col, y="LTV", color="PD_Band",
                                       title="LTV by LoanType & PD Band"),
                                use_container_width=True)
            if "LTV_Band" in filtered and product_col:
                ctab = pd.crosstab(filtered[product_col], filtered["LTV_Band"])
                st.plotly_chart(
                    px.bar(ctab.reset_index().melt(id_vars=product_col, var_name="LTV_Band", value_name="Count"),
                           x=product_col, y="Count", color="LTV_Band", barmode="stack",
                           title="LTV Bands by LoanType (Counts)"),
                    use_container_width=True
                )

    # =========================
    # ROW 3: Collateral & Haircuts (optional)
    # =========================
    st.subheader("Collateral Coverage & Haircuts")
    st.markdown("**Coverage proxy** = 1 / LTV. Haircut & netting visuals activate if columns are present.")

    c5, c6 = st.columns(2)

    with c5:
        if "LTV" in filtered:
            tmp = filtered.copy()
            tmp["Coverage_Proxy"] = np.where(tmp["LTV"] > 0, 1.0 / tmp["LTV"], np.nan)
            if "CollateralQuality" in tmp:
                st.plotly_chart(
                    px.bar(tmp, x="CollateralQuality", y="Coverage_Proxy", color="CollateralQuality",
                           title="Median Coverage Proxy by CollateralQuality"),
                    use_container_width=True
                )
            else:
                st.plotly_chart(px.histogram(tmp, x="Coverage_Proxy", nbins=30, title="Coverage Proxy Distribution"),
                                use_container_width=True)
        st.markdown(
            "> **Interpretation:** Higher coverage (lower LTV) generally reduces loss severity and capital."
        )

    with c6:
        haircut_col = "Haircut_%" if "Haircut_%" in filtered.columns else ("Haircut" if "Haircut" in filtered.columns else None)
        coll_val_col = "CollateralValue" if "CollateralValue" in filtered.columns else ("Collateral_Amount" if "Collateral_Amount" in filtered.columns else None)
        if haircut_col and coll_val_col:
            tmp = filtered.copy()
            tmp["Haircut_dec"] = np.where(tmp[haircut_col] > 1, tmp[haircut_col] / 100.0, tmp[haircut_col])
            tmp["AdjCollateral"] = tmp[coll_val_col] * (1 - tmp["Haircut_dec"].fillna(0))
            tmp["NetExposure_calc"] = np.maximum(tmp["EAD"] - tmp["AdjCollateral"], 0.0)
            st.plotly_chart(px.histogram(tmp, x=haircut_col, nbins=30, title="Haircut Distribution"),
                            use_container_width=True)
            if product_col:
                gross_net = tmp.groupby(product_col).agg(
                    Gross_EAD=("EAD", "sum"), Net_Exposure=("NetExposure_calc", "sum")
                ).reset_index()
                st.plotly_chart(
                    px.bar(gross_net.melt(id_vars=product_col, var_name="Type", value_name="Amount"),
                           x=product_col, y="Amount", color="Type", barmode="group",
                           title="Gross EAD vs Net Exposure (Haircuts applied)"),
                    use_container_width=True
                )
            st.caption("Comprehensive approach conceptually uses:  E* = E(1+He) − C(1−Hc−Hfx) with supervisory haircuts.")
        else:
            st.info("Add columns `CollateralValue` (or `Collateral_Amount`) and `Haircut` / `Haircut_%` to enable haircut views.")

    # =========================
    # ROW 4: Basel IV Band Impacts & Policy vs Regulation
    # =========================
    c7, c8 = st.columns(2)

    with c7:
        st.subheader("Basel IV – PD Bands")
        if "PD_Band" in filtered:
            band_counts = filtered["PD_Band"].value_counts().reindex(["<10%", "10–20%", "20–50%", ">50%"])
            st.plotly_chart(px.bar(band_counts, title="Loan Count by PD Band"), use_container_width=True)

            rf = filtered.groupby("PD_Band").agg(
                Total_RWA=("RWA", "sum"),
                Total_RWA_Calc=("Calculated_RWA", "sum")
            ).reindex(["<10%", "10–20%", "20–50%", ">50%"])
            st.plotly_chart(px.bar(rf, barmode="group", title="Total RWA by PD Band (Reported vs Calculated)"),
                            use_container_width=True)
        st.markdown("> Banding helps visualize how risk clustering affects capital.")

    with c8:
        st.subheader("Policy vs Regulation")
        sc = px.scatter(filtered, x="RWA", y="Calculated_RWA", color=product_col if product_col else None,
                        title="Reported vs Calculated RWA (diagonal = match)")
        if "RWA" in filtered:
            minv, maxv = float(filtered["RWA"].min()), float(filtered["RWA"].max())
            sc.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(color="red", dash="dash"))
        st.plotly_chart(sc, use_container_width=True)

        cap_cols = [c for c in ["CapitalRequirement", "InternalCapital", "OutputFloorCapital", "OptimizedCapital"] if c in filtered.columns]
        if cap_cols and product_col:
            st.plotly_chart(
                px.bar(filtered.groupby(product_col)[cap_cols].median().reset_index(),
                       x=product_col, y=cap_cols, barmode="group",
                       title="Capital (Median) by LoanType – Internal vs Regulatory"),
                use_container_width=True
            )
        else:
            st.info("Add internal capital fields to compare policy vs regulation (e.g., InternalCapital, OutputFloorCapital).")

    # =========================
    # ROW 5: Top Deviations
    # =========================
    st.subheader("Top 15 RWA Deviations (Calc − Reported)")
    if "RWA_Difference" in filtered:
        dev = filtered.assign(AbsDev=filtered["RWA_Difference"].abs()).sort_values("AbsDev", ascending=False).head(15)
        bar = px.bar(dev.reset_index(drop=True), y="RWA_Difference", color=product_col if product_col else None,
                     title="Top 15 – RWA Difference")
        bar.add_hline(y=0, line_dash="dash")
        st.plotly_chart(bar, use_container_width=True)
        st.markdown("> Use this to triage data quality, approach mismatches, CCF/collateral effects, or RW mapping issues.")

    # =========================
    # IRB SECTION (Expander)
    # =========================
    st.subheader("Basel IRB – K × EAD (optional)")
    with st.expander("Compute IRB capital (K) and IRB RWA — needs PD, LGD, EAD; uses M if present"):
        st.markdown(
            """
            **Formula (ASRF):**  
            \\[
            K = LGD \\cdot \\big( N\\!(\\tfrac{1}{\\sqrt{1-R}}G(PD) + \\sqrt{\\tfrac{R}{1-R}}G(0.999)) - PD \\big)
            \\]  
            **IRB RWA** = 12.5 × K × EAD  
            Retail presets use standard supervisory correlations; Corporate adds a maturity adjustment b(PD).  
            """
        )
        # IRB presets
        preset = st.selectbox(
            "IRB Preset",
            [
                "Retail – Residential Mortgage (R=0.15, no maturity adj)",
                "Retail – Qualifying Revolving (R=0.04, no maturity adj)",
                "Retail – Other (R=0.03 + 0.16×PD, no maturity adj)",
                "Corporate (supervisory R(PD) + maturity adj)",
                "Custom (set R)"
            ],
            index=0
        )
        R_custom = st.number_input("Custom R", min_value=0.0, max_value=0.999, value=0.15, step=0.001,
                                   disabled=(not preset.startswith("Custom")))
        M_default = st.number_input("Default M (years)", min_value=0.25, max_value=7.0, value=2.5, step=0.25)
        run_irb = st.button("Compute IRB K & RWA")

        if run_irb:
            try:
                from scipy.stats import norm
                N, G = norm.cdf, norm.ppf
                data = filtered.copy()
                data["PD"] = data["PD"].clip(lower=1e-6)  # avoid log(0)

                # Correlation R
                def R_corr(pd_v):
                    if preset.startswith("Retail – Residential"): return 0.15
                    if preset.startswith("Retail – Qualifying"):  return 0.04
                    if preset.startswith("Retail – Other"):       return 0.03 + 0.16 * pd_v
                    if preset.startswith("Corporate"):
                        # Corporate supervisory correlation function (Basel)
                        return 0.12 * (1 - np.exp(-50 * pd_v)) / (1 - np.exp(-50)) + \
                               0.24 * (1 - (1 - np.exp(-50 * pd_v)) / (1 - np.exp(-50)))
                    return R_custom

                data["R_corr"] = data["PD"].apply(R_corr)
                data["M_eff"] = data["M"] if "M" in data.columns else (data["Maturity"] if "Maturity" in data.columns else M_default)

                # Capital function K
                x = (1 / np.sqrt(1 - data["R_corr"])) * G(data["PD"]) + np.sqrt(data["R_corr"] / (1 - data["R_corr"])) * G(0.999)
                cond = N(x)
                data["K"] = data["LGD"] * (cond - data["PD"])

                # Maturity adj for Corporate
                if preset.startswith("Corporate"):
                    # b(PD) per Basel
                    b = (0.11852 - 0.05478 * np.log(data["PD"])) ** 2
                    MA = (1 + (data["M_eff"] - 2.5) * b) / (1 - 1.5 * b)
                    data["K"] = data["K"] * MA

                data["IRB_RWA"] = 12.5 * data["K"] * data["EAD"]

                st.dataframe(
                    data[["EAD", "PD", "LGD", "R_corr", "M_eff", "K", "IRB_RWA", "RWA", "Calculated_RWA"] +
                         ([product_col] if product_col else [])].head(50)
                )
                st.plotly_chart(px.scatter(data, x="RWA", y="IRB_RWA", color=product_col if product_col else None,
                                           title="Reported RWA vs IRB RWA (K×EAD)"),
                                use_container_width=True)
                if product_col:
                    agg = data.groupby(product_col)[["IRB_RWA", "RWA"]].sum().reset_index()
                    st.plotly_chart(px.bar(agg.melt(id_vars=product_col, var_name="Type", value_name="Amount"),
                                           x=product_col, y="Amount", color="Type", barmode="group",
                                           title="IRB vs Reported – by Product"),
                                    use_container_width=True)
                st.success("IRB results computed. Use filters to focus by product / PD / LTV.")
            except Exception as e:
                st.warning("Install SciPy to enable IRB calculator:  pip install scipy")
                st.exception(e)

    # =========================
    # DATA TABLE + DOWNLOAD
    # =========================
    st.subheader("Filtered Data (sample)")
    st.dataframe(filtered.head(200))
    st.download_button("Download filtered dataset (CSV)", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_dataset.csv", mime="text/csv")

# =========================
# OTHER MODULE HEADERS (placeholders)
# =========================
elif selected_short == "CRR3 Cards":
    st.header("Credit Card CRR3 Treatment")
    # =========================================================
    # CREDIT CARD CRR3 TREATMENT — QRRE / SME / CORPORATE
    # Tabs • Downloads • Policy vs Reg (SA proxy)
    # =========================================================
    import io


    # ---------- Data loader ----------
    @st.cache_data
    def load_cards_data(path="Basel_Combined_Datasets.xlsx", sheet="Merged_RWA_OBS"):
        try:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        except Exception as e:
            st.warning(
                "Could not read Excel. Please upload **Basel_Combined_Datasets.xlsx** or adjust the path.\n\n"
                f"Details: {e}"
            )
            return pd.DataFrame()
        # Normalize columns
        df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)
        # Coerce numerics if present
        for c in ["PD", "RWA", "Annual_Salary", "Asset_Value", "FICO_Score",
                  "Credit_Bureau_Score", "Time_on_Books"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Simple PD banding for visuals
        if "PD" in df.columns:
            df["PD_Band"] = pd.cut(
                df["PD"], [0, 0.05, 0.10, 0.20, 0.50, 1.0],
                labels=["<5%", "5–10%", "10–20%", "20–50%", ">50%"], include_lowest=True
            )
        # Harmonize core labels
        if "Product_Type" not in df.columns and "Product" in df.columns:
            df["Product_Type"] = df["Product"]
        if "Business_Type" not in df.columns:
            df["Business_Type"] = "Unknown"
        if "Customer_ID" not in df.columns:
            df["Customer_ID"] = range(1, len(df) + 1)
        return df

    df_cards = load_cards_data()
    if df_cards.empty:
        st.stop()

    # ---------- Sidebar: Filters & Assumptions ----------
    st.sidebar.subheader("Filters (CRR3)")

    # Product filter
    prods = sorted(df_cards["Product_Type"].dropna().unique().tolist())
    sel_prods = st.sidebar.multiselect("Prod", prods, default=prods[:10] if len(prods) > 10 else prods)

    # Business type filter
    biz_types = sorted(df_cards["Business_Type"].dropna().unique().tolist())
    sel_biz = st.sidebar.multiselect("Biz", biz_types, default=biz_types)

    # PD range
    pd_min = float(df_cards["PD"].min()) if "PD" in df_cards else 0.0
    pd_max = float(df_cards["PD"].max()) if "PD" in df_cards else 1.0
    pd_rng = st.sidebar.slider("PD", 0.0, 1.0, (round(pd_min, 3), round(pd_max, 3)), 0.001)

    # FICO range
    if "FICO_Score" in df_cards:
        fico_min, fico_max = int(df_cards["FICO_Score"].min()), int(df_cards["FICO_Score"].max())
        fico_rng = st.sidebar.slider("FICO", int(fico_min), int(fico_max), (int(fico_min), int(fico_max)))
    else:
        fico_rng = (0, 1000)

    # --- Assumptions you can tune ---
    with st.sidebar.expander("CRR3 Settings / Assumptions", expanded=True):
        st.caption("Tune these for eligibility logic and the ‘Policy vs Reg’ proxy.")

        # Revolving/credit-card product identification
        revolving_defaults = [p for p in prods if any(k in p.lower() for k in ["card", "overdraft", "revolving"])]
        revolving_products = st.multiselect("Revolving products", prods, default=revolving_defaults)

        # Unsecured assumption for revolving
        assume_unsecured = st.checkbox("Assume revolving products are unsecured", value=True)

        # 'Amount owed' proxy (until you add Drawn/EAD)
        amt_field = st.selectbox("Proxy for 'amount owed' (for €1m SME retail test)",
                                 [c for c in ["Asset_Value", "RWA"] if c in df_cards.columns])

        one_million_cap = st.number_input("SME retail cap (€)", min_value=100_000, max_value=5_000_000,
                                          value=1_000_000, step=50_000)

        # SME mapping proxy (use Turnover when you have it)
        st.caption("SME proxy: treat firms as SME if the selected field ≤ threshold (turnover-like).")
        sme_proxy_field = st.selectbox("SME proxy field",
                                       [c for c in ["Asset_Value", "Annual_Salary", "RWA"] if c in df_cards.columns])
        sme_proxy_cap = st.number_input("SME proxy cap", min_value=50_000, max_value=200_000_000,
                                        value=50_000_000, step=1_000_000)

        # QRRE low-volatility proxy (std of PD inside product)
        max_pd_std = st.number_input("QRRE: max PD std by product (volatility proxy)",
                                     min_value=0.0, max_value=1.0, value=0.10, step=0.01)

    # ---------- Apply filters ----------
    f = df_cards.copy()
    if sel_prods: f = f[f["Product_Type"].isin(sel_prods)]
    if sel_biz:   f = f[f["Business_Type"].isin(sel_biz)]
    if "PD" in f: f = f[(f["PD"] >= pd_rng[0]) & (f["PD"] <= pd_rng[1])]
    if "FICO_Score" in f:
        f = f[(f["FICO_Score"] >= fico_rng[0]) & (f["FICO_Score"] <= fico_rng[1])]

    # ---------- CRR3 scaffolding logic ----------
    # Obligor type: simple map (individuals vs firms). Replace with your obligor column when available.
    f["Obligor_Type"] = np.where(f["Business_Type"].str.lower().eq("retail"), "Individual", "Firm")

    # SME vs Corporate (proxy)
    if sme_proxy_field in f:
        f["SME_Flag"] = np.where((f["Obligor_Type"] == "Firm") & (f[sme_proxy_field] <= sme_proxy_cap), True, False)
    else:
        f["SME_Flag"] = False

    # Retail eligibility (CRR retail class)
    # Individuals: retail-eligible; SMEs: retail-eligible if 'amount owed' ≤ €1m; others: not retail
    if amt_field in f:
        f["_Amt_Owed_Proxy"] = f[amt_field].fillna(0.0)
    else:
        f["_Amt_Owed_Proxy"] = np.nan

    f["Retail_Eligible"] = np.where(
        f["Obligor_Type"].eq("Individual"),
        True,
        np.where((f["SME_Flag"]) & (f["_Amt_Owed_Proxy"] <= one_million_cap), True, False)
    )

    # QRRE identification (approx.)
    if "PD" in f:
        pd_std_by_prod = f.groupby("Product_Type")["PD"].std().fillna(0.0)
    else:
        pd_std_by_prod = pd.Series(0.0, index=f["Product_Type"].unique())

    f["Revolving_Flag"] = f["Product_Type"].isin(revolving_products)
    f["LowVol_Flag"] = f["Product_Type"].map(lambda p: (pd_std_by_prod.get(p, 0.0) <= max_pd_std))
    f["Unsecured_Flag"] = assume_unsecured   # global toggle for demo

    f["QRRE_Flag"] = np.where(
        f["Retail_Eligible"] & f["Revolving_Flag"] & f["Unsecured_Flag"] & f["LowVol_Flag"],
        True, False
    )

    # Final segment label
    def segment_row(row):
        if row["QRRE_Flag"]:
            return "Retail–QRRE"
        if row["Retail_Eligible"]:
            return "Retail–Other"
        return "Non‑Retail (SME/Corp)"

    f["Segment"] = f.apply(segment_row, axis=1)

    # ---------- Tabs ----------
    tab_seg, tab_elig, tab_diag, tab_pvrs = st.tabs(
        ["Segments", "Eligibility", " Diagnostics", " Policy vs Reg"]
    )

    # =========================================================
    # TAB 1 — Segments (KPIs + RWA & Mix + Top Products)
    # =========================================================
    with tab_seg:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("# Exposures", f"{len(f):,}")
        k2.metric("Total RWA", f"{f['RWA'].sum():,.0f}" if "RWA" in f else "NA")
        k3.metric("QRRE Count", f"{(f['Segment']=='Retail–QRRE').sum():,}")
        if "RWA" in f:
            qrre_rwa = f.loc[f["Segment"]=="Retail–QRRE", "RWA"].sum()
            total_rwa = f["RWA"].sum() or 1
            k4.metric("QRRE % of RWA", f"{100 * qrre_rwa / total_rwa:.1f}%")
        else:
            k4.metric("QRRE % of RWA", "NA")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("RWA by Segment")
            if "RWA" in f:
                seg_rwa = f.groupby("Segment")["RWA"].sum().sort_values(ascending=False).reset_index()
                st.plotly_chart(
                    px.bar(seg_rwa, x="Segment", y="RWA", color="Segment", text="RWA",
                           title="Total RWA by Segment"),
                    use_container_width=True
                )
            st.markdown("> **Why:** Compare capital across QRRE vs other segments to spot concentration.")

        with c2:
            st.subheader("Exposure Mix (Count)")
            seg_cnt = f["Segment"].value_counts().reset_index()
            seg_cnt.columns = ["Segment", "Count"]
            st.plotly_chart(px.pie(seg_cnt, names="Segment", values="Count", title="Count by Segment"),
                            use_container_width=True)
            st.markdown("> **Tip:** Use the left filters to zoom into card / overdraft products.")

        st.subheader("Top Products by RWA (Segmented)")
        if "RWA" in f:
            top_prod = (f.groupby(["Product_Type", "Segment"])["RWA"].sum()
                        .reset_index().sort_values("RWA", ascending=False).head(15))
            st.plotly_chart(
                px.bar(top_prod, x="Product_Type", y="RWA", color="Segment",
                       title="Top Products by RWA"),
                use_container_width=True
            )

    # =========================================================
    # TAB 2 — Eligibility (rule flags + downloads)
    # =========================================================
    with tab_elig:
        st.subheader("QRRE Eligibility – Rule Flags")
        show_cols = ["Customer_ID", "Product_Type", "Business_Type", "PD", "RWA",
                     "_Amt_Owed_Proxy", "Obligor_Type", "SME_Flag",
                     "Retail_Eligible", "Revolving_Flag", "Unsecured_Flag", "LowVol_Flag", "Segment"]
        show_cols = [c for c in show_cols if c in f.columns]
        table = f[show_cols].copy()
        st.dataframe(table.head(500))

        st.markdown(
            """
            **Reading guide**
            - **Retail_Eligible**: Individuals = retail; SMEs = retail if **amount owed ≤ €1m** (drawn basis).  
            - **QRRE_Flag**: Retail + **revolving** + **unsecured** + **low volatility** proxy.  
            """
        )

        # Downloads
        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download QRRE table (CSV)", data=csv_bytes,
                           file_name="qrre_eligibility_table.csv", mime="text/csv")

        # Optional XLSX
        try:
            import pandas as pd
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                table.to_excel(writer, index=False, sheet_name="QRRE_Eligibility")
            st.download_button("⬇️ Download QRRE table (XLSX)", data=xlsx_buf.getvalue(),
                               file_name="qrre_eligibility_table.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.info("Install **openpyxl** to enable XLSX download: `pip install openpyxl`")

    # =========================================================
    # TAB 3 — Diagnostics (PD & Score behaviour)
    # =========================================================
    with tab_diag:
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("PD Distribution by Segment")
            if "PD" in f:
                st.plotly_chart(px.violin(f, y="PD", x="Segment", color="Segment", box=True, points="all",
                                          title="PD by Segment"),
                                use_container_width=True)
            else:
                st.info("Add **PD** to enable this view.")
            st.markdown("> QRRE pools should show relatively **stable loss** behaviour (lower volatility).")

        with c4:
            st.subheader("FICO vs PD (color = Segment)")
            if {"FICO_Score", "PD"}.issubset(f.columns):
                st.plotly_chart(px.scatter(f, x="FICO_Score", y="PD", color="Segment",
                                           hover_data=["Customer_ID", "Product_Type", "Business_Type"],
                                           trendline="lowess",
                                           title="Score vs PD"),
                                use_container_width=True)
            else:
                st.info("Add **FICO_Score** to enable this diagnostic.")

        st.subheader("PD Bands by Segment")
        if "PD_Band" in f:
            band = (f.groupby(["Segment", "PD_Band"]).size()
                    .reset_index(name="Count"))
            st.plotly_chart(px.bar(band, x="PD_Band", y="Count", color="Segment", barmode="group",
                                   title="Count by PD Band & Segment"),
                            use_container_width=True)

    # =========================================================
    # TAB 4 — Policy vs Regulation (SA proxy)
    # =========================================================
    with tab_pvrs:
        st.subheader("Policy vs Regulation — Standardised Approach (proxy)")
        st.markdown(
            """
            This demo estimates **Standardised Approach (SA) RWA** using simple benchmark risk weights:  
            - **Retail & QRRE** → **75%** RW (non-mortgage retail under SA)  
            - **Non‑Retail (SME/Corp)** → **100%** RW (unrated corporate baseline)  
            - Optional: apply an **SME supporting‑factor** to SME‑flagged exposures (demo only).  
            > We use **Asset_Value** (or your chosen field) as a **proxy for exposure** (until EAD/Drawn is available).  
            """
        )

        # Choose exposure proxy for SA calc
        exp_proxy_col = st.selectbox("Exposure proxy for SA calc",
                                     [c for c in ["Asset_Value", "RWA", "_Amt_Owed_Proxy"] if c in f.columns],
                                     index=0)

        # SME factor demo
        colA, colB = st.columns(2)
        with colA:
            apply_sme_factor = st.checkbox("Apply SME supporting‑factor demo to SME_Flag", value=False)
        with colB:
            sme_factor = st.number_input("SME factor (demo)", min_value=0.50, max_value=1.00, value=0.7619, step=0.01)

        # Benchmark RW mapping
        base_rw = np.where(f["Segment"].isin(["Retail–QRRE", "Retail–Other"]), 0.75, 1.00)
        f["_Exp_Proxy"] = f[exp_proxy_col].fillna(0.0)

        # Apply SME demo
        if apply_sme_factor:
            adj_rw = np.where((f["Segment"]=="Non‑Retail (SME/Corp)") & (f["SME_Flag"]), base_rw * sme_factor, base_rw)
        else:
            adj_rw = base_rw

        f["_SA_RW_Benchmark"] = adj_rw
        f["_SA_RWA_Proxy"] = f["_Exp_Proxy"] * f["_SA_RW_Benchmark"]

        # Deltas (needs reported RWA)
        if "RWA" in f:
            f["_Delta_RWA"] = f["RWA"] - f["_SA_RWA_Proxy"]
            f["_Delta_%"] = np.where(f["_SA_RWA_Proxy"] != 0, 100 * f["_Delta_RWA"] / f["_SA_RWA_Proxy"], np.nan)

        # Visuals
        c5, c6 = st.columns(2)
        with c5:
            if "RWA" in f:
                st.plotly_chart(
                    px.scatter(f, x="_SA_RWA_Proxy", y="RWA", color="Segment",
                               hover_data=["Customer_ID", "Product_Type", "SME_Flag"],
                               title="SA Proxy RWA vs Reported RWA (Diagonal = match)")
                    .update_traces(opacity=0.8),
                    use_container_width=True
                )
            else:
                st.info("Add **RWA** to compare against the SA proxy.")

        with c6:
            comp = (f.groupby("Segment")[["_SA_RWA_Proxy"] + (["RWA"] if "RWA" in f else [])]
                    .sum().reset_index())
            st.plotly_chart(
                px.bar(comp.melt(id_vars="Segment", var_name="Type", value_name="Amount"),
                       x="Segment", y="Amount", color="Type", barmode="group",
                       title="SA Proxy vs Reported — by Segment"),
                use_container_width=True
            )

        # Top deltas
        if "RWA" in f:
            st.subheader("Top 20 Divergences (Reported − SA Proxy)")
            topdev = (f.assign(AbsDev=f["_Delta_RWA"].abs())
                        .sort_values("AbsDev", ascending=False)
                        .head(20))
            st.plotly_chart(
                px.bar(topdev, x="Customer_ID", y="_Delta_RWA", color="Segment",
                       title="Largest absolute differences"),
                use_container_width=True
            )
            st.dataframe(
                topdev[["Customer_ID", "Product_Type", "Segment", "SME_Flag",
                        "_Exp_Proxy", "_SA_RW_Benchmark", "_SA_RWA_Proxy", "RWA", "_Delta_RWA", "_Delta_%"]]
                .head(20)
            )

        st.caption(
            "⚠️ **Demo only:** SA proxy uses broad benchmark RWs and an **exposure proxy**. "
            "For production, compute SA RWA on **EAD** with regulatory CCFs, CRM, and collateral rules."
        )

    # =========================================================
    # TAB 5 — Docs (plain-English notes + sources)
    # =========================================================
    # with tab_docs:
    #     st.markdown(
    #         """
    #         ### What this module checks
    #         - **Retail eligibility** & **QRRE** detection for likely **credit‑card/overdraft** portfolios.  
    #         - **SME vs Corporate** mapping (proxy) and the **€1m** “amount owed” check for SME retail.  
    #         - **Policy vs Regulation** using a simple **SA proxy** (75% retail; 100% non‑retail) with an **SME supporting‑factor** demo.  

    #         ### Why these rules
    #         - Under **CRR**, retail exposures include sub‑classes such as **QRRE**, retail secured by residential property, and others (see **Article 147**). [1](https://en.wikipedia.org/wiki/Advanced_IRB)  
    #         - The EU’s **CRR3** (Regulation (EU) **2024/1623**) updates the credit‑risk chapters from 2025 onward. [2](https://financetrain.com/basel-ii-internal-ratings-based-irb-approach)  
    #         - For the **€1m** SME retail test, the **EBA** clarifies it’s based on the **total amount owed (drawn)**; exposures to **natural persons** are retail‑eligible outside that cap (IRB). [3](https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-217/subpart-D/subject-group-ECFR90182ef648cc7a4/section-217.37)  
    #         - **QRRE** is expected to be **revolving**, typically **credit cards/overdrafts**, with **relatively low loss‑rate volatility**; supervisors (e.g., PRA) expect evidence. [4](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054xAYUAY.pdf)  
    #         - Historical Basel FAQs discuss **FMI vs EL** and volatility expectations for QRRE pools. [5](https://basel-ii-association.com/United_States_of_America/Basel_II_USA_Retail_Exposures.htm)  
    #         - **SME supporting‑factor** examples and thresholds (illustrative): see EBA/industry summaries. [6](https://www.fsb.org/uploads/r_141013a.pdf)

    #         ### What to improve next
        #     - Replace proxies with real **Drawn/EAD**, **Turnover**, **Unsecured/Collateral** flags, **CCFs**.  
        #     - Add **loss‑rate volatility** (mean/stdev across time) for an auditable **QRRE** volatility test.  
        #     - Build a **full SA** calculator with CRM (haircuts, guarantees), product‑level **CCFs**, and **netting**.
        #     """
        # )
    # st.info("Add CRR3 logic, QRRE identification, SME vs corp mapping, and eligibility checks here.")
elif selected_opt.startswith("SME/Corporate"):
    st.header("SME/Corporate Lending Contracts Optimization")
    # ==============================================
    # SME/Corporate Lending Contracts Optimization
    # Structure • Mezz Tranches • Collateral & Haircuts • Guarantees • Netting • SME Factor
    # ==============================================

    # ---------- Data loader ----------
    @st.cache_data
    def load_data(path="Basel_Combined_Datasets.xlsx"):
        import pandas as pd
        # Try preferred sheets/columns; fall back gracefully
        try:
            df = pd.read_excel(path, sheet_name="Merged_RWA_OBS", engine="openpyxl")
        except Exception:
            try:
                df = pd.read_excel(path, sheet_name="Basel_IV_Model", engine="openpyxl")
            except Exception as e:
                st.error(f"Could not read Excel file. Upload **Basel_Combined_Datasets.xlsx**. Details: {e}")
                return pd.DataFrame()
        df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)
        # Coerce key numerics
        for c in ["PD", "LGD", "RWA", "EAD", "Asset_Value", "Annual_Salary"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Exposure proxy
        exp_col = "EAD" if "EAD" in df.columns else ("Asset_Value" if "Asset_Value" in df.columns else None)
        if exp_col is None:
            # No EAD nor Asset_Value → last resort proxy from RWA
            df["Exposure_Proxy"] = df.get("RWA", pd.Series(0.0, index=df.index)).abs()
        else:
            df["Exposure_Proxy"] = df[exp_col].abs()
        # Approximate base RW% (for simulation) = RWA / Exposure_Proxy (clipped)
        if "RWA" in df.columns:
            base_rw = (df["RWA"] / df["Exposure_Proxy"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df["Base_RW"] = base_rw.clip(lower=0.0, upper=2.0)  # 0–200% cap for sanity
        else:
            df["Base_RW"] = 1.0  # if no RWA, assume 100% until user tunes assumptions
        # Light product/sector labels
        if "Product_Type" not in df.columns:
            df["Product_Type"] = df.get("LoanType", "Corporate Loan")
        if "Business_Type" not in df.columns:
            df["Business_Type"] = "Corporate"
        if "Customer_ID" not in df.columns:
            df["Customer_ID"] = np.arange(1, len(df)+1)
        return df

    df = load_data()
    if df.empty:
        st.stop()

    # ---------- Sidebar: Filters & Scenario dials ----------
    st.sidebar.subheader("Filters")
    prods = sorted(df["Product_Type"].dropna().astype(str).unique().tolist())
    sectors = sorted(df["Business_Type"].dropna().astype(str).unique().tolist())
    sel_prod = st.sidebar.multiselect("Prod", prods, default=prods[:10] if len(prods) > 10 else prods)
    sel_sector = st.sidebar.multiselect("Biz", sectors, default=sectors)

    f = df.copy()
    if sel_prod:   f = f[f["Product_Type"].isin(sel_prod)]
    if sel_sector: f = f[f["Business_Type"].isin(sel_sector)]

    # === Scenario dials ===
    st.sidebar.subheader("Scenario Dials")

    # 1) Netting (simplified: % exposure reduction on eligible set)
    netting_share = st.sidebar.slider("Netting – eligible share of exposure", 0.0, 1.0, 0.30, 0.05)
    netting_recog = st.sidebar.slider("Netting – recognition on eligible share", 0.0, 0.50, 0.15, 0.05)

    # 2) Collateral & haircuts (comprehensive approach)
    st.sidebar.markdown("**Collateral & Haircuts (Comprehensive)**")
    collat_pct = st.sidebar.slider("Collateral as % of exposure", 0.0, 1.0, 0.40, 0.05)
    Hc = st.sidebar.slider("Haircut on collateral (Hc)", 0.0, 0.60, 0.15, 0.01)    # e.g., equity 15–25%, debt varies
    He = st.sidebar.slider("Haircut on exposure (He)", 0.0, 0.10, 0.00, 0.01)     # often 0 for loans
    Hfx = st.sidebar.slider("FX mismatch haircut (Hfx)", 0.0, 0.20, 0.08, 0.01)   # typical 8% if mismatch

    # 3) Guarantees – substitution approach
    st.sidebar.markdown("**Guarantees (Substitution)**")
    guar_cov = st.sidebar.slider("Guarantee coverage of residual exposure", 0.0, 1.0, 0.50, 0.05)
    guar_rw_choice = st.sidebar.selectbox(
        "Guarantor RW%",
        options=["0% (Sov)", "20% (Banks/RGLA strong)", "50%", "75% (Retail-like)", "100%"],
        index=1
    )
    guar_rw_map = {"0% (Sov)": 0.00, "20% (Banks/RGLA strong)": 0.20, "50%": 0.50, "75% (Retail-like)": 0.75, "100%": 1.00}
    guar_rw = guar_rw_map[guar_rw_choice]

    # 4) Tranching – allocate exposure into S / M / E and assign RW multipliers
    st.sidebar.markdown("**Tranching**")
    senior_pct = st.sidebar.slider("Senior %", 0.0, 1.0, 0.70, 0.05)
    mezz_pct  = st.sidebar.slider("Mezz %",  0.0, 1.0, 0.20, 0.05)
    equity_pct= st.sidebar.slider("Equity %", 0.0, 1.0, 0.10, 0.05)
    if abs((senior_pct + mezz_pct + equity_pct) - 1.0) > 1e-6:
        st.sidebar.error("Tranche shares must sum to 100%. Adjust sliders.")
    # RW multipliers applied to (remaining) base RW
    senior_mult = st.sidebar.slider("Senior RW ×", 0.10, 1.00, 0.60, 0.05)
    mezz_mult   = st.sidebar.slider("Mezz RW ×",   0.50, 2.50, 1.20, 0.05)
    equity_mult = st.sidebar.slider("Equity RW ×", 1.00, 3.00, 2.50, 0.05)

    # 5) SME support factor (EU CRR Art. 501) – illustrative
    st.sidebar.markdown("**SME Support Factor (EU CRR)**")
    apply_sme = st.sidebar.checkbox("Apply SME support factor (illustrative)", value=False)
    sme_share = st.sidebar.slider("Share of portfolio treated as SME", 0.0, 1.0, 0.30, 0.05)
    # split of E* between ≤2.5m and >2.5m for factor application
    sme_small_ratio = st.sidebar.slider("Within SME: share ≤ €2.5m", 0.0, 1.0, 0.70, 0.05)

    # ---------- Calculations (pipeline) ----------
    E0  = f["Exposure_Proxy"].fillna(0.0).values
    RW0 = f["Base_RW"].fillna(1.0).values

    # Base RWA
    RWA_base = float(np.sum(E0 * RW0))

    # Step 1: Netting (exposure-only recognition)
    E1 = E0 * (1.0 - netting_share * netting_recog)

    # Step 2: Collateral haircuts (comprehensive approach)
    C1 = collat_pct * E1
    E2 = E1 * (1.0 + He) - C1 * (1.0 - Hc - Hfx)
    E2 = np.clip(E2, 0.0, None)  # ≥0
    RWA_after_collat = float(np.sum(E2 * RW0))

    # Step 3: Guarantees (substitution on covered slice of residual exposure)
    cov_amt   = guar_cov * E2
    uncov_amt = (1.0 - guar_cov) * E2
    RWA_after_guar = float(np.sum(uncov_amt * RW0 + cov_amt * guar_rw))

    # Step 4: Tranching (apply RW multipliers to (post-guarantee) exposure using base RW)
    # We illustrate portfolio structuring within loan agreements (not securitization framework).
    Es, Em, Ee = E2 * senior_pct, E2 * mezz_pct, E2 * equity_pct
    RW_s, RW_m, RW_e = RW0 * senior_mult, RW0 * mezz_mult, RW0 * equity_mult
    RWA_after_tranche = float(np.sum(Es*RW_s + Em*RW_m + Ee*RW_e))

    # Step 5: SME support factor (illustrative portfolio-level discount)
    if apply_sme:
        E_meas = E2  # apply to post-collateral, pre-tranche residual exposure (common practice varies)
        E_sme  = sme_share * E_meas
        E_non  = (1.0 - sme_share) * E_meas
        # split SME exposure for 2.5m rule (portfolio-level simplification)
        E_sme_small = sme_small_ratio * E_sme
        E_sme_large = (1.0 - sme_small_ratio) * E_sme
        # Apply factors to RWA contribution (RW × E)
        # factors: 0.7619 for ≤2.5m; 0.85 for >2.5m (illustrative under CRR2/Art. 501)
        RWA_tranche_core = Es*RW_s + Em*RW_m + Ee*RW_e
        # proportionally scale the tranche RWA by SME/non-SME weights
        total_E = np.sum(E_meas) + 1e-12
        prop_sme_small = float(np.sum(E_sme_small) / total_E)
        prop_sme_large = float(np.sum(E_sme_large) / total_E)
        prop_non       = float(np.sum(E_non) / total_E)
        RWA_after_sme = float(np.sum(RWA_tranche_core) * (
            prop_sme_small*0.7619 + prop_sme_large*0.85 + prop_non*1.0
        ))
        RWA_final = RWA_after_sme
    else:
        RWA_final = RWA_after_tranche

    # ---------- KPIs ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposures (n)", f"{len(f):,}")
    c2.metric("Base RWA", f"{RWA_base:,.0f}")
    c3.metric("Final RWA (Scenario)", f"{RWA_final:,.0f}")
    c4.metric("Δ RWA", f"{RWA_final - RWA_base:,.0f}")

    st.caption(
        "• **Base RWA** uses your file's `RWA` when available; otherwise an approximate RW% (=RWA/Exposure) is inferred.\n"
        "• **Exposure proxy** uses `EAD` if present, else `Asset_Value`. Adjust sliders to test CRM and structure levers."
    )

    # ---------- Row 1: Waterfall of RWA changes ----------
    import plotly.graph_objects as go
    steps = [
        ("Base RWA", RWA_base),
        ("Netting benefit", RWA_base - float(np.sum(E1*RW0))),
        ("Collateral benefit", float(np.sum(E1*RW0)) - RWA_after_collat),
        ("Guarantee benefit", RWA_after_collat - RWA_after_guar),
        ("Tranche re‑weight", RWA_after_guar - RWA_after_tranche),
    ]
    if apply_sme:
        steps.append(("SME factor", RWA_after_tranche - RWA_final))

    x = []
    y = []
    measure = []
    cum = RWA_base
    x.append(steps[0][0]); y.append(steps[0][1]); measure.append("absolute")
    for name, delta in steps[1:]:
        x.append(name)
        y.append(delta if name.endswith("benefit") or name=="SME factor" else delta)
        measure.append("relative")
    fig_w = go.Figure(go.Waterfall(
        name="RWA",
        orientation="v",
        measure=measure,
        x=x,
        text=[f"{v:,.0f}" for v in y],
        y=y
    ))
    fig_w.update_layout(title="RWA Waterfall – from Base to Final", showlegend=False)
    st.plotly_chart(fig_w, use_container_width=True)

    # ---------- Row 2: Tranche stack & Collateral effect ----------
    t1, t2 = st.columns(2)

    with t1:
        st.subheader("RWA by Tranche (Scenario)")
        df_tr = pd.DataFrame({
            "Tranche": ["Senior", "Mezz", "Equity"],
            "RWA": [float(np.sum(Es*RW_s)), float(np.sum(Em*RW_m)), float(np.sum(Ee*RW_e))]
        })
        st.plotly_chart(px.bar(df_tr, x="Tranche", y="RWA", color="Tranche",
                               title="Contribution of Tranches to RWA"),
                        use_container_width=True)
        st.markdown(
            "> **Why:** Structuring a facility into senior/mezz/equity layers re‑weights capital consumption even if total exposure is unchanged."
        )

    with t2:
        st.subheader("Collateral Haircuts – Exposure before/after")
        df_coll = pd.DataFrame({
            "Label": ["Pre‑CRM Exposure", "Post‑Haircut Exposure (E*)"],
            "Amount": [float(np.sum(E1)), float(np.sum(E2))]
        })
        st.plotly_chart(px.bar(df_coll, x="Label", y="Amount", color="Label",
                               title="Comprehensive Approach: E → E*"),
                        use_container_width=True)
        st.markdown(
            "> **Formula:**  `E* = E(1 + He) − C(1 − Hc − Hfx)`  (floored at 0). Increase **Hc**/**Hfx** or **collateral %** to test coverage depth."
        )

    # ---------- Row 3: Top RWA savers ----------
    st.subheader("Top Facilities by RWA Saved (Base → Scenario)")
    # per-row base vs final (using tranche scenario)
    # For listing, we compare Base vs Post-Tranche (and SME if chosen)
    RWA_row_base = E0*RW0
    RWA_row_after_tranche = Es*RW_s + Em*RW_m + Ee*RW_e
    if apply_sme:
        # scale row RWA by portfolio SME factor mix — illustrative
        row_scale = (sme_share*sme_small_ratio*0.7619 +
                     sme_share*(1-sme_small_ratio)*0.85 +
                     (1-sme_share)*1.0)
        RWA_row_final = RWA_row_after_tranche * row_scale
    else:
        RWA_row_final = RWA_row_after_tranche

    f2 = f.copy()
    f2["RWA_Base"]  = RWA_row_base
    f2["RWA_Final"] = RWA_row_final
    f2["RWA_Saved"] = f2["RWA_Base"] - f2["RWA_Final"]
    show_cols = ["Customer_ID", "Product_Type", "Business_Type", "RWA_Base", "RWA_Final", "RWA_Saved"]
    st.dataframe(f2[show_cols].sort_values("RWA_Saved", ascending=False).head(15), use_container_width=True)

    # ---------- Guidance / documentation ----------
    with st.expander("How this module applies the rules (plain-English)"):
        st.markdown(
            """
**Netting (exposure recognition):** a simplified reduction against exposure value in line with CRM’s recognition via exposure rather than risk weight.

**Collateral (comprehensive approach):** we apply supervisory haircuts to exposure and collateral, including optional FX mismatch haircut *Hfx*, then risk‑weight the adjusted exposure **E\***.

**Guarantees (substitution):** for the guaranteed slice, we swap the obligor’s risk weight with the guarantor’s risk weight; the uncovered slice keeps the original risk weight.

**Tranches:** a contract‑level structuring concept that applies different RW multipliers to senior/mezz/equity shares to explore capital allocation outcomes (illustrative — not the securitisation framework).

**SME support factor (EU CRR Art. 501):** optional, illustrative RWA discount for eligible SME portions (e.g., 0.7619 up to €2.5m and 0.85 above), applied here as a portfolio‑level scaling.
            """
        )
    # st.info("Add tranche waterfalls, guarantees, haircuts & RWA impact simulations here.")
elif selected_opt.startswith("Checking currency"):
    st.header("FX Hedging & Collateral Mismatch")
    # st.info("Add FX mismatch, currency haircut views, and sensitivity analysis here.")
    # =========================================================
    # FX HEDGING & COLLATERAL CURRENCY MISMATCH – INTERACTIVE LAB
    # =========================================================
    # ---------- Data loader ----------
    @st.cache_data
    def load_fx_data(path="Basel_Combined_Datasets.xlsx", sheet="Merged_RWA_OBS"):
        import pandas as pd
        try:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        except Exception:
            # fallback
            df = pd.read_excel(path, engine="openpyxl")
        df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

        # Numeric casts
        for c in ["EAD", "RWA", "Asset_Value", "PD"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Exposure proxy
        if "EAD" in df.columns:
            df["Exposure_Proxy"] = df["EAD"].abs()
        elif "Asset_Value" in df.columns:
            df["Exposure_Proxy"] = df["Asset_Value"].abs()
        else:
            df["Exposure_Proxy"] = df.get("RWA", 0.0).abs()

        # Base RW% (for visual simulation only)
        if "RWA" in df.columns:
            base = (df["RWA"] / df["Exposure_Proxy"]).replace([np.inf, -np.inf], np.nan)
            df["Base_RW"] = base.fillna(base.median()).clip(0, 2.0)  # 0–200% cap
        else:
            df["Base_RW"] = 1.0

        # Light labels
        if "Product_Type" not in df.columns:
            df["Product_Type"] = df.get("LoanType", "Facility")
        if "Business_Type" not in df.columns:
            df["Business_Type"] = "Unknown"
        if "Customer_ID" not in df.columns:
            df["Customer_ID"] = np.arange(1, len(df)+1)

        return df

    df = load_fx_data()
    if df.empty:
        st.stop()

    # ---------- Sidebar: Filters & Column mapping ----------
    st.sidebar.subheader("Filters")
    prods = sorted(df["Product_Type"].astype(str).unique())
    sects = sorted(df["Business_Type"].astype(str).unique())
    sel_prod = st.sidebar.multiselect("Prod", prods, default=prods[:10] if len(prods) > 10 else prods)
    sel_sect = st.sidebar.multiselect("Biz", sects, default=sects)

    st.sidebar.subheader("Column mapping")
    # Try to auto-detect common names
    likely_curr = [c for c in df.columns if "curr" in c.lower() or c.endswith("_Currency")]
    exp_curr_col = st.sidebar.selectbox("Exposure Currency", options=[*likely_curr, "None"], index=0 if likely_curr else len(likely_curr))
    col_curr_col = st.sidebar.selectbox("Collateral Currency", options=[*likely_curr, "None"], index=1 if len(likely_curr) > 1 else (0 if likely_curr else len(likely_curr)))
    # Optional hedge fields
    num_cols = [c for c in df.columns if df[c].dtype.kind in "if"]
    hedge_notional_col = st.sidebar.selectbox("Hedge Notional (optional)", options=["None", *num_cols], index=0)
    hedge_curr_col = st.sidebar.selectbox("Hedge Currency (optional)", options=[*likely_curr, "None"], index=len(likely_curr))

    # Scenario dials
    st.sidebar.subheader("Scenario dials")
    Hfx = st.sidebar.slider("Currency mismatch haircut (Hfx)", 0.00, 0.20, 0.08, 0.01)  # 8% common baseline
    He  = st.sidebar.slider("Exposure haircut (He)", 0.00, 0.10, 0.00, 0.01)           # often 0 for loans
    Hc  = st.sidebar.slider("Collateral haircut (Hc)", 0.00, 0.60, 0.15, 0.01)         # collateral quality
    coll_pct = st.sidebar.slider("Collateral as % of exposure", 0.0, 1.0, 0.50, 0.05)

    # ---------- Apply filters ----------
    f = df.copy()
    if sel_prod: f = f[f["Product_Type"].isin(sel_prod)]
    if sel_sect: f = f[f["Business_Type"].isin(sel_sect)]
    # Fill mapping
    f["_Exp_CCY"] = f[exp_curr_col] if exp_curr_col in f.columns else "UNK"
    f["_Col_CCY"] = f[col_curr_col] if col_curr_col in f.columns else "UNK"
    f["_Hedge_Notional"] = f[hedge_notional_col] if hedge_notional_col in f.columns else 0.0
    f["_Hedge_CCY"] = f[hedge_curr_col] if hedge_curr_col in f.columns else "UNK"

    # ---------- FX mismatch & hedging metrics ----------
    def mismatch_pair(e, c):
        return f"{e}->{c}" if isinstance(e, str) and isinstance(c, str) else "UNK"
    f["CCY_Pair"] = [mismatch_pair(e, c) for e, c in zip(f["_Exp_CCY"], f["_Col_CCY"])]
    f["FX_Mismatch"] = (f["_Exp_CCY"] != f["_Col_CCY"]).astype(int)

    # Hedge ratio (only if hedge currency matches exposure currency; else still show)
    f["Hedge_Ratio"] = (f["_Hedge_Notional"] / f["Exposure_Proxy"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 2.0)
    f["Hedge_Aligned"] = (f["_Hedge_CCY"] == f["_Exp_CCY"]).astype(int)

    # ---------- CRM comprehensive approach with FX haircut ----------
    # E* = E(1 + He) − C(1 − Hc − Hfx_mis), where Hfx applies only when currencies mismatch
    E = f["Exposure_Proxy"].values
    C = (coll_pct * E)
    Hfx_vec = np.where(f["FX_Mismatch"].values == 1, Hfx, 0.0)
    E_star = E * (1.0 + He) - C * (1.0 - Hc - Hfx_vec)
    E_star = np.clip(E_star, 0.0, None)

    # Base and post-haircut RWA (illustrative using Base_RW)
    RW = f["Base_RW"].values
    RWA_base = float(np.sum(E * RW))
    RWA_after_fx = float(np.sum(E_star * RW))

    # ---------- KPIs ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Exposures (n)", f"{len(f):,}")
    k2.metric("FX Mismatch (count)", f"{int(f['FX_Mismatch'].sum()):,}")
    k3.metric("Base RWA", f"{RWA_base:,.0f}")
    k4.metric("RWA after FX haircuts", f"{RWA_after_fx:,.0f}")

    st.caption(
        "• **FX mismatch** = Exposure currency ≠ Collateral currency → apply **Hfx** in the comprehensive approach.\n"
        "• Results use your file’s `EAD` (or `Asset_Value`) as **E** and a simple portfolio collateral % as **C**."
    )

    # ---------- ROW 1: Currency landscape ----------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Exposure by Currency Pair (Exposure→Collateral)")
        cnt = (f.groupby("CCY_Pair")["Exposure_Proxy"]
               .sum().sort_values(ascending=False).reset_index())
        st.plotly_chart(px.bar(cnt.head(15), x="CCY_Pair", y="Exposure_Proxy", color="CCY_Pair",
                               title="Top Currency Pairs by Exposure"), use_container_width=True)
        st.markdown("> **Why:** Identifies where mismatches concentrate and where Hfx matters most.")

    with c2:
        st.subheader("Hedge Ratio vs FX Mismatch")
        st.plotly_chart(
            px.scatter(f, x="Hedge_Ratio", y="Exposure_Proxy", color="FX_Mismatch",
                       color_continuous_scale=["#2ca02c", "#d62728"],
                       labels={"FX_Mismatch": "Mismatch (1=yes)"},
                       hover_data=["Customer_ID", "Product_Type", "_Exp_CCY", "_Col_CCY", "_Hedge_CCY"],
                       title="Are large positions hedged and aligned in currency?"),
            use_container_width=True
        )
        st.markdown(
            "> **Read:** Points at high **Hedge_Ratio** and **Mismatch=1** may still be exposed if hedges are in a different currency."
        )

    # ---------- ROW 2: Haircut impact & Waterfall ----------
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Comprehensive Approach – Exposure before/after")
        df_before_after = pd.DataFrame({
            "Label": ["Pre‑haircut E", "Post‑haircut E*"],
            "Amount": [float(np.sum(E)), float(np.sum(E_star))]
        })
        st.plotly_chart(px.bar(df_before_after, x="Label", y="Amount", color="Label",
                               title="E → E* with Hc/He/Hfx"), use_container_width=True)
        st.markdown("`E* = E(1 + He) − C(1 − Hc − Hfx)`; **Hfx** applies only where currencies mismatch.")

    with c4:
        import plotly.graph_objects as go
        st.subheader("RWA Waterfall (FX haircuts only)")
        rwa0 = RWA_base
        rwa1 = RWA_after_fx
        fig = go.Figure(go.Waterfall(
            measure=["absolute", "relative"],
            x=["Base RWA", "FX haircuts impact"],
            y=[rwa0, rwa1 - rwa0],
            text=[f"{rwa0:,.0f}", f"{(rwa1 - rwa0):,.0f}"],
        ))
        fig.update_layout(title="Base → After Hfx/Hc/He", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("> **Why:** Shows portfolio‑level capital sensitivity to currency mismatch haircuts.")

    # ---------- ROW 3: Sensitivity – vary Hfx ----------
    st.subheader("Sensitivity to Currency Mismatch Haircut (Hfx)")
    grid = np.linspace(0.00, 0.15, 16)  # 0% to 15%
    rows = []
    for h in grid:
        E_star_h = E * (1.0 + He) - C * (1.0 - Hc - np.where(f["FX_Mismatch"].values==1, h, 0.0))
        E_star_h = np.clip(E_star_h, 0.0, None)
        rows.append({"Hfx": h, "Sum_E*": float(np.sum(E_star_h)), "RWA": float(np.sum(E_star_h * RW))})
    sens = pd.DataFrame(rows)
    st.plotly_chart(px.line(sens, x="Hfx", y=["Sum_E*", "RWA"], markers=True,
                            title="Hfx Sensitivity: Exposure After Haircuts & RWA"),
                    use_container_width=True)
    st.markdown(
        "> **Tip:** Start at **Hfx≈8%** (standard supervisory setting) and test governance scenarios with ±2–4%.\n"
        "> **Note:** If you remargin less frequently/longer holding periods, supervisory haircuts are scaled upward."
    )

    # ---------- ROW 4: Drill – worst mismatches ----------
    st.subheader("Top Mismatch Positions (by exposure)")
    cols = ["Customer_ID", "Product_Type", "Business_Type", "_Exp_CCY", "_Col_CCY",
            "Exposure_Proxy", "Hedge_Ratio", "_Hedge_CCY"]
    view = f[f["FX_Mismatch"]==1][cols].sort_values("Exposure_Proxy", ascending=False).head(20)
    st.dataframe(view, use_container_width=True)

    # ---------- Documentation ----------
    with st.expander("What this dashboard checks "):
        st.markdown(
            """
**1) FX mismatch:** We flag facilities where **Exposure currency ≠ Collateral currency**.  
**2) Haircuts:** Under the **comprehensive approach**, we apply base haircuts (**Hc** on collateral, optional **He** on exposure) and add **Hfx** for currency mismatch. This increases the effective exposure **E\*** before risk‑weighting.  
**3) Hedging:** If hedge size and currency are available, **Hedge_Ratio** helps assess if the FX risk is neutralised and whether the hedge currency is aligned to the exposure.  
**4) Sensitivity:** The line chart shows portfolio **E\*** and **RWA** as **Hfx** varies (e.g., policy or market stress).
            """
        )
elif selected_opt.startswith("Review/enhancement"):
    st.header("Standard RWA Process")
    # =========================================================
    # STANDARD RWA CALCULATION – REVIEW & ENHANCEMENT
    # Automation audit • Δ% checks • Reconciliation to rules
    # =========================================================
    # ---------- Data loader ----------
    @st.cache_data
    def load_any(path="Basel_Combined_Datasets.xlsx"):
        import pandas as pd
        # Try common sheets in your upload; normalize names
        for sheet in ["Basel_IV_Model", "Merged_RWA_OBS", 0]:
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
                df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)
                return df
            except Exception:
                continue
        return pd.DataFrame()

    df = load_any()
    if df.empty:
        st.warning("Upload **Basel_Combined_Datasets.xlsx** or set the correct path/sheet.")
        st.stop()

    # ---------- Lightweight harmonization ----------
    # Use LoanType / Product_Type for grouping if available
    if "LoanType" not in df.columns:
        if "Product_Type" in df.columns:
            df["LoanType"] = df["Product_Type"]
        else:
            df["LoanType"] = "Loan"
    for c in ["PD", "LGD", "EAD", "RiskWeight", "RWA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Exposure proxy (if EAD missing)
    if "EAD" not in df.columns or df["EAD"].isna().all():
        # fall back to Asset_Value; if missing, scale from RWA
        if "Asset_Value" in df.columns:
            df["EAD"] = pd.to_numeric(df["Asset_Value"], errors="coerce")
        else:
            # if all else fails, infer a proxy assuming 100% RW baseline
            df["EAD"] = df.get("RWA", 0.0).abs()

    # ---------- Sidebar filters ----------
    st.sidebar.subheader("Filters")
    loan_types = sorted(df["LoanType"].dropna().astype(str).unique().tolist())
    sel_types = st.sidebar.multiselect("Loan Type", loan_types, default=loan_types[:10] if len(loan_types)>10 else loan_types)

    if sel_types:
        df = df[df["LoanType"].isin(sel_types)]

    # PD filter if present
    if "PD" in df.columns:
        pd_min, pd_max = float(df["PD"].min(skipna=True) or 0.0), float(df["PD"].max(skipna=True) or 1.0)
        pd_rng = st.sidebar.slider("PD range", 0.0, 1.0, (round(max(0.0, pd_min),3), round(min(1.0, pd_max),3)), 0.001)
        df = df[(df["PD"].fillna(0.0) >= pd_rng[0]) & (df["PD"].fillna(0.0) <= pd_rng[1])]

    # ---------- Standardized validator (simple PD→RW buckets) ----------
    # Your reference rule:
    # Estimated_RW = 0.30 if PD<0.20; 0.50 if PD<0.50; else 0.75
    if "PD" not in df.columns:
        df["PD"] = np.nan
    df["Estimated_RW"] = df["PD"].apply(lambda x: 0.3 if pd.notna(x) and x < 0.2 else (0.5 if pd.notna(x) and x < 0.5 else (0.75 if pd.notna(x) else np.nan)))

    # Calculated RWA from validator
    df["Calculated_RWA"] = df["EAD"].fillna(0.0) * df["Estimated_RW"].fillna(0.0)

    # Differences vs reported RWA
    if "RWA" not in df.columns:
        df["RWA"] = np.nan
    df["RWA_Difference"] = df["Calculated_RWA"] - df["RWA"]
    df["RWA_Delta_%"] = np.where(df["RWA"].abs()>0, 100.0 * df["RWA_Difference"] / df["RWA"], np.nan)

    # Optional reconciliation to rule if RiskWeight exists
    if "RiskWeight" in df.columns:
        df["Rule_RWA"] = df["EAD"].fillna(0.0) * df["RiskWeight"].fillna(0.0)
        df["Rule_Diff"] = df["Rule_RWA"] - df["RWA"]
        df["Rule_Delta_%"] = np.where(df["RWA"].abs()>0, 100.0 * df["Rule_Diff"]/df["RWA"], np.nan)
    else:
        df["Rule_RWA"] = np.nan
        df["Rule_Diff"] = np.nan
        df["Rule_Delta_%"] = np.nan

    # ---------- KPI cards ----------
    tot_ead = float(df["EAD"].sum(skipna=True))
    rep_rwa = float(df["RWA"].sum(skipna=True)) if df["RWA"].notna().any() else np.nan
    calc_rwa = float(df["Calculated_RWA"].sum(skipna=True))
    med_delta = float(df["RWA_Delta_%"].median(skipna=True)) if df["RWA_Delta_%"].notna().any() else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("# Loans", f"{len(df):,}")
    k2.metric("Total EAD", f"{tot_ead:,.0f}")
    k3.metric("RWA (Reported)", "NA" if np.isnan(rep_rwa) else f"{rep_rwa:,.0f}")
    k4.metric("RWA (Calculated)", f"{calc_rwa:,.0f}")
    k5.metric("Median Δ% (Calc vs Rep)", "NA" if np.isnan(med_delta) else f"{med_delta:,.1f}%")

    st.caption(
        "We recompute a **simplified standardized RWA** as `EAD × Estimated_RW(PD bucket)` to help **validate automation**. "
        "Use the visuals below to spot differences and data issues."
    )

    # =========================================================
    # ROW 1 — Reported vs Calculated + Δ% histogram
    # =========================================================
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Reported vs Calculated RWA")
        if df["RWA"].notna().any():
            # Scatter with 45° line
            fig_sc = px.scatter(
                df, x="RWA", y="Calculated_RWA", color="LoanType",
                hover_data=[col for col in ["LoanID","PD","LGD","EAD","RiskWeight"] if col in df.columns],
                title="Reported vs Calculated (PD-bucket validator)"
            )
            # add diagonal
            if np.isfinite(rep_rwa) and np.isfinite(calc_rwa):
                max_axis = float(np.nanmax([df["RWA"].max(), df["Calculated_RWA"].max()]))
                fig_sc.add_shape(type="line", x0=0, y0=0, x1=max_axis, y1=max_axis, line=dict(color="red", dash="dash"))
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("No reported RWA in file to compare against.")

        st.markdown("> **Why:** points far from the red diagonal suggest mapping or data issues (PD, EAD, RW).")

    with c2:
        st.subheader("Δ% Distribution (Calculated vs Reported)")
        if df["RWA_Delta_%"].notna().any():
            fig_hist = px.histogram(df, x="RWA_Delta_%",
                                    nbins=60, color="LoanType",
                                    title="Percentage deviation histogram")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No Δ% available (missing RWA).")
        st.markdown("> **Check:** wide tails or strong skew imply inconsistent automation or input anomalies.")

    # =========================================================
    # ROW 2 — Avg RWA by Loan Type + Top‑15 deviations (bars)
    # =========================================================
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Average RWA by Loan Type (interactive)")
        if df["RWA"].notna().any():
            avg_rwa = (df.groupby("LoanType", as_index=False)["RWA"]
                         .mean().sort_values("RWA", ascending=False))
            fig_avg = px.bar(avg_rwa, x="LoanType", y="RWA", color="LoanType",
                             title="Average RWA by Loan Type")
            st.plotly_chart(fig_avg, use_container_width=True)
        else:
            st.info("No reported RWA found to compute averages.")

        st.markdown(
            "> **Note:** this is the interactive equivalent of your seaborn bar chart. "
            "Use the **sidebar filter** to focus on a subset."
        )

    with c4:
        st.subheader("Top 15 Deviations — Reported vs Calculated")
        if df["RWA"].notna().any():
            top = df.reindex(df["RWA_Difference"].abs().sort_values(ascending=False).head(15).index)
            # Create id labels
            lab = (top["LoanID"].astype(str) if "LoanID" in top.columns else top.index.astype(str))
            df_top = pd.DataFrame({
                "Label": lab,
                "Reported_RWA": top["RWA"].values,
                "Calculated_RWA": top["Calculated_RWA"].values
            })
            fig_top = px.bar(
                df_top.melt(id_vars="Label", var_name="Series", value_name="Amount"),
                x="Label", y="Amount", color="Series", barmode="group",
                title="Reported vs Calculated – Top Deviating Loans"
            )
            fig_top.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("Cannot show top deviations when reported RWA is missing.")

        st.markdown("> **Action:** drill into these facilities first; mismatches often come from RW mapping or PD/EAD inputs.")

    # =========================================================
    # ROW 3 — Automation Audit & Reconciliation to Rules
    # =========================================================
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("Automation Audit – Data Quality")
        # Simple rule checks
        checks = {}
        checks["Missing PD"] = int(df["PD"].isna().sum())
        checks["PD out of [0,1]"] = int(((df["PD"]<0) | (df["PD"]>1)).sum()) if "PD" in df else 0
        checks["Missing EAD"] = int(df["EAD"].isna().sum())
        checks["EAD < 0"] = int((df["EAD"]<0).sum())
        checks["LGD missing"] = int(df["LGD"].isna().sum()) if "LGD" in df else 0
        checks["LGD out of [0,1]"] = int(((df["LGD"]<0)|(df["LGD"]>1)).sum()) if "LGD" in df else 0
        checks["RWA missing"] = int(df["RWA"].isna().sum())
        checks["RWA < 0"] = int((df["RWA"].fillna(0)<0).sum())

        # Δ% flags
        big_25 = int((df["RWA_Delta_%"].abs()>25).sum())
        big_50 = int((df["RWA_Delta_%"].abs()>50).sum())
        checks["|Δ%| > 25"] = big_25
        checks["|Δ%| > 50"] = big_50

        audit_tbl = pd.DataFrame(
            {"Check": list(checks.keys()), "Count": list(checks.values())}
        ).sort_values("Count", ascending=False)
        st.dataframe(audit_tbl, use_container_width=True, hide_index=True)

        st.markdown(
            "> **Interpretation:** Fix **missing/invalid PD, EAD, LGD** first; then review outliers with **|Δ%| > 25/50**."
        )

    with c6:
        st.subheader("Reconciliation to Rules (if RiskWeight present)")
        if "RiskWeight" in df.columns and df["RiskWeight"].notna().any():
            # mismatch rate using 5% tolerance
            tol = 5.0
            mis = int((df["Rule_Delta_%"].abs() > tol).sum())
            tot = int(df["Rule_Delta_%"].notna().sum())
            rate = 100.0 * mis / max(tot,1)
            m1, m2 = st.columns(2)
            m1.metric("Obs with RW", f"{tot:,}")
            m2.metric("Mismatch rate (>5%)", f"{rate:.1f}%")

            fig_rule = px.histogram(df, x="Rule_Delta_%",
                                    nbins=60, title="Δ% to Rule: Reported RWA vs EAD×RiskWeight")
            st.plotly_chart(fig_rule, use_container_width=True)
            st.markdown(
                "> **Why:** If many deals breach a small tolerance (e.g., **5%**), check **RW mapping**, **EAD units**, and **rounding**."
            )
        else:
            st.info("RiskWeight column not found — skipping rule reconciliation histogram.")

    # ---------- Download results ----------
    with st.expander("Download audit & deltas"):
        out_cols = [c for c in ["LoanID","LoanType","PD","LGD","EAD","RiskWeight","RWA",
                                "Estimated_RW","Calculated_RWA","RWA_Difference","RWA_Delta_%",
                                "Rule_RWA","Rule_Diff","Rule_Delta_%"] if c in df.columns]
        csv_bytes = df[out_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (audit_results.csv)", data=csv_bytes, file_name="audit_results.csv", mime="text/csv")

    # ---------- Help ----------
    with st.expander("What each check means "):
        st.markdown(
            """
- **Validator formula:** a simple, transparent rule `EAD × Estimated_RW(PD bucket)` to sanity‑check automation (not a regulatory model).
- **Scatter (Reported vs Calculated):** points should cluster near the red diagonal if the pipeline is sound.
- **Δ% histogram:** should center near 0%. Long tails indicate mapping/ingestion errors.
- **Rule reconciliation:** if `RiskWeight` is present, verify `Reported RWA ≈ EAD × RiskWeight`. High mismatch → fix **RW** assignment or **EAD** scaling.
- **Audit table:** prioritize **missing/invalid PD/EAD/LGD** and **large |Δ%|** items for correction.
            """
        )
    # st.info("Add automation audit, delta % checks, and reconciliation to rules here.")
elif selected_opt.startswith("Assessment of potential"):
    st.header("NPE, Pre-Default, Recovery")
   # =========================================================
    # NPE (Non-Performing Exposures), Pre-Default Forecasting,
    # Defaulted Exposures & Early-Recovery Modelling
    # =========================================================

    # ---------- Data loader ----------
    @st.cache_data
    def load_any(path="Basel_Combined_Datasets.xlsx"):
        import pandas as pd, numpy as np
        try:
            df = pd.read_excel(path, sheet_name="Merged_RWA_OBS", engine="openpyxl")
        except Exception:
            try:
                df = pd.read_excel(path, engine="openpyxl")
            except Exception as e:
                st.error(f"Upload **Basel_Combined_Datasets.xlsx**. Details: {e}")
                return pd.DataFrame()
        df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

        # Exposure proxy
        if "EAD" in df.columns:
            df["EAD"] = pd.to_numeric(df["EAD"], errors="coerce").abs()
            df["Exposure_Proxy"] = df["EAD"]
        elif "Asset_Value" in df.columns:
            df["Exposure_Proxy"] = pd.to_numeric(df["Asset_Value"], errors="coerce").abs()
        else:
            # last resort: from RWA base if available
            if "RWA" in df.columns:
                df["Exposure_Proxy"] = pd.to_numeric(df["RWA"], errors="coerce").abs()
            else:
                df["Exposure_Proxy"] = 0.0

        # LGD
        if "LGD" in df.columns:
            df["LGD"] = pd.to_numeric(df["LGD"], errors="coerce").clip(0, 1)
        else:
            df["LGD"] = 0.45  # neutral fallback; can change via sidebar
        # PD (point-in-time)
        if "PD" in df.columns:
            df["PD"] = pd.to_numeric(df["PD"], errors="coerce").clip(0, 1)
        return df

    df = load_any()
    if df.empty:
        st.stop()

    # ---------- Sidebar: cohort filters ----------
    st.sidebar.subheader("Filters")
    # Generic categorical filters if present
    cat_cols = [c for c in ["Product_Type", "Business_Type", "Region", "Segment", "Industry"]
                if c in df.columns]
    f = df.copy()
    for c in cat_cols:
        vals = sorted(f[c].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect(c, vals, default=vals[:10] if len(vals) > 10 else vals)
        if sel:
            f = f[f[c].astype(str).isin(sel)]

    # EAD & LGD filters (robust)
    st.sidebar.subheader("Exposure & LGD")
    ead_series = f.get("EAD", f["Exposure_Proxy"]).replace([np.inf, -np.inf], np.nan).dropna()
    if not ead_series.empty:
        ead_min, ead_max = float(ead_series.min()), float(ead_series.max())
        ead_low, ead_high = st.sidebar.slider(
            "EAD range", min_value=float(np.floor(ead_min)), max_value=float(np.ceil(ead_max)),
            value=(float(np.floor(ead_min)), float(np.ceil(ead_max)))
        )
        f = f[(f.get("EAD", f["Exposure_Proxy"]) >= ead_low) & (f.get("EAD", f["Exposure_Proxy"]) <= ead_high)]
    lgd_series = f["LGD"].replace([np.inf, -np.inf], np.nan).dropna()
    if not lgd_series.empty:
        lgd_low, lgd_high = st.sidebar.slider("LGD range (0–1)", 0.0, 1.0,
                                              value=(float(lgd_series.min()), float(lgd_series.max())), step=0.01)
        f = f[(f["LGD"] >= lgd_low) & (f["LGD"] <= lgd_high)]

    # ---------- Target label: defaults / NPE ----------
    st.sidebar.subheader("Target (Defaults/NPE)")
    default_cols = [c for c in f.columns if c.lower() in
                    ["default_flag","default","in_default","npe","npl","status","dpd","days_past_due"]]
    method = st.sidebar.selectbox(
        "How to define default:",
        ["Auto (DPD≥90 / status contains 'default')", "Pick a column", "Threshold on PD"]
    )

    if method == "Pick a column" and default_cols:
        col = st.sidebar.selectbox("Select column", default_cols)
        if col.lower() in ["dpd","days_past_due"]:
            dpd_thr = st.sidebar.slider("DPD ≥", 30, 180, 90, 5)
            f["TARGET_DEFAULT"] = (pd.to_numeric(f[col], errors="coerce") >= dpd_thr).astype(int)
        elif col.lower() in ["status"]:
            f["TARGET_DEFAULT"] = f[col].astype(str).str.lower().str.contains("default|npl|npe").astype(int)
        else:
            f["TARGET_DEFAULT"] = pd.to_numeric(f[col], errors="coerce").fillna(0).clip(0,1).astype(int)
    elif method == "Threshold on PD":
        if "PD" not in f.columns:
            st.warning("No PD column available; cannot threshold. Falling back to Auto.")
            method = "Auto (DPD≥90 / status contains 'default')"
        else:
            pd_thr = st.sidebar.slider("Mark default if PD ≥", 0.01, 1.0, 0.20, 0.01)
            f["TARGET_DEFAULT"] = (f["PD"].fillna(0) >= pd_thr).astype(int)
    if method.startswith("Auto"):
        if "DPD" in f.columns:
            f["TARGET_DEFAULT"] = (pd.to_numeric(f["DPD"], errors="coerce") >= 90).astype(int)
        elif "Days_Past_Due" in f.columns:
            f["TARGET_DEFAULT"] = (pd.to_numeric(f["Days_Past_Due"], errors="coerce") >= 90).astype(int)
        elif "Status" in f.columns:
            f["TARGET_DEFAULT"] = f["Status"].astype(str).str.lower().str.contains("default|npl|npe").astype(int)
        else:
            # Fall back to PD threshold of 20% if nothing else (user can change to "Threshold on PD")
            f["TARGET_DEFAULT"] = (f.get("PD", 0).fillna(0) >= 0.20).astype(int)

    # ---------- Feature set ----------
    import numpy as np, pandas as pd
    # Choose numeric features; drop obvious non-features
    drop_like = {"TARGET_DEFAULT","EAD","Exposure_Proxy","LGD","RWA","Customer_ID"}
    num_cols = [c for c in f.select_dtypes(include=[np.number]).columns if c not in drop_like]
    # If none, stop gracefully
    if not num_cols:
        st.warning("No numeric features to train a PD model. Please add numeric drivers (utilization, DTI, DPD, scores...).")
        st.stop()

    # Train / test split (random if no date)
    from sklearn.model_selection import train_test_split
    X = f[num_cols].fillna(f[num_cols].median())
    y = f["TARGET_DEFAULT"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Standardize & Logistic Regression (class_weight balanced)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(class_weight='balanced', max_iter=500, solver="lbfgs")
    )
    pipe.fit(X_train, y_train)

    # Predictions / metrics
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    # PD bands & transitions (current PD vs predicted PD_hat)
    def band(x):
        if x < 0.01: return "A: <1%"
        if x < 0.03: return "B: 1–3%"
        if x < 0.07: return "C: 3–7%"
        if x < 0.12: return "D: 7–12%"
        if x < 0.20: return "E: 12–20%"
        return "F: ≥20%"

    test_df = f.loc[X_test.index].copy()
    test_df["PD_hat"] = y_prob
    if "PD" in test_df.columns:
        test_df["PD_band_now"] = test_df["PD"].fillna(0).apply(band)
    else:
        # If no current PD, use model PD on train set as proxy baseline
        test_df["PD_band_now"] = test_df["PD_hat"].apply(band)
    test_df["PD_band_12m"] = test_df["PD_hat"].apply(band)
    trans = pd.crosstab(test_df["PD_band_now"], test_df["PD_band_12m"], normalize="index") * 100.0

    # Simplified IRB K & RWA (illustrative; not the full formula)
    # (Reference snippet you provided)
    try:
        from scipy.stats import norm
        R = 0.15
        z_pd = norm.ppf(np.clip(test_df["PD_hat"], 1e-6, 1-1e-6))
        K = test_df["LGD"].values * norm.cdf(z_pd / np.sqrt(1 - R))
        EADv = test_df.get("EAD", test_df["Exposure_Proxy"]).fillna(0).values
        RWA = 12.5 * K * EADv
        rwa_sum = float(np.sum(RWA))
    except Exception:
        K = np.zeros(len(test_df))
        rwa_sum = 0.0

    # ---------- KPIs ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records (cohort)", f"{len(f):,}")
    k2.metric("AUC (Hold‑out)", f"{auc:.3f}")
    k3.metric("Precision / Recall", f"{prec:.2f} / {rec:.2f}")
    k4.metric("Scenario RWA (illustrative)", f"{rwa_sum:,.0f}")

    st.caption(
        "• **Target (default/NPE)** uses your DPD/status/flag if present (90+ DPD backstop common in EU). "
        "• **AUC/PR**: model discrimination on hold‑out. "
        "• **RWA** uses the simplified K form for teaching; see expander for the full IRB reference."
    )

    # ---------- Row 1: ROC and Confusion Matrix ----------
    import plotly.express as px, plotly.graph_objects as go
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fpr, tpr, thr = roc_curve(y_test, y_prob)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        st.subheader("ROC Curve")
        st.plotly_chart(px.line(roc_df, x="FPR", y="TPR", title=f"ROC (AUC={auc:.3f})").update_layout(yaxis=dict(range=[0,1]), xaxis=dict(range=[0,1])), use_container_width=True)
        st.markdown("> **Why this matters:** AUC shows how well your early‑warning signals separate future defaults from non‑defaults.")

    with r1c2:
        st.subheader("Confusion Matrix (0.5 threshold)")
        cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
        st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                                  title="Confusion Matrix"), use_container_width=True)
        st.markdown("> Tune the threshold if you want to prioritise **recall** (catching more NPE inflows) over precision.")

    # ---------- Row 2: PD Transitions & Early-Warning Drivers ----------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("PD Band Transitions (current → 12m model)")
        st.plotly_chart(px.imshow(trans.reindex(sorted(trans.index), columns=sorted(trans.columns)),
                                  text_auto=".1f", color_continuous_scale="Viridis",
                                  title="Rows sum to 100%").update_xaxes(title="12m PD band").update_yaxes(title="Current band"),
                        use_container_width=True)
        st.markdown("> **PD migration** helps quantify inflows into higher‑risk bands — a leading indicator for NPE strategy.")

    with r2c2:
        st.subheader("Top Early‑Warning Signals (Logistic coefficients)")
        # get standardized coefficients
        lr = pipe.named_steps["logisticregression"]
        sc = pipe.named_steps["standardscaler"]
        coefs = pd.Series(lr.coef_.ravel(), index=num_cols)
        drivers = coefs.abs().sort_values(ascending=False).head(10).index.tolist()
        imp = coefs[drivers].sort_values(key=np.abs)
        st.plotly_chart(px.bar(x=imp.index, y=imp.values,
                               title="Largest magnitude standardized coefficients",
                               labels={"x":"Feature","y":"Coefficient"}), use_container_width=True)
        st.markdown("> Use these drivers to build **early‑warning rules** (watchlists / limit tightening / covenants).")

    # ---------- Row 3: Collections & Recovery Path Simulator ----------
    st.subheader("Early‑Recovery Paths — Cure / Restructure / Liquidation (NPV & LGD)")
    s1, s2, s3, s4, s5 = st.columns(5)
    cure_share = s1.slider("Cure %", 0, 100, 25, 5)
    restr_share = s2.slider("Restructure %", 0, 100, 50, 5)
    liq_share = 100 - (cure_share + restr_share)
    s3.metric("Liquidation %", f"{max(liq_share,0)}%")
    disc_rate = s4.number_input("Discount rate (annual %)", 0.0, 50.0, 10.0, 0.5) / 100.0
    horizon_m = s5.number_input("Horizon (months)", 1, 60, 24, 1)

    # Path assumptions
    st.markdown("**Path assumptions (set recoveries & timing)**")
    p1, p2, p3 = st.columns(3)
    cure_rec = p1.slider("Cure: recovery % of EAD", 0, 100, 80, 5) / 100.0
    cure_m   = p1.slider("Cure months", 1, 24, 6)
    restr_rec= p2.slider("Restructure: recovery %", 0, 100, 60, 5) / 100.0
    restr_m  = p2.slider("Restructure months", 1, 36, 18)
    liq_rec  = p3.slider("Liquidation: recovery %", 0, 100, 40, 5) / 100.0
    liq_m    = p3.slider("Liquidation months", 1, 48, 24)

    # Defaulted cohort
    def_df = f[f["TARGET_DEFAULT"]==1].copy()
    EAD_def = def_df.get("EAD", def_df["Exposure_Proxy"]).fillna(0).sum()
    # NPV function
    def npv(amount, months, r):
        return amount / ((1+r/12.0)**months)

    weights = np.array([cure_share, restr_share, max(liq_share,0)]) / max(cure_share + restr_share + max(liq_share,0), 1)
    rec_vals = np.array([cure_rec, restr_rec, liq_rec])
    rec_mths = np.array([cure_m, restr_m, liq_m])

    rec_npv = np.sum([npv(rec_vals[i]*EAD_def*weights[i], rec_mths[i], disc_rate) for i in range(3)])
    lgd_implied = 1.0 - (rec_npv / EAD_def if EAD_def>0 else 0.0)

    cA, cB = st.columns(2)
    with cA:
        chart = pd.DataFrame({
            "Path":["Cure","Restructure","Liquidation"],
            "NPV":[npv(cure_rec*EAD_def*weights[0], cure_m, disc_rate),
                   npv(restr_rec*EAD_def*weights[1], restr_m, disc_rate),
                   npv(liq_rec*EAD_def*weights[2], liq_m, disc_rate)]
        })
        st.plotly_chart(px.bar(chart, x="Path", y="NPV", title=f"NPV by Recovery Path (Total defaulted EAD: {EAD_def:,.0f})"),
                        use_container_width=True)
    with cB:
        # Simple monthly curve
        months = np.arange(1, int(horizon_m)+1)
        flow = (months==cure_m)*cure_rec*EAD_def*weights[0] + \
               (months==restr_m)*restr_rec*EAD_def*weights[1] + \
               (months==liq_m)*liq_rec*EAD_def*weights[2]
        cum = np.cumsum(flow)
        st.plotly_chart(px.line(x=months, y=cum, labels={"x":"Month","y":"Cumulative cash"},
                                title="Cumulative Recoveries Over Time"), use_container_width=True)

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Recovery NPV (defaults)", f"{rec_npv:,.0f}")
    kpi2.metric("Implied LGD (defaults)", f"{lgd_implied:.2%}")

    st.markdown(
        "> **Interpretation**: Adjust the mix and timing to compare strategies. "
        "Higher, earlier cash‑flows **lower LGD** and improve expected loss; they also matter for IRB **LGD‑in‑default** and provisioning under IFRS 9."
    )

    # ---------- Explanations ----------
    with st.expander("What this dashboard shows (plain English)"):
        st.markdown("""
- **NPE identification**: We use your **DPD/status flags** (90+ DPD is the common backstop) or a PD threshold to define defaults/NPE.  
- **Pre‑default forecasting**: A **balanced logistic regression** produces a 12‑month **PD** and highlights **top drivers** you can turn into early‑warning indicators.  
- **PD migrations**: The **transitions heatmap** estimates how accounts migrate across **PD bands**, signalling inflows to higher risk.  
- **Collections / Recovery**: The simulator compares **cure, restructure, liquidation** mixes, computes **NPV of recoveries**, and the **implied LGD** on the defaulted cohort.  
- **Capital view (illustrative)**: We include a **simplified** IRB **K** and **RWA** calculation for teaching; production models should use the full ASRF formula with supervisory correlation, maturity, and downturn LGD.
        """)
    # st.info("Add PD transitions, early-warning, and collections recovery paths here.")
elif selected_opt.startswith("Strategic recommendations"):
    st.header("Securitization & Private Credit Synergies")
    # =========================================================
    # STRATEGIC SECURITIZATION & PRIVATE CREDIT SYNERGIES
    # Tranche-level capital relief + investor return overlays
    # =========================================================

    # ---------- Data loader ----------
    @st.cache_data
    def load_pool(path="Basel_Combined_Datasets.xlsx"):
        import pandas as pd
        try:
            df = pd.read_excel(path, sheet_name="Merged_RWA_OBS", engine="openpyxl")
        except Exception:
            try:
                df = pd.read_excel(path, engine="openpyxl")
            except Exception as e:
                st.error(f"Upload **Basel_Combined_Datasets.xlsx** or adjust the path. Details: {e}")
                return pd.DataFrame()
        df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

        # Numeric clean-up
        for c in ["PD", "LGD", "EAD", "RWA"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Fallbacks
        if "EAD" not in df.columns or df["EAD"].isna().all():
            if "Asset_Value" in df.columns:
                df["EAD"] = pd.to_numeric(df["Asset_Value"], errors="coerce")
            else:
                df["EAD"] = df.get("RWA", 0.0)
        if "LGD" not in df.columns:
            df["LGD"] = 0.45  # neutral default, editable via sidebar later if needed
        if "PD" not in df.columns:
            df["PD"] = 0.02  # neutral default PD

        if "LoanType" not in df.columns:
            df["LoanType"] = df.get("Product_Type", "Loan")

        # Base RW proxy if missing (RWA/EAD)
        if "Base_RW" not in df.columns:
            df["Base_RW"] = (df.get("RWA", 0.0) / df["EAD"]).replace([np.inf, -np.inf], np.nan)
            # cap & fill
            df["Base_RW"] = df["Base_RW"].fillna(df["Base_RW"].median()).clip(0.0, 2.0)

        return df

    df = load_pool()
    if df.empty:
        st.stop()

    # ---------- Sidebar filters ----------
    st.sidebar.subheader("Filters")
    products = sorted(df["LoanType"].dropna().astype(str).unique())
    sel_products = st.sidebar.multiselect("Prod", products, default=products[:10] if len(products)>10 else products)

    f = df.copy()
    if sel_products:
        f = f[f["LoanType"].isin(sel_products)]

    # PD filter (if present)
    if "PD" in f.columns:
        pd_min, pd_max = float(f["PD"].min()), float(f["PD"].max())
        pd_rng = st.sidebar.slider("PD", 0.0, 1.0,
                                   value=(round(max(0.0, pd_min),3), round(min(1.0, pd_max),3)),
                                   step=0.001)
        f = f[(f["PD"].fillna(0.0)>=pd_rng[0]) & (f["PD"].fillna(0.0)<=pd_rng[1])]

    # ---------- Portfolio snapshot ----------
    EAD_pool = float(f["EAD"].sum())
    # Base RWA from your file if present; else EAD * Base_RW
    if "RWA" in f.columns and f["RWA"].notna().any():
        RWA_base = float(f["RWA"].sum())
    else:
        RWA_base = float(np.sum(f["EAD"] * f["Base_RW"]))

    # EL% (pool) ≈ weighted PD×LGD (simple proxy); can be swapped with internal EL
    w = f["EAD"] / max(EAD_pool, 1e-12)
    EL_pool_pct = float((w * f["PD"].clip(0,1) * f["LGD"].clip(0,1)).sum())  # (0–1)
    EL_amt = EL_pool_pct * EAD_pool

    # ---------- Structure dials ----------
    st.sidebar.subheader("Structure (tranche shares)")
    eq_pct = st.sidebar.slider("Equity %", 0.0, 0.20, 0.05, 0.01)
    mz_pct = st.sidebar.slider("Mezz %",  0.0, 0.50, 0.15, 0.01)
    sr_pct = 1.0 - (eq_pct + mz_pct)
    if sr_pct < 0:
        st.sidebar.error("Equity% + Mezz% must be ≤ 100%")
        st.stop()

    st.sidebar.subheader("Bank retention of each tranche")
    retain_sr = st.sidebar.slider("Retain Senior", 0.0, 1.0, 1.00, 0.05)
    retain_mz = st.sidebar.slider("Retain Mezz",  0.0, 1.0, 0.05, 0.05)  # keep 5% unless selling
    retain_eq = st.sidebar.slider("Retain Equity",0.0, 1.0, 0.05, 0.05)

    # Notionals
    N_eq  = eq_pct * EAD_pool
    N_mz  = mz_pct * EAD_pool
    N_sr  = sr_pct * EAD_pool
    # Retained / Sold
    N_sr_ret, N_sr_sold = retain_sr*N_sr, (1-retain_sr)*N_sr
    N_mz_ret, N_mz_sold = retain_mz*N_mz, (1-retain_mz)*N_mz
    N_eq_ret, N_eq_sold = retain_eq*N_eq, (1-retain_eq)*N_eq

    # ---------- Risk weights (illustrative, configurable) ----------
    st.sidebar.subheader("Risk Weights (illustrative)")
    rw_sr = st.sidebar.number_input("Senior RW %", 0.0, 1250.0, 15.0, 1.0)   # eg 15%
    rw_mz = st.sidebar.number_input("Mezz RW %",   0.0, 1250.0, 350.0, 10.0) # eg 350%
    rw_eq = st.sidebar.number_input("Equity RW %", 0.0, 1250.0, 1250.0, 10.0)# eg 1250%

    # Convert to decimals
    rw_sr_d = rw_sr/100.0; rw_mz_d = rw_mz/100.0; rw_eq_d = rw_eq/100.0

    # ---------- Private credit overlay (for SOLD tranches) ----------
    st.sidebar.subheader("Private Credit Overlay")
    coup_mz = st.sidebar.number_input("Sold Mezz coupon % (annual)", 0.0, 50.0, 10.0, 0.5)/100.0
    price_mz= st.sidebar.number_input("Mezz price (% of par)", 50.0, 105.0, 99.0, 0.5)/100.0
    fee_bp  = st.sidebar.number_input("Servicing/struct fee (bps)", 0.0, 500.0, 50.0, 5.0)/10000.0
    # Loss allocation: EL waterfall (Equity → Mezz → Senior)
    loss = EL_amt
    loss_eq = min(loss, N_eq); loss -= loss_eq
    loss_mz = min(loss, N_mz); loss -= loss_mz
    loss_sr = min(loss, N_sr); loss -= loss_sr
    # Expected loss rates per tranche
    EL_eq_pct = (loss_eq / N_eq) if N_eq>0 else 0.0
    EL_mz_pct = (loss_mz / N_mz) if N_mz>0 else 0.0
    EL_sr_pct = (loss_sr / N_sr) if N_sr>0 else 0.0

    # Investor expected yield (1-year simple) on SOLD mezz (transparent proxy)
    # Net = coupon – expected loss – fee; Cash invested = price
    inv_mezz_yield = None
    if N_mz_sold > 0:
        exp_loss_y = EL_mz_pct
        net_yield = (coup_mz - exp_loss_y - fee_bp) / max(price_mz, 1e-6)
        inv_mezz_yield = net_yield

    # ---------- Capital: BEFORE vs AFTER securitization ----------
    # BEFORE: bank holds whole pool capital = RWA_base
    # AFTER: remove SOLD tranches; risk-weight only RETAINED tranches with chosen RWs
    RWA_after = (N_sr_ret*rw_sr_d + N_mz_ret*rw_mz_d + N_eq_ret*rw_eq_d)
    cap_relief = RWA_base - RWA_after
    cap_relief_pct = 100.0 * cap_relief / max(RWA_base, 1e-9)

    # ---------- KPIs ----------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pool EAD", f"{EAD_pool:,.0f}")
    k2.metric("Base RWA", f"{RWA_base:,.0f}")
    k3.metric("RWA After", f"{RWA_after:,.0f}")
    k4.metric("Capital Relief (Δ)", f"{cap_relief:,.0f}")
    k5.metric("Relief %", f"{cap_relief_pct:.1f}%")

    st.caption(
        "• **Base RWA** comes from your data (or `EAD×Base_RW` if RWA missing). "
        "• **RWA After** risk‑weights only **retained** tranches; **sold** notional leaves the bank’s balance sheet. "
        "• Numbers here are **illustrative**, not regulatory SSFA/SEC‑ERBA."
    )

    # =========================================================
    # ROW 1 — Capital Waterfall & Tranche Holding Mix
    # =========================================================
    c1, c2 = st.columns(2)
    with c1:
        import plotly.graph_objects as go
        st.subheader("Capital Relief — Waterfall")
        fig_w = go.Figure(go.Waterfall(
            measure=["absolute","relative"],
            x=["Base RWA","Securitization effect"],
            y=[RWA_base, (RWA_after - RWA_base)],
            text=[f"{RWA_base:,.0f}", f"{(RWA_after - RWA_base):,.0f}"]
        ))
        fig_w.update_layout(title="Before → After", showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)
        st.markdown("> **Why:** Quickly shows **capital released** by selling/down‑weighting tranches.")

    with c2:
        st.subheader("Who Holds What (Notional)")
        df_hold = pd.DataFrame({
            "Tranche":["Senior","Mezz","Equity","Senior","Mezz","Equity"],
            "Holder":["Bank","Bank","Bank","Private Credit","Private Credit","Private Credit"],
            "Notional":[N_sr_ret, N_mz_ret, N_eq_ret, N_sr_sold, N_mz_sold, N_eq_sold]
        })
        st.plotly_chart(px.bar(df_hold, x="Tranche", y="Notional", color="Holder", barmode="stack",
                               title="Holdings by Tranche"),
                        use_container_width=True)
        st.markdown("> **Mix:** You can size **mezz for sale** to private credit while keeping a minimum **risk‑retention** slice.")

    # =========================================================
    # ROW 2 — Expected Loss Allocation & Investor Overlay
    # =========================================================
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Expected Loss (EL) – Waterfall")
        df_el = pd.DataFrame({
            "Tranche":["Equity","Mezz","Senior"],
            "EL_amount":[loss_eq, loss_mz, loss_sr],
            "EL_%":[EL_eq_pct, EL_mz_pct, EL_sr_pct]
        })
        st.plotly_chart(px.bar(df_el, x="Tranche", y="EL_amount", color="Tranche",
                               title=f"Pool EL = {EL_pool_pct:.2%} of EAD"),
                        use_container_width=True)
        st.markdown("> **Waterfall:** Losses first hit **Equity**, then **Mezz**, finally **Senior**.")

    with c4:
        st.subheader("Private‑Credit Mezz — Expected Net Yield (1‑yr proxy)")
        if inv_mezz_yield is not None:
            y = inv_mezz_yield*100.0
            st.metric("Expected Net Yield", f"{y:.1f}%")
            st.markdown(
                f"""
                **Calculation:** Net ≈ coupon − expected loss − fees, divided by price.  
                - Coupon: **{coup_mz*100:.1f}%**  
                - Expected loss: **{EL_mz_pct*100:.1f}%**  
                - Fees: **{fee_bp*10000:.0f} bps**  
                - Price: **{price_mz*100:.1f}% of par**
                """
            )
        else:
            st.info("Sell part of **Mezz** to see investor‑yield overlay.")

    # =========================================================
    # ROW 3 — Download & Recommendations
    # =========================================================
    with st.expander("Download scenario results (CSV)"):
        out = pd.DataFrame({
            "Metric":["Pool_EAD","Base_RWA","After_RWA","Capital_Relief","Relief_%","Equity_%","Mezz_%","Senior_%",
                      "Retain_Sr","Retain_Mz","Retain_Eq","EL_pool_%","EL_Equity_%","EL_Mezz_%","EL_Senior_%",
                      "Mezz_Coupon_%","Mezz_Price_%","Mezz_ExpNetYield_%"],
            "Value":[EAD_pool, RWA_base, RWA_after, cap_relief, cap_relief_pct, eq_pct*100, mz_pct*100, sr_pct*100,
                     retain_sr*100, retain_mz*100, retain_eq*100, EL_pool_pct*100, EL_eq_pct*100, EL_mz_pct*100,
                     EL_sr_pct*100, coup_mz*100 if inv_mezz_yield else np.nan, price_mz*100 if inv_mezz_yield else np.nan,
                     inv_mezz_yield*100 if inv_mezz_yield else np.nan]
        })
        st.download_button("Download CSV", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="securitization_scenario.csv", mime="text/csv")

    with st.expander("How to read this (plain‑English)"):
        st.markdown(
            """
- **Goal**: free up capital by **selling mezz/equity** (to private credit) and **retaining senior** at a low RW.  
- **Inputs**: set **tranche %**, **retention %**, and **RW assumptions** (illustrative).  
- **Capital relief**: *After RWA* re‑weights only **retained** tranches; *sold* tranches drop out.  
- **Investor overlay**: for the **sold mezz**, we show a **simple net yield** = coupon − EL − fees, / price.  
- **Caution**: This is **not** a regulatory SSFA/SEC‑ERBA calculator; use for **structuring insight** and **stakeholder dialogue**.
            """
        )

    # st.info("Add tranche-level capital relief and investor return overlays here.")
elif selected_opt.startswith("Analysis of off-balance"):
    st.header("Off-Balance Sheet & CCFs")
    # =========================================================
    # OFF-BALANCE SHEET (OBS) • CCFs & SA-CCR (simplified view)
    # Full, fixed module — robust fallbacks, synthetic fill, clean UI
    # =========================================================
    import os
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go


    # ---------- helpers ----------
    def num_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
        """Return df[col] coerced to numeric if present, else a default Series aligned to df.index."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    # ---------- data loader (CSV → Excel fallbacks) ----------
    @st.cache_data
    def load_obs() -> pd.DataFrame:
        df = None
        # Try CSV (as per your note)
        if os.path.exists("Merged_RWA_OBS_Dataset (2).csv"):
            try:
                df = pd.read_csv("Merged_RWA_OBS_Dataset (2).csv")
            except Exception:
                df = None
        # Else, try Excel
        if df is None and os.path.exists("Basel_Combined_Datasets.xlsx"):
            for sheet in ["Merged_RWA_OBS", "Basel_IV_Model","Synthetic_OBS_Exposures",0]:
                try:
                    tmp = pd.read_excel("Basel_Combined_Datasets.xlsx", sheet_name=sheet, engine="openpyxl")
                    df = tmp; break
                except Exception:
                    continue

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize columns
        df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

        # Numeric casts (if present)
        for c in [
            "EAD","RWA","RiskWeight","PD","LGD","CCF","Undrawn_Amount","Notional",
            "Exposure_At_Default_OBS","FICO_Score","Credit_Bureau_Score","Annual_Salary",
            "Time_on_Books","Asset_Value","MtM","Collateral","AddOn_Factor"
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Date columns
        date_col = None
        for d in ["AsOfDate","Report_Date","Date","Quarter"]:
            if d in df.columns:
                df[d] = pd.to_datetime(df[d], errors="coerce")
                if df[d].notna().any() and date_col is None:
                    date_col = d

        # OBS notional base (Series-safe fallbacks)
        s_ead_obs  = num_series(df, "Exposure_At_Default_OBS")
        s_undrawn  = num_series(df, "Undrawn_Amount")
        s_notional = num_series(df, "Notional")
        s_ead      = num_series(df, "EAD", default=0.0)

        base_ead_obs = s_ead_obs
        base_ead_obs = base_ead_obs.where(base_ead_obs.notna(), s_undrawn)
        base_ead_obs = base_ead_obs.where(base_ead_obs.notna(), s_notional)
        base_ead_obs = base_ead_obs.where(base_ead_obs.notna(), s_ead)
        df["EAD_OBS"] = base_ead_obs.abs()

        # CCF (numeric parse or from CCF_Source)
        df["CCF"] = num_series(df, "CCF", default=np.nan)
        if df["CCF"].isna().all() and "CCF_Source" in df.columns:
            parsed = df["CCF_Source"].astype(str).str.extract(r"(\d+)")
            df["CCF"] = pd.to_numeric(parsed[0], errors="coerce")/100.0
        df["CCF"] = df["CCF"].fillna(0.50).clip(0.0, 1.0)

        # EAD after CCF
        df["EAD_Post_CCF"] = df["EAD_OBS"] * df["CCF"]

        # Risk weight (Series-safe) default 100%
        df["RiskWeight"] = num_series(df, "RiskWeight", default=np.nan).fillna(1.00)

        # RWA keep/compute
        rwa_series = num_series(df, "RWA", default=np.nan)
        df["RWA"] = np.where(rwa_series.notna(), rwa_series, df["EAD_Post_CCF"]*df["RiskWeight"])

        # Friendly labels
        if "Product_Type" not in df.columns:
            df["Product_Type"] = df.get("LoanType","OBS")
        if "OBS_Category" not in df.columns:
            df["OBS_Category"] = df.get("Exposure_Type","Commitments/Contingents")
        if "ZIP" not in df.columns and "Zip" in df.columns:
            df["ZIP"] = df["Zip"]
        if "Business_Type" not in df.columns:
            df["Business_Type"] = df.get("Segment","Unknown")

        df["_Date"] = df[date_col] if date_col else pd.NaT
        return df

    df = load_obs()
    if df.empty:
        st.warning("Upload **Merged_RWA_OBS_Dataset (2).csv** or **Basel_Combined_Datasets.xlsx**.")
        st.stop()

    # ---------- sidebar filters ----------
    st.sidebar.subheader("Filters")
    prod_vals = sorted(df["Product_Type"].dropna().astype(str).unique())
    sel_prod = st.sidebar.multiselect("Prod", prod_vals, default=prod_vals[:10] if len(prod_vals)>10 else prod_vals)

    basel_vals = ["All"] + sorted(pd.Series(df.get("CCF_Source","Unknown")).dropna().astype(str).unique())
    sel_basel = st.sidebar.selectbox("Basel ref in CCF_Source", basel_vals, index=0)

    # Date range
    if df["_Date"].notna().any():
        mind, maxd = df["_Date"].min(), df["_Date"].max()
        d1, d2 = st.sidebar.date_input("Date range", (mind.date(), maxd.date()))
    else:
        d1 = d2 = None

    # Apply filters
    f = df.copy()
    if sel_prod: f = f[f["Product_Type"].isin(sel_prod)]
    if sel_basel != "All": f = f[f.get("CCF_Source","").astype(str).str.contains(sel_basel, case=False, na=False)]
    if d1 and d2 and f["_Date"].notna().any():
        f = f[(f["_Date"]>=pd.to_datetime(d1)) & (f["_Date"]<=pd.to_datetime(d2))]

    # ---------- Synthetic Imputation Panel ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Synthetic fill for unstructured fields")
    enable_synth = st.sidebar.checkbox("Fill missing ZIP with synthetic codes (demo)", value=True)
    seed_val = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    enable_synth_salary = st.sidebar.checkbox("Also fill missing Salary (demo)", value=False)
    enable_synth_tob    = st.sidebar.checkbox("Also fill missing Time on Books (demo)", value=False)

    def synthetic_zip_series(df_in: pd.DataFrame, seed: int) -> pd.Series:
        rs = np.random.default_rng(seed)
        if "ZIP" not in df_in.columns:
            zip_col = pd.Series(index=df_in.index, dtype="object")
        else:
            zip_col = df_in["ZIP"]
        is_miss = zip_col.isna() | (zip_col.astype(str).str.strip().eq("")) | (zip_col.astype(str).str.lower().eq("unknown"))
        n_missing = int(is_miss.sum())
        if n_missing == 0:
            return zip_col
        synth_vals = rs.integers(1000, 99000, size=n_missing)
        synth_str = pd.Series([f"{v:05d}" for v in synth_vals], index=zip_col[is_miss].index)
        out = zip_col.copy()
        out.loc[is_miss] = synth_str
        return out

    def synthetic_numeric_series(base: pd.Series, seed: int, low: float, high: float, round_to: int | None = None) -> pd.Series:
        rs = np.random.default_rng(seed)
        ser = pd.to_numeric(base, errors="coerce")
        is_miss = ser.isna()
        if is_miss.any():
            draws = rs.uniform(low, high, size=int(is_miss.sum()))
            if round_to is not None:
                draws = np.round(draws / round_to) * round_to
            ser = ser.copy()
            ser.loc[is_miss] = draws
        return ser

    # Apply synthetic fills (only where missing)
    f["_ZIP_Synthetic"] = 0
    if enable_synth:
        if "ZIP" not in f.columns:
            f["ZIP"] = np.nan
        before = f["ZIP"].isna() | (f["ZIP"].astype(str).str.strip().eq("")) | (f["ZIP"].astype(str).str.lower().eq("unknown"))
        f["ZIP"] = synthetic_zip_series(f, seed_val)
        after  = f["ZIP"].isna() | (f["ZIP"].astype(str).str.strip().eq("")) | (f["ZIP"].astype(str).str.lower().eq("unknown"))
        f.loc[before & (~after), "_ZIP_Synthetic"] = 1  # mark newly filled rows

    if enable_synth_salary:
        if "Annual_Salary" not in f.columns: f["Annual_Salary"] = np.nan
        f["Annual_Salary"] = synthetic_numeric_series(f["Annual_Salary"], seed_val+7, low=18_000, high=220_000, round_to=500)

    if enable_synth_tob:
        if "Time_on_Books" not in f.columns: f["Time_on_Books"] = np.nan
        f["Time_on_Books"] = synthetic_numeric_series(f["Time_on_Books"], seed_val+11, low=0.5, high=120.0, round_to=1)

    # ---------- KPI tiles ----------
    total_obs = float(f["EAD_OBS"].sum())
    total_ead = float(f["EAD_Post_CCF"].sum())
    total_rwa = float(f["RWA"].sum())
    n = len(f)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("# OBS items", f"{n:,}")
    k2.metric("Notional (OBS)", f"{total_obs:,.0f}")
    k3.metric("EAD (after CCF)", f"{total_ead:,.0f}")
    k4.metric("RWA (OBS)", f"{total_rwa:,.0f}")
    st.caption("**EAD after CCF** = Notional × **CCF**. **RWA** = EAD × **Risk Weight** (dataset or baseline).")

    # ---------- chart switcher ----------
    st.subheader("Insights")
    view = st.selectbox(
        "Select a view",
        [
            "CCF Distribution (counts)",
            "OBS Breakdown by Product",
            "RWA Contribution by OBS Category",
            "FICO vs RWA",
            "Credit Bureau Score vs PD",
            "Credit Bureau Score vs RWA",
            "RWA by ZIP",
            "RWA by Salary Band",
            "RWA by Time on Books (trend)",
            "RWA by Business Type",
            "RWA vs Asset Value",
            "OBS Trends Over Time",
            "SA‑CCR (Derivatives) – simplified"
        ],
        index=0
    )

    def explain(txt: str): st.markdown(f"> {txt}")

    # ---------- renderers ----------
    if view == "CCF Distribution (counts)":
        bins = [0,0.01,0.21,0.51,0.76,1.01]  # 0 / 20 / 50 / 75 / 100
        labels = ["0%","20%","50%","75%","100%"]
        b = pd.cut(f["CCF"].fillna(0.0), bins=bins, labels=labels, right=False)
        st.plotly_chart(
            px.histogram(x=b, title="Credit Conversion Factor (CCF) Buckets", labels={"x":"CCF bucket","y":"Count"}),
            use_container_width=True
        )
        explain("Shows where OBS items sit by **CCF bucket** — shifting to **lower CCFs** reduces **EAD** and **RWA**.")

    elif view == "OBS Breakdown by Product":
        g = (f.groupby("Product_Type")[["EAD_Post_CCF","RWA"]]
               .sum().reset_index().sort_values("RWA", ascending=False))
        fig = px.bar(g.melt(id_vars="Product_Type", var_name="Metric", value_name="Amount"),
                     x="Product_Type", y="Amount", color="Metric", barmode="group",
                     title="OBS EAD & RWA by Product")
        st.plotly_chart(fig, use_container_width=True)
        explain("Highlights **which products** drive OBS capital after **CCFs**.")

    elif view == "RWA Contribution by OBS Category":
        g = (f.groupby("OBS_Category")[["RWA"]].sum().reset_index()
               .sort_values("RWA", ascending=False))
        st.plotly_chart(
            px.bar(g, x="OBS_Category", y="RWA", color="OBS_Category", title="RWA by OBS Category"),
            use_container_width=True
        )
        explain("Which **OBS categories** (e.g., undrawn lines, guarantees, trade finance) consume the most capital.")

    elif view == "FICO vs RWA" and {"FICO_Score","RWA"}.issubset(f.columns):
        st.plotly_chart(
            px.scatter(f, x="FICO_Score", y="RWA", color="Product_Type",
                       hover_data=["OBS_Category","CCF","EAD_Post_CCF"],
                       trendline="lowess", title="FICO Score vs RWA"),
            use_container_width=True
        )
        explain("Lower FICO generally ↗ **RWA**; check also **CCF** and **RiskWeight** assignments.")

    elif view == "Credit Bureau Score vs PD" and {"Credit_Bureau_Score","PD"}.issubset(f.columns):
        st.plotly_chart(
            px.scatter(f, x="Credit_Bureau_Score", y="PD", color="Product_Type",
                       hover_data=["OBS_Category","CCF"], trendline="lowess",
                       title="Credit Bureau Score vs PD"),
            use_container_width=True
        )
        explain("Quick validation that **score** ranks **PD** as expected across products and OBS types.")

    elif view == "Credit Bureau Score vs RWA" and {"Credit_Bureau_Score","RWA"}.issubset(f.columns):
        st.plotly_chart(
            px.scatter(f, x="Credit_Bureau_Score", y="RWA", color="Product_Type",
                       hover_data=["OBS_Category","CCF","EAD_Post_CCF"], trendline="lowess",
                       title="Credit Bureau Score vs RWA"),
            use_container_width=True
        )
        explain("Connects external score to **capital** post‑CCF. Outliers → RW/CCF mapping checks.")

    elif view == "RWA by ZIP" and "ZIP" in f.columns:
        z = f.copy()
        # drop any residual Unknown/blank
        mask_known = ~(z["ZIP"].astype(str).str.strip().eq("")) & ~(z["ZIP"].astype(str).str.lower().eq("unknown")) & z["ZIP"].notna()
        z = z[mask_known]
        g = (z.groupby("ZIP")[["RWA"]]
               .sum().reset_index().sort_values("RWA", ascending=False).head(30))
        fig = px.bar(g, x="ZIP", y="RWA", color="ZIP", title="Top ZIPs by RWA (Structured Only)")
        st.plotly_chart(fig, use_container_width=True)
        if "_ZIP_Synthetic" in f.columns:
            synth_ratio = 100.0 * float(f["_ZIP_Synthetic"].sum()) / max(len(f),1)
            st.caption(f"**Synthetic ZIP share in current filter:** {synth_ratio:.1f}% of rows")
        explain("Geographic **hot‑spots** for OBS capital. Synthetic ZIPs (if enabled) prevent 'Unknown' from cluttering visuals.")

    elif view == "RWA by Salary Band" and "Annual_Salary" in f.columns:
        bands = pd.cut(f["Annual_Salary"], bins=[0,25_000,50_000,75_000,100_000,200_000,1e12],
                       labels=["≤25k","25–50k","50–75k","75–100k","100–200k",">200k"])
        g = (f.groupby(bands, observed=False)[["RWA"]].sum().reset_index())
        st.plotly_chart(
            px.bar(g, x="Annual_Salary", y="RWA", color="Annual_Salary", title="RWA by Salary Band"),
            use_container_width=True
        )
        explain("Salary bands proxy **capacity to pay**; supports line‑management strategy.")

    elif view == "RWA by Time on Books (trend)" and "Time_on_Books" in f.columns:
        st.plotly_chart(
            px.scatter(f, x="Time_on_Books", y="RWA", color="Product_Type",
                       trendline="lowess", title="RWA by Time on Books"),
            use_container_width=True
        )
        explain("Seasoning: compare **new vs seasoned** exposures for capital spikes.")

    elif view == "RWA by Business Type":
        g = (f.groupby("Business_Type")[["RWA"]].sum().reset_index()
               .sort_values("RWA", ascending=False))
        st.plotly_chart(
            px.bar(g, x="Business_Type", y="RWA", color="Business_Type", title="RWA by Business Type"),
            use_container_width=True
        )
        explain("Which **segments** (Retail/SME/Corporate) drive OBS capital.")

    elif view == "RWA vs Asset Value" and "Asset_Value" in f.columns:
        st.plotly_chart(
            px.scatter(f, x="Asset_Value", y="RWA", color="Product_Type",
                       hover_data=["OBS_Category","CCF","RiskWeight"], trendline="lowess",
                       title="RWA vs Asset Value"),
            use_container_width=True
        )
        explain("Scale effect: larger assets/limits ↗ capital even after **CCF** scaling.")

    elif view == "OBS Trends Over Time" and f["_Date"].notna().any():
        g = (f.groupby("_Date")[["EAD_OBS","EAD_Post_CCF","RWA"]]
               .sum().reset_index().sort_values("_Date"))
        st.plotly_chart(
            px.line(g, x="_Date", y=["EAD_OBS","EAD_Post_CCF","RWA"], title="OBS Exposure & RWA Trends Over Time"),
            use_container_width=True
        )
        explain("Are **OBS exposures** growing, and are optimization efforts **reducing RWA** over time?")

    elif view == "SA‑CCR (Derivatives) – simplified":
        st.markdown("**SA‑CCR (simplified)** — transparent demo (for derivatives rows only).")
        # Identify derivatives rows
        dmask = f.get("Is_Derivative", pd.Series(False, index=f.index))
        if isinstance(dmask, (pd.Series, np.ndarray)):
            df_der = f[dmask==1].copy()
        else:
            df_der = f.head(0).copy()
        if df_der.empty and "AssetClass" in f.columns:
            df_der = f[f["AssetClass"].notna()].copy()

        if df_der.empty:
            st.info("No derivative rows found (need fields like **Is_Derivative**, **MtM**, **Collateral**, **AddOn_Factor**/**AssetClass**).")
        else:
            colA, colB, colC = st.columns(3)
            alpha = colA.number_input("α (multiplier)", 1.0, 2.0, 1.4, 0.05)
            mult  = colB.number_input("PFE Multiplier", 0.0, 1.0, 1.0, 0.05)

            default_addons = {"IR":0.015,"FX":0.04,"Credit":0.10,"Equity":0.06,"Commodity":0.10}
            def add_on_row(r):
                if not np.isnan(r.get("AddOn_Factor", np.nan)):
                    return r["AddOn_Factor"]
                ac = str(r.get("AssetClass","IR"))
                return default_addons.get(ac, 0.06)

            df_der["AddOn_used"] = df_der.apply(add_on_row, axis=1)
            V = num_series(df_der, "MtM", default=0.0).values
            C = num_series(df_der, "Collateral", default=0.0).values
            RC = np.maximum(V - C, 0.0)
            Notional = num_series(df_der, "Notional", default=0.0).values
            PFE = mult * df_der["AddOn_used"].values * Notional
            EAD_saccr = alpha * (RC + PFE)

            df_der["_EAD_SACCR"] = EAD_saccr
            df_der["_RWA_SACCR"] = EAD_saccr * num_series(df_der, "RiskWeight", default=1.0)

            st.plotly_chart(
                px.scatter(
                    df_der,
                    x="_EAD_SACCR",
                    y=df_der.get("EAD_Post_CCF", df_der.get("EAD_OBS", pd.Series(0,index=df_der.index))),
                    color=df_der.get("AssetClass","AssetClass"),
                    hover_data=["Product_Type","OBS_Category","AddOn_used"],
                    title="SA‑CCR EAD (demo) vs Reported/CCF EAD"
                ),
                use_container_width=True
            )
        explain("**EAD_SA‑CCR ≈ α × (RC + PFE)** — didactic benchmark; use full SA‑CCR for production.")

    else:
        st.info("Columns required for this view are missing in the current dataset/filters.")

    # ---------- Structuring strategies comparison (mini panel) ----------
    st.markdown("---")
    st.subheader("Impact of Structuring Strategies on RWA (illustrative)")
    st.caption("Compares **average RWA per exposure** under simple labels. Replace with your true structuring flags.")

    s_cols = []
    if "CollateralFlag" in f.columns: s_cols.append("CollateralFlag")
    if "GuaranteeFlag"  in f.columns: s_cols.append("GuaranteeFlag")
    if "NettingFlag"    in f.columns: s_cols.append("NettingFlag")

    if s_cols:
        f["Structure"] = np.select(
            [
                (f.get("CollateralFlag",0)==1),
                (f.get("NettingFlag",0)==1),
                (f.get("GuaranteeFlag",0)==1)
            ],
            ["Collateralized","Netting","Guarantee‑Enhanced"],
            default="Unstructured"
        )
    else:
        f["Structure"] = np.where(
            f["OBS_Category"].astype(str).str.contains("guarantee", case=False), "Guarantee‑Enhanced",
            np.where(
                f["OBS_Category"].astype(str).str.contains("trade|cash|secur|collat", case=False), "Collateralized",
                "Unstructured"
            )
        )

    g = (f.groupby("Structure")[["RWA"]].mean().reset_index().sort_values("RWA", ascending=False))
    st.plotly_chart(
        px.bar(g, x="Structure", y="RWA", color="Structure", title="Avg RWA per exposure — by strategy"),
        use_container_width=True
    )

    # ---------- Download filtered subset ----------
    with st.expander("Download filtered OBS subset"):
        cols_out = [c for c in [
            "Product_Type","OBS_Category","CCF_Source","CCF","EAD_OBS","EAD_Post_CCF",
            "RiskWeight","RWA","FICO_Score","Credit_Bureau_Score","Annual_Salary",
            "Time_on_Books","ZIP","_Date","_ZIP_Synthetic"
        ] if c in f.columns]
        st.download_button(
            "Download CSV",
            data=f[cols_out].to_csv(index=False).encode("utf-8"),
            file_name="obs_filtered_subset.csv",
            mime="text/csv"
        )
    # st.info("Add exposure category breakdowns, CCF multipliers, and SA-CCR views here.")
elif selected_opt.startswith("Examine impact"):
    st.header("Capital Buffer Impacts")
    st.markdown("""
    ### Basel IV Capital Buffer Analysis
    This section provides:
    - **Current buffer vs Basel minimum**
    - **Stress scenarios** (economic downturn, credit shocks)
    - **Buffer trend tracking**
    Charts below are arranged for clarity and stakeholder engagement.
    """)

    # Sample Data
    np.random.seed(42)
    months = pd.date_range("2024-01", periods=12, freq="M")
    buffer = np.random.uniform(10, 15, size=12)
    stress_buffer = buffer - np.random.uniform(1, 3, size=12)
    rwa = np.random.uniform(1000, 1500, size=12)

    df = pd.DataFrame({
        "Month": months,
        "Buffer": buffer,
        "StressBuffer": stress_buffer,
        "RWA": rwa
    })

    # KPIs at the top
    st.markdown("#### Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Buffer", f"{buffer[-1]:.2f}%")
    col2.metric("Stress Buffer", f"{stress_buffer[-1]:.2f}%")
    col3.metric("Basel Minimum", "8.0%")

    st.markdown("---")

    # Row 1: Capital Buffer Overview
    st.subheader("Capital Buffer Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Current vs Stress Buffer Comparison**")
        fig1 = px.bar(df, x="Month", y=["Buffer", "StressBuffer"],
                      barmode="group", title="Capital Buffer Comparison",
                      labels={"value": "Capital Buffer (%)", "variable": "Scenario"})
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        st.markdown("**RWA Trend**")
        fig2 = px.line(df, x="Month", y="RWA", title="Risk-Weighted Assets Trend",
                       labels={"RWA": "RWA (in million)"})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Row 2: Stress Scenario Impact
    st.subheader("Stress Scenario Impact")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Impact of Stress Scenarios on Buffer**")
        fig3 = px.line(df, x="Month", y=["Buffer", "StressBuffer"],
                       title="Buffer Under Stress vs Normal",
                       labels={"value": "Capital Buffer (%)"})
        st.plotly_chart(fig3, use_container_width=True)
    with col_d:
        st.markdown("**Buffer Trend Over Time**")
        fig4 = px.area(df, x="Month", y="Buffer", title="Capital Buffer Trend",
                       labels={"Buffer": "Capital Buffer (%)"})
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    ---
    **Interpretation for Stakeholders:**
    - **Buffer**: Current capital buffer maintained by the bank.
    - **StressBuffer**: Buffer under stress scenarios (economic downturn, credit shocks).
    - Basel IV requires maintaining adequate buffers to absorb losses during stress.
    - **RWA Trend**: Higher RWA increases capital requirements.
    """)
    # st.info("Add stress scenarios and buffer tracking under Basel IV reforms here.")
elif selected_opt.startswith("Portfolio-level"):
    st.header("Portfolio-Level Optimization")
    st.markdown("""
    ### Strategic Portfolio Optimization
    This section provides:
    - **Retail, SME, and Industry sector breakdown**
    - **Capital allocation efficiency**
    - **Cross-segment strategy simulation**
    Use sliders in the sidebar to simulate allocation changes and observe impact.
    """)

    # Mock Data for Portfolio
    np.random.seed(42)
    sectors = ["Retail", "SME", "Industry"]
    capital_req = [40, 35, 25]  # Basel Capital %
    ead_values = [500, 400, 300]  # Exposure at Default (in millions)

    # Sidebar sliders for strategy simulation
    st.sidebar.subheader("Adjust Capital Allocation Strategy")
    retail_alloc = st.sidebar.slider("Retail Allocation (%)", 0, 100, 40)
    sme_alloc = st.sidebar.slider("SME Allocation (%)", 0, 100, 35)
    industry_alloc = st.sidebar.slider("Industry Allocation (%)", 0, 100, 25)

    total_alloc = retail_alloc + sme_alloc + industry_alloc
    if total_alloc != 100:
        st.warning(f"⚠ Total allocation should equal 100%. Current: {total_alloc}%")

    st.markdown("---")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Retail Capital", f"{retail_alloc}%")
    col2.metric("SME Capital", f"{sme_alloc}%")
    col3.metric("Industry Capital", f"{industry_alloc}%")

    st.markdown("---")

    # Charts in rows and columns
    st.subheader("Portfolio Visualizations")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("**Capital Requirement by Sector**")
        fig1 = px.pie(names=sectors, values=capital_req, title="Basel Capital Requirement")
        st.plotly_chart(fig1, use_container_width=True)
    with row1_col2:
        st.markdown("**Exposure at Default (EAD)**")
        fig2 = px.bar(x=sectors, y=ead_values, title="EAD by Sector", labels={"x": "Sector", "y": "EAD (in millions)"})
        st.plotly_chart(fig2, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown("**Simulated Allocation Impact**")
        fig3 = px.bar(x=sectors, y=[retail_alloc, sme_alloc, industry_alloc],
                      title="Simulated Capital Allocation", labels={"x": "Sector", "y": "Allocation (%)"})
        st.plotly_chart(fig3, use_container_width=True)
    with row2_col2:
        st.markdown("**Capital Efficiency Analysis**")
        efficiency = [retail_alloc / ead_values[0], sme_alloc / ead_values[1], industry_alloc / ead_values[2]]
        fig4 = px.line(x=sectors, y=efficiency, title="Capital Efficiency by Sector",
                       labels={"x": "Sector", "y": "Efficiency Ratio"})
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    ---
    **Interpretation for Stakeholders:**
    - Basel IV requires efficient capital allocation across sectors.
    - Use sliders to simulate allocation changes and observe efficiency ratios.
    - Higher efficiency means better utilization of capital relative to exposure.
    """)
    # st.info("Add cross-segment strategy simulation and capital allocation efficiency here.")

st.markdown("---")
# st.markdown(
#     "Use the **left menu** to switch modules. The **Retail/LTV/CRM** module includes filters, visuals, IRB calculator, "
#     "and clear explanations for Basel IV analysis."
# )

# Optional references (appear under the app)
# with st.expander("References"):
#     st.markdown(
#         """
# - **Basel IRB formula & bands**: *Basel Committee on Banking Supervision — Explanatory Note on IRB Risk Weight Functions (July 2005)*  
#   https://www.bis.org/bcbs/irbriskweight.pdf  
# - **IRB ASRF form and discussion** (conditional PD, 0.999 quantile):  
#   https://quant.stackexchange.com/questions/34420/difference-between-the-basel-irb-and-the-vasicek-formula  
# - **Credit Risk Mitigation & Haircuts (Comprehensive Approach)** — Basel Framework (CRE22):  
#   https://www.bis.org/basel_framework/chapter/CRE/22.htm
#         """
#     )
