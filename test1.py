
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Basel IV Dashboard", layout="wide")

# Sidebar Menu
optimization_points = [
    "Retail Lending",
    "Credit Card CRR3",
    "SME Lending",
    "Currency Hedging",
    "RWA Automation",
    "NPE & Recovery",
    "Securitization",
    "Off-Balance Sheet",
    "Capital Buffer",
    "Portfolio Strategy"
]

st.sidebar.title("Basel IV Optimization Menu")
selected_opt = st.sidebar.radio("Select optimization to view:", optimization_points)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Basel Combined Dataset (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
else:
    st.warning("Please upload the Basel Combined Dataset to proceed.")
    df = pd.DataFrame()  # Empty placeholder

st.title("📊 Basel IV Credit Risk and Optimization Dashboard")
st.markdown("---")

# Mock data for demo if no file uploaded
if df.empty:
    np.random.seed(42)
    months = pd.date_range("2024-01", periods=12, freq="M")
    buffer = np.random.uniform(10, 15, size=12)
    stress_buffer = buffer - np.random.uniform(1, 3, size=12)
    rwa = np.random.uniform(1000, 1500, size=12)
    df_demo = pd.DataFrame({"Month": months, "Buffer": buffer, "StressBuffer": stress_buffer, "RWA": rwa})
else:
    df_demo = df

# Portfolio Strategy Section
if selected_opt == "Portfolio Strategy":
    st.header("Portfolio-Level Strategic Optimization")
    st.markdown("""
    ### Strategic Portfolio Optimization
    - Retail, SME, and Industry sector breakdown
    - Capital allocation efficiency
    - Cross-segment strategy simulation
    Use sliders in the sidebar to simulate allocation changes and observe impact.
    """)

    sectors = ["Retail", "SME", "Industry"]
    capital_req = [40, 35, 25]
    ead_values = [500, 400, 300]

    st.sidebar.subheader("Adjust Capital Allocation Strategy")
    retail_alloc = st.sidebar.slider("Retail Allocation (%)", 0, 100, 40)
    sme_alloc = st.sidebar.slider("SME Allocation (%)", 0, 100, 35)
    industry_alloc = st.sidebar.slider("Industry Allocation (%)", 0, 100, 25)

    total_alloc = retail_alloc + sme_alloc + industry_alloc
    if total_alloc != 100:
        st.warning(f"⚠ Total allocation should equal 100%. Current: {total_alloc}%")

    col1, col2, col3 = st.columns(3)
    col1.metric("Retail Capital", f"{retail_alloc}%")
    col2.metric("SME Capital", f"{sme_alloc}%")
    col3.metric("Industry Capital", f"{industry_alloc}%")

    st.markdown("---")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("**Capital Requirement by Sector**")
        fig1 = px.pie(names=sectors, values=capital_req, title="Basel Capital Requirement")
        st.plotly_chart(fig1, use_container_width=True)
    with row1_col2:
        st.markdown("**Exposure at Default (EAD)**")
        fig2 = px.bar(x=sectors, y=ead_values, title="EAD by Sector")
        st.plotly_chart(fig2, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown("**Simulated Allocation Impact**")
        fig3 = px.bar(x=sectors, y=[retail_alloc, sme_alloc, industry_alloc], title="Simulated Capital Allocation")
        st.plotly_chart(fig3, use_container_width=True)
    with row2_col2:
        st.markdown("**Capital Efficiency Analysis**")
        efficiency = [retail_alloc / ead_values[0], sme_alloc / ead_values[1], industry_alloc / ead_values[2]]
        fig4 = px.line(x=sectors, y=efficiency, title="Capital Efficiency by Sector")
        st.plotly_chart(fig4, use_container_width=True)

# Capital Buffer Section
if selected_opt == "Capital Buffer":
    st.header("Impact on Capital Buffer")
    st.markdown("""
    ### Basel IV Capital Buffer Analysis
    - Current buffer vs Basel minimum
    - Stress scenarios (economic downturn, credit shocks)
    - Buffer trend tracking
    """)

    severity = st.sidebar.slider("Adjust Stress Severity (%)", 0, 50, 20)
    adjusted_stress = df_demo["Buffer"] - (df_demo["Buffer"] * severity / 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Buffer", f"{df_demo['Buffer'].iloc[-1]:.2f}%")
    col2.metric("Stress Buffer", f"{adjusted_stress.iloc[-1]:.2f}%")
    col3.metric("Basel Minimum", "8.0%")

    if adjusted_stress.iloc[-1] < 8:
        st.error("⚠ Stress buffer falls below Basel minimum! Immediate action required.")
    else:
        st.success("✅ Buffer remains above Basel minimum under current stress scenario.")

    st.markdown("---")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        fig1 = px.bar(df_demo, x="Month", y="Buffer", title="Current Buffer Trend")
        st.plotly_chart(fig1, use_container_width=True)
    with row1_col2:
        fig2 = px.line(x=df_demo["Month"], y=adjusted_stress, title="Stress Buffer Trend")
        st.plotly_chart(fig2, use_container_width=True)
