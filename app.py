"""
HDFC Bank — Autonomous AI Employee Productivity, Risk & Workload Planning Dashboard
Interactive Streamlit application providing real-time workforce telemetry, operational risk scoring,
department capacity balancing, and automated PDF report downloads.
"""

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CORPORATE THEMING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HDFC Bank — Employee Productivity & Risk AI System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS
st.markdown("""
<style>
    /* Global Styles */
    .main-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #003366;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #92400E;
        border-radius: 4px;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-top: 3px solid #003366;
        text-align: center;
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #003366;
        margin-bottom: 2px;
    }
    .kpi-lbl {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING HELPERS
# -----------------------------------------------------------------------------
# Resolve base path relative to dashboard script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_employee_data.csv")
PREDS_PATH = os.path.join(BASE_DIR, "outputs", "employee_predictions.csv")
DEPT_PATH = os.path.join(BASE_DIR, "outputs", "department_summary.csv")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "model_metrics.json")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "employee_reports")
MGMT_REPORT_PATH = os.path.join(BASE_DIR, "outputs", "HDFC_Bank_Management_Report.pdf")
ZIP_PATH = os.path.join(REPORTS_DIR, "HDFC_Bank_400_Employee_Reports.zip")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH) or not os.path.exists(PREDS_PATH):
        return None, None, None, None
    df_clean = pd.read_csv(DATA_PATH)
    df_preds = pd.read_csv(PREDS_PATH)
    dept_summary = pd.read_csv(DEPT_PATH) if os.path.exists(DEPT_PATH) else None
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    return df_clean, df_preds, dept_summary, metrics

df_clean, df_preds, dept_summary, metrics = load_data()

# Header & Disclaimer
st.markdown('<div class="main-title">🏦 HDFC Bank — Autonomous Employee Productivity & Workload Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Operational Risk Telemetry, Capacity Balancing & Algorithmic Workload Management Engine</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <b>ACADEMIC RESEARCH DISCLAIMER:</b> The performance weights and data used in this project are academic assumptions created 
    for synthetic-data analysis and are not official HDFC Bank employee evaluation criteria. 
    This system is intended strictly as an operational decision-support tool for capacity planning and fatigue prevention.
</div>
""", unsafe_allow_html=True)

if df_clean is None or df_preds is None:
    st.error("Dataset not detected. Please run 'python pipeline.py' first to generate synthetic data, train models, and create reports.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS (Section 23)
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/28/HDFC_Bank_Logo.svg", width=190)
st.sidebar.markdown("### Operational Filters")

all_depts = ["All Departments"] + sorted(df_preds["Department"].unique().tolist())
selected_dept = st.sidebar.selectbox("Filter Department", all_depts)

all_risk_levels = ["All Risk Tiers", "High Risk", "Medium Risk", "Low Risk"]
selected_risk = st.sidebar.selectbox("Filter Risk Level", all_risk_levels)

all_workload_status = ["All Workload Statuses", "Overloaded", "Balanced", "Available Capacity"]
selected_wl = st.sidebar.selectbox("Filter Workload Status", all_workload_status)

all_months = sorted(df_clean["Month"].unique().tolist())
selected_month = st.sidebar.selectbox("Evaluation Month", all_months, index=len(all_months)-1)

# Apply filters
filtered_df = df_preds.copy()
if selected_dept != "All Departments":
    filtered_df = filtered_df[filtered_df["Department"] == selected_dept]
if selected_risk != "All Risk Tiers":
    filtered_df = filtered_df[filtered_df["Risk_Level"] == selected_risk]
if selected_wl != "All Workload Statuses":
    filtered_df = filtered_df[filtered_df["Workload_Status"] == selected_wl]

# Sidebar Quick Action
st.sidebar.markdown("---")
st.sidebar.markdown("### Master Downloads")
if os.path.exists(ZIP_PATH):
    with open(ZIP_PATH, "rb") as zf:
        st.sidebar.download_button(
            label="📦 Download All 400 Employee PDFs (ZIP)",
            data=zf.read(),
            file_name="HDFC_Bank_400_Employee_Reports.zip",
            mime="application/zip",
            use_container_width=True
        )

if os.path.exists(MGMT_REPORT_PATH):
    with open(MGMT_REPORT_PATH, "rb") as mf:
        st.sidebar.download_button(
            label="📄 Download Executive Management PDF",
            data=mf.read(),
            file_name="HDFC_Bank_Management_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# TABS INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Analytics & 10 Charts",
    "👤 Individual Employee PDF & Telemetry",
    "🏢 Department Capacity Balancing",
    "🤖 AI Risk Model & Clustering",
    "📁 Reports Index & Export"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE ANALYTICS & 10 CHARTS (Section 23)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### High-Level Workforce KPI Metrics")
    
    # 8 Required KPI Cards
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    
    total_count = len(filtered_df)
    avg_p = filtered_df["Performance_Score"].mean() if total_count > 0 else 0
    avg_prod = filtered_df["Productivity_Score"].mean() if total_count > 0 else 0
    avg_ot = filtered_df["Overtime_Hours"].mean() if total_count > 0 else 0
    avg_eng = filtered_df["Engagement_Score"].mean() if total_count > 0 else 0
    
    hr_count = (filtered_df["Risk_Level"] == "High Risk").sum()
    mr_count = (filtered_df["Risk_Level"] == "Medium Risk").sum()
    lr_count = (filtered_df["Risk_Level"] == "Low Risk").sum()
    
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_count}</div><div class="kpi-lbl">Total Headcount</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_p:.1f}</div><div class="kpi-lbl">Avg Performance</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_prod:.1f}</div><div class="kpi-lbl">Avg Productivity</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_ot:.1f}h</div><div class="kpi-lbl">Avg Overtime</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{avg_eng:.1f}</div><div class="kpi-lbl">Avg Engagement</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="kpi-card" style="border-top-color:#D32F2F;"><div class="kpi-val" style="color:#D32F2F;">{hr_count}</div><div class="kpi-lbl">High Risk</div></div>', unsafe_allow_html=True)
    with col7:
        st.markdown(f'<div class="kpi-card" style="border-top-color:#F57C00;"><div class="kpi-val" style="color:#F57C00;">{mr_count}</div><div class="kpi-lbl">Medium Risk</div></div>', unsafe_allow_html=True)
    with col8:
        st.markdown(f'<div class="kpi-card" style="border-top-color:#388E3C;"><div class="kpi-val" style="color:#388E3C;">{lr_count}</div><div class="kpi-lbl">Low Risk</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### Operational Analytics Visualizations (All 10 Core Charts)")
    
    # ROW 1: Charts 1 & 2
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("#### 1. Operational Risk Distribution")
        risk_counts = filtered_df["Risk_Level"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4.5, 3), dpi=120)
        colors_dict = {"Low Risk": "#388E3C", "Medium Risk": "#F57C00", "High Risk": "#D32F2F"}
        chart_colors = [colors_dict.get(k, "#003366") for k in risk_counts.index]
        ax1.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%", startangle=140, colors=chart_colors,
                wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
        ax1.set_title(f"Risk Tier Breakdown (N={total_count})", fontsize=9, fontweight="bold", color="#003366")
        st.pyplot(fig1)
        plt.close(fig1)
        
    with r1c2:
        st.markdown("#### 2. Performance by Department")
        fig2, ax2 = plt.subplots(figsize=(4.5, 3), dpi=120)
        dept_perf = filtered_df.groupby("Department")["Performance_Score"].mean().sort_values()
        ax2.barh(dept_perf.index, dept_perf.values, color="#003366", alpha=0.85)
        ax2.set_xlim(50, 100)
        ax2.set_xlabel("Average Performance Score", fontsize=8)
        ax2.tick_params(axis="both", labelsize=7.5)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="x")
        st.pyplot(fig2)
        plt.close(fig2)

    # ROW 2: Charts 3 & 4
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("#### 3. Overtime vs Performance (The Fatigue Paradox)")
        fig3, ax3 = plt.subplots(figsize=(4.5, 3), dpi=120)
        scatter_colors = filtered_df["Risk_Level"].map({"High Risk": "#D32F2F", "Medium Risk": "#F57C00", "Low Risk": "#388E3C"})
        ax3.scatter(filtered_df["Overtime_Hours"], filtered_df["Performance_Score"], c=scatter_colors, alpha=0.65, s=25)
        if len(filtered_df) > 1:
            z = np.polyfit(filtered_df["Overtime_Hours"], filtered_df["Performance_Score"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(filtered_df["Overtime_Hours"].min(), filtered_df["Overtime_Hours"].max(), 40)
            ax3.plot(x_line, p(x_line), "r--", linewidth=1.5, label="Trendline")
        ax3.set_xlabel("Overtime Hours / Month", fontsize=8)
        ax3.set_ylabel("Performance Score", fontsize=8)
        ax3.tick_params(axis="both", labelsize=7.5)
        ax3.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig3)
        plt.close(fig3)

    with r2c2:
        st.markdown("#### 4. Workload Index vs Performance")
        fig4, ax4 = plt.subplots(figsize=(4.5, 3), dpi=120)
        ax4.scatter(filtered_df["Workload_Index"], filtered_df["Performance_Score"], c="#004C8F", alpha=0.6, s=25)
        ax4.axvline(x=70, color="#D32F2F", linestyle=":", label="Workload Strain Threshold (>70)")
        ax4.set_xlabel("Workload Index (0-100)", fontsize=8)
        ax4.set_ylabel("Performance Score", fontsize=8)
        ax4.tick_params(axis="both", labelsize=7.5)
        ax4.legend(fontsize=7, loc="lower left")
        ax4.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig4)
        plt.close(fig4)

    # ROW 3: Charts 5 & 6
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.markdown("#### 5. Engagement Score vs Performance")
        fig5, ax5 = plt.subplots(figsize=(4.5, 3), dpi=120)
        ax5.scatter(filtered_df["Engagement_Score"], filtered_df["Performance_Score"], c="#10B981", alpha=0.65, s=25)
        ax5.set_xlabel("Engagement Score (0-100)", fontsize=8)
        ax5.set_ylabel("Performance Score", fontsize=8)
        ax5.tick_params(axis="both", labelsize=7.5)
        ax5.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig5)
        plt.close(fig5)

    with r3c2:
        st.markdown("#### 6. Absenteeism vs Performance")
        fig6, ax6 = plt.subplots(figsize=(4.5, 3), dpi=120)
        abs_grp = filtered_df.groupby("Absence_Days")["Performance_Score"].mean()
        ax6.bar(abs_grp.index, abs_grp.values, color="#F57C00", alpha=0.85, width=0.6)
        ax6.set_xlabel("Absence Days in Reporting Month", fontsize=8)
        ax6.set_ylabel("Average Performance Score", fontsize=8)
        ax6.set_ylim(40, 100)
        ax6.tick_params(axis="both", labelsize=7.5)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="y")
        st.pyplot(fig6)
        plt.close(fig6)

    # ROW 4: Charts 7 & 8
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.markdown("#### 7. Monthly Performance & Overtime Trend (6 Months)")
        monthly_trend = df_clean.groupby("Month").agg(
            Avg_Perf=("Performance_Score", "mean"),
            Avg_OT=("Overtime_Hours", "mean"),
            Avg_Comp=("Task_Completion_Rate", "mean")
        ).reset_index()
        fig7, ax7_1 = plt.subplots(figsize=(4.5, 3), dpi=120)
        ax7_1.plot(monthly_trend["Month"], monthly_trend["Avg_Perf"], color="#003366", marker="o", label="Avg Performance")
        ax7_1.plot(monthly_trend["Month"], monthly_trend["Avg_Comp"], color="#2E7D32", marker="s", linestyle="--", label="Task Completion %")
        ax7_1.set_ylabel("Score / Completion (%)", fontsize=7.5, color="#003366")
        ax7_1.set_ylim(60, 100)
        ax7_1.tick_params(axis="both", labelsize=7)
        ax7_1.grid(True, linestyle=":", alpha=0.5)
        
        ax7_2 = ax7_1.twinx()
        ax7_2.plot(monthly_trend["Month"], monthly_trend["Avg_OT"], color="#D32F2F", marker="^", label="Avg Overtime (h)")
        ax7_2.set_ylabel("Overtime Hours", fontsize=7.5, color="#D32F2F")
        ax7_2.tick_params(axis="y", labelsize=7)
        ax7_1.legend(loc="lower left", fontsize=6.5)
        st.pyplot(fig7)
        plt.close(fig7)

    with r4c2:
        st.markdown("#### 8. Department Risk Tier Comparison")
        fig8, ax8 = plt.subplots(figsize=(4.5, 3), dpi=120)
        risk_dept = pd.crosstab(filtered_df["Department"], filtered_df["Risk_Level"], normalize="index") * 100
        # Reorder columns if present
        cols_order = [c for c in ["Low Risk", "Medium Risk", "High Risk"] if c in risk_dept.columns]
        risk_dept = risk_dept[cols_order]
        colors_stack = [colors_dict.get(c, "#999999") for c in cols_order]
        risk_dept.plot(kind="bar", stacked=True, color=colors_stack, ax=ax8, alpha=0.9)
        ax8.set_ylabel("% Employees", fontsize=8)
        ax8.set_xticklabels(ax8.get_xticklabels(), rotation=35, ha="right", fontsize=7)
        ax8.legend(loc="upper right", fontsize=7)
        ax8.grid(True, linestyle=":", alpha=0.4, axis="y")
        st.pyplot(fig8)
        plt.close(fig8)

    # ROW 5: Charts 9 & 10
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        st.markdown("#### 9. Employee Behavioral Clusters (K-Means)")
        fig9, ax9 = plt.subplots(figsize=(4.5, 3), dpi=120)
        cluster_palette = {
            "Healthy & Productive": "#2E7D32",
            "Overloaded & At Risk": "#D32F2F",
            "Under-Utilized": "#004C8F",
            "Low Productivity / Support Required": "#F57C00"
        }
        for cname, cgrp in filtered_df.groupby("Cluster_Name"):
            c_color = cluster_palette.get(cname, "#666666")
            ax9.scatter(cgrp["Workload_Index"], cgrp["Performance_Score"], label=cname, color=c_color, alpha=0.7, s=25)
        ax9.set_xlabel("Workload Index", fontsize=8)
        ax9.set_ylabel("Performance Score", fontsize=8)
        ax9.tick_params(axis="both", labelsize=7.5)
        ax9.legend(fontsize=6.5, loc="lower left")
        ax9.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig9)
        plt.close(fig9)

    with r5c2:
        st.markdown("#### 10. Top Predictive Risk Drivers (Feature Importance)")
        fig10, ax10 = plt.subplots(figsize=(4.5, 3), dpi=120)
        if metrics and "feature_importance" in metrics:
            fi_df = pd.DataFrame(metrics["feature_importance"]).head(7)
            ax10.barh(fi_df["feature"].str.replace("_", " "), fi_df["importance"], color="#003366", alpha=0.85)
            ax10.set_xlabel("Predictive Importance (%)", fontsize=8)
            ax10.tick_params(axis="both", labelsize=7.5)
            ax10.invert_yaxis()
            ax10.grid(True, linestyle=":", alpha=0.5, axis="x")
        else:
            ax10.text(0.5, 0.5, "Model metrics pending training", ha="center")
        st.pyplot(fig10)
        plt.close(fig10)

# -----------------------------------------------------------------------------
# TAB 2: INDIVIDUAL EMPLOYEE PDF & TELEMETRY (Section 24)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### Individual Employee Performance Telemetry & PDF Report")
    
    col_sel, col_stat = st.columns([1.5, 3])
    with col_sel:
        emp_list = sorted(filtered_df["Employee_ID"].unique().tolist())
        if not emp_list:
            st.warning("No employees match the current filter selection.")
            st.stop()
            
        selected_emp = st.selectbox("Select Employee for Inspection:", emp_list, index=0)
        emp_row = filtered_df[filtered_df["Employee_ID"] == selected_emp].iloc[0]
        emp_hist = df_clean[df_clean["Employee_ID"] == selected_emp].sort_values(by="Month")
        
        # Download Individual PDF Button
        emp_pdf_path = os.path.join(REPORTS_DIR, f"{selected_emp}_Report.pdf")
        if os.path.exists(emp_pdf_path):
            with open(emp_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label=f"📥 Download {selected_emp} Official PDF Report",
                data=pdf_bytes,
                file_name=f"{selected_emp}_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning(f"PDF report not found at {emp_pdf_path}. Generate via pipeline.")
            
    with col_stat:
        st.markdown(f"#### Profile: **{selected_emp}** &nbsp;|&nbsp; Department: **{emp_row['Department']}** &nbsp;|&nbsp; Cluster: **{emp_row.get('Cluster_Name', 'General')}**")
        
        # Badges
        r_badge = f'<span class="badge-high">HIGH RISK</span>' if "High" in emp_row["Risk_Level"] else (f'<span class="badge-med">MEDIUM RISK</span>' if "Med" in emp_row["Risk_Level"] else f'<span class="badge-low">LOW RISK</span>')
        wl_badge = f'<span class="badge-high">OVERLOADED</span>' if "Overloaded" in emp_row["Workload_Status"] else (f'<span class="badge-low">BALANCED</span>' if "Balanced" in emp_row["Workload_Status"] else f'<span class="badge-med">AVAILABLE CAPACITY</span>')
        
        st.markdown(f"Status: {r_badge} &nbsp;&nbsp; Workload Status: {wl_badge} &nbsp;&nbsp; Performance: **{emp_row['Performance_Category']}**", unsafe_allow_html=True)
        
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        m_c1.metric("Performance", f"{emp_row['Performance_Score']:.1f}")
        m_c2.metric("Risk Score", f"{emp_row['Risk_Score']:.1f} / 100")
        m_c3.metric("Overtime", f"{emp_row['Overtime_Hours']:.1f} hrs")
        m_c4.metric("Completion %", f"{emp_row['Task_Completion_Rate']:.1f}%")
        m_c5.metric("Engagement", f"{emp_row['Engagement_Score']:.1f}")

    st.markdown("---")
    
    # Detail columns
    det_c1, det_c2 = st.columns([1.2, 1])
    with det_c1:
        st.markdown("#### Six-Month Performance & Overtime History")
        fig_ind, ax_ind = plt.subplots(figsize=(6, 2.5), dpi=120)
        ax_ind.plot(emp_hist["Month"], emp_hist["Performance_Score"], marker="o", color="#003366", label="Performance Score")
        ax_ind.plot(emp_hist["Month"], emp_hist["Task_Completion_Rate"], marker="s", color="#2E7D32", linestyle="--", label="Task Completion %")
        ax_ind.set_ylabel("Score / Completion %", fontsize=8)
        ax_ind.set_ylim(20, 105)
        ax_ind.grid(True, linestyle=":", alpha=0.5)
        
        ax_ind_ot = ax_ind.twinx()
        ax_ind_ot.plot(emp_hist["Month"], emp_hist["Overtime_Hours"], marker="^", color="#D32F2F", label="Overtime (hrs)")
        ax_ind_ot.set_ylabel("Overtime Hours", fontsize=8, color="#D32F2F")
        ax_ind.legend(loc="lower left", fontsize=7)
        st.pyplot(fig_ind)
        plt.close(fig_ind)
        
    with det_c2:
        st.markdown("#### Capacity & Workload Balance")
        cur_t = int(emp_row["Current_Tasks"])
        rec_t = int(emp_row["Recommended_Tasks"])
        diff_t = int(emp_row["Difference_Tasks"])
        
        st.write(f"- **Current Tasks Assigned:** {cur_t}")
        st.write(f"- **Recommended Tasks:** {rec_t}")
        st.write(f"- **Capacity Difference:** {'+' if diff_t > 0 else ''}{diff_t} tasks")
        st.write(f"- **Absence Days:** {int(emp_row['Absence_Days'])} days/mo")
        st.write(f"- **Quality Score:** {emp_row['Quality_Score']:.1f} / 100")
        st.write(f"- **Deadline Adherence:** {emp_row['Deadline_Adherence']:.1f}%")
        
        st.markdown("#### AI Workload Recommendation")
        st.info(f"**Directive:** {emp_row['Recommendation']}\n\n**Reason:** {emp_row['Reason']}")

# -----------------------------------------------------------------------------
# TAB 3: DEPARTMENT CAPACITY BALANCING (Section 25)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Department-Level Analytics & Workload Pressure Ranking")
    st.markdown(
        "Automated aggregation across all 7 operational banking divisions identifies operational bottlenecks, "
        "workload stress concentrations, and capacity surpluses ready for cross-functional redeployment."
    )
    
    if dept_summary is not None:
        st.dataframe(
            dept_summary[[
                "Workload_Pressure_Rank", "Department", "Employee_Count", "Average_Performance",
                "Average_Overtime_Hours", "Average_Workload_Index", "High_Risk_Percentage",
                "Overloaded_Count", "Available_Capacity_Count", "Workload_Pressure_Score", "Department_Recommendation"
            ]],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("#### Strategic Department Observations")
        for _, d in dept_summary.iterrows():
            with st.expander(f"#{int(d['Workload_Pressure_Rank'])} {d['Department']} (Pressure Score: {d['Workload_Pressure_Score']})"):
                st.write(f"- **Workforce Size:** {int(d['Employee_Count'])} synthetic staff")
                st.write(f"- **High-Risk Density:** {d['High_Risk_Percentage']}% ({int(d['High_Risk_Count'])} staff)")
                st.write(f"- **Overloaded Personnel:** {int(d['Overloaded_Count'])} vs **Available Capacity:** {int(d['Available_Capacity_Count'])}")
                st.write(f"- **Average Monthly Overtime:** {d['Average_Overtime_Hours']} hours/employee")
                st.info(f"**AI Strategy:** {d['Department_Recommendation']}")
    else:
        st.warning("Department summary not loaded.")

# -----------------------------------------------------------------------------
# TAB 4: AI RISK MODEL & CLUSTERING (Section 8 & 10)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### Machine Learning Model Diagnostics & Validation")
    
    if metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Model Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col_m2.metric("Precision (Macro)", f"{metrics['precision_macro']*100:.2f}%")
        col_m3.metric("Recall (Macro)", f"{metrics['recall_macro']*100:.2f}%")
        col_m4.metric("F1 Score (Macro)", f"{metrics['f1_macro']*100:.2f}%")
        
        st.markdown("---")
        c_fi, c_cm = st.columns(2)
        with c_fi:
            st.markdown("#### Feature Importance Rankings")
            fi_df = pd.DataFrame(metrics["feature_importance"])
            st.dataframe(fi_df, use_container_width=True, hide_index=True)
            
        with c_cm:
            st.markdown("#### Confusion Matrix (Held-out Test Cohort)")
            cm_df = pd.DataFrame(
                metrics["confusion_matrix"],
                columns=["Pred Low Risk", "Pred Med Risk", "Pred High Risk"],
                index=["True Low Risk", "True Med Risk", "True High Risk"]
            )
            st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Train the model via pipeline.py to display detailed machine learning diagnostics.")

# -----------------------------------------------------------------------------
# TAB 5: REPORTS INDEX & EXPORT (Section 22 & 24)
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### Master Reports Index & Batch Downloads")
    st.markdown(
        "Access the master index mapping all 400 employees, download the complete ZIP archive containing "
        "all 400 individual reports, or inspect individual PDF generation logs."
    )
    
    down_col1, down_col2 = st.columns(2)
    with down_col1:
        if os.path.exists(ZIP_PATH):
            with open(ZIP_PATH, "rb") as zf:
                st.download_button(
                    label="📦 Download All 400 Individual PDFs (.ZIP)",
                    data=zf.read(),
                    file_name="HDFC_Bank_400_Employee_Reports.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    with down_col2:
        if os.path.exists(MGMT_REPORT_PATH):
            with open(MGMT_REPORT_PATH, "rb") as mf:
                st.download_button(
                    label="📄 Download Master Executive Management PDF",
                    data=mf.read(),
                    file_name="HDFC_Bank_Management_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
    st.markdown("#### Master Report Directory (400 Employees)")
    index_csv_path = os.path.join(REPORTS_DIR, "report_index.csv")
    if os.path.exists(index_csv_path):
        df_idx = pd.read_csv(index_csv_path)
        st.dataframe(df_idx, use_container_width=True, hide_index=True)
    else:
        st.write("Report index file pending generation.")
