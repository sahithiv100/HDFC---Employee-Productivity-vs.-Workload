"""
HDFC Bank AI System - Feature Engineering
Computes business metrics, workload index, performance scores, productivity scores,
and capacity differentials according to HDFC Bank academic project specifications.
"""

import os
import numpy as np
import pandas as pd

def compute_features(
    input_path="data/cleaned_employee_data.csv",
    output_path="data/cleaned_employee_data.csv"
):
    print("=" * 60)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    
    # Ensure chronological order
    df = df.sort_values(by=["Employee_ID", "Month"]).reset_index(drop=True)
    
    # 1. Task Completion Rate (%)
    df["Task_Completion_Rate"] = (
        (df["Tasks_Completed"] / df["Tasks_Assigned"].replace(0, np.nan)) * 100.0
    ).fillna(100.0).clip(0.0, 100.0).round(1)
    
    # 2. Overtime Ratio (%)
    df["Overtime_Ratio"] = (
        (df["Overtime_Hours"] / df["Working_Hours"].replace(0, np.nan)) * 100.0
    ).fillna(0.0).clip(0.0, 100.0).round(1)
    
    # 3. Absenteeism Rate (%) - Assuming 22 available working days per month
    available_working_days = 22.0
    df["Absenteeism_Rate"] = (
        (df["Absence_Days"] / available_working_days) * 100.0
    ).clip(0.0, 100.0).round(1)
    
    # 4. Performance Score (Formula from Section 7):
    # 30% Task Completion + 25% Quality + 20% Deadline Adherence + 15% Engagement + 10% Previous Performance
    df["Performance_Score"] = (
        0.30 * df["Task_Completion_Rate"] +
        0.25 * df["Quality_Score"] +
        0.20 * df["Deadline_Adherence"] +
        0.15 * df["Engagement_Score"] +
        0.10 * df["Previous_Performance"]
    ).clip(0.0, 100.0).round(1)
    
    # Performance Classification:
    # 80-100 = High Performance, 60-79 = Moderate Performance, Below 60 = Low Performance
    conditions_perf = [
        df["Performance_Score"] >= 80.0,
        df["Performance_Score"] >= 60.0
    ]
    choices_perf = ["High Performance", "Moderate Performance"]
    df["Performance_Category"] = np.select(conditions_perf, choices_perf, default="Low Performance")
    
    # 5. Productivity Score:
    # 40% Task Completion Rate + 35% Quality Score + 25% Deadline Adherence
    df["Productivity_Score"] = (
        0.40 * df["Task_Completion_Rate"] +
        0.35 * df["Quality_Score"] +
        0.25 * df["Deadline_Adherence"]
    ).clip(0.0, 100.0).round(1)
    
    # 6. Workload Index (0-100 scale):
    # Normalized composite metric of Tasks Assigned, Working Hours, and Overtime Hours
    tasks_norm = ((df["Tasks_Assigned"] - 20) / (85 - 20)).clip(0.0, 1.0)
    hours_norm = ((df["Working_Hours"] - 150) / (225 - 150)).clip(0.0, 1.0)
    ot_norm = ((df["Overtime_Hours"] - 0) / (45 - 0)).clip(0.0, 1.0)
    
    df["Workload_Index"] = (
        (0.50 * tasks_norm + 0.25 * hours_norm + 0.25 * ot_norm) * 100.0
    ).clip(0.0, 100.0).round(1)
    
    # 7. Performance Change:
    # Month-over-month change
    df["Performance_Change"] = (
        df.groupby("Employee_ID")["Performance_Score"].diff().fillna(
            df["Performance_Score"] - df["Previous_Performance"]
        )
    ).round(1)
    
    # 8. Six-Month Trend Tracking (Month 6 vs Month 1 difference per employee)
    m1_perf = df.groupby("Employee_ID")["Performance_Score"].transform("first")
    m6_perf = df.groupby("Employee_ID")["Performance_Score"].transform("last")
    df["6M_Performance_Change"] = (m6_perf - m1_perf).round(1)
    
    m1_ot = df.groupby("Employee_ID")["Overtime_Hours"].transform("first")
    m6_ot = df.groupby("Employee_ID")["Overtime_Hours"].transform("last")
    df["6M_Overtime_Change"] = (m6_ot - m1_ot).round(1)
    
    m1_comp = df.groupby("Employee_ID")["Task_Completion_Rate"].transform("first")
    m6_comp = df.groupby("Employee_ID")["Task_Completion_Rate"].transform("last")
    df["6M_Completion_Change"] = (m6_comp - m1_comp).round(1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Feature engineering complete. Processed {len(df)} records across {df['Employee_ID'].nunique()} employees.")
    print(f"  * Performance Score mean: {df['Performance_Score'].mean():.1f} (High: {(df['Performance_Category'] == 'High Performance').sum()}, Moderate: {(df['Performance_Category'] == 'Moderate Performance').sum()}, Low: {(df['Performance_Category'] == 'Low Performance').sum()})")
    print(f"  * Productivity Score mean: {df['Productivity_Score'].mean():.1f}")
    print(f"  * Workload Index mean: {df['Workload_Index'].mean():.1f}")
    print("=" * 60)
    return df

if __name__ == "__main__":
    compute_features()
