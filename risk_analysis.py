"""
HDFC Bank AI System - Operational Risk Scoring & Factor Identification
Computes explainable 0-100 operational risk scores from multi-dimensional operational metrics
and extracts dynamic, employee-specific risk factors.
"""

import os
import numpy as np
import pandas as pd

def calculate_risk_scores(
    input_path="data/cleaned_employee_data.csv",
    output_path="data/cleaned_employee_data.csv"
):
    print("=" * 60)
    print("STARTING OPERATIONAL RISK SCORING")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    
    # 1. Overtime Risk (0-100): Overtime > 35h is extreme (100), 0h is 0
    ot_risk = (df["Overtime_Hours"] / 35.0 * 100.0).clip(0.0, 100.0)
    
    # 2. Workload Risk (0-100): Direct Workload_Index
    wl_risk = df["Workload_Index"].clip(0.0, 100.0)
    
    # 3. Absenteeism Risk (0-100): Absence > 5 days in a month represents high operational disruption
    abs_risk = (df["Absence_Days"] / 5.0 * 100.0).clip(0.0, 100.0)
    
    # 4. Engagement Risk (0-100): Inversely related to engagement score
    eng_risk = ((100.0 - df["Engagement_Score"]) / 60.0 * 100.0).clip(0.0, 100.0)
    
    # 5. Performance Trend Risk (0-100): Negative trend yields elevated risk
    # If 6M drop >= 18 points -> 100 risk; drop <= 0 -> proportional; positive change -> 0-15 risk
    trend_risk = (
        np.where(
            df["6M_Performance_Change"] < 0,
            (-df["6M_Performance_Change"] / 20.0 * 100.0),
            np.maximum(0.0, 15.0 - df["6M_Performance_Change"])
        )
    )
    trend_risk = np.clip(trend_risk, 0.0, 100.0)
    
    # 6. Task Completion Risk (0-100): Inversely related to task completion rate
    comp_risk = (((100.0 - df["Task_Completion_Rate"]) / 40.0) * 100.0).clip(0.0, 100.0)
    
    # Combine sub-risks using explainable project-defined weights
    # 20% Overtime + 20% Workload + 15% Absenteeism + 15% Engagement + 15% Trend + 15% Completion
    df["Overtime_Risk"] = ot_risk.round(1)
    df["Workload_Risk"] = wl_risk.round(1)
    df["Absenteeism_Risk"] = abs_risk.round(1)
    df["Engagement_Risk"] = eng_risk.round(1)
    df["Trend_Risk"] = trend_risk.round(1)
    df["Completion_Risk"] = comp_risk.round(1)
    
    raw_risk = (
        0.20 * df["Overtime_Risk"] +
        0.20 * df["Workload_Risk"] +
        0.15 * df["Absenteeism_Risk"] +
        0.15 * df["Engagement_Risk"] +
        0.15 * df["Trend_Risk"] +
        0.15 * df["Completion_Risk"]
    )
    df["Risk_Score"] = raw_risk.clip(0.0, 100.0).round(1)
    
    # Classify Risk Level:
    # 0-39 = Low Risk, 40-69 = Medium Risk, 70-100 = High Risk
    conditions_risk = [
        df["Risk_Score"] >= 70.0,
        df["Risk_Score"] >= 40.0
    ]
    choices_risk = ["High Risk", "Medium Risk"]
    df["Risk_Level"] = np.select(conditions_risk, choices_risk, default="Low Risk")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary per month 6 (current state)
    current_df = df[df["Month"] == "2026-06"]
    risk_dist = current_df["Risk_Level"].value_counts().to_dict()
    print(f"[SUCCESS] Operational Risk Scoring complete.")
    print(f"  * Month 6 Employee Risk Breakdown: {risk_dist}")
    print(f"  * Average Risk Score: {current_df['Risk_Score'].mean():.1f} / 100")
    print("=" * 60)
    return df

def identify_employee_risk_factors(row):
    """
    Dynamically identifies top 3-5 specific, quantitative risk factors for an employee.
    Used for individual PDF reports and dashboard inspection.
    """
    factors = []
    
    # 1. Overtime
    if row["Overtime_Hours"] >= 25.0:
        factors.append(f"Excessive monthly overtime ({row['Overtime_Hours']} hrs, {row['Overtime_Ratio']}% of working hours).")
    elif row["Overtime_Hours"] >= 15.0:
        factors.append(f"Elevated overtime level ({row['Overtime_Hours']} hrs/mo), indicating capacity pressure.")
        
    # 2. Overtime trend
    if row.get("6M_Overtime_Change", 0) >= 12.0:
        factors.append(f"Overtime surged by +{row['6M_Overtime_Change']} hrs over the 6-month evaluation period.")
        
    # 3. Workload
    if row["Workload_Index"] >= 70.0:
        factors.append(f"Severe workload index ({row['Workload_Index']}/100) exceeding sustainable threshold.")
    elif row["Workload_Index"] >= 55.0:
        factors.append(f"Moderately heavy workload ({row['Tasks_Assigned']} tasks assigned).")
        
    # 4. Task Completion
    if row["Task_Completion_Rate"] < 70.0:
        factors.append(f"Critical task completion deficit ({row['Task_Completion_Rate']}%), with {row['Tasks_Assigned'] - row['Tasks_Completed']} incomplete tasks.")
    elif row["Task_Completion_Rate"] < 85.0:
        factors.append(f"Suboptimal task completion rate ({row['Task_Completion_Rate']}%).")
        
    # 5. Engagement
    if row["Engagement_Score"] < 60.0:
        factors.append(f"Low employee engagement index ({row['Engagement_Score']}/100), flagging burnout risk.")
    elif row["Engagement_Score"] < 75.0:
        factors.append(f"Declining employee engagement ({row['Engagement_Score']}/100).")
        
    # 6. Performance Trend
    if row.get("6M_Performance_Change", 0) <= -12.0:
        factors.append(f"Pronounced 6-month performance decline ({row['6M_Performance_Change']} points).")
    elif row.get("6M_Performance_Change", 0) < -5.0:
        factors.append(f"Downward trajectory in overall performance score ({row['6M_Performance_Change']} pts).")
        
    # 7. Absenteeism
    if row["Absence_Days"] >= 4:
        factors.append(f"Frequent absenteeism ({row['Absence_Days']} days/mo, {row['Absenteeism_Rate']}% rate).")
    elif row["Absence_Days"] >= 2:
        factors.append(f"Moderate absence level ({row['Absence_Days']} days in the reporting period).")
        
    # 8. Deadline Adherence
    if row["Deadline_Adherence"] < 70.0:
        factors.append(f"High operational delay frequency (Deadline Adherence: {row['Deadline_Adherence']}%).")
        
    # Fallback if employee is healthy/low risk
    if len(factors) == 0:
        factors.append("No critical operational risks identified; metrics remain within healthy operating parameters.")
        factors.append("Sustainable working hours and manageable task distribution observed.")
        factors.append("Strong task completion and deadline compliance maintained.")
        
    return factors[:5]

if __name__ == "__main__":
    calculate_risk_scores()
