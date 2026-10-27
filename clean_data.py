"""
HDFC Bank AI System - Automated Data Cleaning Pipeline
Validates, cleans, imputes, deduplicates, and sanitizes synthetic employee records.
"""

import os
import numpy as np
import pandas as pd

def clean_employee_data(
    input_path="data/synthetic_employee_data.csv",
    output_path="data/cleaned_employee_data.csv"
):
    print("=" * 60)
    print("STARTING AUTOMATED DATA CLEANING PIPELINE")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    print(f"[INFO] Loaded raw synthetic data: {initial_rows} rows, {len(df.columns)} columns.")
    
    audit_log = []
    
    # 1. Deduplication
    dup_mask = df.duplicated(subset=["Employee_ID", "Month"], keep="first")
    dup_count = dup_mask.sum()
    if dup_count > 0:
        audit_log.append(f"Removed {dup_count} duplicate employee-month records.")
        df = df[~dup_mask].copy()
    else:
        audit_log.append("No duplicate records found.")
        
    # 2. Type Enforcement
    numeric_cols = [
        "Working_Hours", "Overtime_Hours", "Tasks_Assigned", "Tasks_Completed",
        "Absence_Days", "Engagement_Score", "Quality_Score", "Deadline_Adherence",
        "Previous_Performance", "Performance_Score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip()
    df["Department"] = df["Department"].astype(str).str.strip()
    df["Month"] = df["Month"].astype(str).str.strip()
    
    # 3. Missing Value Detection & Imputation
    null_counts = df[numeric_cols].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # Impute with Department median, fallback to global median
                dept_medians = df.groupby("Department")[col].transform("median")
                df[col] = df[col].fillna(dept_medians).fillna(df[col].median())
                audit_log.append(f"Imputed {null_counts[col]} missing values in '{col}' using department-level median.")
    else:
        audit_log.append("No missing values found.")
        
    # 4. Negative Value Correction
    for col in ["Working_Hours", "Overtime_Hours", "Tasks_Assigned", "Tasks_Completed", "Absence_Days"]:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            df[col] = df[col].clip(lower=0)
            audit_log.append(f"Corrected {neg_count} negative values in '{col}' to zero.")
            
    # 5. Outlier Detection & Impossible Value Capping
    # Monthly working hours cap: standard full month is ~176h, absolute maximum realistic is 240h
    wh_outliers = (df["Working_Hours"] > 250.0).sum()
    if wh_outliers > 0:
        # Cap to realistic maximum
        df["Working_Hours"] = df["Working_Hours"].clip(lower=120.0, upper=240.0)
        audit_log.append(f"Clamped {wh_outliers} unrealistic Working_Hours records to maximum 240h.")
    else:
        df["Working_Hours"] = df["Working_Hours"].clip(lower=120.0, upper=240.0)
        
    ot_outliers = (df["Overtime_Hours"] > 70.0).sum()
    if ot_outliers > 0:
        df["Overtime_Hours"] = df["Overtime_Hours"].clip(lower=0.0, upper=70.0)
        audit_log.append(f"Clamped {ot_outliers} extreme Overtime_Hours records to 70h.")
    else:
        df["Overtime_Hours"] = df["Overtime_Hours"].clip(lower=0.0, upper=70.0)
        
    # Tasks logic: Tasks_Assigned >= 1, Tasks_Completed <= Tasks_Assigned
    df["Tasks_Assigned"] = df["Tasks_Assigned"].clip(lower=15, upper=100).astype(int)
    df["Tasks_Completed"] = np.minimum(df["Tasks_Completed"], df["Tasks_Assigned"]).astype(int)
    
    # Absence days cap: month has ~22 working days
    df["Absence_Days"] = df["Absence_Days"].clip(lower=0, upper=12).astype(int)
    
    # 6. Score Bounds [0, 100]
    score_cols = ["Engagement_Score", "Quality_Score", "Deadline_Adherence", "Previous_Performance", "Performance_Score"]
    for col in score_cols:
        df[col] = df[col].clip(lower=0.0, upper=100.0).round(1)
        
    # Sort logically
    df = df.sort_values(by=["Employee_ID", "Month"]).reset_index(drop=True)
    
    # 7. Verification checks
    unique_emps = df["Employee_ID"].nunique()
    total_records = len(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("[CLEANING AUDIT LOG]")
    for item in audit_log:
        print(f"  * {item}")
    print(f"[SUCCESS] Cleaned data saved: {total_records} records for {unique_emps} employees -> {output_path}")
    
    assert unique_emps == 400, f"Expected 400 employees, found {unique_emps}"
    assert total_records == 2400, f"Expected 2400 records, found {total_records}"
    print("=" * 60)
    return df

if __name__ == "__main__":
    clean_employee_data()
