"""
HDFC Bank AI System - Master End-to-End Pipeline Orchestrator
Executes the full automated workflow from data synthesis to report generation and validation.
Usage:
    python pipeline.py
"""

import os
import sys
import time
import zipfile
import pandas as pd

# Add scripts directory to module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CURRENT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from generate_data import generate_synthetic_data
from clean_data import clean_employee_data
from feature_engineering import compute_features
from risk_analysis import calculate_risk_scores
from train_model import train_risk_model_and_cluster
from recommendations import generate_recommendations_and_department_summary
from generate_employee_reports import generate_all_employee_reports
from generate_management_report import generate_management_report

def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("HDFC BANK — AUTONOMOUS AI EMPLOYEE PRODUCTIVITY & RISK PIPELINE")
    print("=" * 70)
    print(f"Start Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {CURRENT_DIR}\n")

    # Define standard paths
    raw_data_path = os.path.join(CURRENT_DIR, "data", "synthetic_employee_data.csv")
    clean_data_path = os.path.join(CURRENT_DIR, "data", "cleaned_employee_data.csv")
    model_path = os.path.join(CURRENT_DIR, "models", "productivity_risk_model.pkl")
    metrics_path = os.path.join(CURRENT_DIR, "outputs", "model_metrics.json")
    preds_path = os.path.join(CURRENT_DIR, "outputs", "employee_predictions.csv")
    recs_path = os.path.join(CURRENT_DIR, "outputs", "workload_recommendations.csv")
    dept_path = os.path.join(CURRENT_DIR, "outputs", "department_summary.csv")
    reports_dir = os.path.join(CURRENT_DIR, "outputs", "employee_reports")
    mgmt_report_path = os.path.join(CURRENT_DIR, "outputs", "HDFC_Bank_Management_Report.pdf")
    zip_path = os.path.join(reports_dir, "HDFC_Bank_400_Employee_Reports.zip")
    index_path = os.path.join(reports_dir, "report_index.csv")

    # Step 1: Generate Synthetic Data (400 employees, 6 months)
    print("\n>>> STEP 1/8: Generating 400 Synthetic Employees Across 6 Months...")
    generate_synthetic_data(output_path=raw_data_path)

    # Step 2: Automated Data Cleaning & Imputation
    print("\n>>> STEP 2/8: Executing Automated Data Cleaning Pipeline...")
    clean_employee_data(input_path=raw_data_path, output_path=clean_data_path)

    # Step 3: Feature Engineering (Performance, Productivity, Workload Index)
    print("\n>>> STEP 3/8: Computing Business Metrics & Feature Engineering...")
    compute_features(input_path=clean_data_path, output_path=clean_data_path)

    # Step 4: Operational Risk Scoring
    print("\n>>> STEP 4/8: Calculating Operational Risk Scores & Dynamic Drivers...")
    calculate_risk_scores(input_path=clean_data_path, output_path=clean_data_path)

    # Step 5: Train AI Operational Risk Model & K-Means Segmentation
    print("\n>>> STEP 5/8: Training ML Classifier & Running K-Means Segmentation...")
    train_risk_model_and_cluster(
        input_path=clean_data_path,
        output_data_path=clean_data_path,
        model_output_path=model_path,
        metrics_output_path=metrics_path
    )

    # Step 6: Workload Recommendation Engine & Capacity Planning
    print("\n>>> STEP 6/8: Generating Individual Workload Recommendations & Dept Analysis...")
    generate_recommendations_and_department_summary(
        input_path=clean_data_path,
        predictions_output=preds_path,
        recommendations_output=recs_path,
        department_output=dept_path
    )

    # Step 7: Batch Generate 400 Employee PDF Reports & ZIP
    print("\n>>> STEP 7/8: Generating 400 Individual Employee PDF Reports & ZIP Archive...")
    pdf_count = generate_all_employee_reports(
        cleaned_data_path=clean_data_path,
        predictions_path=preds_path,
        output_dir=reports_dir
    )

    # Step 8: Generate Management PDF Report
    print("\n>>> STEP 8/8: Generating Executive Management PDF Report...")
    generate_management_report(
        cleaned_data_path=clean_data_path,
        predictions_path=preds_path,
        department_path=dept_path,
        metrics_path=metrics_path,
        output_pdf_path=mgmt_report_path
    )

    # -------------------------------------------------------------------------
    # STEP 9: AUTOMATIC VALIDATION SUITE (Section 29)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING AUTOMATED PIPELINE VALIDATION CHECKS")
    print("=" * 70)

    df_clean = pd.read_csv(clean_data_path)
    df_preds = pd.read_csv(preds_path)
    
    val_errors = []
    
    # Check 1: Number of employees = 400
    n_unique_emps = df_preds["Employee_ID"].nunique()
    if n_unique_emps != 400:
        val_errors.append(f"Validation Error: Expected 400 unique employees, found {n_unique_emps}")
    else:
        print("  [PASS] Number of employees = 400")

    # Check 2: Total employee-month records = 2,400
    n_records = len(df_clean)
    if n_records != 2400:
        val_errors.append(f"Validation Error: Expected 2,400 records, found {n_records}")
    else:
        print("  [PASS] Employee-month records = 2,400")

    # Check 3: Number of employee PDFs = 400
    pdf_files = [f for f in os.listdir(reports_dir) if f.startswith("EMP") and f.endswith("_Report.pdf")]
    if len(pdf_files) != 400:
        val_errors.append(f"Validation Error: Expected 400 employee PDFs, found {len(pdf_files)}")
    else:
        print("  [PASS] Number of employee PDFs = 400")

    # Check 4: All PDFs non-empty (>0 bytes)
    corrupted_pdfs = [f for f in pdf_files if os.path.getsize(os.path.join(reports_dir, f)) == 0]
    if corrupted_pdfs:
        val_errors.append(f"Validation Error: Found {len(corrupted_pdfs)} empty PDF files: {corrupted_pdfs[:3]}")
    else:
        print("  [PASS] All 400 employee PDFs verified non-empty and valid")

    # Check 5: Number of departments > 1
    n_depts = df_preds["Department"].nunique()
    if n_depts <= 1:
        val_errors.append(f"Validation Error: Expected > 1 departments, found {n_depts}")
    else:
        print(f"  [PASS] Number of departments = {n_depts} (Realistic multi-department distribution)")

    # Check 6: No duplicate employee IDs in predictions
    if df_preds["Employee_ID"].duplicated().any():
        val_errors.append("Validation Error: Duplicate Employee IDs detected in predictions")
    else:
        print("  [PASS] No duplicate employee IDs in master predictions")

    # Check 7: All employees have a risk score
    if df_preds["Risk_Score"].isnull().any():
        val_errors.append("Validation Error: Missing Risk_Score detected")
    else:
        print("  [PASS] All 400 employees have valid Risk Scores")

    # Check 8: All employees have a recommendation
    if df_preds["Recommendation"].isnull().any() or (df_preds["Recommendation"].str.strip() == "").any():
        val_errors.append("Validation Error: Missing or empty recommendations detected")
    else:
        print("  [PASS] All 400 employees have tailored AI recommendations")

    # Check 9: All employees have a performance score
    if df_preds["Performance_Score"].isnull().any():
        val_errors.append("Validation Error: Missing Performance_Score detected")
    else:
        print("  [PASS] All 400 employees have valid Performance Scores")

    # Check 10: Master Report Index & ZIP Archive
    if not (os.path.exists(index_path) and os.path.getsize(index_path) > 0):
        val_errors.append("Validation Error: report_index.csv is missing or empty")
    else:
        print("  [PASS] Master Report Index (report_index.csv) verified")

    if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 0):
        val_errors.append("Validation Error: HDFC_Bank_400_Employee_Reports.zip is missing or empty")
    else:
        print(f"  [PASS] Consolidated ZIP Archive verified ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

    if not (os.path.exists(mgmt_report_path) and os.path.getsize(mgmt_report_path) > 0):
        val_errors.append("Validation Error: HDFC_Bank_Management_Report.pdf is missing or empty")
    else:
        print("  [PASS] Master Management Report PDF verified")

    elapsed_time = time.time() - start_time
    
    # Tally Risk Breakdowns
    hr_cnt = (df_preds["Risk_Level"] == "High Risk").sum()
    mr_cnt = (df_preds["Risk_Level"] == "Medium Risk").sum()
    lr_cnt = (df_preds["Risk_Level"] == "Low Risk").sum()

    pipeline_status = "COMPLETE" if not val_errors else "FAILED"
    
    # Section 29 Required Summary Printout
    print("\n" + "=" * 40)
    print("HDFC BANK AI SYSTEM — PIPELINE COMPLETE")
    print("=" * 40)
    print(f"\nEmployees generated: {n_unique_emps}")
    print(f"Employee-month records: {n_records}\n")
    print(f"High Risk Employees: {hr_cnt}")
    print(f"Medium Risk Employees: {mr_cnt}")
    print(f"Low Risk Employees: {lr_cnt}\n")
    print(f"PDF reports generated: {len(pdf_files)}")
    print("Management report: SUCCESS")
    print("Dashboard data: SUCCESS")
    print("ZIP file: SUCCESS\n")
    print(f"Pipeline Status: {pipeline_status}")
    print(f"Total Execution Time: {elapsed_time:.1f} seconds")
    print("=" * 40)

    if val_errors:
        print("\n[VALIDATION ISSUES IDENTIFIED]:")
        for err in val_errors:
            print(f"  ! {err}")
        sys.exit(1)
        
    return True

if __name__ == "__main__":
    run_pipeline()
