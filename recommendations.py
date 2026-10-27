"""
HDFC Bank AI System - Workload Recommendation Engine & Department Capacity Planning
Calculates personalized employee capacity, workload adjustments (Current vs Recommended),
generates dynamic tailored AI recommendations, and computes department-level analytics.
"""

import os
import numpy as np
import pandas as pd
from risk_analysis import identify_employee_risk_factors

def generate_recommendations_and_department_summary(
    input_path="data/cleaned_employee_data.csv",
    predictions_output="outputs/employee_predictions.csv",
    recommendations_output="outputs/workload_recommendations.csv",
    department_output="outputs/department_summary.csv"
):
    print("=" * 60)
    print("STARTING WORKLOAD RECOMMENDATION & CAPACITY ANALYSIS ENGINE")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    
    # Filter for latest reporting month (Month 6 = 2026-06) for current capacity planning
    latest_month = "2026-06"
    df_m6 = df[df["Month"] == latest_month].copy().sort_values(by="Employee_ID").reset_index(drop=True)
    
    # Also calculate 6-month historical averages per employee
    m6_agg = df.groupby("Employee_ID").agg(
        Avg_Working_Hours=("Working_Hours", "mean"),
        Avg_Overtime_Hours=("Overtime_Hours", "mean"),
        Avg_Tasks_Assigned=("Tasks_Assigned", "mean"),
        Avg_Tasks_Completed=("Tasks_Completed", "mean"),
        Avg_Task_Completion_Rate=("Task_Completion_Rate", "mean"),
        Total_Absence_Days=("Absence_Days", "sum"),
        Avg_Engagement_Score=("Engagement_Score", "mean"),
        Avg_Quality_Score=("Quality_Score", "mean"),
        Avg_Deadline_Adherence=("Deadline_Adherence", "mean"),
        Avg_Performance_Score=("Performance_Score", "mean"),
        Avg_Productivity_Score=("Productivity_Score", "mean"),
        Avg_Workload_Index=("Workload_Index", "mean"),
        Avg_Risk_Score=("Risk_Score", "mean")
    ).round(1).reset_index()
    
    df_current = pd.merge(df_m6, m6_agg, on="Employee_ID")
    
    rec_records = []
    
    # Capacity Baseline: Standard sustainable task volume is 44 tasks/month
    STANDARD_CAPACITY = 44
    
    for idx, row in df_current.iterrows():
        emp_id = row["Employee_ID"]
        dept = row["Department"]
        current_tasks = int(row["Tasks_Assigned"])
        ot_hours = float(row["Overtime_Hours"])
        perf_score = float(row["Performance_Score"])
        risk_score = float(row["Risk_Score"])
        risk_level = str(row["Risk_Level"])
        comp_rate = float(row["Task_Completion_Rate"])
        eng_score = float(row["Engagement_Score"])
        absence = int(row["Absence_Days"])
        perf_change = float(row.get("6M_Performance_Change", 0.0))
        cluster_name = str(row.get("Cluster_Name", "General"))
        
        # ---------------------------------------------------------
        # Dynamic Capacity & Recommended Workload Calculation (Section 12)
        # ---------------------------------------------------------
        if risk_level == "High Risk" or current_tasks >= 60:
            # Overloaded: reduce workload by 15% to 25%
            reduction_ratio = 0.20 if current_tasks >= 68 else 0.15
            recommended_tasks = int(np.round(current_tasks * (1.0 - reduction_ratio)))
            recommended_tasks = max(38, min(recommended_tasks, 48))
            diff = recommended_tasks - current_tasks # negative number = reduce
            workload_status = "Overloaded"
            
        elif current_tasks <= 34 and perf_score >= 70.0 and ot_hours <= 5.0:
            # Under-utilized: increase workload by 15% to 30%
            increase_tasks = int(np.round(min(12, max(6, (STANDARD_CAPACITY - current_tasks)))))
            recommended_tasks = current_tasks + increase_tasks
            recommended_tasks = min(recommended_tasks, 48)
            diff = recommended_tasks - current_tasks # positive number = absorb work
            workload_status = "Available Capacity"
            
        elif current_tasks > 52 and comp_rate < 80.0:
            recommended_tasks = int(np.round(current_tasks * 0.85))
            diff = recommended_tasks - current_tasks
            workload_status = "Overloaded"
            
        elif abs(current_tasks - STANDARD_CAPACITY) <= 5 and ot_hours <= 15.0:
            recommended_tasks = current_tasks
            diff = 0
            workload_status = "Balanced"
            
        else:
            # Moderate adjustments toward sustainable baseline
            target = STANDARD_CAPACITY
            diff = target - current_tasks
            if diff < -3:
                workload_status = "Overloaded"
                recommended_tasks = current_tasks + diff
            elif diff > 3:
                workload_status = "Available Capacity"
                recommended_tasks = current_tasks + diff
            else:
                workload_status = "Balanced"
                recommended_tasks = current_tasks
                diff = 0
                
        # ---------------------------------------------------------
        # AI Recommendation Rationale & Suggested Actions (Section 11)
        # ---------------------------------------------------------
        actions = []
        if workload_status == "Overloaded":
            tasks_to_cut = abs(diff)
            pct = int(round((tasks_to_cut / current_tasks) * 100))
            recommendation = (
                f"Reduce assigned workload by approximately {pct}% (-{tasks_to_cut} tasks) "
                f"and redistribute active volume to peer colleagues with available capacity."
            )
            reason = (
                f"Employee is experiencing severe capacity strain with {ot_hours} overtime hours, "
                f"a workload index of {row['Workload_Index']}/100, and a task completion rate of {comp_rate}%. "
                f"Without immediate task shedding, performance degradation and operational error rates will compound."
            )
            actions = [
                f"Immediately reassign {tasks_to_cut} operational deliverables to under-utilized peers in {dept}.",
                "Enforce a mandatory ceiling of no more than 10 overtime hours per month.",
                "Conduct a bi-weekly workflow review to prioritize core regulatory and customer deliverables.",
                "Monitor task completion stability and error rates over the next 30-day operating cycle."
            ]
            
        elif workload_status == "Available Capacity":
            tasks_to_add = diff
            recommendation = (
                f"Gradually expand task allocation by +{tasks_to_add} tasks to optimize capacity utilization "
                f"and relieve pressure from overloaded team members."
            )
            reason = (
                f"Employee demonstrates high operational efficiency (Task Completion: {comp_rate}%, "
                f"Performance: {perf_score}/100) with minimal overtime ({ot_hours} hrs) and available headroom. "
                f"Current task volume ({current_tasks} tasks) is well below the departmental capacity baseline."
            )
            actions = [
                f"Progressively allocate {tasks_to_add} cross-functional tasks over the next 2 cycles.",
                f"Partner with senior team leads in {dept} to mentor and absorb overflow operational tasks.",
                "Review skill readiness for specialized, high-impact banking deliverables.",
                "Ensure workload expansion remains calibrated to maintain current high quality scores."
            ]
            
        else: # Balanced
            recommendation = (
                "Maintain current balanced workload allocation and continue standard operational cadence."
            )
            reason = (
                f"Employee maintains an optimal operational equilibrium (Performance: {perf_score}/100, "
                f"Workload Index: {row['Workload_Index']}/100, Overtime: {ot_hours} hrs). "
                f"Task throughput and quality metrics remain well within healthy bank tolerances."
            )
            actions = [
                "Preserve existing task assignment ratios and monthly working hour schedules.",
                "Maintain periodic monthly performance touchpoints to sustain high engagement.",
                "Encourage participation in departmental process improvement initiatives.",
                "Evaluate for leadership development or specialized domain certifications."
            ]
            
        # Specific overrides for Low Engagement or High Absenteeism
        if eng_score < 60.0 and perf_change < -5.0:
            actions.insert(0, "Schedule an immediate supportive 1-on-1 manager check-in to address engagement barriers.")
        if absence >= 4:
            actions.insert(1, "Review leave and attendance patterns; structure task handovers during scheduled absences.")
            
        risk_factors = identify_employee_risk_factors(row)
        
        # Short AI summary (100-150 words) for Section 19
        overall_summary = (
            f"Employee {emp_id} within the {dept} department currently reflects a {perf_score:.1f}/100 performance score "
            f"and an operational risk rating of {risk_score:.1f}/100 ({risk_level}). "
            f"Over the 6-month analysis period, task completion reached {comp_rate:.1f}% alongside an overtime commitment "
            f"of {ot_hours:.1f} hours per month. The employee is segmented into the '{cluster_name}' cluster with a "
            f"workload profile characterized as {workload_status.upper()}. "
            f"Based on synthetic operational modeling, the management recommendation is to {recommendation.lower()} "
            f"This intervention is designed to stabilize individual productivity, preserve operational quality standards, "
            f"and balance departmental delivery risks. Final personnel actions remain subject to human managerial oversight."
        )
        
        rec_record = {
            "Employee_ID": emp_id,
            "Department": dept,
            "Current_Tasks": current_tasks,
            "Recommended_Tasks": recommended_tasks,
            "Difference_Tasks": diff,
            "Workload_Status": workload_status,
            "Recommendation": recommendation,
            "Reason": reason,
            "Suggested_Actions": " | ".join(actions),
            "Risk_Factors": " | ".join(risk_factors),
            "Overall_Summary": overall_summary
        }
        rec_records.append(rec_record)
        
    df_recs = pd.DataFrame(rec_records)
    
    # Merge with df_current for full employee predictions table
    df_full_predictions = pd.merge(df_current, df_recs, on=["Employee_ID", "Department"])
    
    os.makedirs(os.path.dirname(predictions_output), exist_ok=True)
    df_full_predictions.to_csv(predictions_output, index=False)
    print(f"[SUCCESS] Saved full employee predictions to {predictions_output} ({len(df_full_predictions)} records)")
    
    os.makedirs(os.path.dirname(recommendations_output), exist_ok=True)
    df_recs.to_csv(recommendations_output, index=False)
    print(f"[SUCCESS] Saved workload recommendations to {recommendations_output} ({len(df_recs)} records)")
    
    # -------------------------------------------------------------
    # DEPARTMENT ANALYSIS (Section 25)
    # -------------------------------------------------------------
    print("\nComputing Department-Level Aggregations & Workload Pressure Analysis...")
    dept_groups = df_full_predictions.groupby("Department")
    
    dept_summary = dept_groups.agg(
        Employee_Count=("Employee_ID", "count"),
        Average_Performance=("Performance_Score", "mean"),
        Average_Productivity=("Productivity_Score", "mean"),
        Average_Overtime_Hours=("Overtime_Hours", "mean"),
        Average_Workload_Index=("Workload_Index", "mean"),
        Average_Engagement_Score=("Engagement_Score", "mean"),
        Average_Absenteeism_Rate=("Absenteeism_Rate", "mean"),
        Average_Task_Completion=("Task_Completion_Rate", "mean"),
        High_Risk_Count=("Risk_Level", lambda s: (s == "High Risk").sum()),
        Medium_Risk_Count=("Risk_Level", lambda s: (s == "Medium Risk").sum()),
        Low_Risk_Count=("Risk_Level", lambda s: (s == "Low Risk").sum()),
        Overloaded_Count=("Workload_Status", lambda s: (s == "Overloaded").sum()),
        Balanced_Count=("Workload_Status", lambda s: (s == "Balanced").sum()),
        Available_Capacity_Count=("Workload_Status", lambda s: (s == "Available Capacity").sum()),
        Total_Current_Tasks=("Current_Tasks", "sum"),
        Total_Recommended_Tasks=("Recommended_Tasks", "sum")
    ).round(1).reset_index()
    
    dept_summary["High_Risk_Percentage"] = (
        (dept_summary["High_Risk_Count"] / dept_summary["Employee_Count"]) * 100.0
    ).round(1)
    
    dept_summary["Net_Task_Discrepancy"] = (
        dept_summary["Total_Recommended_Tasks"] - dept_summary["Total_Current_Tasks"]
    )
    
    # Workload pressure score combining Overtime, Workload Index, and High-Risk %
    dept_summary["Workload_Pressure_Score"] = (
        0.40 * (dept_summary["Average_Workload_Index"]) +
        0.35 * (dept_summary["Average_Overtime_Hours"] / 35.0 * 100.0) +
        0.25 * (dept_summary["High_Risk_Percentage"])
    ).round(1)
    
    dept_summary = dept_summary.sort_values(by="Workload_Pressure_Score", ascending=False).reset_index(drop=True)
    dept_summary["Workload_Pressure_Rank"] = dept_summary.index + 1
    
    # Department-level strategic recommendations
    dept_recs = []
    for _, drow in dept_summary.iterrows():
        d_name = drow["Department"]
        hr_pct = drow["High_Risk_Percentage"]
        over_cnt = drow["Overloaded_Count"]
        avail_cnt = drow["Available_Capacity_Count"]
        
        if hr_pct >= 30.0 or over_cnt > avail_cnt:
            rec = (
                f"Urgent operational rebalancing required in {d_name}: {over_cnt} overloaded staff vs "
                f"{avail_cnt} with capacity. Reallocate tasks internally and cap overtime."
            )
        elif avail_cnt > over_cnt:
            rec = (
                f"Capacity surplus identified in {d_name}: {avail_cnt} employees can absorb cross-functional "
                f"tasks to alleviate high-pressure bank units."
            )
        else:
            rec = (
                f"{d_name} is operating at near-optimal equilibrium. Maintain workload allocation and monitor "
                f"quarterly productivity benchmarks."
            )
        dept_recs.append(rec)
        
    dept_summary["Department_Recommendation"] = dept_recs
    
    os.makedirs(os.path.dirname(department_output), exist_ok=True)
    dept_summary.to_csv(department_output, index=False)
    print(f"[SUCCESS] Saved department summary to {department_output} ({len(dept_summary)} departments)")
    
    print("\nDepartment Workload Pressure Ranking:")
    for _, drow in dept_summary.iterrows():
        print(f"  #{int(drow['Workload_Pressure_Rank'])} {drow['Department']}: Pressure {drow['Workload_Pressure_Score']} | High Risk: {drow['High_Risk_Percentage']}% | Overloaded: {int(drow['Overloaded_Count'])}")
    print("=" * 60)
    
    return df_full_predictions, dept_summary

if __name__ == "__main__":
    generate_recommendations_and_department_summary()
