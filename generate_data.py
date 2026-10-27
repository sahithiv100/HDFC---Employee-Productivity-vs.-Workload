"""
HDFC Bank AI System - Synthetic Employee Data Generation
Generates realistic monthly data for 400 synthetic employees across 6 months (2,400 records)
with realistic patterns reflecting 5 distinct operational archetypes.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_data(output_path="data/synthetic_employee_data.csv", seed=42):
    np.random.seed(seed)
    
    # 400 Unique Synthetic Employees
    num_employees = 400
    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    num_months = len(months)
    
    # Realistic Department Distribution (Total = 400)
    departments = [
        ("Retail Banking", 80),
        ("Customer Service", 70),
        ("Operations", 80),
        ("Credit & Loans", 60),
        ("Corporate Banking", 40),
        ("Digital Banking", 40),
        ("Risk & Compliance", 30)
    ]
    
    dept_assignments = []
    for dept_name, count in departments:
        dept_assignments.extend([dept_name] * count)
    
    # Shuffle department assignments reproducibly
    np.random.shuffle(dept_assignments)
    
    # 5 Realistic Archetypes:
    # 1: Healthy & Productive (120 employees = 30%)
    # 2: Overloaded (100 employees = 25%)
    # 3: Under-utilized (60 employees = 15%)
    # 4: Productivity Decline (60 employees = 15%)
    # 5: High Performer (60 employees = 15%)
    archetype_counts = [
        ("Healthy & Productive", 120),
        ("Overloaded", 100),
        ("Under-utilized", 60),
        ("Productivity Decline", 60),
        ("High Performer", 60)
    ]
    
    archetype_assignments = []
    for arch_name, count in archetype_counts:
        archetype_assignments.extend([arch_name] * count)
    np.random.shuffle(archetype_assignments)
    
    records = []
    
    for emp_idx in range(1, num_employees + 1):
        emp_id = f"EMP{emp_idx:03d}"
        dept = dept_assignments[emp_idx - 1]
        archetype = archetype_assignments[emp_idx - 1]
        
        # Base baseline parameters per employee
        if archetype == "Healthy & Productive":
            base_work_hours = np.random.uniform(162, 172)
            base_ot = np.random.uniform(3, 8)
            base_tasks = np.random.randint(42, 50)
            base_comp_ratio = np.random.uniform(0.90, 0.96)
            base_absent = np.random.choice([0, 1, 2], p=[0.7, 0.25, 0.05])
            base_engagement = np.random.uniform(82, 92)
            base_quality = np.random.uniform(84, 94)
            base_adherence = np.random.uniform(86, 96)
            base_prev_perf = np.random.uniform(82, 90)
            
            # Month over month trend
            work_hours_trend = np.random.uniform(-0.5, 0.5)
            ot_trend = np.random.uniform(-0.2, 0.2)
            tasks_trend = np.random.uniform(-0.5, 0.5)
            comp_trend = np.random.uniform(0.002, 0.008)
            engagement_trend = np.random.uniform(0.2, 0.8)
            quality_trend = np.random.uniform(0.2, 0.6)
            adherence_trend = np.random.uniform(0.1, 0.5)
            
        elif archetype == "Overloaded":
            base_work_hours = np.random.uniform(175, 185)
            base_ot = np.random.uniform(18, 26)
            base_tasks = np.random.randint(60, 75)
            base_comp_ratio = np.random.uniform(0.85, 0.90)
            base_absent = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
            base_engagement = np.random.uniform(75, 82)
            base_quality = np.random.uniform(80, 88)
            base_adherence = np.random.uniform(82, 88)
            base_prev_perf = np.random.uniform(76, 85)
            
            # Worsening trend: overtime & hours spike, completion & engagement drop
            work_hours_trend = np.random.uniform(5.0, 7.5)   # up to 210-220h
            ot_trend = np.random.uniform(3.5, 5.5)           # up to 35-48h
            tasks_trend = np.random.uniform(1.5, 3.0)        # tasks continue piling
            comp_trend = np.random.uniform(-0.045, -0.025)   # drop to 65-72%
            engagement_trend = np.random.uniform(-4.5, -2.5) # drops towards 50-60
            quality_trend = np.random.uniform(-3.5, -1.8)    # drops towards 65-72
            adherence_trend = np.random.uniform(-4.0, -2.0)  # drops towards 62-70
            
        elif archetype == "Under-utilized":
            base_work_hours = np.random.uniform(155, 164)
            base_ot = np.random.uniform(0, 3)
            base_tasks = np.random.randint(22, 32)
            base_comp_ratio = np.random.uniform(0.94, 0.99)
            base_absent = np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])
            base_engagement = np.random.uniform(72, 84)
            base_quality = np.random.uniform(85, 94)
            base_adherence = np.random.uniform(88, 96)
            base_prev_perf = np.random.uniform(76, 86)
            
            work_hours_trend = np.random.uniform(-0.4, 0.4)
            ot_trend = np.random.uniform(-0.1, 0.1)
            tasks_trend = np.random.uniform(-0.3, 0.5)
            comp_trend = np.random.uniform(-0.002, 0.002)
            engagement_trend = np.random.uniform(-0.5, 0.5)
            quality_trend = np.random.uniform(0.0, 0.5)
            adherence_trend = np.random.uniform(0.0, 0.5)
            
        elif archetype == "Productivity Decline":
            base_work_hours = np.random.uniform(165, 175)
            base_ot = np.random.uniform(10, 18)
            base_tasks = np.random.randint(46, 56)
            base_comp_ratio = np.random.uniform(0.82, 0.88)
            base_absent = np.random.choice([1, 2], p=[0.6, 0.4])
            base_engagement = np.random.uniform(76, 84)
            base_quality = np.random.uniform(80, 88)
            base_adherence = np.random.uniform(80, 88)
            base_prev_perf = np.random.uniform(78, 86)
            
            # Steep decline in engagement, performance, completion; rising absence
            work_hours_trend = np.random.uniform(3.0, 5.5)
            ot_trend = np.random.uniform(2.5, 4.0)
            tasks_trend = np.random.uniform(0.0, 1.5)
            comp_trend = np.random.uniform(-0.055, -0.035)   # drops to 52-64%
            engagement_trend = np.random.uniform(-5.5, -3.5) # drops to 40-52
            quality_trend = np.random.uniform(-4.5, -2.5)    # drops to 55-66
            adherence_trend = np.random.uniform(-5.0, -3.0)  # drops to 50-62
            
        else: # High Performer
            base_work_hours = np.random.uniform(168, 178)
            base_ot = np.random.uniform(6, 12)
            base_tasks = np.random.randint(48, 58)
            base_comp_ratio = np.random.uniform(0.95, 0.99)
            base_absent = np.random.choice([0, 1], p=[0.85, 0.15])
            base_engagement = np.random.uniform(88, 96)
            base_quality = np.random.uniform(90, 98)
            base_adherence = np.random.uniform(92, 98)
            base_prev_perf = np.random.uniform(88, 96)
            
            work_hours_trend = np.random.uniform(0.0, 0.8)
            ot_trend = np.random.uniform(-0.2, 0.5)
            tasks_trend = np.random.uniform(0.2, 1.0)
            comp_trend = np.random.uniform(0.001, 0.005)
            engagement_trend = np.random.uniform(0.3, 0.9)
            quality_trend = np.random.uniform(0.2, 0.6)
            adherence_trend = np.random.uniform(0.2, 0.5)
            
        # Generate 6 monthly snapshots
        prev_perf_tracker = base_prev_perf
        for m_idx, month_str in enumerate(months):
            m = m_idx  # 0 to 5
            
            wh = base_work_hours + work_hours_trend * m + np.random.normal(0, 1.5)
            ot = max(0.0, base_ot + ot_trend * m + np.random.normal(0, 1.0))
            tasks_assigned = int(np.round(base_tasks + tasks_trend * m + np.random.normal(0, 1.0)))
            tasks_assigned = max(15, tasks_assigned)
            
            comp_ratio = np.clip(base_comp_ratio + comp_trend * m + np.random.normal(0, 0.015), 0.35, 1.0)
            tasks_completed = int(np.round(tasks_assigned * comp_ratio))
            tasks_completed = min(tasks_assigned, max(5, tasks_completed))
            
            # Absence days
            if archetype in ["Overloaded", "Productivity Decline"]:
                absent_days = int(np.clip(base_absent + int(m * 0.7) + np.random.choice([0, 1], p=[0.6, 0.4]), 0, 7))
            else:
                absent_days = int(np.clip(base_absent + np.random.choice([0, 1], p=[0.8, 0.2]), 0, 4))
                
            eng = np.clip(base_engagement + engagement_trend * m + np.random.normal(0, 1.5), 30.0, 99.0)
            qual = np.clip(base_quality + quality_trend * m + np.random.normal(0, 1.5), 40.0, 99.0)
            adh = np.clip(base_adherence + adherence_trend * m + np.random.normal(0, 1.5), 35.0, 99.0)
            
            # Calculate standard performance score for this month according to Section 7 weights:
            # 30% Task Completion, 25% Quality, 20% Deadline Adherence, 15% Engagement, 10% Previous Performance
            task_comp_pct = (tasks_completed / tasks_assigned) * 100.0
            current_perf = (
                0.30 * task_comp_pct +
                0.25 * qual +
                0.20 * adh +
                0.15 * eng +
                0.10 * prev_perf_tracker
            )
            current_perf = np.clip(current_perf, 30.0, 99.5)
            
            record = {
                "Employee_ID": emp_id,
                "Department": dept,
                "Month": month_str,
                "Working_Hours": round(float(wh), 1),
                "Overtime_Hours": round(float(ot), 1),
                "Tasks_Assigned": int(tasks_assigned),
                "Tasks_Completed": int(tasks_completed),
                "Absence_Days": int(absent_days),
                "Engagement_Score": round(float(eng), 1),
                "Quality_Score": round(float(qual), 1),
                "Deadline_Adherence": round(float(adh), 1),
                "Previous_Performance": round(float(prev_perf_tracker), 1),
                "Performance_Score": round(float(current_perf), 1),
                "Archetype": archetype  # helpful synthetic ground truth for validation
            }
            records.append(record)
            
            # Update previous performance tracker for next month
            prev_perf_tracker = current_perf

    df = pd.DataFrame(records)
    
    # Introduce controlled anomalies to demonstrate automated data cleaning pipeline
    # 1. A couple of missing values in non-key columns (e.g. Engagement_Score or Overtime_Hours)
    df.loc[15, "Engagement_Score"] = np.nan
    df.loc[45, "Overtime_Hours"] = np.nan
    df.loc[98, "Quality_Score"] = np.nan
    
    # 2. A duplicate row
    dup_row = df.iloc[[20]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    
    # 3. An impossible working hours outlier & a negative overtime value
    df.loc[72, "Working_Hours"] = 850.0  # impossible monthly hours
    df.loc[110, "Overtime_Hours"] = -12.5 # negative hours
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Generated synthetic dataset with {len(df)} records (including controlled test anomalies) saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
