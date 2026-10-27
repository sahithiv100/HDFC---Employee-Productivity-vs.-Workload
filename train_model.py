"""
HDFC Bank AI System - AI Operational Risk Model & Employee Segmentation
Trains a Random Forest Classifier to predict employee risk levels,
evaluates precision/recall/F1/confusion matrix, and performs K-Means clustering for employee segmentation.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_risk_model_and_cluster(
    input_path="data/cleaned_employee_data.csv",
    output_data_path="data/cleaned_employee_data.csv",
    model_output_path="models/productivity_risk_model.pkl",
    metrics_output_path="outputs/model_metrics.json"
):
    print("=" * 60)
    print("TRAINING AI OPERATIONAL RISK MODEL & K-MEANS SEGMENTATION")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    
    feature_cols = [
        "Overtime_Hours",
        "Overtime_Ratio",
        "Workload_Index",
        "Task_Completion_Rate",
        "Absence_Days",
        "Absenteeism_Rate",
        "Engagement_Score",
        "Quality_Score",
        "Deadline_Adherence",
        "Performance_Change",
        "Working_Hours",
        "Productivity_Score",
        "Performance_Score"
    ]
    
    target_col = "Risk_Level"
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    
    # Predictions & Evaluation
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average="macro"))
    recall_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    precision_weighted = float(precision_score(y_test, y_pred, average="weighted"))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    clf_report = classification_report(y_test, y_pred, labels=labels, output_dict=True)
    
    # Feature Importance
    importances = rf.feature_importances_
    feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp) * 100.0, 2)} for f, imp in zip(feature_cols, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    print(f"Model Performance Evaluation (Test Set N={len(y_test)}):")
    print(f"  * Accuracy:           {accuracy * 100:.2f}%")
    print(f"  * Precision (Macro):  {precision_macro * 100:.2f}%")
    print(f"  * Recall (Macro):     {recall_macro * 100:.2f}%")
    print(f"  * F1 Score (Macro):   {f1_macro * 100:.2f}%")
    print(f"  * Confusion Matrix:   {cm}")
    print("\nTop 5 Predictive Features:")
    for item in feat_imp[:5]:
        print(f"  - {item['feature']}: {item['importance']}%")
        
    # Full dataset prediction
    df["AI_Predicted_Risk"] = rf.predict(df[feature_cols])
    
    # -------------------------------------------------------------
    # K-MEANS EMPLOYEE SEGMENTATION (Section 10)
    # 4 Clusters:
    # 1. Healthy & Productive
    # 2. Overloaded & At Risk
    # 3. Under-Utilized
    # 4. Low Productivity / Support Required
    # -------------------------------------------------------------
    print("\nRunning K-Means Clustering for Employee Segmentation...")
    cluster_features = [
        "Workload_Index", "Performance_Score", "Productivity_Score",
        "Overtime_Hours", "Engagement_Score", "Absenteeism_Rate"
    ]
    
    scaler = StandardScaler()
    X_clust_scaled = scaler.fit_transform(df[cluster_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_clust_scaled)
    df["Cluster_ID"] = cluster_labels
    
    # Dynamically assign meaningful names based on cluster centroids
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=cluster_features
    )
    
    cluster_mapping = {}
    assigned_names = set()
    
    for c_id in range(4):
        c_row = centroids.iloc[c_id]
        wl = c_row["Workload_Index"]
        perf = c_row["Performance_Score"]
        ot = c_row["Overtime_Hours"]
        eng = c_row["Engagement_Score"]
        
        # Rule-based archetype mapping based on center metrics
        if wl >= 60.0 and ot >= 18.0:
            name = "Overloaded & At Risk"
            desc = "Employees facing high task volume, excessive overtime, and declining performance stability."
        elif wl < 40.0 and perf >= 70.0:
            name = "Under-Utilized"
            desc = "Employees with low assigned task volume, high available capacity, and strong potential to absorb work."
        elif perf < 68.0 or eng < 65.0:
            name = "Low Productivity / Support Required"
            desc = "Employees struggling with task delivery, deadline adherence, or disengagement needing managerial intervention."
        else:
            name = "Healthy & Productive"
            desc = "Employees operating at optimal workload equilibrium with high quality, low overtime, and steady performance."
            
        # Ensure distinct assignments if any collision
        if name in assigned_names:
            remaining_names = [
                ("Healthy & Productive", "Employees operating at optimal workload equilibrium with high quality, low overtime, and steady performance."),
                ("Overloaded & At Risk", "Employees facing high task volume, excessive overtime, and declining performance stability."),
                ("Under-Utilized", "Employees with low assigned task volume, high available capacity, and strong potential to absorb work."),
                ("Low Productivity / Support Required", "Employees struggling with task delivery, deadline adherence, or disengagement needing managerial intervention.")
            ]
            for r_name, r_desc in remaining_names:
                if r_name not in assigned_names:
                    name, desc = r_name, r_desc
                    break
                    
        assigned_names.add(name)
        cluster_mapping[c_id] = {"name": name, "description": desc}
        
    df["Cluster_Name"] = df["Cluster_ID"].map(lambda cid: cluster_mapping[cid]["name"])
    df["Cluster_Description"] = df["Cluster_ID"].map(lambda cid: cluster_mapping[cid]["description"])
    
    # Save model and artifacts
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model_payload = {
        "classifier": rf,
        "feature_cols": feature_cols,
        "labels": labels,
        "kmeans": kmeans,
        "scaler": scaler,
        "cluster_features": cluster_features,
        "cluster_mapping": cluster_mapping,
        "metrics": {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "feature_importance": feat_imp
        }
    }
    with open(model_output_path, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"[SUCCESS] Trained model package saved to {model_output_path}")
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(model_payload["metrics"], f, indent=2)
    print(f"[SUCCESS] Metrics JSON saved to {metrics_output_path}")
    
    df.to_csv(output_data_path, index=False)
    print(f"[SUCCESS] Cleaned data enriched with predictions and clusters saved to {output_data_path}")
    print("=" * 60)
    return df, model_payload

if __name__ == "__main__":
    train_risk_model_and_cluster()
