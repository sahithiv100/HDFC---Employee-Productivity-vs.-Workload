"""
HDFC Bank AI System - Automated Employee PDF Report Generator
Generates exactly 400 individual 2-page professional corporate PDF reports (EMP001 to EMP400),
builds the Master Report Index (report_index.csv), creates README.txt, and packages the complete ZIP archive.
"""

import os
import io
import zipfile
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

# Corporate Color Palette (HDFC Bank Inspired)
HDFC_NAVY = colors.HexColor("#003366")
HDFC_BLUE = colors.HexColor("#004C8F")
HDFC_RED = colors.HexColor("#ED232A")
DARK_GRAY = colors.HexColor("#2C3E50")
LIGHT_BG = colors.HexColor("#F8F9FA")
BORDER_GRAY = colors.HexColor("#E2E8F0")

COLOR_HIGH_RISK = colors.HexColor("#D32F2F")
COLOR_MED_RISK = colors.HexColor("#F57C00")
COLOR_LOW_RISK = colors.HexColor("#388E3C")

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' and running headers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(HDFC_NAVY)
        
        # Running Top Header
        self.drawString(36, 762, "HDFC BANK — AUTONOMOUS EMPLOYEE ANALYTICS SYSTEM")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawRightString(576, 762, "CONFIDENTIAL & PROPRIETARY — ACADEMIC USE")
        
        self.setStrokeColor(BORDER_GRAY)
        self.setLineWidth(0.75)
        self.line(36, 756, 576, 756)
        
        # Running Footer
        self.line(36, 42, 576, 42)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(36, 30, "Generated on 2026-10-05 | Autonomous Risk & Workload Planning Engine")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, page_str)
        self.restoreState()


def create_trend_chart_image(emp_history_df):
    """
    Renders a crisp 6-month historical trend chart (Performance, Productivity, Completion, Overtime)
    and returns it as an in-memory PNG BytesIO object for ReportLab.
    """
    months = emp_history_df["Month"].tolist()
    # Format labels e.g. "Jan", "Feb", etc.
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"][:len(months)]
    
    perf = emp_history_df["Performance_Score"].tolist()
    prod = emp_history_df["Productivity_Score"].tolist()
    comp = emp_history_df["Task_Completion_Rate"].tolist()
    ot = emp_history_df["Overtime_Hours"].tolist()
    
    fig, ax1 = plt.subplots(figsize=(6.8, 2.2), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FAFCFE")
    
    # Primary Axis: Scores (%)
    line1 = ax1.plot(month_labels, perf, color="#003366", marker="o", linewidth=2.0, markersize=4, label="Performance Score")
    line2 = ax1.plot(month_labels, prod, color="#008080", marker="s", linewidth=1.8, markersize=4, linestyle="--", label="Productivity Score")
    line3 = ax1.plot(month_labels, comp, color="#2E7D32", marker="^", linewidth=1.8, markersize=4, linestyle=":", label="Task Completion %")
    
    ax1.set_ylabel("Score / Completion (%)", fontsize=8, fontweight="bold", color="#1A365D")
    ax1.set_ylim(20, 105)
    ax1.tick_params(axis="both", labelsize=7.5)
    ax1.grid(True, linestyle=":", alpha=0.5, color="#CBD5E0")
    
    # Secondary Axis: Overtime (Hours)
    ax2 = ax1.twinx()
    line4 = ax2.plot(month_labels, ot, color="#ED232A", marker="d", linewidth=2.0, markersize=4, label="Overtime Hours")
    ax2.set_ylabel("Overtime (Hours)", fontsize=8, fontweight="bold", color="#C53030")
    ax2.set_ylim(0, max(50, max(ot) + 10))
    ax2.tick_params(axis="y", labelsize=7.5)
    
    # Combined Legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=4, fontsize=7, frameon=True, edgecolor="#E2E8F0")
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def generate_employee_report(emp_id, emp_history_df, emp_pred_row, output_pdf_path):
    """
    Builds an individual 2-page executive report for a single employee using ReportLab.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=HDFC_NAVY,
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=DARK_GRAY,
        spaceAfter=6
    )
    
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#555555")
    )
    
    sec_heading = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=HDFC_NAVY,
        spaceBefore=6,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=DARK_GRAY
    )
    
    badge_style = ParagraphStyle(
        "Badge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        alignment=1, # Center
        textColor=colors.white
    )
    
    kpi_val_style = ParagraphStyle(
        "KPIVal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        alignment=1,
        textColor=HDFC_NAVY
    )
    
    kpi_lbl_style = ParagraphStyle(
        "KPILbl",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=8.5,
        alignment=1,
        textColor=colors.HexColor("#718096")
    )
    
    story = []
    
    # -------------------------------------------------------------------------
    # PAGE 1: Employee Overview, Status Badges, KPI Matrix, 6-Month Trend Chart
    # -------------------------------------------------------------------------
    story.append(Paragraph("HDFC Bank — Employee Productivity & Workload Risk Report", title_style))
    story.append(Paragraph(
        f"<b>Employee ID:</b> {emp_id} &nbsp;|&nbsp; <b>Department:</b> {emp_pred_row['Department']} &nbsp;|&nbsp; "
        f"<b>Analysis Period:</b> January–June 2026 &nbsp;|&nbsp; <b>Cluster:</b> {emp_pred_row.get('Cluster_Name', 'General')}",
        subtitle_style
    ))
    
    # Academic Disclaimer Box
    disclaimer_text = (
        "<b>ACADEMIC DISCLAIMER:</b> This report is generated strictly using synthetic data for academic research "
        "and algorithmic modeling. It does not represent actual HDFC Bank employee information, personal records, "
        "or official bank performance standards. Performance weights and risk models are project assumptions."
    )
    disclaimer_table = Table(
        [[Paragraph(disclaimer_text, disclaimer_style)]],
        colWidths=[540]
    )
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BORDER", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 6))
    
    # Status Badges: Risk Level, Performance, Workload Status
    risk_level = str(emp_pred_row["Risk_Level"]).upper()
    perf_cat = str(emp_pred_row.get("Performance_Category", "MODERATE")).upper()
    workload_status = str(emp_pred_row.get("Workload_Status", "BALANCED")).upper()
    
    risk_color = COLOR_HIGH_RISK if "HIGH" in risk_level else (COLOR_MED_RISK if "MED" in risk_level else COLOR_LOW_RISK)
    perf_color = COLOR_LOW_RISK if "HIGH" in perf_cat else (COLOR_MED_RISK if "MOD" in perf_cat else COLOR_HIGH_RISK)
    wl_color = COLOR_HIGH_RISK if "OVERLOADED" in workload_status else (COLOR_LOW_RISK if "BALANCED" in workload_status else HDFC_BLUE)
    
    badge_data = [
        [
            Paragraph(f"RISK LEVEL<br/><b>{risk_level}</b>", badge_style),
            Paragraph(f"PERFORMANCE<br/><b>{perf_cat}</b>", badge_style),
            Paragraph(f"WORKLOAD STATUS<br/><b>{workload_status}</b>", badge_style)
        ]
    ]
    badge_table = Table(badge_data, colWidths=[180, 180, 180])
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), risk_color),
        ("BACKGROUND", (1, 0), (1, 0), perf_color),
        ("BACKGROUND", (2, 0), (2, 0), wl_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_GRAY)
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 7))
    
    # KPI Grid (7 Key Metrics)
    perf_sc = emp_pred_row["Performance_Score"]
    prod_sc = emp_pred_row["Productivity_Score"]
    risk_sc = emp_pred_row["Risk_Score"]
    eng_sc = emp_pred_row["Engagement_Score"]
    ot_hrs = emp_pred_row["Overtime_Hours"]
    comp_rt = emp_pred_row["Task_Completion_Rate"]
    abs_dys = int(emp_pred_row["Absence_Days"])
    
    kpi_cells = [
        [
            Paragraph(f"{perf_sc:.1f}", kpi_val_style),
            Paragraph(f"{prod_sc:.1f}", kpi_val_style),
            Paragraph(f"{risk_sc:.1f}", kpi_val_style),
            Paragraph(f"{eng_sc:.1f}", kpi_val_style),
            Paragraph(f"{ot_hrs:.1f}h", kpi_val_style),
            Paragraph(f"{comp_rt:.1f}%", kpi_val_style),
            Paragraph(f"{abs_dys} d", kpi_val_style),
        ],
        [
            Paragraph("Performance", kpi_lbl_style),
            Paragraph("Productivity", kpi_lbl_style),
            Paragraph("Risk Score", kpi_lbl_style),
            Paragraph("Engagement", kpi_lbl_style),
            Paragraph("Overtime", kpi_lbl_style),
            Paragraph("Task Compl.", kpi_lbl_style),
            Paragraph("Absence Days", kpi_lbl_style),
        ]
    ]
    kpi_col_width = 540 / 7
    kpi_table = Table(kpi_cells, colWidths=[kpi_col_width] * 7)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 6))
    
    # 6-Month Performance Trend Chart
    story.append(Paragraph("Six-Month Performance & Operational Trend", sec_heading))
    chart_buf = create_trend_chart_image(emp_history_df)
    trend_img = Image(chart_buf, width=540, height=170)
    story.append(trend_img)
    story.append(Spacer(1, 4))
    
    # Dynamic Trend Explanation (Section 15)
    m1_perf = emp_history_df.iloc[0]["Performance_Score"]
    m6_perf = emp_history_df.iloc[-1]["Performance_Score"]
    m1_ot = emp_history_df.iloc[0]["Overtime_Hours"]
    m6_ot = emp_history_df.iloc[-1]["Overtime_Hours"]
    m1_comp = emp_history_df.iloc[0]["Task_Completion_Rate"]
    m6_comp = emp_history_df.iloc[-1]["Task_Completion_Rate"]
    
    perf_direction = "decreased" if m6_perf < m1_perf else ("increased" if m6_perf > m1_perf else "remained flat")
    ot_direction = "increased" if m6_ot > m1_ot else ("decreased" if m6_ot < m1_ot else "remained unchanged")
    comp_direction = "fell" if m6_comp < m1_comp else "improved"
    
    trend_narrative = (
        f"<b>Trend Analysis:</b> Over the six-month analysis period (January to June 2026), employee performance "
        f"<b>{perf_direction} from {m1_perf:.1f} to {m6_perf:.1f}</b> while monthly overtime "
        f"<b>{ot_direction} from {m1_ot:.1f} to {m6_ot:.1f} hours</b>. Task completion rate {comp_direction} from "
        f"{m1_comp:.1f}% to {m6_comp:.1f}%. "
    )
    if m6_ot >= 25.0 and m6_perf < m1_perf:
        trend_narrative += (
            "This empirical pattern confirms management's core hypothesis: increasing working hours and overtime "
            "coincide with declining output quality and task throughput due to acute fatigue."
        )
    elif m6_perf >= 85.0:
        trend_narrative += (
            "The employee exhibits consistent high operational stability with sustainable task completion, "
            "disciplined work hours, and strong delivery quality."
        )
    else:
        trend_narrative += (
            "Monthly metrics demonstrate steady operational output with minor variance within allowable operational tolerances."
        )
        
    trend_table = Table([[Paragraph(trend_narrative, body_style)]], colWidths=[540])
    trend_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BORDER", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(trend_table)
    
    # -------------------------------------------------------------------------
    # PAGE 2: Risk Factors, Workload Analysis, AI Recommendations, Assessment
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    
    story.append(Paragraph("Employee Risk Diagnostics & Operational Drivers", sec_heading))
    risk_factors_list = str(emp_pred_row.get("Risk_Factors", "")).split(" | ")
    
    rf_rows = []
    for i, rf in enumerate(risk_factors_list[:5], 1):
        rf_rows.append([
            Paragraph(f"<b>#{i}</b>", ParagraphStyle("RNum", fontName="Helvetica-Bold", fontSize=8, textColor=HDFC_RED)),
            Paragraph(rf, body_style)
        ])
    rf_table = Table(rf_rows, colWidths=[30, 510])
    rf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(rf_table)
    story.append(Spacer(1, 6))
    
    # Workload Analysis (Section 17)
    story.append(Paragraph("Workload & Capacity Analysis (Current vs Recommended)", sec_heading))
    cur_tasks = int(emp_pred_row["Current_Tasks"])
    rec_tasks = int(emp_pred_row["Recommended_Tasks"])
    diff_tasks = int(emp_pred_row["Difference_Tasks"])
    diff_label = f"+{diff_tasks} tasks (absorb capacity)" if diff_tasks > 0 else (f"{diff_tasks} tasks (reduce volume)" if diff_tasks < 0 else "0 tasks (maintain)")
    
    wl_headers = ["Metric", "Current Monthly Value", "Benchmark / Recommended", "Variance / Action"]
    wl_data = [
        [Paragraph(f"<b>{h}</b>", ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=8, textColor=colors.white)) for h in wl_headers],
        [Paragraph("Tasks Assigned", body_style), Paragraph(f"{cur_tasks} tasks", body_style), Paragraph(f"{rec_tasks} tasks", body_style), Paragraph(f"<b>{diff_label}</b>", body_style)],
        [Paragraph("Tasks Completed", body_style), Paragraph(f"{int(emp_pred_row['Tasks_Completed'])} tasks", body_style), Paragraph(f"{rec_tasks} tasks", body_style), Paragraph(f"Completion: {comp_rt:.1f}%", body_style)],
        [Paragraph("Monthly Working Hours", body_style), Paragraph(f"{emp_pred_row['Working_Hours']:.1f} hrs", body_style), Paragraph("160 - 176 hrs (standard)", body_style), Paragraph(f"Overtime: {ot_hrs:.1f} hrs", body_style)],
        [Paragraph("Workload Index (0-100)", body_style), Paragraph(f"{emp_pred_row['Workload_Index']:.1f}", body_style), Paragraph("40.0 - 55.0 (balanced)", body_style), Paragraph(f"Status: <b>{workload_status}</b>", body_style)]
    ]
    wl_table = Table(wl_data, colWidths=[135, 135, 135, 135])
    wl_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HDFC_NAVY),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(wl_table)
    story.append(Spacer(1, 6))
    
    # AI-Generated Workload Recommendation (Section 18)
    story.append(Paragraph("AI-Generated Workload Recommendation", sec_heading))
    rec_text = emp_pred_row.get("Recommendation", "Maintain current operational cadence.")
    reason_text = emp_pred_row.get("Reason", "Metrics within balanced tolerances.")
    actions_raw = emp_pred_row.get("Suggested_Actions", "Review monthly performance.")
    actions_list = actions_raw.split(" | ") if isinstance(actions_raw, str) else [str(actions_raw)]
    
    rec_cell_content = [
        Paragraph(f"<b>Primary Directive:</b> {rec_text}", ParagraphStyle("RD", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=HDFC_NAVY)),
        Spacer(1, 3),
        Paragraph(f"<b>Operational Rationale:</b> {reason_text}", body_style),
        Spacer(1, 3),
        Paragraph("<b>Suggested Managerial Action Steps:</b>", ParagraphStyle("SH", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=DARK_GRAY))
    ]
    for act in actions_list[:4]:
        rec_cell_content.append(Paragraph(f"• {act}", body_style))
        
    rec_box = Table([[rec_cell_content]], colWidths=[540])
    rec_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(rec_box)
    story.append(Spacer(1, 6))
    
    # Overall Assessment & Executive Summary (Section 19)
    story.append(Paragraph("Overall Assessment & Synthesis", sec_heading))
    summary_text = emp_pred_row.get("Overall_Summary", "")
    
    eval_matrix = [
        [
            Paragraph(f"<b>Performance:</b> {perf_cat}", body_style),
            Paragraph(f"<b>Risk Rating:</b> {risk_level}", body_style),
            Paragraph(f"<b>Workload:</b> {workload_status}", body_style),
            Paragraph(f"<b>Engagement:</b> {eng_sc:.1f}/100", body_style),
        ]
    ]
    eval_table = Table(eval_matrix, colWidths=[135, 135, 135, 135])
    eval_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(eval_table)
    story.append(Spacer(1, 4))
    
    sum_table = Table([[Paragraph(summary_text, body_style)]], colWidths=[540])
    sum_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(sum_table)
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)


def generate_all_employee_reports(
    cleaned_data_path="data/cleaned_employee_data.csv",
    predictions_path="outputs/employee_predictions.csv",
    output_dir="outputs/employee_reports"
):
    print("=" * 60)
    print("STARTING BATCH GENERATION OF 400 EMPLOYEE PDF REPORTS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    df_clean = pd.read_csv(cleaned_data_path)
    df_preds = pd.read_csv(predictions_path)
    
    employees = sorted(df_preds["Employee_ID"].unique())
    total_emps = len(employees)
    print(f"[INFO] Loaded {total_emps} unique employees. Beginning automated generation...")
    
    generated_count = 0
    error_count = 0
    report_index_records = []
    
    # Pre-index history by employee for high speed lookup
    history_groups = {emp_id: group.sort_values(by="Month") for emp_id, group in df_clean.groupby("Employee_ID")}
    preds_indexed = {row["Employee_ID"]: row for _, row in df_preds.iterrows()}
    
    start_time = datetime.now()
    
    for idx, emp_id in enumerate(employees, 1):
        pdf_filename = f"{emp_id}_Report.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        try:
            emp_pred = preds_indexed[emp_id]
            if not (os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0):
                emp_hist = history_groups[emp_id]
                generate_employee_report(emp_id, emp_hist, emp_pred, pdf_path)
            
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                generated_count += 1
            else:
                error_count += 1
                print(f"[WARNING] Generated PDF empty or missing for {emp_id}")
                
            report_index_records.append({
                "Employee_ID": emp_id,
                "Department": emp_pred["Department"],
                "Risk_Level": emp_pred["Risk_Level"],
                "Risk_Score": emp_pred["Risk_Score"],
                "Performance_Score": emp_pred["Performance_Score"],
                "Workload_Status": emp_pred["Workload_Status"],
                "Recommended_Action": emp_pred["Recommendation"],
                "PDF_File": pdf_filename
            })
            
            if idx % 50 == 0 or idx == total_emps:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"  * Generated {idx}/{total_emps} PDFs ({rate:.1f} reports/sec)...")
                
        except Exception as e:
            error_count += 1
            print(f"[ERROR] Failed to generate PDF for {emp_id}: {str(e)}")
            traceback.print_exc()
            
    # Save Master Report Index (Section 22)
    index_path = os.path.join(output_dir, "report_index.csv")
    df_index = pd.DataFrame(report_index_records)
    df_index.to_csv(index_path, index=False)
    print(f"\n[SUCCESS] Master Report Index saved to {index_path} ({len(df_index)} records)")
    
    # Save README.txt (Section 22)
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("HDFC BANK — AUTONOMOUS EMPLOYEE ANALYTICS SYSTEM\n")
        f.write("INDIVIDUAL EMPLOYEE PERFORMANCE & WORKLOAD RISK REPORTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Synthetic Employees: 400\n")
        f.write(f"Total Generated PDF Reports: {generated_count}\n")
        f.write(f"Generation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("FILE NAMING CONVENTION:\n")
        f.write("  - EMP001_Report.pdf to EMP400_Report.pdf\n\n")
        f.write("FOLDER CONTENTS:\n")
        f.write("  - report_index.csv: Master directory mapping every employee to risk, performance, and actions.\n")
        f.write("  - HDFC_Bank_400_Employee_Reports.zip: Complete consolidated archive of all 400 reports.\n")
        f.write("  - README.txt: This structural guide.\n\n")
        f.write("IMPORTANT ACADEMIC DISCLAIMER:\n")
        f.write("  All employee data, metrics, performance scores, and risk classifications contained\n")
        f.write("  in these reports are completely synthetic academic models created for algorithmic\n")
        f.write("  workload balancing research. They do NOT reflect actual HDFC Bank employees or standards.\n")
        f.write("=" * 70 + "\n")
    print(f"[SUCCESS] Documentation saved to {readme_path}")
    
    # Create consolidated ZIP file (Section 24)
    zip_path = os.path.join(output_dir, "HDFC_Bank_400_Employee_Reports.zip")
    print(f"\nPackaging all {generated_count} employee PDF reports into ZIP archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for emp_id in employees:
            fname = f"{emp_id}_Report.pdf"
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=fname)
        # Also include index and readme in the zip
        if os.path.exists(index_path):
            zipf.write(index_path, arcname="report_index.csv")
        if os.path.exists(readme_path):
            zipf.write(readme_path, arcname="README.txt")
            
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[SUCCESS] Consolidated ZIP archive created: {zip_path} ({zip_size_mb:.2f} MB)")
    
    print(f"\nFinal Generation Summary:")
    print(f"  * Total Expected:  400")
    print(f"  * Total Generated: {generated_count}")
    print(f"  * Errors:          {error_count}")
    print("=" * 60)
    
    return generated_count

if __name__ == "__main__":
    generate_all_employee_reports()
