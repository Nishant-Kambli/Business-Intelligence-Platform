"""Executive report text and CSV, Excel, and PDF export helpers."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape
import zipfile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from .analytics import attrition_by, executive_insights, field_distribution, workforce_kpis

plt.style.use("default")


def executive_report(frame: pd.DataFrame, config: Any) -> dict[str, Any]:
    """Build a concise descriptive report from fields available in this dataset."""
    kpis = workforce_kpis(frame)
    summary = [f"Employees represented: {kpis['employees']:,}."]
    if kpis["departments"] is not None:
        summary.append(f"Departments represented: {kpis['departments']}.")
    if kpis["average_age"] is not None:
        summary.append(f"Average age: {kpis['average_age']:.1f}.")
    if kpis["average_salary"] is not None:
        summary.append(f"Average salary in dataset units: {kpis['average_salary']:,.0f}.")
    if kpis["attrition_rate"] is not None:
        summary.append(f"Historical attrition rate: {kpis['attrition_rate']:.1f}%.")
    for field, label in (("engagement_score", "Average engagement score"), ("satisfaction_score", "Average satisfaction score"), ("work_life_balance_score", "Average work-life balance score")):
        if field in frame:
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            if not values.empty:
                summary.append(f"{label}: {values.mean():.2f} on the source scale.")
                break
    if "training_outcome" in frame:
        summary.append(f"Training outcome records: {int(frame['training_outcome'].notna().sum()):,}.")
    dimensions = [field for field in (config.department_column, config.role_column) if field and field in frame]
    tables = {}
    if config.target_column and config.target_column in frame:
        for dimension in dimensions:
            tables[dimension] = attrition_by(frame, dimension).sort_values("Attrition rate", ascending=False)
    insights = executive_insights(frame, config.satisfaction_columns)
    charts = []
    chart_fields = list(dict.fromkeys((
        config.department_column, config.salary_column, "age", "experience", config.target_column,
        *config.satisfaction_columns, "performance_score", "employee_rating", "performance_rating",
        "training_program", "training_type", "training_outcome", "training_cost", "employment_status",
    )))
    for field in chart_fields:
        if field and field in frame:
            data = field_distribution(frame, field, limit=12)
            if not data.empty:
                charts.append({"title": f"{field.replace('_', ' ').title()} distribution", "data": data})
    recommendations = []
    if tables:
        group_name, attrition_table = next(iter(tables.items()))
        if not attrition_table.empty:
            highest = attrition_table.iloc[0]
            recommendations.append(f"Review local context for {highest['Group']} with HR partners; this group has the highest observed attrition rate in the snapshot.")
    if config.salary_column and config.salary_column in frame:
        recommendations.append("Review pay distributions alongside role and tenure before making compensation decisions.")
    if config.satisfaction_columns and any(field in frame for field in config.satisfaction_columns):
        recommendations.append("Use employee feedback to understand engagement scores and identify supportive follow-up options.")
    if "training_outcome" in frame:
        recommendations.append("Review training outcomes alongside program, duration, and employee context before changing learning plans.")
    if any(field in frame for field in ("performance_score", "employee_rating", "performance_rating")):
        recommendations.append("Interpret performance ratings with their source definitions and review context with managers before making workforce decisions.")
    if not recommendations:
        recommendations.append("Collect consistent workforce snapshots and employee feedback to guide future review.")
    recommendations.append("Treat descriptive patterns as prompts for investigation; they do not establish causes or individual risk.")
    return {
        "title": "Executive Workforce Report",
        "summary": summary,
        "insights": insights,
        "tables": tables,
        "charts": charts,
        "recommendations": recommendations,
        "limitations": "This report summarizes one dataset snapshot. It cannot establish change over time or causal effects.",
    }


def report_rows(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten report narrative into a portable section/item table."""
    rows = []
    for section in ("summary", "insights", "recommendations"):
        for item in report[section]:
            rows.append({"Section": section.title(), "Item": item})
    for name, table in report.get("tables", {}).items():
        for record in table.to_dict(orient="records"):
            rows.append({"Section": f"Historical attrition by {name}", "Item": "; ".join(f"{key}: {value}" for key, value in record.items())})
    for chart in report.get("charts", []):
        for record in chart["data"].to_dict(orient="records"):
            rows.append({"Section": chart["title"], "Item": "; ".join(f"{key}: {value}" for key, value in record.items())})
    rows.append({"Section": "Limitations", "Item": report["limitations"]})
    return pd.DataFrame(rows)


def _cell(value: Any, row: int, column: int) -> str:
    cell_ref = f"{chr(64 + column)}{row}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def excel_bytes(report: dict[str, Any]) -> bytes:
    """Create a minimal standards-compliant XLSX workbook using only stdlib ZIP/XML."""
    output = BytesIO()
    rows = [[report["title"]], ["Section", "Item"]]
    rows.extend([["Summary", item] for item in report["summary"]])
    rows.extend([["Key insight", item] for item in report["insights"]])
    for name, table in report["tables"].items():
        rows.append([f"Historical attrition by {name}"])
        rows.append(list(table.columns))
        rows.extend(table.astype(object).where(pd.notna(table), "").values.tolist())
    for chart in report.get("charts", []):
        rows.append([chart["title"]])
        rows.append(list(chart["data"].columns))
        rows.extend(chart["data"].astype(object).where(pd.notna(chart["data"]), "").values.tolist())
    rows.extend([["Recommendation", item] for item in report["recommendations"]])
    rows.append(["Limitations", report["limitations"]])
    sheet_rows = []
    for row_idx, values in enumerate(rows, 1):
        sheet_rows.append(f'<row r="{row_idx}">' + "".join(_cell(value, row_idx, col_idx) for col_idx, value in enumerate(values, 1)) + "</row>")
    sheet = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(sheet_rows) + "</sheetData></worksheet>"
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Executive Report" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml": sheet,
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return output.getvalue()


def pdf_bytes(report: dict[str, Any]) -> bytes:
    """Render a narrative page followed by dataset-driven chart pages."""
    output = BytesIO()
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        y = 0.95
        fig.text(0.08, y, report["title"], fontsize=20, weight="bold", color="#142b35")
        y -= 0.07
        for heading, items in (("Summary", report["summary"]), ("Key Insights", report["insights"]), ("Recommendations", report["recommendations"])):
            fig.text(0.08, y, heading, fontsize=13, weight="bold", color="#087e8b")
            y -= 0.035
            for item in items:
                fig.text(0.10, y, "• " + item, fontsize=10, color="#142b35", wrap=True)
                y -= 0.045
            y -= 0.02
        fig.text(0.08, max(y, 0.08), "Limitations", fontsize=13, weight="bold", color="#087e8b")
        fig.text(0.08, max(y - 0.04, 0.03), report["limitations"], fontsize=9, color="#60747c", wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for chart in report.get("charts", []):
            data = chart["data"]
            if data.empty or "Value" not in data or "Employees" not in data:
                continue
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            fig.set_facecolor("white")
            ax.set_facecolor("white")
            plot = data.sort_values("Employees")
            ax.barh(plot["Value"].astype(str), plot["Employees"], color="#087e8b")
            ax.set_title(chart["title"], color="#142b35", fontsize=16, pad=18)
            ax.set_xlabel("Employees")
            ax.tick_params(colors="#1f2937")
            ax.xaxis.label.set_color("#1f2937")
            ax.yaxis.label.set_color("#1f2937")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#cbd5e1")
            ax.grid(axis="x", color="#e2e8f0", alpha=1)
            ax.set_axisbelow(True)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return output.getvalue()
