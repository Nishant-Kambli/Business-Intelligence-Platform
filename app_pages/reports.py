"""Executive workforce report generation and file downloads."""

import streamlit as st

from ai_bi.app_state import active_dataset
from ai_bi.report_service import executive_report, excel_bytes, pdf_bytes, report_rows
from ai_bi.utils import page_header, setup_page


@st.cache_data(show_spinner=False)
def cached_pdf(report_data):
    return pdf_bytes(report_data)


@st.cache_data(show_spinner=False)
def cached_excel(report_data):
    return excel_bytes(report_data)


setup_page("Executive Reports")
frame, config, report, source_name = active_dataset()
page_header("People Intelligence / Reports", "Executive Workforce Report", "Create a concise, shareable summary from the selected dataset.")

report_data = executive_report(frame, config)
st.markdown("### Summary")
for item in report_data["summary"]:
    st.write(f"• {item}")
st.markdown("### Key Insights")
for item in report_data["insights"]:
    st.info(item)
st.markdown("### Workforce Charts")
for chart in report_data.get("charts", []):
    st.markdown(f"#### {chart['title']}")
    st.bar_chart(chart["data"], x="Value", y="Employees", alt=chart["title"])
for dimension, table in report_data["tables"].items():
    st.markdown(f"### Historical attrition by {dimension.replace('_', ' ')}")
    st.dataframe(table, hide_index=True, alt=f"Historical attrition rates grouped by {dimension}.")
st.markdown("### Recommendations")
for item in report_data["recommendations"]:
    st.write(f"• {item}")
st.caption(report_data["limitations"])

rows = report_rows(report_data)
csv_data = rows.to_csv(index=False).encode("utf-8-sig")
st.markdown("### Export report")
left, middle, right = st.columns(3)
left.download_button("Download CSV", csv_data, "executive_workforce_report.csv", "text/csv", type="primary")
middle.download_button("Download Excel", cached_excel(report_data), "executive_workforce_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
right.download_button("Download PDF", cached_pdf(report_data), "executive_workforce_report.pdf", "application/pdf")
