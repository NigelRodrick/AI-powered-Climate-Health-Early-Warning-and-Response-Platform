from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf(output_path: Path) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story = []
    story.append(Paragraph("System Usage and Goal Achievement Report", title))
    story.append(Paragraph("Project: AI-powered Climate-Health Early Warning and Response Platform", body))
    story.append(Paragraph(f"Organization: Fidinsky Tech Solutions | Date: {datetime.now():%Y-%m-%d}", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Executive Overview", h2))
    story.append(
        Paragraph(
            "This report explains how the prototype platform is used by frontline stakeholders and how its design "
            "supports measurable progress toward child-health and climate resilience goals. The system combines climate "
            "signals and health-service indicators, computes localized risk levels, and provides action-oriented alerts.",
            body,
        )
    )

    story.append(Paragraph("2. Problem Context and Intended Outcome", h2))
    story.append(
        Paragraph(
            "In climate-vulnerable settings, delayed information often causes late response to outbreaks, stock pressure, "
            "and disruptions in child-focused services. The platform addresses this by shifting teams from reactive response "
            "to early warning and early action.",
            body,
        )
    )

    story.append(Paragraph("3. Current System Components", h2))
    components = [
        ["Component", "Current Function"],
        ["Data ingestion", "Reads climate-health indicators from structured datasets and validates required fields."],
        ["Risk scoring", "Calculates transparent weighted risk score and classifies into low/moderate/high/critical."],
        ["Alerting engine", "Generates role-based actions for clinics, communities, and local government users."],
        ["API service", "Accepts new records via POST /score and returns standardized alert outputs."],
        ["Dashboard", "Displays counts by risk level and detailed recommended actions by location and period."],
    ]
    table = Table(components, colWidths=[5.2 * cm, 11 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94A3B8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("4. System Usage Workflow", h2))
    usage_steps = [
        "Step 1: Data inputs are collected from climate observations and facility/community health indicators.",
        "Step 2: System normalizes bounded indicators and computes a composite risk score for each location.",
        "Step 3: Score is mapped to risk class (low, moderate, high, critical).",
        "Step 4: Alerting module produces audience-specific recommendations.",
        "Step 5: Dashboard and API outputs support rapid planning, communication, and local response decisions.",
    ]
    for step in usage_steps:
        story.append(Paragraph(f"- {step}", body))

    story.append(Paragraph("5. How the System Achieves Project Goals", h2))
    goal_rows = [
        ["Goal", "Mechanism in System", "Expected Improvement"],
        ["Earlier detection", "Risk scoring from combined climate-health signals", "Improved warning lead time"],
        ["Faster action", "Predefined response recommendations by risk level", "Reduced delay to frontline action"],
        ["Service continuity", "Clinic and supply-focused alert actions", "Lower disruption during shocks"],
        ["Scalable adoption", "Open-source modular architecture and API", "Replication across regions"],
        ["Child-centered impact", "Prioritization of vulnerable hotspots", "Reduced avoidable child health harm"],
    ]
    goal_table = Table(goal_rows, colWidths=[4.2 * cm, 6.4 * cm, 5.6 * cm])
    goal_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCFCE7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94A3B8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(goal_table)

    story.append(Paragraph("6. Monitoring and Evaluation Framework", h2))
    story.append(
        Paragraph(
            "The implementation tracks quantitative and qualitative indicators to validate effectiveness. Core indicators include: "
            "risk-alert lead time, proportion of alerts acted on within target time, facility readiness score, and service continuity "
            "during climate events. Feedback loops from users are incorporated to refine thresholds, messaging, and usability.",
            body,
        )
    )

    story.append(Paragraph("7. Risks and Mitigation", h2))
    story.append(Paragraph("- Data quality variance: mitigated through validation checks and local calibration.", body))
    story.append(Paragraph("- Connectivity constraints: mitigated via low-bandwidth delivery channels and offline workflow support.", body))
    story.append(Paragraph("- Model trust and adoption: mitigated through transparent scoring logic and human-in-the-loop oversight.", body))
    story.append(Paragraph("- Privacy concerns: mitigated through non-PII datasets in prototype and privacy-by-design controls for pilots.", body))

    story.append(Paragraph("8. 12-Month Implementation Milestones", h2))
    story.append(
        Paragraph(
            "Months 1-3: harden MVP and finalize baseline metrics. Months 4-6: deploy live pilot and train users. "
            "Months 7-9: evaluate performance and optimize models. Months 10-12: prepare multi-site scale-up and "
            "release strengthened open-source implementation materials.",
            body,
        )
    )

    story.append(Paragraph("9. Conclusion", h2))
    story.append(
        Paragraph(
            "The platform is technically aligned with the objective of protecting children from climate-driven health risks "
            "through early warning, data-guided coordination, and scalable open-source deployment. The current prototype "
            "provides a credible implementation base for pilot execution and measurable impact validation.",
            body,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    output = root / "docs" / "System_Usage_and_Goal_Achievement_Report.pdf"
    build_pdf(output)
    print(output)
