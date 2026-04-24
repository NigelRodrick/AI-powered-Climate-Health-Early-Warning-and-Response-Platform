from __future__ import annotations

from typing import Any


TEMPLATES = {
    "critical": {
        "clinic": "[CRITICAL] {city}: Outbreak risk score {risk_score}. Activate surge staff, pediatric stock buffer, and emergency triage now.",
        "community": "[CRITICAL ALERT] {city}: High disease risk detected. Follow prevention guidance immediately and report severe symptoms early.",
        "local_government": "[CRITICAL] {city} ({region}) requires immediate response. Trigger coordination protocol and deploy rapid support teams.",
    },
    "high": {
        "clinic": "[HIGH] {city}: Elevated outbreak probability ({outbreak_probability}). Verify staff roster, medicine stock, and referral readiness.",
        "community": "[HIGH ALERT] {city}: Increased disease risk. Reinforce hygiene, safe water, and early care-seeking.",
        "local_government": "[HIGH] {city} risk increasing. Prioritize preparedness actions and targeted risk communication.",
    },
    "moderate": {
        "clinic": "[MODERATE] {city}: Continue surveillance and validate reporting quality this week.",
        "community": "[NOTICE] {city}: Moderate health risk. Maintain prevention measures and monitor symptoms.",
        "local_government": "[MODERATE] {city}: Keep routine readiness checks active and monitor trend signals.",
    },
    "low": {
        "clinic": "[LOW] {city}: Maintain routine monitoring.",
        "community": "[LOW] {city}: Continue standard prevention practices.",
        "local_government": "[LOW] {city}: No escalation needed; keep baseline surveillance.",
    },
}


def generate_message_templates(alerts: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for alert in alerts:
        level = alert.get("risk_level", "low")
        level_templates = TEMPLATES.get(level, TEMPLATES["low"])
        for audience, template in level_templates.items():
            rows.append(
                {
                    "city": str(alert.get("city", "")),
                    "region": str(alert.get("region", "")),
                    "risk_level": level.upper(),
                    "audience": audience,
                    "message": template.format(
                        city=alert.get("city", "Unknown"),
                        region=alert.get("region", "Unknown"),
                        risk_score=alert.get("risk_score", "N/A"),
                        outbreak_probability=alert.get("outbreak_probability", "N/A"),
                    ),
                }
            )
    return rows
