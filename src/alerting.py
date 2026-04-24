from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class AlertAction:
    audience: str
    action: str


RISK_ACTIONS = {
    "critical": [
        AlertAction("clinic", "Pre-position pediatric supplies and activate surge roster."),
        AlertAction("community", "Send urgent risk communication and prevention guidance."),
        AlertAction("local_government", "Trigger emergency coordination check-in within 6 hours."),
    ],
    "high": [
        AlertAction("clinic", "Validate stock buffer and staffing for the next 7 days."),
        AlertAction("community", "Push prevention messages through trusted channels."),
        AlertAction("local_government", "Review hotspot list and update response plan."),
    ],
    "moderate": [
        AlertAction("clinic", "Monitor indicators and verify reporting continuity."),
        AlertAction("community", "Share practical prevention tips for households."),
    ],
    "low": [
        AlertAction("clinic", "Continue routine surveillance and weekly review."),
    ],
}


def build_alerts(scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for _, row in scored_df.iterrows():
        level = row["risk_level"]
        actions = RISK_ACTIONS[level]
        alerts.append(
            {
                "location_id": row["location_id"],
                "week": row["week"],
                "risk_score": row["risk_score"],
                "risk_level": level,
                "actions": [{"audience": a.audience, "action": a.action} for a in actions],
            }
        )
    return alerts
