from src.risk_scoring import classify_risk


def test_classify_risk_bands() -> None:
    assert classify_risk(0.20) == "low"
    assert classify_risk(0.40) == "moderate"
    assert classify_risk(0.60) == "high"
    assert classify_risk(0.80) == "critical"
