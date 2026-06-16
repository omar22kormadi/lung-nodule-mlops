"""
alerts.py
=========
Checks the latest drift report and prints/logs alert messages
if any monitored metric exceeds thresholds.

Designed to be run periodically (e.g., cron, scheduled task, or manually).
Extend send_alert() with email/Slack/Teams notifications as needed.

Run:
    python alerts.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

from drift_detection import detect_drift

# ---------------------------------------------------------------------------
# Config — adjust thresholds here
# ---------------------------------------------------------------------------
ALERT_RULES = {
    # Flag if KS p-value < 0.05 (configured inside drift_detection.py)
    "yolo_confidence":  {"enabled": True},
    "mal_probability":  {"enabled": True},
    "diameter_mm":      {"enabled": True},
    "malignant_rate":   {"enabled": True, "max_delta": 0.15},
}

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
ALERT_LOG_PATH = PROJECT_ROOT / "data" / "monitoring" / "alerts.log"


# ---------------------------------------------------------------------------
# Alert sender — extend this with email / Slack / Teams
# ---------------------------------------------------------------------------
def send_alert(message: str) -> None:
    """
    Sends an alert. Currently prints to stdout and appends to alerts.log.
    To add email/Slack: add your notification logic here.
    """
    timestamp = datetime.datetime.utcnow().isoformat()
    full_msg = f"[{timestamp}] ALERT: {message}"

    print(f"\n🚨  {full_msg}")

    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def run_alerts() -> None:
    print("[Alerts] Running drift checks...")
    report = detect_drift(n=100)

    if report["status"] == "no_data":
        print("[Alerts] No data yet — skipping.")
        return

    triggered = False
    for metric, result in report.get("checks", {}).items():
        rule = ALERT_RULES.get(metric, {})
        if not rule.get("enabled", True):
            continue

        if result.get("is_drifted", False):
            triggered = True
            rec  = result.get("recent_mean", result.get("recent_rate", "N/A"))
            base = result.get("baseline_mean", result.get("baseline_rate", "N/A"))
            p    = result.get("p_value", "N/A")
            send_alert(
                f"Drift in [{metric}] — "
                f"recent={rec}, baseline={base}, p-value={p}"
            )

    if not triggered:
        print("[Alerts] ✅  All checks passed. No alerts.")


if __name__ == "__main__":
    run_alerts()
