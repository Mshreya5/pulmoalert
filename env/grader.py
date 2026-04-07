from typing import List, Dict


def grade_episode(history: List[Dict]) -> float:
    if not history:
        return 0.0

    total_steps = len(history)
    oxygen_depleted = any(h["observation"]["oxygen_level"] <= 0.0 for h in history)
    if oxygen_depleted:
        safety = 0.0
    else:
        safety = 1.0

    correct_emergency = 0
    false_emergency = 0
    proactive_refill = 0
    unnecessary_alerts = 0

    for h in history:
        obs = h["observation"]
        action = h["action"]
        risk = obs.get("risk_level", "low")

        if action == "EMERGENCY_ALERT":
            if risk == "critical":
                correct_emergency += 1
            else:
                false_emergency += 1

        if action == "ALERT_REFILL":
            if obs["oxygen_level"] < 25:
                proactive_refill += 1
            else:
                unnecessary_alerts += 1

        if action == "NOTIFY_CAREGIVER" and risk == "low":
            unnecessary_alerts += 1

    emergency_precision = correct_emergency / (correct_emergency + false_emergency + 1e-8)
    emergency_recall = correct_emergency / max(1, 1 + int(any(h["observation"]["risk_level"] == "critical" for h in history)))
    emergency_score = (emergency_precision + emergency_recall) / 2.0

    refill_efficiency = max(0.0, 1.0 - unnecessary_alerts / max(1, total_steps * 0.25))
    alert_efficiency = max(0.0, 1.0 - (false_emergency + unnecessary_alerts) / max(1, total_steps * 0.2))

    score = 0.5 * safety + 0.25 * emergency_score + 0.25 * min(refill_efficiency, alert_efficiency)
    return round(max(0.0, min(1.0, score)), 4)


def grade_run(episodes: List[List[Dict]]) -> Dict[str, float]:
    scores = [grade_episode(ep) for ep in episodes]
    avg = sum(scores) / max(1, len(scores))
    return {"average_score": round(avg, 4), "episode_scores": scores}
