from typing import List, Dict

from env.models import EpisodeSummary, StepRecord


_WEIGHTS = {"critical": 2.0, "high": 1.5, "moderate": 1.0, "low": 0.5}
_MAX_WEIGHT = 2.0


def grade_episode(history: List[StepRecord], task: str = "unknown") -> EpisodeSummary:
    if not history:
        return EpisodeSummary(
            task=task, total_steps=0, score=0.0,
            total_reward=0.0, false_alarms=0,
            critical_misses=0, oxygen_depletions=0,
        )

    total_steps = len(history)
    total_possible = total_steps * _MAX_WEIGHT
    earned = 0.0
    total_reward = 0.0
    false_alarms = 0
    critical_misses = 0
    oxygen_depletions = 0

    for record in history:
        obs = record.observation if isinstance(record, StepRecord) else _dict_to_obs(record)
        action = record.action if isinstance(record, StepRecord) else record["action"]
        risk = record.risk if isinstance(record, StepRecord) else record.get("info", {}).get("risk", "low")
        reward = record.reward if isinstance(record, StepRecord) else record["reward"]
        done = record.done if isinstance(record, StepRecord) else record["done"]

        oxygen = obs.oxygen_level if hasattr(obs, "oxygen_level") else obs.get("oxygen_level", 100.0)
        spo2 = obs.spo2 if hasattr(obs, "spo2") else obs.get("spo2", 100.0)
        hr = obs.heart_rate if hasattr(obs, "heart_rate") else obs.get("heart_rate", 80.0)
        rr = obs.respiratory_rate if hasattr(obs, "respiratory_rate") else obs.get("respiratory_rate", 16.0)

        weight = _WEIGHTS.get(risk, 0.5)
        total_reward += reward

        if _is_correct(action, risk, oxygen, spo2, hr, rr):
            earned += weight
        if _is_false_alarm(action, risk, oxygen):
            earned -= 1.0
            false_alarms += 1
        if action == "WAIT" and risk == "critical":
            critical_misses += 1
        if done and oxygen <= 0.0:
            oxygen_depletions += 1

    score = earned / total_possible
    return EpisodeSummary(
        task=task,
        total_steps=total_steps,
        score=round(max(0.0, min(1.0, score)), 4),
        total_reward=round(total_reward, 3),
        false_alarms=false_alarms,
        critical_misses=critical_misses,
        oxygen_depletions=oxygen_depletions,
    )


def grade_run(episodes: List, task_names: List[str] = None) -> Dict:
    names = task_names or ["unknown"] * len(episodes)
    summaries = [grade_episode(ep, task=n) for ep, n in zip(episodes, names)]
    scores = [s.score for s in summaries]
    avg = sum(scores) / max(1, len(scores))
    return {
        "average_score": round(avg, 4),
        "episode_scores": scores,
        "summaries": [s.dict() for s in summaries],
    }


def _is_correct(action: str, risk: str, oxygen: float, spo2: float,
                hr: float, rr: float) -> bool:
    if risk == "critical":
        return action in ("EMERGENCY_ALERT", "NOTIFY_CAREGIVER")
    if risk == "high":
        return action in ("ALERT_REFILL", "EMERGENCY_ALERT", "NOTIFY_CAREGIVER")
    if oxygen < 25:
        return action == "ALERT_REFILL"
    if spo2 < 92 or hr > 110 or hr < 55 or rr > 25:
        return action in ("NOTIFY_CAREGIVER", "ALERT_REFILL")
    return action == "WAIT"


def _is_false_alarm(action: str, risk: str, oxygen: float) -> bool:
    if action == "EMERGENCY_ALERT" and risk not in ("critical", "high"):
        return True
    if action == "ALERT_REFILL" and oxygen >= 40 and risk == "low":
        return True
    return False


def _dict_to_obs(record: dict):
    class _Obs:
        pass
    o = _Obs()
    for k, v in record.get("observation", {}).items():
        setattr(o, k, v)
    return o
