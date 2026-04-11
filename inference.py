import os
from env.tasks import list_tasks, get_task
from env.models import Action, ActionType
from env.grader import grade_run
from env.logger import EpisodeLogger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def baseline_policy(obs) -> Action:
    risk_score = {"low": 0.1, "moderate": 0.4, "high": 0.75, "critical": 0.95}
    risk = risk_score.get(obs.risk_level, 0.0)

    if risk > 0.7:
        return Action(action=ActionType.EMERGENCY_ALERT)
    if obs.oxygen_level < 25:
        return Action(action=ActionType.ALERT_REFILL)
    if (obs.risk_level == "moderate" and obs.spo2_trend == -1) or \
       obs.heart_rate > 110 or obs.heart_rate < 55 or obs.respiratory_rate > 25:
        return Action(action=ActionType.NOTIFY_CAREGIVER)
    return Action(action=ActionType.WAIT)


def llm_policy(obs):
    if OpenAI is None:
        return None
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    if not base_url or not api_key:
        return None

    client = OpenAI(base_url=base_url, api_key=api_key)

    system = (
        "You are a clinical decision-support AI monitoring an ICU patient on supplemental oxygen. "
        "Your goal is to take the safest, most proportionate action at each step. "
        "Avoid false alarms. Escalate only when vitals indicate genuine risk."
    )
    user = (
        f"Current vitals:\n"
        f"  O2 cylinder: {obs.oxygen_level:.1f}%  (depletes in ~{obs.predicted_depletion_time:.0f} steps)\n"
        f"  SpO2: {obs.spo2:.1f}%  trend: {'+' if obs.spo2_trend == 1 else ('-' if obs.spo2_trend == -1 else '=')}\n"
        f"  Heart rate: {obs.heart_rate:.0f} bpm\n"
        f"  Respiratory rate: {obs.respiratory_rate:.0f} breaths/min\n"
        f"  Risk level: {obs.risk_level}\n\n"
        "Choose ONE action: WAIT | ALERT_REFILL | EMERGENCY_ALERT | NOTIFY_CAREGIVER\n"
        "Reply with ONLY the action name, nothing else."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip().upper()
        if text in ActionType.__members__:
            return Action(action=ActionType[text])
    except Exception:
        pass
    return None


def run_task(task_name: str, max_steps: int = 120) -> dict:
    env = get_task(task_name)
    obs = env.reset()

    print(f"[START] task={task_name}", flush=True)

    with EpisodeLogger(log_dir="logs", task=task_name) as logger:
        for step_num in range(1, max_steps + 1):
            action = llm_policy(obs) or baseline_policy(obs)
            obs, reward, done, info = env.step(action)

            action_str = action.action if isinstance(action.action, str) else action.action.value
            logger.log_step(obs.dict(), action_str, reward.value, done, info)

            print(f"[STEP] step={step_num} reward={reward.value:.3f}", flush=True)

            if done:
                break

    scores = grade_run([env.history], task_names=[task_name])
    final_score = scores["average_score"]
    total_steps = len(env.history)

    print(f"[END] task={task_name} score={final_score:.2f} steps={total_steps}", flush=True)
    return scores


if __name__ == "__main__":
    for task in list_tasks():
        run_task(task)
