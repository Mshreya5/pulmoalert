import os
from env.tasks import list_tasks, get_task
from env.models import Action, ActionType
from env.grader import grade_run

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# -------------------------
# Heuristic fallback policy
# -------------------------
def heuristic_policy(obs):
    if obs.risk_level == "critical":
        return Action(action=ActionType.EMERGENCY_ALERT)
    if obs.oxygen_level < 25 or obs.spo2 < 92:
        return Action(action=ActionType.ALERT_REFILL)
    if obs.heart_rate < 55 or obs.heart_rate > 110:
        return Action(action=ActionType.NOTIFY_CAREGIVER)
    return Action(action=ActionType.WAIT)


# -------------------------
# LLM policy using proxy
# -------------------------
def llm_policy(obs):
    if OpenAI is None:
        return None

    # Robust environment variable handling
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")

    if not base_url or not api_key:
        return None

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    prompt = (
        f"Patient state:\n"
        f"Oxygen level: {obs.oxygen_level}\n"
        f"SpO2: {obs.spo2}\n"
        f"Heart rate: {obs.heart_rate}\n"
        f"Risk level: {obs.risk_level}\n\n"
        "Choose ONE action from:\n"
        "WAIT, ALERT_REFILL, EMERGENCY_ALERT, NOTIFY_CAREGIVER\n"
        "Respond ONLY with the action name."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )

        print("LLM call made", flush=True)  # Debug proof

        action_text = response.choices[0].message.content.strip().upper()

        if action_text in ActionType.__members__:
            return Action(action=ActionType[action_text])

    except Exception:
        return None

    return None


# -------------------------
# Run a single task
# -------------------------
def run_task(task_name: str, max_steps: int = 120):
    env = get_task(task_name)
    history = []

    print(f"[START] task={task_name}", flush=True)
    obs = env.reset()

    for step_number in range(1, max_steps + 1):

        # Try LLM first
        action = llm_policy(obs)

        # Fallback if LLM fails
        if action is None:
            action = heuristic_policy(obs)

        obs, reward, done, info = env.step(action)

        history.append({
            "observation": obs.dict(),
            "action": action.action.value,
            "reward": reward.value,
            "done": done,
            "info": info,
        })

        print(f"[STEP] step={step_number} reward={reward.value:.3f}", flush=True)

        if done:
            break

    scores = grade_run([history])
    final_score = scores["average_score"]
    total_steps = len(history)

    print(f"[END] task={task_name} score={final_score:.4f} steps={total_steps}", flush=True)

    return scores


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    for task_name in list_tasks():
        run_task(task_name)