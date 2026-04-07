import os
from env.tasks import list_tasks, get_task
from env.models import Action, ActionType
from env.grader import grade_run

try:
    import openai
except ImportError:
    openai = None


def heuristic_policy(obs):
    if obs.risk_level == "critical":
        return Action(action=ActionType.EMERGENCY_ALERT)
    if obs.oxygen_level < 25 or obs.spo2 < 92:
        return Action(action=ActionType.ALERT_REFILL)
    if obs.heart_rate < 55 or obs.heart_rate > 110:
        return Action(action=ActionType.NOTIFY_CAREGIVER)
    return Action(action=ActionType.WAIT)


def run_task(task_name: str, max_steps: int = 120):
    env = get_task(task_name)
    history = []

    print(f"[START] task={task_name}", flush=True)
    obs = env.reset()

    for step_number in range(1, max_steps + 1):
        action = heuristic_policy(obs)
        if openai is not None:
            try:
                ai_action = openai_policy(obs)
                if ai_action is not None:
                    action = ai_action
            except Exception:
                pass

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


def openai_policy(obs):
    if not openai:
        return None

    api_base = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")
    if not api_base or not model_name or not hf_token:
        return None

    openai.api_base = api_base
    openai.api_key = hf_token

    prompt = (
        "You are a PulmoAlert agent. Observation: "
        f"oxygen_level={obs.oxygen_level}, spo2={obs.spo2}, heart_rate={obs.heart_rate}, "
        f"consumption_rate={obs.consumption_rate}, risk_level={obs.risk_level}. "
        "Choose one action from [WAIT, ALERT_REFILL, EMERGENCY_ALERT, NOTIFY_CAREGIVER]. "
        "Respond with the exact action string only."
    )

    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        result = completion.choices[0].message.content.strip().upper()
        if result in ActionType.__members__:
            return Action(action=ActionType[result])
    except Exception:
        return None

    return None


if __name__ == "__main__":
    for task_name in list_tasks():
        run_task(task_name)
