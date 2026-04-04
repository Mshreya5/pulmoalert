import os
import time
import logging

from env.tasks import list_tasks, get_task
from env.models import Action, ActionType
from env.grader import grade_run

try:
    import openai
except ImportError:
    openai = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pulmoalert")


def heuristic_policy(obs):
    if obs.risk_level == "critical":
        return Action(action=ActionType.EMERGENCY_ALERT)
    if obs.oxygen_level < 25 or obs.spo2 < 92:
        return Action(action=ActionType.ALERT_REFILL)
    if obs.heart_rate < 55 or obs.heart_rate > 110:
        return Action(action=ActionType.NOTIFY_CAREGIVER)
    return Action(action=ActionType.WAIT)


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


def run_task(task_name: str, episodes: int = 20, max_steps: int = 120):
    env = get_task(task_name)
    episode_histories = []
    total_reward = 0.0
    logger.info("[START] task=%s episodes=%d", task_name, episodes)

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        history = []
        for step in range(max_steps):
            action = heuristic_policy(obs)
            if openai is not None:
                ai_action = openai_policy(obs)
                if ai_action is not None:
                    action = ai_action
            obs, r, done, info = env.step(action)
            ep_reward += r.value
            logger.info(
                "[STEP] task=%s episode=%d step=%d action=%s reward=%.3f done=%s",
                task_name,
                ep + 1,
                step + 1,
                action.action.value,
                r.value,
                done,
            )
            history.append({
                "observation": obs.model_dump(),
                "action": action.action.value,
                "reward": r.value,
                "done": done,
                "info": info,
            })
            if done:
                break
        total_reward += ep_reward
        episode_histories.append(history)

    scores = grade_run(episode_histories)
    avg_reward = total_reward / episodes
    logger.info(
        "[END] task=%s avg_reward=%.3f avg_score=%.4f",
        task_name,
        avg_reward,
        scores["average_score"],
    )
    return scores


if __name__ == "__main__":
    logger.info("[START] PulmoAlert inference run")
    start = time.time()
    results = {}
    for task in list_tasks():
        results[task] = run_task(task, episodes=12, max_steps=150)
    logger.info("[END] PulmoAlert inference complete runtime=%.2fs", time.time() - start)
