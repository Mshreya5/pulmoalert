from env.environment import PulmoAlertEnv

_TASK_SEEDS = {"easy": 100, "medium": 200, "hard": 300}


def get_task(task_name: str) -> PulmoAlertEnv:
    seed = _TASK_SEEDS.get(task_name, 42)
    return PulmoAlertEnv(task_name=task_name, seed=seed)


def list_tasks() -> list:
    return ["easy", "medium", "hard"]
