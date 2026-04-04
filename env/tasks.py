from env.environment import PulmoAlertEnv


def get_task(task_name: str) -> PulmoAlertEnv:
    """Return a configured environment for the requested difficulty."""
    return PulmoAlertEnv(task_name=task_name, seed=1234)


def list_tasks() -> list:
    return ["easy", "medium", "hard"]
