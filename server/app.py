from fastapi import FastAPI, HTTPException

from env.environment import PulmoAlertEnv
from env.models import Action

app = FastAPI(title="PulmoAlert API")

env = PulmoAlertEnv(task_name="easy")


@app.post("/reset")
def reset():
    observation = env.reset()
    return {"observation": observation.model_dump()}


@app.post("/step")
def step(action: Action):
    try:
        observation, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


def main():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
