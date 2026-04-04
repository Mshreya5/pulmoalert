import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from env.environment import PulmoAlertEnv
from env.models import Action

app = FastAPI(title="PulmoAlert API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = PulmoAlertEnv(task_name="easy")

_INDEX = os.path.join(os.path.dirname(__file__), "..", "index.html")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.abspath(_INDEX))


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
