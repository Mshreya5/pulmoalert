import sys
import os
import uuid
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env.environment import PulmoAlertEnv
from env.models import Action
from env.grader import grade_episode
from env.tasks import list_tasks

app = FastAPI(
    title="PulmoAlert",
    description="OpenEnv ICU oxygen monitoring — session-isolated REST API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, PulmoAlertEnv] = {}
_INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))


def _get_session(session_id: str) -> PulmoAlertEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset first.",
        )
    return env


@app.get("/", include_in_schema=False)
def root():
    if os.path.exists(_INDEX):
        return FileResponse(_INDEX)
    return {
        "name": "PulmoAlert",
        "version": "2.0.0",
        "description": "ICU oxygen monitoring OpenEnv",
        "tasks": list_tasks(),
    }


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.get("/tasks")
def get_tasks():
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(
    task: str = Query(default="easy", description="Task name: easy | medium | hard"),
    seed: int = Query(default=42, description="Random seed for reproducibility"),
):
    if task not in list_tasks():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Valid: {list_tasks()}",
        )
    session_id = str(uuid.uuid4())
    env = PulmoAlertEnv(task_name=task, seed=seed)
    obs = env.reset()
    _sessions[session_id] = env
    obs_dict = obs.dict()
    obs_dict["time_step"] = obs_dict["time_elapsed"]
    return {"session_id": session_id, "observation": obs_dict}


@app.get("/state")
def get_state(session_id: str = Query(...)):
    env = _get_session(session_id)
    return {"observation": env.state().dict()}


class StepRequest(BaseModel):
    action: str
    session_id: str = ""


@app.post("/step")
def step(
    body: StepRequest,
    session_id: str = Query(default=""),
):
    sid = session_id or body.session_id
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")

    env = _get_session(sid)
    action = Action(action=body.action)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if done:
        _sessions.pop(sid, None)

    obs_dict = obs.dict()
    obs_dict["time_step"] = obs_dict["time_elapsed"]
    return {
        "observation": obs_dict,
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/grade")
def grade(session_id: str = Query(...)):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    summary = grade_episode(env.history, task=env.task_name)
    return summary.dict()


@app.get("/metrics")
def metrics():
    return {
        "active_sessions": len(_sessions),
        "tasks_in_flight": {
            t: sum(1 for e in _sessions.values() if e.task_name == t)
            for t in list_tasks()
        },
    }


@app.delete("/session")
def delete_session(session_id: str = Query(...)):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/info")
def info():
    return {
        "name": "PulmoAlert",
        "version": "2.0.0",
        "description": "ICU oxygen monitoring OpenEnv",
        "tasks": list_tasks(),
    }


def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
