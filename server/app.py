import sys
import os
import uuid
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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


@app.get("/")
def root():
    return {
        "name": "PulmoAlert",
        "version": "2.0.0",
        "description": "ICU oxygen monitoring OpenEnv",
        "tasks": list_tasks(),
        "docs": "/docs",
        "health": "/health",
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
    return {"session_id": session_id, "observation": obs.dict()}


@app.get("/state")
def get_state(session_id: str = Query(...)):
    env = _get_session(session_id)
    return {"observation": env.state().dict()}


@app.post("/step")
def step(action: Action, session_id: str = Query(...)):
    env = _get_session(session_id)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if done:
        _sessions.pop(session_id, None)

    return {
        "observation": obs.dict(),
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


@app.get("/ui", include_in_schema=False)
def serve_ui():
    if os.path.exists(_INDEX):
        return FileResponse(_INDEX)
    return {"detail": "No UI available"}


def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
