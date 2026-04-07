---
title: PulmoAlert
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# PulmoAlert — OpenEnv Medical Monitoring RL Environment

Production-ready reinforcement learning environment for oxygen cylinder and patient vitals monitoring with multi-difficulty tasks and deterministic grading.

## Overview

PulmoAlert simulates real-world medical support scenarios where an AI agent monitors oxygen levels and patient vitals, making critical decisions to maintain safety while minimizing unnecessary alerts.

**Observation Space:**
- `oxygen_level` (0–100): Oxygen cylinder percentage
- `spo2` (0–100): Patient blood oxygen saturation
- `heart_rate` (bpm): Patient heart rate
- `consumption_rate` (%/step): Oxygen consumption rate
- `time_step` (int): Simulation time step
- `risk_level` (str): low, moderate, high, critical

**Action Space:**
- `WAIT`: No intervention
- `ALERT_REFILL`: Request oxygen refill
- `EMERGENCY_ALERT`: Trigger emergency response
- `NOTIFY_CAREGIVER`: Alert medical staff

**Reward Function:**
- Safe WAIT in low-risk: +1.0
- WAIT under high-risk: -3.0
- Timely refill alert: +2.5
- Unnecessary early refill: -2.0
- Correct emergency detection: +4.0
- False emergency alert: -3.0
- Oxygen depletion: -20.0 (critical failure)

## Tasks

Three difficulty levels with increasing complexity:

| Task | Oxygen Range | Consumption | Noise | Objective |
|------|--------------|-------------|-------|-----------|
| easy | 70–100% | 0.8–1.5%/step | low | Stable management |
| medium | 55–90% | 1.5–2.8%/step | medium | Balancing alerts |
| hard | 40–80% | 2.5–4.5%/step | high | Critical response |

## Running Inference

```bash
cd pulmoalert-openenv
python inference.py
```

**Output Format:**
```text
[START] task=easy
[STEP] step=1 reward=1.000
[STEP] step=2 reward=0.500
[END] task=easy score=0.9375 steps=104
[START] task=medium
[STEP] step=1 reward=1.000
...
[END] task=medium score=0.9375 steps=56
[START] task=hard
...
[END] task=hard score=0.7651 steps=29
```

## Project Structure

```
pulmoalert-openenv/
├── env/
│   ├── environment.py      # PulmoAlertEnv class
│   ├── models.py           # Pydantic data models
│   ├── tasks.py            # Task factory
│   ├── grader.py           # Episode grading
├── inference.py            # Evaluator script
├── server/
│   ├── app.py              # FastAPI server
│   └── __init__.py
├── index.html              # Dashboard frontend
├── pyproject.toml          # Package metadata
├── requirements.txt        # Dependencies
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Container config
└── README.md               # This file
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- pydantic>=1.10.0
- fastapi
- uvicorn
- openai>=0.27.0
- openenv-core

## Grading

Deterministic evaluation (0.0–1.0):
- **Safety** (50%): No oxygen depletion
- **Emergency Precision** (25%): Correct vs false alerts
- **Efficiency** (25%): Minimize unnecessary alerts

## OpenEnv Compliance

✓ Full OpenEnv multi-mode deployment support  
✓ Validated with `openenv validate`  
✓ Dockerized for Hugging Face Spaces  
✓ Structured stdout output for evaluators  

## Dashboard

Optional FastAPI dashboard for interactive testing:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8080
```

Access at: http://localhost:8080

## References

- [OpenEnv Documentation](https://github.com/openenvhub/openenv)
- [Hugging Face Spaces Config](https://huggingface.co/docs/hub/spaces-config-reference)
